"""Held-out analysis under FINAL_PROTOCOL_V1: judged effectiveness, primary contrasts, RQ1 equality.

The primary family, panel rule, contrast, inference settings and margin are fixed here and
in FINAL_PROTOCOL_V1.md. The same core also runs on development queries as a pipeline test
(`dry-run`); that output carries no significance or generalization claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile

import numpy as np

from .evaluation import ann_recall, evaluate, read_run, validate_run
from .paired_inference import holm_adjust, paired_query_estimate
from .release import MODELS, checksum, manifest_entries, parse_qrels
from .storage import atomic_json

PRIMARY_FAMILY = 'gpt-oss-120b'
FAMILIES = ('gpt-oss-120b', 'raw', 'dctr', 'human', 'gpt-oss-20b')
MARGIN = 0.01
ALPHA = 0.05
RNG_SEED = 20261119
BOOTSTRAP_DRAWS = 10_000
RANDOMIZATION_DRAWS = 100_000
SEEDS = (11, 29, 47)
SNAPSHOTS = (1, 2, 3)
DEPTH = 100
HNSW_MODES = ('incremental', 'replay')
IVF_MODES = ('frozen_incremental', 'frozen_rebuild', 'refreshed_rebuild')
HNSW_REGIMES = ('frozen', 'matched_recall')
IVF_REGIMES = ('frozen_snapshot1', 'current_calibration', 'all_lists')
# Primary Holm family: calibration-selected settings, incrementally maintained indexes.
PRIMARY_ARMS = {'epzero_hnsw': ('epzero', 'incremental', 'matched_recall'),
                'faiss_hnsw': ('faiss_hnsw', 'incremental', 'matched_recall'),
                'ivf_frozen': ('faiss_ivf', 'frozen_incremental', 'current_calibration')}
SECONDARY_ARMS = {'epzero_hnsw_fixed_ef': ('epzero', 'incremental', 'frozen'),
                  'faiss_hnsw_fixed_ef': ('faiss_hnsw', 'incremental', 'frozen'),
                  'ivf_frozen_fixed_nprobe': ('faiss_ivf', 'frozen_incremental', 'frozen_snapshot1'),
                  'ivf_refreshed': ('faiss_ivf', 'refreshed_rebuild', 'current_calibration')}
EQUALITY_PAIRS = ((('epzero', 'incremental'), ('epzero', 'replay')),
                  (('faiss_hnsw', 'incremental'), ('faiss_hnsw', 'replay')),
                  (('faiss_ivf', 'frozen_incremental'), ('faiss_ivf', 'frozen_rebuild')))


def expected_conditions():
    hnsw = {(b, m, s) for b in ('epzero', 'faiss_hnsw') for m in HNSW_MODES for s in SEEDS}
    ivf = {('faiss_ivf', m, s) for m in IVF_MODES for s in SEEDS}
    return hnsw | ivf


def load_test_qrels(release):
    """All released test judgment families, checksum-pinned; LLM members read in memory only."""
    release = Path(release)
    entries = manifest_entries(release)
    qrels, identity = {}, {}
    for family in ('raw', 'dctr'):
        qrels[family] = {}
        for step in SNAPSHOTS:
            item = entries[f'test_qrels_snapshot_{step}_{family}']
            path = release / item['cache_path']
            if checksum(path, 'md5') != item['expected_md5']:
                raise ValueError('test judgment identity mismatch')
            qrels[family][step] = parse_qrels(path.read_text(encoding='utf-8-sig'))
            identity[f'{family}/snapshot-{step}'] = checksum(path)
    item = entries['test_qrels_snapshot_3_human']
    path = release / item['cache_path']
    if checksum(path, 'md5') != item['expected_md5']:
        raise ValueError('human judgment identity mismatch')
    qrels['human'] = {3: parse_qrels(path.read_text(encoding='utf-8-sig'))}
    identity['human/snapshot-3'] = checksum(path)
    item = entries['llm-synthetic-qrels']
    archive = release / item['cache_path']
    if checksum(archive, 'md5') != item['expected_md5']:
        raise ValueError('LLM judgment archive identity mismatch')
    with zipfile.ZipFile(archive) as z:
        for model in ('gpt-oss-120b', 'gpt-oss-20b'):
            qrels[model] = {}
            for step in SNAPSHOTS:
                name = f'llm_synthetic_qrels/{MODELS[model]}-umbrella_zeroshot_no_desc_no_narrative-snapshot-{step}.qrels.txt'
                if z.getinfo(name).file_size > 16 << 20:
                    raise ValueError('judgment member exceeds size budget')
                raw = z.read(name)
                qrels[model][step] = parse_qrels(raw.decode('utf-8-sig'))
                identity[f'{model}/snapshot-{step}'] = hashlib.sha256(raw).hexdigest()
    return qrels, identity


def panel(qrels_by_snapshot, query_ids):
    """Fixed panel: queries with a positive judgment at every judged snapshot of the family."""
    return sorted(q for q in query_ids
                  if all(any(r > 0 for r in qrels_by_snapshot[s].get(q, {}).values()) for s in qrels_by_snapshot))


def score(run, qrels, queries):
    """Local metrics with independent pytrec_eval parity on nDCG@10 and R@100."""
    import pytrec_eval
    local = evaluate({q: run.get(q, []) for q in queries}, qrels, queries, population='explicit', ndcg_k=10, recall_k=DEPTH)
    judged = {q: qrels.get(q, {}) for q in queries}
    other = pytrec_eval.RelevanceEvaluator(judged, {'ndcg_cut.10', 'recall.100'}).evaluate(
        {q: dict(run[q]) for q in queries if run.get(q)})
    for q in queries:
        for ours, theirs in (('nDCG@10', 'ndcg_cut_10'), ('R@100', 'recall_100')):
            if not np.isclose(local['per_query'][q][ours], other.get(q, {}).get(theirs, 0.), atol=1e-12, rtol=0):
                raise ValueError('independent evaluator parity failed')
    return local['per_query']


def _estimate(query_ids, seed_ids, values):
    return paired_query_estimate(query_ids, list(seed_ids), values, bootstrap_draws=BOOTSTRAP_DRAWS,
                                 randomization_draws=RANDOMIZATION_DRAWS, rng_seed=RNG_SEED)


def _summary(estimate):
    return {k: estimate[k] for k in ('query_count', 'seed_count', 'mean_difference', 'percentile_interval',
                                     'two_sided_randomization_p', 'difference_bytes_sha256')}


def decision(interval, holm_p, margin=MARGIN, alpha=ALPHA):
    low, high = interval
    within = -margin <= low and high <= margin
    significant = holm_p < alpha
    label = 'negligible' if within else ('changed' if significant else 'inconclusive')
    return {'within_margin': within, 'holm_significant': significant, 'label': label,
            'rule': f'negligible iff the whole 95% interval lies within +/-{margin}; '
                    f'changed iff not negligible and Holm p < {alpha}; otherwise inconclusive'}


def ann_summary(run, reference):
    values = np.array(list(ann_recall(run, reference).values()))
    worst = np.sort(values)[:max(1, len(values) // 10)]
    return {'mean': float(values.mean()), 'worst_decile_mean': float(worst.mean()),
            'min': float(values.min()), 'zero_recall_queries': int((values == 0).sum()),
            'below_0.5_queries': int((values < .5).sum())}


def analyze_core(runs, exact, qrels, query_ids, *, primary_family, families, equality=None, context=None):
    """runs[(backend, mode, seed, snapshot, regime)] -> run; exact[snapshot] -> run; qrels[family][snapshot].

    Seeds are averaged within query by paired_query_estimate; queries are the resampling unit.
    """
    if primary_family not in families or set(families) - set(qrels):
        raise ValueError('primary family and families must have judgments')
    conditions = sorted({k[:3] for k in runs})
    result = {'context': context or {}, 'primary_family': primary_family, 'families': list(families),
              'margin': MARGIN, 'alpha': ALPHA, 'rng_seed': RNG_SEED, 'bootstrap_draws': BOOTSTRAP_DRAWS,
              'randomization_draws': RANDOMIZATION_DRAWS, 'panels': {}, 'exact_trend': {}, 'ann': {},
              'primary': {}, 'secondary': {}, 'equality': equality}
    rows, metrics, exact_metrics = [], {}, {}
    for family in families:
        population = panel(qrels[family], query_ids)
        result['panels'][family] = {'count': len(population), 'query_ids': population,
                                    'snapshots': sorted(qrels[family]),
                                    'rule': 'positive judgment at every judged snapshot of this family'}
        if len(population) < 2:
            raise ValueError(f'panel too small for {family}')
        for step in qrels[family]:
            exact_metrics[family, step] = score(exact[step], qrels[family][step], population)
            for q, values in exact_metrics[family, step].items():
                rows.append({'family': family, 'backend': 'exact', 'mode': 'exact', 'seed': 0, 'snapshot': step,
                             'regime': 'exact', 'query_id': q, **values})
            for key, run in runs.items():
                if key[3] != step:
                    continue
                metrics[family, key] = score(run, qrels[family][step], population)
                for q, values in metrics[family, key].items():
                    rows.append({'family': family, 'backend': key[0], 'mode': key[1], 'seed': key[2],
                                 'snapshot': step, 'regime': key[4], 'query_id': q, **values,
                                 'ANN_minus_exact_nDCG@10': values['nDCG@10'] - exact_metrics[family, step][q]['nDCG@10']})
        # Exact-search trend: a property of representation and labels, not of the index.
        means = {step: float(np.mean([exact_metrics[family, step][q]['nDCG@10'] for q in population])) for step in qrels[family]}
        trend = {'nDCG@10_mean': means,
                 'judged@10_mean': {step: float(np.mean([exact_metrics[family, step][q]['judged@10'] for q in population]))
                                    for step in qrels[family]}}
        if {1, 3} <= set(qrels[family]):
            change = np.array([[exact_metrics[family, 3][q]['nDCG@10'] - exact_metrics[family, 1][q]['nDCG@10']] for q in population])
            trend['snapshot3_minus_snapshot1'] = _summary(_estimate(population, ['exact'], change))
        result['exact_trend'][family] = trend
    for key, run in sorted(runs.items()):
        result['ann'][' '.join(map(str, key))] = ann_summary(run, {q: exact[key[3]][q] for q in run})

    def gap(family, arm, step, q, seed):
        backend, mode, regime = arm
        return metrics[family, (backend, mode, seed, step, regime)][q]['nDCG@10'] - exact_metrics[family, step][q]['nDCG@10']

    def arm_seeds(arm):
        return [s for s in SEEDS if (arm[0], arm[1], s) in conditions]

    for family in families:
        population = result['panels'][family]['query_ids']
        steps = sorted(qrels[family])
        block = {}
        for name, arm in {**PRIMARY_ARMS, **SECONDARY_ARMS}.items():
            seeds = arm_seeds(arm)
            if not seeds:
                continue
            item = {'arm': list(arm), 'seeds': seeds, 'gap_by_snapshot': {}}
            for step in steps:
                values = np.array([[gap(family, arm, step, q, s) for s in seeds] for q in population])
                item['gap_by_snapshot'][step] = _summary(_estimate(population, seeds, values))
            if {1, 3} <= set(steps):
                values = np.array([[gap(family, arm, 3, q, s) - gap(family, arm, 1, q, s) for s in seeds] for q in population])
                item['gap_change_D'] = _summary(_estimate(population, seeds, values))
            block[name] = item
        if family == primary_family:
            names = [n for n in PRIMARY_ARMS if n in block]
            if names != list(PRIMARY_ARMS):
                raise ValueError('primary family requires all three primary arms')
            adjusted = holm_adjust([block[n]['gap_change_D']['two_sided_randomization_p'] for n in names])
            for name, p in zip(names, adjusted):
                result['primary'][name] = {**block[name]['gap_change_D'], 'arm': block[name]['arm'],
                                           'holm_p': p, **decision(block[name]['gap_change_D']['percentile_interval'], p)}
        result['secondary'][family] = block
    return result, rows


def _run_sha(folder, item):
    path = folder / item['file']
    if checksum(path) != item['sha256']:
        raise ValueError(f'run checksum mismatch: {path}')
    return path


def collect(report_paths, role, query_ids, exact, *, expected=None, config_sha256=None, inputs_sha256=None):
    """Verify returned evidence and gather runs; recompute ANN recall against the exact references."""
    expected = expected_conditions() if expected is None else expected
    reports, runs, excluded = {}, {}, []
    for path in map(Path, report_paths):
        report = json.loads(path.read_text())
        if report.get('evaluation_role', 'development') != role or 'backend' not in report:
            continue
        identity = (report['backend'], report['mode'], report['seed'])
        if identity not in expected:
            continue
        if report['status'] != 'completed':
            # Failed attempts are reported, never analyzed.
            excluded.append({'condition': list(identity), 'status': report['status'],
                             'error_type': report.get('error_type'), 'report': path.as_posix()})
            continue
        if identity in reports:
            raise ValueError(f'duplicate evidence for {identity}')
        if [s['snapshot'] for s in report['snapshots']] != list(SNAPSHOTS):
            raise ValueError(f'incomplete evidence for {identity}')
        held = report['provenance'].get('heldout_inputs', {})
        if config_sha256 is not None and held.get('config_sha256') != config_sha256[report['backend'] == 'faiss_ivf']:
            raise ValueError(f'configuration differs from the frozen one: {identity}')
        if inputs_sha256 is not None and any(held.get(k) != v for k, v in inputs_sha256.items()):
            raise ValueError(f'held-out inputs differ: {identity}')
        reports[identity] = report
        for stage in report['snapshots']:
            step = stage['snapshot']
            for regime, item in stage['runs'].items():
                run = read_run(_run_sha(path.parent, item))
                feasible = item.get('feasible', True)
                if feasible:
                    # Liveness was checked in-job against the snapshot; here: population, depth, uniqueness, finiteness.
                    validate_run(run, query_ids, {d for pairs in run.values() for d, _ in pairs}, DEPTH)
                elif set(run) - set(query_ids):
                    raise ValueError('unknown query in infeasible run')
                run = {q: run.get(q, []) for q in query_ids}
                recall = ann_recall(run, {q: exact[step][q] for q in query_ids})
                if any(recall[q] != item['ann_recall_per_query'][q] for q in query_ids):
                    raise ValueError(f'reported ANN recall differs from recount: {identity} {step} {regime}')
                runs[(*identity, step, regime)] = run
    missing = expected - reports.keys()
    if missing:
        raise ValueError(f'missing evidence: {sorted(missing)}')
    return reports, runs, excluded


def equality_checks(reports, runs):
    """RQ1: incremental vs identical-history rebuild must be identical in state and output."""
    ignore = {'parent_state_sha256', 'state_sha256'}
    out = []
    for left, right in EQUALITY_PAIRS:
        for seed in SEEDS:
            a, b = reports.get((*left, seed)), reports.get((*right, seed))
            if a is None or b is None:
                continue
            for sa, sb in zip(a['snapshots'], b['snapshots']):
                step = sa['snapshot']
                if left[0] == 'faiss_ivf':
                    keys = ('serialized_index_sha256', 'live_ids_sha256', 'insertion_order_sha256')
                    state = all(sa[k] == sb[k] for k in keys)
                    regimes = IVF_REGIMES
                else:
                    state = {k: v for k, v in sa['state'].items() if k not in ignore} == \
                            {k: v for k, v in sb['state'].items() if k not in ignore}
                    regimes = HNSW_REGIMES
                outputs = all(runs[(*left, seed, step, r)] == runs[(*right, seed, step, r)] for r in regimes)
                out.append({'pair': [list(left), list(right)], 'seed': seed, 'snapshot': step,
                            'state_identical': bool(state), 'rankings_and_scores_identical': bool(outputs)})
    return {'comparisons': len(out), 'all_identical': all(x['state_identical'] and x['rankings_and_scores_identical'] for x in out),
            'details': out}


def write_outputs(output, result, rows, extra):
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'per-query.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in (
            'family', 'backend', 'mode', 'seed', 'snapshot', 'regime', 'query_id'), k)))
        writer.writeheader()
        writer.writerows(rows)
    report = {**extra, **result, 'source_sha256': checksum(__file__),
              'paired_inference_sha256': checksum(Path(__file__).with_name('paired_inference.py')),
              'evaluator_version': importlib.metadata.version('pytrec-eval-terrier'),
              'artifacts': {'per-query.csv': checksum(output / 'per-query.csv')}, 'status': 'completed'}
    atomic_json(output / 'report.json', report)
    return report


def analyze_heldout(inputs, release, evidence, output, *, hnsw_config, ivf_config):
    from .heldout import ROLE, load_heldout
    inputs, release, evidence, output = map(Path, (inputs, release, evidence, output))
    queries, refs, receipts, loaded = load_heldout(inputs)
    query_ids = queries[ROLE].ids
    exact = {step: {q: refs[step - 1][q] for q in query_ids} for step in SNAPSHOTS}
    reports, runs, excluded = collect(sorted(evidence.rglob('report.json')), ROLE, query_ids, exact,
                            config_sha256=(checksum(hnsw_config), checksum(ivf_config)),
                            inputs_sha256={k: loaded[k] for k in ('test_inputs_sha256', 'test_encoding_sha256')})
    binaries = {r['provenance']['epzero_binary_sha256'] for r in reports.values()}
    qrels, identity = load_test_qrels(release)
    result, rows = analyze_core(runs, exact, qrels, query_ids, primary_family=PRIMARY_FAMILY, families=FAMILIES,
                                equality=equality_checks(reports, runs),
                                context={'kind': 'held-out evaluation under FINAL_PROTOCOL_V1',
                                         'query_count': len(query_ids)})
    extra = {'judgment_sha256': identity, 'inputs': loaded, 'epzero_binaries': sorted(binaries),
             'conditions': len(reports), 'runs': len(runs), 'excluded_attempts': excluded}
    return write_outputs(output, result, rows, extra)


def dry_run(root, output):
    """Pipeline test on the 50 development queries and verified development evidence.

    Training judgments exist for snapshot 1 only; they are reused at every snapshot here to
    exercise the code. The numbers are not a result of any kind.
    """
    from .query_diagnostic import validate_inputs
    root, output = Path(root), Path(output)
    build = root / 'build'
    contract, _ = validate_inputs(build / 'longeval-training-query-inputs-01')
    query_ids = contract['roles']['development']
    exact = {}
    for step in SNAPSHOTS:
        folder = build / 'longeval-query-sensitivity-01' / f'snapshot-{step}'
        receipt = json.loads((folder / 'report.json').read_text())
        if receipt['artifacts']['float32.run'] != checksum(folder / 'float32.run'):
            raise ValueError('development exact reference checksum mismatch')
        run = read_run(folder / 'float32.run')
        exact[step] = {q: run[q] for q in query_ids}
    paths = sorted(build.glob('longeval-pace-trajectory-evidence-*/*/trajectory/report.json'))
    paths += sorted(build.glob('longeval-pace-ivf-evidence-*/*/experiment/*/report.json'))
    reports, runs, excluded = collect(paths, 'development', query_ids, exact)
    entries = manifest_entries(build / 'longeval-release/files')
    qrels = {}
    for family in ('raw', 'dctr'):
        item = entries[f'longeval_sci_training_2026_qrels_{family}']
        path = build / 'longeval-release/files' / item['cache_path']
        if checksum(path, 'md5') != item['expected_md5']:
            raise ValueError('training judgment identity mismatch')
        judged = parse_qrels(path.read_text(encoding='utf-8-sig'))
        qrels[family] = {step: judged for step in SNAPSHOTS}
    result, rows = analyze_core(runs, exact, qrels, query_ids, primary_family='raw', families=('raw', 'dctr'),
                                equality=equality_checks(reports, runs),
                                context={'kind': 'PIPELINE TEST ONLY on development queries; snapshot-1 training '
                                                 'judgments reused at every snapshot; no claims', 'query_count': len(query_ids)})
    return write_outputs(output, result, rows, {'pipeline_test_only': True, 'conditions': len(reports), 'runs': len(runs),
                                                'excluded_attempts': excluded})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('heldout')
    for name in ('inputs', 'release', 'evidence', 'output', 'hnsw_config', 'ivf_config'):
        p.add_argument(name, type=Path)
    p = sub.add_parser('dry-run')
    p.add_argument('output', type=Path)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    a = parser.parse_args()
    if a.command == 'heldout':
        r = analyze_heldout(a.inputs, a.release, a.evidence, a.output, hnsw_config=a.hnsw_config, ivf_config=a.ivf_config)
    else:
        r = dry_run(a.root, a.output)
    print(json.dumps({'status': r['status'], 'primary': {k: {x: v[x] for x in ('mean_difference', 'percentile_interval',
                      'holm_p', 'label')} for k, v in r['primary'].items()},
                      'equality': {k: r['equality'][k] for k in ('comparisons', 'all_identical')}}, indent=2))


if __name__ == '__main__':
    main()
