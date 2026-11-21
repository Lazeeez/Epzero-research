"""Post-freeze EXPLORATORY robustness analyses of the held-out evidence (not part of FINAL_PROTOCOL_V1).

1. Official baseline: does our exact search reproduce the organisers' published nDCG@10 for
   `baseline-qwen3-4b` (CEUR Vol-4283, paper 374, Table 3)? This validates the pipeline externally.
2. Leaderboard impact: where would that baseline rank among the 21 other official systems, and
   what Relative Change (RC, the lab's robustness measure) would it get, if it were served
   through each approximate index instead of exact search?
3. Tail: the per-query distribution of the ANN-minus-exact nDCG@10 difference behind the mean.
4. Calibration: are the training (calibration/development) queries easier for the index than the
   test queries, how large must a calibration set be to detect the recall drift, how much
   smaller is the uncertainty of the paired change between snapshots, and what safety margin on
   the recall target would a small calibration set need?
5. Search width: at each fixed ef of the held-out sweep, does the effectiveness gap still change
   between snapshots 1 and 3? This separates the implementation from its calibration outcome.

Everything here describes existing, verified evidence. No new retrieval is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import evaluate
from .heldout import ROLE, load_heldout
from .heldout_analysis import SEEDS, SNAPSHOTS, collect, load_test_qrels
from .release import checksum
from .storage import atomic_json

# nDCG@10 from the extended overview of the CLEF 2026 LongEval lab (CEUR Vol-4283, paper 374, Table 3),
# snapshots 1-3. `baseline-qwen3-4b` is exact search over the released Qwen3-Embedding-4B vectors.
OFFICIAL = {
    'custom-lexicalft-tc': {'dctr': (.300, .284, .195), 'gpt-oss-120b': (.698, .709, .713)},
    'custom-lexical-FT-T': {'dctr': (.300, .284, .195), 'gpt-oss-120b': (.698, .709, .713)},
    'Log-boost-03': {'dctr': (.263, .253, .182), 'gpt-oss-120b': (.648, .711, .747)},
    'Dense-Fusion-Run': {'dctr': (.326, .274, .182), 'gpt-oss-120b': (.746, .752, .756)},
    'binary-boost-09': {'dctr': (.257, .241, .168), 'gpt-oss-120b': (.646, .712, .748)},
    'custom-lexical-FT': {'dctr': (.290, .256, .166), 'gpt-oss-120b': (.706, .714, .717)},
    'bm25-ft-additive': {'dctr': (.291, .257, .166), 'gpt-oss-120b': (.706, .714, .717)},
    'bm25-ft-router': {'dctr': (.290, .256, .166), 'gpt-oss-120b': (.707, .714, .717)},
    'temporal-fusion': {'dctr': (.277, .268, .161), 'gpt-oss-120b': (.745, .750, .752)},
    'QwenTemporalIR': {'dctr': (.264, .249, .160), 'gpt-oss-120b': (.761, .769, .769)},
    'b-bm25-dense-rerank-v2': {'dctr': (.088, .260, .155), 'gpt-oss-120b': (.200, .752, .754)},
    'b-bm25-dense-rerank': {'dctr': (.089, .264, .154), 'gpt-oss-120b': (.200, .752, .755)},
    'baseline-bm25': {'dctr': (.279, .254, .150), 'gpt-oss-120b': (.743, .748, .751)},
    'kq-rm3-pl2': {'dctr': (.276, .233, .142), 'gpt-oss-120b': (.803, .811, .812)},
    'kq-rm3-bm25': {'dctr': (.273, .219, .139), 'gpt-oss-120b': (.814, .823, .823)},
    'rm3-conservative': {'dctr': (.271, .249, .134), 'gpt-oss-120b': (.741, .750, .753)},
    'bm25-dense-rerank': {'dctr': (.225, .215, .134), 'gpt-oss-120b': (.846, .850, .848)},
    'baseline-qwen3-4b': {'dctr': (.234, .201, .133), 'gpt-oss-120b': (.791, .796, .800)},
    'put-to-front': {'dctr': (.261, .217, .132), 'gpt-oss-120b': (.914, .926, .925)},
    'ce-reranked': {'dctr': (.218, .215, .109), 'gpt-oss-120b': (.679, .690, .697)},
    'Timely-Boost-V2': {'dctr': (.130, .106, .108), 'gpt-oss-120b': (.525, .590, .598)},
    'kq-rm3-dirichlet': {'dctr': (.181, .123, .079), 'gpt-oss-120b': (.666, .672, .656)},
}
BASELINE = 'baseline-qwen3-4b'
ARMS = {'epzero_hnsw_ef200': ('epzero', 'incremental', 'frozen'),
        'faiss_hnsw_ef200': ('faiss_hnsw', 'incremental', 'frozen'),
        'faiss_hnsw_calibrated': ('faiss_hnsw', 'incremental', 'matched_recall'),
        'ivf_frozen_calibrated': ('faiss_ivf', 'frozen_incremental', 'current_calibration')}


def official_ndcg(run, qrels):
    """Mean nDCG@10 over every query the family judged at this snapshot (the official population)."""
    queries = sorted(qrels)
    scores = evaluate({q: run.get(q, []) for q in queries}, qrels, queries, population='explicit')['per_query']
    return float(np.mean([scores[q]['nDCG@10'] for q in queries])), len(queries)


def relative_change(first, later):
    """The lab's robustness measure: 0 is stable, positive is impaired."""
    return (first - later) / first


def leaderboard(runs, exact, qrels):
    out = {'reproduction': {}, 'arms': {}}
    for family in ('gpt-oss-120b', 'dctr'):
        ours = [official_ndcg(exact[s], qrels[family][s]) for s in SNAPSHOTS]
        official = OFFICIAL[BASELINE][family]
        out['reproduction'][family] = {
            'ours': [round(v, 4) for v, _ in ours], 'official': list(official), 'queries': [n for _, n in ours],
            'max_abs_difference': float(max(abs(v - o) for (v, _), o in zip(ours, official))),
            'relative_change_ours': [relative_change(ours[0][0], ours[i][0]) for i in (1, 2)],
            'relative_change_official': [relative_change(official[0], official[i]) for i in (1, 2)]}
        others = {name: scores[family] for name, scores in OFFICIAL.items() if name != BASELINE}
        for arm, (backend, mode, regime) in ARMS.items():
            values = np.array([[official_ndcg(runs[(backend, mode, seed, s, regime)], qrels[family][s])[0]
                                for s in SNAPSHOTS] for seed in SEEDS])
            mean = values.mean(axis=0)

            def rank(value, i):
                return int(1 + sum(score[i] > value for score in others.values()))
            out['arms'].setdefault(arm, {})[family] = {
                'ndcg_by_snapshot': mean.round(4).tolist(),
                'difference_from_exact': (mean - np.array([v for v, _ in ours])).round(4).tolist(),
                'rank_exact': [rank(ours[i][0], i) for i in range(3)],
                'rank_ann_mean': [rank(mean[i], i) for i in range(3)],
                'rank_ann_by_seed': [[rank(values[k, i], i) for i in range(3)] for k in range(len(SEEDS))],
                'relative_change': [relative_change(mean[0], mean[i]) for i in (1, 2)],
                'relative_change_exact': out['reproduction'][family]['relative_change_ours']}
    return out


def tail(runs, exact, qrels, population):
    """How the mean gap is distributed over queries (seeds averaged within query)."""
    out = {}
    judged = qrels['gpt-oss-120b']
    for arm, (backend, mode, regime) in ARMS.items():
        out[arm] = {}
        for s in SNAPSHOTS:
            base = evaluate(exact[s], judged[s], population, population='explicit')['per_query']
            per_seed = []
            for seed in SEEDS:
                scores = evaluate(runs[(backend, mode, seed, s, regime)], judged[s], population,
                                  population='explicit')['per_query']
                per_seed.append([scores[q]['nDCG@10'] - base[q]['nDCG@10'] for q in population])
            diffs = np.mean(per_seed, axis=0)
            order = np.sort(diffs)
            worst = order[:max(1, len(order) // 20)]
            out[arm][s] = {'mean': float(diffs.mean()), 'unchanged_share': float(np.mean(np.abs(diffs) < 1e-12)),
                           'worse_share': float(np.mean(diffs < -1e-12)), 'better_share': float(np.mean(diffs > 1e-12)),
                           'loss_over_0.05_share': float(np.mean(diffs < -.05)),
                           'loss_over_0.10_share': float(np.mean(diffs < -.10)),
                           'worst_5pct_mean': float(worst.mean()), 'min': float(order[0]),
                           'share_of_total_loss_from_worst_5pct': float(worst.sum() / diffs[diffs < 0].sum())}
    return out


def _recall_table(paths, role, regime='frozen'):
    """{(backend, seed, snapshot): {query: recall}} for incremental HNSW runs at fixed ef."""
    table = {}
    for path in map(Path, paths):
        report = json.loads(path.read_text())
        if (report.get('status') != 'completed' or report.get('mode') != 'incremental'
                or report.get('evaluation_role', 'development') != role or report.get('backend') == 'faiss_ivf'):
            continue
        for stage in report['snapshots']:
            table[(report['backend'], report['seed'], stage['snapshot'])] = stage['runs'][regime]['ann_recall_per_query']
    return table


def query_shift(dev_paths, test_paths, *, draws=20_000, seed=20261120):
    """Mean neighbour recall at fixed ef 200 on training (development) queries vs test queries."""
    dev, test = _recall_table(dev_paths, 'development'), _recall_table(test_paths, ROLE)
    rng = np.random.default_rng(seed)
    out = []
    for backend in ('epzero', 'faiss_hnsw'):
        for s in SNAPSHOTS:
            a = np.mean([[dev[(backend, k, s)][q] for q in sorted(dev[(backend, k, s)])] for k in SEEDS], axis=0)
            b = np.mean([[test[(backend, k, s)][q] for q in sorted(test[(backend, k, s)])] for k in SEEDS], axis=0)
            boot = (a[rng.integers(0, len(a), (draws, len(a)))].mean(axis=1)
                    - b[rng.integers(0, len(b), (draws, len(b)))].mean(axis=1))
            pooled = np.concatenate([a, b])
            observed = a.mean() - b.mean()
            perm = np.array([(lambda p: p[:len(a)].mean() - p[len(a):].mean())(rng.permutation(pooled)) for _ in range(5000)])
            out.append({'backend': backend, 'snapshot': s, 'development_queries': len(a), 'test_queries': len(b),
                        'development_recall': float(a.mean()), 'test_recall': float(b.mean()),
                        'difference': float(observed), 'interval_95': np.quantile(boot, [.025, .975]).tolist(),
                        'permutation_p': float((np.sum(np.abs(perm) >= abs(observed)) + 1) / (len(perm) + 1)),
                        'test_recall_sd': float(b.std(ddof=1))})
    return out


# One-sided 5% critical values of Student's t for n - 1 degrees of freedom.
T_CRITICAL = {25: 1.7109, 50: 1.6766, 100: 1.6604, 200: 1.6525, 400: 1.6487}


def recall_change(test_paths, *, sizes=(25, 50, 100, 200, 400), draws=5_000, seed=20261120):
    """Uncertainty of the paired change in neighbour recall at ef 200 between snapshots 1 and 3.

    The calibration in the protocol compares each snapshot's mean recall with a fixed target. The
    same queries are searched at every snapshot, so a monitor could instead test the per-query
    change, which removes the between-query variance. For n queries drawn from the test-query
    distribution this reports how often a one-sided paired t-test at 5% detects the fall.
    """
    table = _recall_table(test_paths, ROLE)
    rng = np.random.default_rng(seed)
    out = []
    for backend in ('epzero', 'faiss_hnsw'):
        queries = sorted(table[(backend, SEEDS[0], 1)])
        level = {(k, s): np.array([table[(backend, k, s)][q] for q in queries]) for k in SEEDS for s in (1, 3)}
        change = {k: level[(k, 3)] - level[(k, 1)] for k in SEEDS}
        detect = {}
        for n in sizes:
            hits = 0
            for k in SEEDS:
                sample = change[k][rng.integers(0, len(queries), (draws, n))]
                sd = sample.std(axis=1, ddof=1)
                t = np.divide(sample.mean(axis=1), sd / np.sqrt(n), out=np.zeros(draws), where=sd > 0)
                hits += int(np.sum(t < -T_CRITICAL[n]))
            detect[str(n)] = hits / (draws * len(SEEDS))
        sd_change = float(np.mean([change[k].std(ddof=1) for k in SEEDS]))
        out.append({'backend': backend, 'queries': len(queries),
                    'recall_snapshot_1': float(np.mean([level[(k, 1)].mean() for k in SEEDS])),
                    'recall_snapshot_3': float(np.mean([level[(k, 3)].mean() for k in SEEDS])),
                    'mean_change': float(np.mean([change[k].mean() for k in SEEDS])),
                    'sd_of_level_snapshot_3': float(np.mean([level[(k, 3)].std(ddof=1) for k in SEEDS])),
                    'sd_of_change': sd_change,
                    'correlation_snapshot_1_and_3': float(np.mean([np.corrcoef(level[(k, 1)], level[(k, 3)])[0, 1] for k in SEEDS])),
                    'standard_error_of_change_at_50': sd_change / np.sqrt(50),
                    'mean_change_by_seed': {str(k): float(change[k].mean()) for k in SEEDS},
                    'sd_of_change_by_seed': {str(k): float(change[k].std(ddof=1)) for k in SEEDS},
                    'paired_t_test_detects_fall': detect})
    return out


def calibration_power(sweep_paths, *, target=.95, sizes=(25, 50, 100, 200, 400, 800), draws=5_000, seed=20261120):
    """If calibration queries came from the test-query distribution, how often would a set of n queries
    select each ef (smallest in the doubling grid whose mean recall reaches the target)?"""
    recalls = {}
    for path in map(Path, sweep_paths):
        report = json.loads(path.read_text())
        if json.loads((path.parents[2] / 'inputs.json').read_text())['mode'] != 'incremental':
            continue
        for ef, item in report['runs'].items():
            if int(ef) in (100, 200, 400, 800):
                recalls[(report['backend'], report['seed'], report['snapshot'], int(ef))] = item['ann_recall_per_query']
    rng = np.random.default_rng(seed)
    out = []
    for backend in ('epzero', 'faiss_hnsw'):
        for s in SNAPSHOTS:
            queries = sorted(recalls[(backend, SEEDS[0], s, 200)])
            # One index per seed; average the selection outcome over the three seeds.
            for n in sizes:
                picks = {100: 0, 200: 0, 400: 0, 800: 0, 'none': 0}
                for k in SEEDS:
                    matrix = np.array([[recalls[(backend, k, s, ef)][q] for q in queries] for ef in (100, 200, 400, 800)])
                    sample = rng.integers(0, len(queries), (draws, n))
                    means = matrix[:, sample].mean(axis=2)            # (ef, draw)
                    reached = means >= target
                    first = np.where(reached.any(axis=0), reached.argmax(axis=0), -1)
                    for i, ef in enumerate((100, 200, 400, 800)):
                        picks[ef] += int(np.sum(first == i))
                    picks['none'] += int(np.sum(first == -1))
                total = draws * len(SEEDS)
                true_recall = {ef: float(np.mean([np.mean(list(recalls[(backend, k, s, ef)].values())) for k in SEEDS]))
                               for ef in (100, 200, 400, 800)}
                out.append({'backend': backend, 'snapshot': s, 'calibration_queries': n,
                            'selects_ef': {str(k): v / total for k, v in picks.items()},
                            'true_test_recall': true_recall})
    return out


def calibration_margin(sweep_paths, *, target=.95, sizes=(50, 100, 200), margins=(0, .005, .01, .015, .02, .03),
                       grid=(100, 200, 400, 800), draws=5_000, seed=20261120):
    """Calibration with a safety margin: choose the smallest ef whose calibration recall reaches
    target + margin. For calibration sets drawn from the test queries, report how often the chosen
    ef truly falls short of the target on all test queries, and the mean chosen ef (the cost)."""
    recalls = {}
    for path in map(Path, sweep_paths):
        report = json.loads(path.read_text())
        if json.loads((path.parents[2] / 'inputs.json').read_text())['mode'] != 'incremental':
            continue
        for ef, item in report['runs'].items():
            if int(ef) in grid:
                recalls[(report['backend'], report['seed'], report['snapshot'], int(ef))] = item['ann_recall_per_query']
    rng = np.random.default_rng(seed)
    out = []
    for backend in ('epzero', 'faiss_hnsw'):
        for s in SNAPSHOTS:
            queries = sorted(recalls[(backend, SEEDS[0], s, grid[0])])
            for n in sizes:
                for margin in margins:
                    short = none = 0
                    chosen = []
                    for k in SEEDS:
                        matrix = np.array([[recalls[(backend, k, s, ef)][q] for q in queries] for ef in grid])
                        truth = matrix.mean(axis=1)                                   # (ef,) on all test queries
                        means = matrix[:, rng.integers(0, len(queries), (draws, n))].mean(axis=2)   # (ef, draw)
                        reached = means >= target + margin
                        first = np.where(reached.any(axis=0), reached.argmax(axis=0), -1)
                        none += int(np.sum(first == -1))
                        picked = first[first >= 0]
                        short += int(np.sum(truth[picked] < target))
                        chosen.extend(np.array(grid)[picked].tolist())
                    total = draws * len(SEEDS)
                    out.append({'backend': backend, 'snapshot': s, 'calibration_queries': n, 'margin': margin,
                                'chosen_ef_falls_short_of_target': short / total, 'no_ef_in_grid_qualifies': none / total,
                                'mean_chosen_ef': float(np.mean(chosen)) if chosen else None,
                                'chosen_ef_share': {str(ef): chosen.count(ef) / total for ef in grid}})
    return out


def change_by_width(sweep, exact, judged, population):
    """ANN-minus-exact nDCG@10 at each fixed ef, and its change from snapshot 1 to 3 (seeds averaged within query).

    `sweep` maps (backend, mode, seed, snapshot, ef) to a run; only incrementally maintained indexes are used.
    """
    from .heldout_analysis import _estimate, _summary, score
    base = {s: score(exact[s], judged[s], population) for s in SNAPSHOTS}
    out, changes = [], {}
    for backend in ('epzero', 'faiss_hnsw'):
        for ef in sorted({key[4] for key in sweep if key[0] == backend and key[1] == 'incremental'}):
            gaps = {}
            for s in (1, 3):
                cols = [score(sweep[(backend, 'incremental', seed, s, ef)], judged[s], population) for seed in SEEDS]
                gaps[s] = np.array([[c[q]['nDCG@10'] - base[s][q]['nDCG@10'] for c in cols] for q in population])
            changes[(backend, ef)] = gaps[3] - gaps[1]
            change = _summary(_estimate(population, list(SEEDS), changes[(backend, ef)]))
            out.append({'backend': backend, 'ef': ef, 'gap_snapshot_1': float(gaps[1].mean()),
                        'gap_snapshot_3': float(gaps[3].mean()), 'change': change['mean_difference'],
                        'change_interval_95': change['percentile_interval'],
                        'change_within_margin_0.01': bool(-.01 < change['percentile_interval'][0]
                                                          and change['percentile_interval'][1] < .01)})
    # The two implementations at the same width, on the same queries: is one change larger than the other?
    between = []
    for ef in sorted({ef for _, ef in changes}):
        contrast = _summary(_estimate(population, list(SEEDS), changes[('epzero', ef)] - changes[('faiss_hnsw', ef)]))
        between.append({'ef': ef, 'epzero_minus_faiss_change': contrast['mean_difference'],
                        'interval_95': contrast['percentile_interval'],
                        'two_sided_randomization_p': contrast['two_sided_randomization_p']})
    return {'by_backend': out, 'epzero_against_faiss': between}


def analyze(inputs, release, evidence, extension_evidence, development_root, output):
    inputs, evidence, extension_evidence, development_root, output = map(
        Path, (inputs, evidence, extension_evidence, development_root, output))
    queries, refs, _, loaded = load_heldout(inputs)
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in SNAPSHOTS}
    _, runs, _ = collect(sorted(evidence.rglob('report.json')), ROLE, ids, exact)
    qrels, identity = load_test_qrels(release)
    from .heldout_analysis import panel
    from .heldout_pool import load
    population = panel(qrels['gpt-oss-120b'], ids)
    # The verified sweep runs: (kind, backend, mode, seed, snapshot, 'ef<N>') -> run.
    sweep = {(key[1], key[2], key[3], key[4], int(key[5][2:])): run
             for key, run in load(inputs, evidence, extension_evidence)[3].items() if key[0] == 'sweep'}
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'completed', 'kind': 'EXPLORATORY post-freeze robustness analyses',
              'source_sha256': checksum(__file__), 'inputs': loaded, 'judgment_sha256': identity,
              'official_source': 'CEUR Vol-4283 paper 374, Table 3 and Table 4',
              'leaderboard': leaderboard(runs, exact, qrels),
              'tail': tail(runs, exact, qrels, population),
              'query_shift': query_shift(sorted(development_root.glob('longeval-pace-trajectory-evidence-*/*/trajectory/report.json')),
                                         sorted(evidence.rglob('trajectory/report.json'))),
              'recall_change': recall_change(sorted(evidence.rglob('trajectory/report.json'))),
              'calibration_margin': calibration_margin(sorted(extension_evidence.rglob('sweep/snapshot-*/report.json'))),
              'change_by_width': change_by_width(sweep, exact, qrels['gpt-oss-120b'], population),
              'calibration_power': calibration_power(sorted(extension_evidence.rglob('sweep/snapshot-*/report.json')))}
    atomic_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('inputs', 'release', 'evidence', 'extension_evidence', 'development_root', 'output'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    r = analyze(a.inputs, a.release, a.evidence, a.extension_evidence, a.development_root, a.output)
    print(json.dumps(r['leaderboard']['reproduction'], indent=2))


if __name__ == '__main__':
    main()
