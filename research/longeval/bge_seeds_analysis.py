"""Ten construction seeds on the BGE vectors (design and predictions: scripts/pace/LONGEVAL_BGE_SEEDS.md,
amendment 7; recorded before the jobs ran).

    python -m research.longeval.bge_seeds_analysis PREPARED RELEASE BGE_FROZEN BGE_SEEDS QWEN_INPUTS QWEN_FROZEN QWEN_SEEDS OUT.json

PREPARED is one BGE held-out job's `prepared` folder; every BGE report is checked against it by content
(amendment 6). BGE_FROZEN and QWEN_FROZEN hold the three frozen seeds' evidence, BGE_SEEDS and QWEN_SEEDS the
seven further seeds'. Per primary arm: D over ten seeds with queries and seeds resampled, beside the
three-seed value; how often calibration widened; and the encoder contrast (BGE minus Qwen, per query, seeds
averaged within each encoder), resampling queries jointly and seeds within each encoder. Aggregates only.

As checks, the three frozen BGE seeds must reproduce E4's D and its encoder contrast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bge_heldout import check_inputs, load_prepared
from .heldout import ROLE, load_heldout
from .heldout_analysis import HNSW_MODES, IVF_MODES, PRIMARY_ARMS, collect, load_test_qrels, panel
from .heldout_seeds import EXTRA_SEEDS
from .heldout_sensitivity import FAMILY
from .release import checksum
from .seeds_analysis import DRAWS, RNG_SEED, d_matrix, two_level, widened, widths

FROZEN_SEEDS = (11, 29, 47)
ALL_SEEDS = FROZEN_SEEDS + tuple(EXTRA_SEEDS)
E4_D = {'epzero_hnsw': 0.0015583, 'faiss_hnsw': -0.0027348, 'ivf_frozen': None}   # IVF read from E4's aggregate
E4_EPZERO_CONTRAST = 0.0074962
MARGIN, POINT_MARGIN = 0.01, 0.006


def _expected(seeds):
    frozen = [s for s in seeds if s in FROZEN_SEEDS]
    extra = [s for s in seeds if s not in FROZEN_SEEDS]
    return ({(b, m, s) for b in ('epzero', 'faiss_hnsw') for m in HNSW_MODES for s in frozen}
            | {('faiss_ivf', m, s) for m in IVF_MODES for s in frozen}
            | {(arm[0], arm[1], s) for arm in PRIMARY_ARMS.values() for s in extra})


def _paths(*folders):
    return [p for folder in folders if folder is not None for p in sorted(Path(folder).rglob('report.json'))]


def _matrices(reports, runs, exact, qrels, population, seeds):
    out = {}
    for name, arm in PRIMARY_ARMS.items():
        missing = [s for s in seeds if (arm[0], arm[1], s) not in reports]
        if missing:
            raise ValueError(f'{name}: no evidence for seeds {missing}')
        out[name] = d_matrix(runs, exact, qrels, population, arm, seeds)
    return out


def bge(prepared, release, frozen, extra, *, seeds, hnsw_config, ivf_config):
    _, queries, refs, identity = load_prepared(prepared)
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in (1, 2, 3)}
    paths = _paths(frozen, extra if any(s not in FROZEN_SEEDS for s in seeds) else None)
    checked = check_inputs(paths, prepared)
    reports, runs, excluded = collect(paths, ROLE, ids, exact, expected=_expected(seeds),
                                      config_sha256=(checksum(hnsw_config), checksum(ivf_config)))
    qrels = load_test_qrels(release)[0][FAMILY]
    population = panel(qrels, ids)
    return population, _matrices(reports, runs, exact, qrels, population, seeds), reports, {'jobs_checked_by_content': checked,
                                                                                         'excluded_reports': excluded}


def qwen(inputs, release, frozen, extra, *, seeds):
    queries, refs, _, _ = load_heldout(inputs)
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in (1, 2, 3)}
    paths = _paths(frozen, extra if any(s not in FROZEN_SEEDS for s in seeds) else None)
    reports, runs, _ = collect(paths, ROLE, ids, exact, expected=_expected(seeds))
    qrels = load_test_qrels(release)[0][FAMILY]
    population = panel(qrels, ids)
    return population, _matrices(reports, runs, exact, qrels, population, seeds)


def contrast(b_pop, b, q_pop, q, *, draws=DRAWS, seed=RNG_SEED):
    """BGE minus Qwen per query (seeds averaged within each encoder); queries resampled jointly, seeds within
    each encoder."""
    row = {x: i for i, x in enumerate(q_pop)}
    common = [x for x in b_pop if x in row]
    b = b[[b_pop.index(x) for x in common]]
    q = q[[row[x] for x in common]]
    rng = np.random.default_rng(seed)
    n = len(common)
    sample = []
    for _ in range(draws):
        r = rng.integers(0, n, n)
        sb = rng.integers(0, b.shape[1], b.shape[1])
        sq = rng.integers(0, q.shape[1], q.shape[1])
        sample.append(b[np.ix_(r, sb)].mean() - q[np.ix_(r, sq)].mean())
    return {'queries': n, 'bge_D': float(b.mean()), 'qwen_D': float(q.mean()), 'bge_minus_qwen': float(b.mean() - q.mean()),
            'interval_95_queries_and_seeds': np.quantile(sample, [.025, .975]).tolist()}


def analyze(prepared, release, bge_frozen, bge_seeds, qwen_inputs, qwen_frozen, qwen_seeds, *, hnsw_config, ivf_config,
            seeds=ALL_SEEDS, e4=None):
    b_pop, b_mat, b_reports, identity = bge(prepared, release, bge_frozen, bge_seeds, seeds=seeds,
                                            hnsw_config=hnsw_config, ivf_config=ivf_config)
    q_pop, q_mat = qwen(qwen_inputs, release, qwen_frozen, qwen_seeds, seeds=seeds)
    e4_d = dict(E4_D)
    if e4 is not None:
        e4_d['ivf_frozen'] = json.loads(Path(e4).read_text())['primary']['ivf_frozen']['mean_difference']
    out = {'kind': 'EXPLORATORY ten construction seeds on the BGE vectors (amendment 7), aggregates only',
           'design': 'scripts/pace/LONGEVAL_BGE_SEEDS.md', 'source_sha256': checksum(__file__), 'family': FAMILY,
           'seeds': list(seeds), 'panel': len(b_pop), 'inputs': identity,
           'bootstrap': {'draws': DRAWS, 'seed': RNG_SEED, 'unit': 'queries and seeds'}, 'arms': {}}
    for name, values in b_mat.items():
        arm = PRIMARY_ARMS[name]
        per_seed = values.mean(axis=0)
        chosen = {s: widths(b_reports, arm, s) for s in seeds}
        frozen_cols = [i for i, s in enumerate(seeds) if s in FROZEN_SEEDS]
        block = {'D': float(values.mean()), 'interval_95_queries_and_seeds': two_level(values),
                 'D_frozen_three_seeds': float(values[:, frozen_cols].mean()),
                 'check_frozen_reproduces_E4': (abs(float(values[:, frozen_cols].mean()) - e4_d[name]) < 5e-6) if e4_d[name] is not None else None,
                 'D_by_seed': {str(s): round(float(v), 5) for s, v in zip(seeds, per_seed)},
                 'sd_across_seeds': float(per_seed.std(ddof=1)) if len(seeds) > 1 else None,
                 'calibrated_widths_by_seed': {str(s): chosen[s] for s in seeds},
                 'seeds_widened': [s for s in seeds if widened(arm, chosen[s])],
                 'encoder_contrast': contrast(b_pop, values, q_pop, q_mat[name])}
        out['arms'][name] = block
    ep = out['arms']['epzero_hnsw']['encoder_contrast']
    out['prediction'] = {
        'statement': 'ten-seed BGE D within ±0.006 for every arm (points); Epzero encoder contrast smaller in magnitude than +0.0075 with its interval within ±0.01',
        'D_points_within_0.006': {n: abs(b['D']) <= POINT_MARGIN for n, b in out['arms'].items()},
        'epzero_contrast_smaller_than_three_seed': abs(ep['bge_minus_qwen']) < abs(E4_EPZERO_CONTRAST),
        'epzero_contrast_interval_within_0.01': all(abs(x) <= MARGIN for x in ep['interval_95_queries_and_seeds'])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('prepared', 'release', 'bge_frozen', 'bge_seeds', 'qwen_inputs', 'qwen_frozen', 'qwen_seeds', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--hnsw-config', type=Path, default=Path(__file__).with_name('trajectory-development.json'))
    parser.add_argument('--ivf-config', type=Path, default=Path(__file__).with_name('ivf-development.json'))
    parser.add_argument('--e4', type=Path, default=Path(__file__).with_name('e4-bge-heldout.json'))
    parser.add_argument('--frozen-only', action='store_true', help='check: the three frozen seeds only')
    a = parser.parse_args()
    out = analyze(a.prepared, a.release, a.bge_frozen, a.bge_seeds, a.qwen_inputs, a.qwen_frozen, a.qwen_seeds,
                  hnsw_config=a.hnsw_config, ivf_config=a.ivf_config, e4=a.e4,
                  seeds=FROZEN_SEEDS if a.frozen_only else ALL_SEEDS)
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({n: {k: b[k] for k in ('D', 'interval_95_queries_and_seeds', 'D_frozen_three_seeds', 'check_frozen_reproduces_E4')}
                      | {'contrast': b['encoder_contrast']['bge_minus_qwen'], 'contrast_interval': b['encoder_contrast']['interval_95_queries_and_seeds']}
                      for n, b in out['arms'].items()} | {'prediction': out['prediction']}, indent=2))


if __name__ == '__main__':
    main()
