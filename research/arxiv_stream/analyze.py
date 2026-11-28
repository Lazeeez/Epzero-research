"""Screen A1: the measures and predictions of docs/screen-a1-arxiv-stream.md, from the returned builds.

    python -m research.arxiv_stream.analyze SAMPLE JOBS OUT.json

JOBS holds one folder per build job (`arm.txt` and `build/`), as `scripts/pace/arxiv-stream.sbatch` writes them.
Queries are the sample's new-content queries (`per-query-ef*.npy`) and its uniform ones (`confirm-` prefix), each
scored on its own. Seeds are averaged within query by summing their per-query tables (the same targets in each);
intervals resample queries (5,000 draws, one draw set for every contrast). Aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.longeval.insertion_strata import risk_difference

EF, DRAWS, RNG = 200, 5000, 20261128
SETS = {'new-content': '', 'uniform': 'confirm-'}


def collect(jobs):
    """{(arm, backend, seed): build folder}, from each job's arm.txt (`appended-epzero-11`, `repaired-epzero-29`)."""
    found = {}
    for arm_file in sorted(Path(jobs).glob('*/arm.txt')):
        arm = arm_file.read_text().strip().split('=', 1)[1]
        if arm == 'sample':
            continue
        order, backend, seed = arm.rsplit('-', 2)
        if (arm_file.parent / 'build' / 'report.json').exists():
            found[(order, backend, int(seed))] = arm_file.parent / 'build'
    return found


def _seed_sum(builds, order, backend, prefix, name='per-query'):
    tables = [np.load(d / f'{prefix}{name}-ef{EF}.npy') for (o, b, _), d in sorted(builds.items()) if (o, b) == (order, backend)]
    return np.sum(tables, axis=0) if tables else None


def _excess(table, draws, target=3, reference=1):
    point = 100 * risk_difference(table, target, reference)
    boot = np.array([100 * risk_difference(table[r], target, reference) for r in draws])
    return point, boot


def _item(point, boot):
    return {'value': float(point), 'interval_95': np.quantile(boot, [.025, .975]).tolist()}


def _oldest(table):
    return table[:, 0, :, 1].sum() / table[:, 0, :, 0].sum()


def exposure_fit(report):
    """R² of the in-degree by insertion decile against a + b·ln(n/i) at the decile midpoints."""
    y = np.asarray(report['indegree_by_insertion_decile'], dtype=np.float64)
    x = np.log(1 / ((np.arange(10) + 0.5) / 10))
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {'slope': float(b), 'intercept': float(a), 'r2': float(r2)}


def age_curve(sample, build):
    """Pooled miss rate of exact targets by arrival year, in one appended build (new-content queries)."""
    month = np.load(Path(sample) / 'month.npy')
    exact = np.load(Path(sample) / 'exact-top100.npy')
    found = np.load(Path(build) / f'labels-ef{EF}.npy')
    year = month[exact] // 100
    missed = np.array([~np.isin(exact[q], found[q]) for q in range(len(exact))])
    return {str(y): {'targets': int((year == y).sum()), 'miss_rate': float(missed[year == y].mean())}
            for y in np.unique(year)}


def analyze(sample, jobs):
    builds = collect(jobs)
    backends = sorted({b for _, b, _ in builds})
    queries = len(np.load(Path(sample) / 'queries.npy'))
    draws = np.random.default_rng(RNG).integers(0, queries, (DRAWS, queries))
    out = {'kind': 'EXPLORATORY screen A1 (docs/screen-a1-arxiv-stream.md), aggregates only', 'ef': EF,
           'builds': sorted(f'{o}-{b}-{s}' for o, b, s in builds), 'by_set': {}, 'exposure': {}, 'age_curve': {},
           'recall_at_100': {}}
    for o, b, s in sorted(builds):
        report = json.loads((builds[(o, b, s)] / 'report.json').read_text())
        widths = report['widths'][str(EF)]
        confirm = report.get('confirm', {}).get('widths', {}).get(str(EF), {})
        out['recall_at_100'][f'{o}-{b}-{s}'] = {'new-content': widths['recall_at_100'], 'uniform': confirm.get('recall_at_100')}
        if o == 'appended':
            out['exposure'][f'{b}-{s}'] = exposure_fit(report)
            if s == 11:
                out['age_curve'][b] = age_curve(sample, builds[(o, b, s)])
    for set_name, prefix in SETS.items():
        block = {}
        for b in backends:
            arms = {o: _seed_sum(builds, o, b, prefix) for o in ('appended', 'reversed', 'shuffled', 'repaired')}
            entry = {}
            if arms['appended'] is not None:
                app, app_boot = _excess(arms['appended'], draws)
                entry['appended_excess_pp'] = _item(app, app_boot)
                if arms['reversed'] is not None:
                    rev, rev_boot = _excess(arms['reversed'], draws)
                    entry['reversed_excess_pp'] = _item(rev, rev_boot)
                    entry['reversed_minus_appended_pp'] = _item(rev - app, rev_boot - app_boot)
                if arms['repaired'] is not None:
                    rep, rep_boot = _excess(arms['repaired'], draws)
                    old = np.array([100 * (_oldest(arms['repaired'][r]) - _oldest(arms['appended'][r])) for r in draws])
                    entry['repaired_excess_pp'] = _item(rep, rep_boot)
                    entry['repaired_minus_appended_pp'] = _item(rep - app, rep_boot - app_boot)
                    entry['repaired_oldest_miss_change_pp'] = {
                        'value': float(100 * (_oldest(arms['repaired']) - _oldest(arms['appended']))),
                        'upper_95': float(np.quantile(old, .975))}
            position = _seed_sum(builds, 'shuffled', b, prefix, 'per-query-position')
            if position is not None:
                entry['shuffled_last_minus_first_tenth_pp'] = _item(*_excess(position, draws, target=10, reference=1))
            block[b] = entry
        out['by_set'][set_name] = block
    out['predictions'] = predictions(out, backends)
    return out


def predictions(out, backends):
    p, sets = {}, out['by_set']
    above = lambda item, floor: item['value'] > floor and item['interval_95'][0] > 0
    p['1_appended_excess_above_1pp_interval_above_0'] = {
        f'{s}/{b}': above(sets[s][b]['appended_excess_pp'], 1.0) for s in sets for b in backends if 'appended_excess_pp' in sets[s][b]}
    p['2_reversal_below_minus_1pp_interval_below_0'] = {
        f'{s}/{b}': (sets[s][b]['reversed_minus_appended_pp']['value'] < -1.0
                     and sets[s][b]['reversed_minus_appended_pp']['interval_95'][1] < 0)
        for s in sets for b in backends if 'reversed_minus_appended_pp' in sets[s][b]}
    p['3_shuffled_position_positive'] = {
        f'{s}/{b}': sets[s][b]['shuffled_last_minus_first_tenth_pp']['interval_95'][0] > 0
        for s in sets for b in backends if 'shuffled_last_minus_first_tenth_pp' in sets[s][b]}
    repair = {}
    for s in sets:
        e = sets[s].get('epzero', {})
        if 'repaired_excess_pp' in e:
            app_recall = np.mean([v[s] for k, v in out['recall_at_100'].items() if k.startswith('appended-epzero-')])
            rep_recall = np.mean([v[s] for k, v in out['recall_at_100'].items() if k.startswith('repaired-epzero-')])
            repair[s] = {'removes_at_least_half': e['repaired_excess_pp']['value'] <= 0.5 * e['appended_excess_pp']['value'],
                         'oldest_rise_at_most_0.75pp': e['repaired_oldest_miss_change_pp']['upper_95'] <= 0.75,
                         'recall_not_lower': bool(rep_recall >= app_recall)}
    p['4_repair'] = repair
    p['5_exposure_r2_at_least_0.9'] = {k: v['r2'] >= 0.9 for k, v in out['exposure'].items()}
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('jobs', type=Path)
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    out = analyze(a.sample, a.jobs)
    a.output.write_text(json.dumps(out, indent=2, default=bool) + '\n')
    print(json.dumps(out['predictions'], indent=2, default=bool))


if __name__ == '__main__':
    main()
