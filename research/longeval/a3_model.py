"""A3: the mean-field in-degree model against measured builds (docs/a3-mechanism-derivation.md).

    python -m research.longeval.a3_model OUT.json BUILD [BUILD ...]

Each BUILD is a screen-M1 build folder (`accumulation.py run`). Per insertion tenth, the model gives the choices
by later arrivals C = k̄ ln(n/i), the kept links from later nodes S and the reverse links R under random
displacement in full lists of M₀ = 2M links, with k̄ the build's own mean number of choices and nothing fitted.
The predictions scored are the ones recorded for M 32 in the derivation note, before those builds ran.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .accumulation import measures


def model_tenths(k, M0, n):
    i = np.arange(1, n + 1, dtype=np.float64)
    T = np.log(n / i)
    lam, tstar = k / (M0 + 1), (M0 - k) / k
    C = k * T
    S = np.where(T <= tstar, k * T, k * (tstar + (1 - np.exp(-lam * (T - tstar))) / lam))
    R = k * (i / n) ** lam
    tenth = lambda a: [float(a[d * n // 10:(d + 1) * n // 10].mean()) for d in range(10)]
    return {'C': tenth(C), 'S': tenth(S), 'R': tenth(R)}


def model_measured(choices, L):
    """Section 8: S and R by insertion tenth from each insertion's own number of choices (`choices`, by insertion
    position) and the measured mean size L of a saturated list, removal rate 1/L per addition."""
    k = np.asarray(choices, dtype=np.float64)
    n = len(k)
    w = np.zeros(n)
    w[1:] = k[1:] / np.arange(1, n)                            # insertion q chooses a given earlier node w.p. k_q/q
    after = np.concatenate([np.cumsum(w[::-1])[::-1][1:], [0.0]])   # A(p): additions to a list after position p
    survive = np.exp(-np.maximum(0.0, after - (L - k)) / L)    # a link made at q, in q's own list
    S = np.concatenate([np.cumsum((w * survive)[::-1])[::-1][1:], [0.0]])
    R = k * np.exp(-after / L)
    tenth = lambda a: [float(a[d * n // 10:(d + 1) * n // 10].mean()) for d in range(10)]
    return {'S': tenth(S), 'R': tenth(R)}


def compare_measured(build, L):
    m = measures(build)
    choices = np.load(Path(build) / 'choices.npy')
    model = model_measured(choices, L)
    S, R = m['later_by_tenth'], m['earlier_by_tenth']
    return {'build': Path(build).name, 'L': L, 'measured': {'S': S, 'R': R}, 'model': model,
            'predictions': {'1_S_within_1.5_every_tenth': all(abs(a - b) <= 1.5 for a, b in zip(S, model['S'])),
                            '2_R_within_1.5_every_tenth': all(abs(a - b) <= 1.5 for a, b in zip(R, model['R'])),
                            '3_first_tenth_S_within_15pct': abs(model['S'][0] - S[0]) <= 0.15 * S[0]}}


def list_profile(links_dir):
    """Section 10: from a link-history build, the number of candidates removed per pruning event (its empirical
    distribution) and the removal probability by age-rank tenth within an event (youngest first)."""
    links_dir = Path(links_dir)
    report = json.loads((links_dir / 'report.json').read_text())
    t = np.load(links_dir / 'events.npy')
    event, age, removed = t[:, 0].astype(np.int64), t[:, 4], t[:, 6].astype(bool)
    counts, by_tenth = [], np.zeros((10, 2))
    for e in np.unique(event):
        rows = np.flatnonzero(event == e)
        counts.append(int(removed[rows].sum()))
        order = np.argsort(age[rows], kind='stable')
        tenth = np.empty(len(rows), dtype=np.int64)
        tenth[order] = np.arange(len(rows)) * 10 // len(rows)
        np.add.at(by_tenth, (tenth, 0), 1)
        np.add.at(by_tenth, (tenth, 1), removed[rows])
    return {'M0': 2 * report['M'], 'removed_per_event': np.bincount(counts).tolist(),
            'removal_by_age_tenth': (by_tenth[:, 1] / by_tenth[:, 0]).tolist()}


def _prune(born, profile, rng, counts):
    """One pruning event on a list of creation stamps (len M0 + 1): remove r, weighted by age-rank tenth."""
    r = rng.choice(len(counts), p=counts)
    ages = -np.asarray(born, dtype=np.float64)                   # larger stamp = younger
    order = np.argsort(ages, kind='stable')                      # youngest first
    tenth = np.empty(len(born), dtype=np.int64)
    tenth[order] = np.arange(len(born)) * 10 // len(born)
    w = np.asarray(profile['removal_by_age_tenth'])[tenth]
    gone = rng.choice(len(born), size=min(r, len(born)), replace=False, p=w / w.sum()) if r else []
    keep = np.ones(len(born), dtype=bool)
    keep[gone] = False
    return [b for b, k in zip(born, keep) if k]


def survival_curves(profile, ks, horizon, *, sims=400, burn=400, seed=20261128):
    """Survival after a additions (0..horizon) of a chooser's own links (per initial size k) and of a reverse link
    entering a list in its steady state, from the simulated list process."""
    rng = np.random.default_rng(seed)
    counts = np.asarray(profile['removed_per_event'], dtype=np.float64)
    counts /= counts.sum()
    full = profile['M0'] + 1
    own = {}
    for k in ks:
        alive = np.zeros(horizon + 1)
        for _ in range(sims):
            born, stamp = list(range(-k, 0)), 0                  # the k own links, stamps below 0
            for a in range(horizon + 1):
                alive[a] += sum(1 for b in born if b < 0) / k
                born.append(stamp)
                stamp += 1
                if len(born) >= full:
                    born = _prune(born, profile, rng, counts)
        own[k] = alive / sims
    alive = np.zeros(horizon + 1)
    for _ in range(sims * 4):
        born, stamp = [], 0
        for _ in range(burn):                                   # reach the steady state
            born.append(stamp)
            stamp += 1
            if len(born) >= full:
                born = _prune(born, profile, rng, counts)
        tag = stamp
        born.append(stamp)
        stamp += 1
        if len(born) >= full:
            born = _prune(born, profile, rng, counts)
        for a in range(horizon + 1):
            alive[a] += tag in born
            born.append(stamp)
            stamp += 1
            if len(born) >= full:
                born = _prune(born, profile, rng, counts)
    return own, alive / (sims * 4)


def model_list_process(choices, profile, **kw):
    k = np.asarray(choices, dtype=np.int64)
    n = len(k)
    w = np.zeros(n)
    w[1:] = k[1:] / np.arange(1, n)
    after = np.concatenate([np.cumsum(w[::-1])[::-1][1:], [0.0]])
    horizon = int(np.ceil(after.max())) + 1
    ks = [int(v) for v in np.unique(k) if (k == v).mean() >= 0.001 and v > 0]
    own, reverse = survival_curves(profile, ks, horizon, **kw)
    grid = np.arange(horizon + 1)
    nearest = np.array([min(ks, key=lambda v: abs(v - x)) for x in range(k.max() + 1)])
    surv_own = np.empty(n)
    for v in ks:
        rows = nearest[k] == v
        surv_own[rows] = np.interp(after[rows], grid, own[v])
    S = np.concatenate([np.cumsum((w * surv_own)[::-1])[::-1][1:], [0.0]])
    R = k * np.interp(after, grid, reverse)
    tenth = lambda a: [float(a[d * n // 10:(d + 1) * n // 10].mean()) for d in range(10)]
    return {'S': tenth(S), 'R': tenth(R)}


def compare_list_process(build, links_dir, measured_rate_r_error=None):
    m = measures(build)
    profile = list_profile(links_dir)
    model = model_list_process(np.load(Path(build) / 'choices.npy'), profile)
    S, R = m['later_by_tenth'], m['earlier_by_tenth']
    worst_r = max(abs(a - b) for a, b in zip(R, model['R']))
    out = {'build': Path(build).name, 'profile': profile, 'measured': {'S': S, 'R': R}, 'model': model,
           'largest_R_error': worst_r,
           'predictions': {'1_S_within_1.5_every_tenth': all(abs(a - b) <= 1.5 for a, b in zip(S, model['S'])),
                           '2_R_within_1.5_every_tenth': all(abs(a - b) <= 1.5 for a, b in zip(R, model['R']))}}
    if measured_rate_r_error is not None:
        out['predictions']['3_R_error_below_half_the_measured_rate_model'] = worst_r < 0.5 * measured_rate_r_error
    return out


def compare(build):
    m = measures(build)
    report = json.loads((Path(build) / 'report.json').read_text())
    k, M0, n = m['mean_choices'], 2 * report['M'], report['count']
    model = model_tenths(k, M0, n)
    C, S, R = m['chosen_by_tenth'], m['later_by_tenth'], m['earlier_by_tenth']
    out = {'build': Path(build).name, 'M': report['M'], 'ef_construction': report['ef_construction'], 'count': n,
           'order': report['order'], 'seed': report['seed'], 'mean_choices': k, 'M0': M0,
           'measured': {'C': C, 'S': S, 'R': R}, 'model': model,
           'surviving_share_of_choices_by_tenth': m['surviving_share_of_choices_by_tenth']}
    out['predictions_m32'] = {
        '1_C_within_3pct_every_tenth': all(abs(c - p) <= 0.03 * p for c, p in zip(C, model['C'])),
        '2_R_within_1_tenths_3_to_10': all(abs(r - p) <= 1 for r, p in zip(R[2:], model['R'][2:])),
        '2_R_below_model_first_tenth': R[0] < model['R'][0],
        '3_S_within_1_tenths_3_to_10': all(abs(s - p) <= 1 for s, p in zip(S[2:], model['S'][2:])),
        '3_S_below_model_first_tenth': S[0] < model['S'][0],
        '4_S_over_C_at_least_0.95_from_third_tenth': all(v >= 0.95 for v in m['surviving_share_of_choices_by_tenth'][2:])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output', type=Path)
    parser.add_argument('builds', type=Path, nargs='*')
    parser.add_argument('--measured-rate', nargs='+', metavar='BUILD=LINKS_REPORT',
                        help='section 8: score BUILD with L from a link-history report.json')
    parser.add_argument('--list-process', nargs='+', metavar='BUILD=LINKS_DIR',
                        help='section 10: score BUILD with the list process measured in LINKS_DIR')
    parser.add_argument('--measured-rate-results', type=Path, help="section 8's aggregates, for prediction 3")
    a = parser.parse_args()
    if a.list_process:
        earlier = {}
        if a.measured_rate_results:
            for b in json.loads(a.measured_rate_results.read_text())['builds']:
                earlier[b['build']] = max(abs(x - y) for x, y in zip(b['measured']['R'], b['model']['R']))
        out = {'kind': 'EXPLORATORY A3 list-process model (docs/a3-mechanism-derivation.md section 10), aggregates only',
               'builds': []}
        for pair in a.list_process:
            build, links = pair.split('=', 1)
            name = Path(build).name
            out['builds'].append(compare_list_process(build, links, earlier.get(name) if 'm32' in name else None))
        a.output.write_text(json.dumps(out, indent=2) + '\n')
        for b in out['builds']:
            print(b['build'], json.dumps(b['predictions']), round(b['largest_R_error'], 2))
        return
    if a.measured_rate:
        out = {'kind': 'EXPLORATORY A3 measured-rate model (docs/a3-mechanism-derivation.md section 8), aggregates only',
               'builds': []}
        for pair in a.measured_rate:
            build, links = pair.split('=', 1)
            L = json.loads(Path(links).read_text())['mean_list_size_pruned']
            out['builds'].append(compare_measured(build, L))
        a.output.write_text(json.dumps(out, indent=2) + '\n')
        for b in out['builds']:
            print(b['build'], f"L={b['L']:.2f}", json.dumps(b['predictions']))
        return
    out = {'kind': 'EXPLORATORY A3 model check (docs/a3-mechanism-derivation.md), aggregates only',
           'builds': [compare(b) for b in a.builds]}
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    for b in out['builds']:
        print(b['build'], f"k={b['mean_choices']:.2f}", json.dumps(b['predictions_m32']))


if __name__ == '__main__':
    main()
