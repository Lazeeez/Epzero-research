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
    a = parser.parse_args()
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
