"""Query-clustered paired inference; implemented before held-out evaluation.

This module does not select a population, contrast, significance family, or practical
margin. Those must be frozen separately. It is not applied to development results here.
"""
from __future__ import annotations

import hashlib

import numpy as np


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('p-values must be a finite one-dimensional probability array')
    order = np.argsort(values, kind='stable')
    adjusted = np.empty_like(values)
    previous = 0.
    for rank, position in enumerate(order):
        previous = min(1., max(previous, (len(values) - rank) * float(values[position])))
        adjusted[position] = previous
    return adjusted.tolist()


def paired_query_estimate(query_ids, seed_ids, differences, *, bootstrap_draws=10_000,
                          randomization_draws=100_000, rng_seed=20261117, confidence=.95, batch=256):
    """Average prespecified seeds within query, then resample/sign-flip queries only.

    differences[q, seed] is an already paired contrast. For a longitudinal contrast,
    compute the complete within-query/seed change first; do not resample snapshots.
    Percentile intervals are nominal, not simultaneous or equivalence intervals.
    The randomization p-value assumes paired sign exchangeability under the null.
    """
    values = np.asarray(differences, dtype=np.float64)
    if (len(query_ids) < 2 or len(set(query_ids)) != len(query_ids) or not seed_ids
            or len(set(seed_ids)) != len(seed_ids) or values.shape != (len(query_ids), len(seed_ids))
            or not np.isfinite(values).all()):
        raise ValueError('requires complete finite paired query-by-seed contrasts and unique IDs')
    if (any(type(v) is not int or v <= 0 for v in (bootstrap_draws, randomization_draws, batch))
            or type(rng_seed) is not int or rng_seed < 0 or not 0 < confidence < 1):
        raise ValueError('invalid inference configuration')
    effects = values.mean(axis=1)
    observed = float(effects.mean())
    bootstrap_rng, randomization_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(rng_seed).spawn(2)]
    sampled = np.empty(bootstrap_draws, dtype=np.float64)
    for start in range(0, bootstrap_draws, batch):
        count = min(batch, bootstrap_draws - start)
        draws = bootstrap_rng.integers(0, len(effects), size=(count, len(effects)))
        sampled[start:start+count] = effects[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    interval = np.quantile(sampled, [tail, 1 - tail]).tolist()
    extreme = 0
    # A small roundoff guard treats arithmetically tied statistics as ties.
    tolerance = 16 * np.finfo(np.float64).eps * max(1., float(np.max(np.abs(effects))))
    for start in range(0, randomization_draws, batch):
        count = min(batch, randomization_draws - start)
        signs = randomization_rng.integers(0, 2, size=(count, len(effects))) * 2 - 1
        statistics = (signs * effects).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(statistics) >= abs(observed) - tolerance))
    return {'query_count': len(query_ids), 'seed_count': len(seed_ids), 'query_ids': list(query_ids),
            'seed_ids': list(seed_ids), 'mean_difference': observed, 'nominal_confidence': confidence,
            'percentile_interval': interval, 'two_sided_randomization_p': (extreme + 1) / (randomization_draws + 1),
            'bootstrap_draws': bootstrap_draws, 'randomization_draws': randomization_draws, 'rng_seed': rng_seed,
            'query_effects': dict(zip(query_ids, map(float, effects))),
            'difference_bytes_sha256': hashlib.sha256(values.astype('<f8').tobytes()).hexdigest(),
            'resampling_unit': 'query; construction seeds averaged within query',
            'limits': ['Nominal percentile interval; not multiplicity-adjusted',
                       'Sign randomization assumes paired exchangeability under the null',
                       'A non-significant difference is not evidence of equivalence',
                       'No generalization over datasets, embeddings or construction-seed distribution']}
