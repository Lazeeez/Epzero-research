import json

import numpy as np

from research.longeval import ngfix_arrival as na


def make_sample(tmp_path, n=2000, dim=16, queries=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = rng.normal(size=(queries, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    exact = np.argsort(-(q.astype(np.float64) @ x.T.astype(np.float64)), axis=1, kind='stable')[:, :100].astype(np.int32)
    s = tmp_path / 'sample'
    s.mkdir()
    np.save(s / 'vectors.npy', x)
    np.save(s / 'queries.npy', q)
    np.save(s / 'exact-top100.npy', exact)
    np.save(s / 'cohort.npy', (1 + np.arange(n) * 3 // n).astype(np.int8))
    np.save(s / 'lexical-rank.npy', np.arange(n, dtype=np.int64))
    (s / 'sample.json').write_text(json.dumps({'metric': 'ip', 'count': n}))
    return s, x, q, exact


def read_matrix(path, dtype):
    raw = np.fromfile(path, dtype='<u4', count=2)
    return np.fromfile(path, dtype=dtype, offset=8).reshape(int(raw[0]), int(raw[1]))


def fake_run(tmp_path, prepared, rankings_for):
    """The adapter's output files, with rankings from `rankings_for(variant, ef, query_row)` in adapter IDs."""
    run = tmp_path / 'run'
    run.mkdir()
    queries = read_matrix(prepared / 'queries.fbin', '<f4')
    with open(run / 'rankings.tsv', 'w') as f:
        f.write('variant\tquery\tef\trank\tid\tdistance\n')
        for variant in na.VARIANTS:
            for ef in na.EFS:
                for qi in range(len(queries)):
                    for rank, doc in enumerate(rankings_for(variant, ef, qi), 1):
                        f.write(f'{variant}\t{qi}\t{ef}\t{rank}\t{doc}\t0.5\n')
    (run / 'stages.tsv').write_text('variant\tstage\tseconds\tserialized_bytes\nbase\tbuild\t1.0\t10\n')
    return run


def test_split_is_disjoint_and_seeded():
    train, test = na.split(381)
    assert len(train) == 190 and len(test) == 191 and not set(train) & set(test)
    assert np.array_equal(train, na.split(381)[0])


def test_prepared_files_pass_the_adapters_checks(tmp_path):
    sample, x, q, _ = make_sample(tmp_path)
    info = na.prepare(sample, tmp_path / 'prep', order='shuffled')
    prep = tmp_path / 'prep'
    base = read_matrix(prep / 'base.fbin', '<f4')
    sequence, train = np.load(prep / 'sequence.npy'), np.load(prep / 'train.npy')
    assert np.array_equal(base, x[sequence]) and info['train_queries'] == 30
    gt = read_matrix(prep / 'calibration_top200.ibin', '<i4')
    queries = read_matrix(prep / 'queries.fbin', '<f4')
    assert gt.shape == (30, 200) and np.array_equal(queries[:30], q[train])
    for qi in range(len(gt)):                                    # the adapter's checks, in the same arithmetic
        assert len(set(gt[qi])) == 200 and gt[qi].min() >= 0 and gt[qi].max() < len(base)
        distance = 1 - base[gt[qi]].astype(np.float64) @ queries[qi].astype(np.float64)
        assert (np.diff(distance) >= -1e-12).all()
        assert set(gt[qi][:50]) <= set(np.argsort(-(base @ queries[qi]))[:60])


def test_scoring_maps_ids_back_and_finds_a_planted_bias(tmp_path):
    sample, x, q, exact = make_sample(tmp_path)
    na.prepare(sample, tmp_path / 'prep', order='appended')
    prep = tmp_path / 'prep'
    sequence, train, test = (np.load(prep / f) for f in ('sequence.npy', 'train.npy', 'test.npy'))
    position = np.empty_like(sequence)
    position[sequence] = np.arange(len(sequence))
    order = np.concatenate([train, test])
    cohort = np.load(sample / 'cohort.npy')

    def rankings(variant, ef, qi):
        targets = exact[order[qi]]
        if variant == 'base':                     # base drops a third of the newest cohort's targets, a twentieth of the oldest's
            kept = [t for i, t in enumerate(targets) if not ((cohort[t] == 3 and i % 3 == 0) or (cohort[t] == 1 and i % 20 == 0))]
            filler = [d for d in range(len(x)) if d not in set(targets)][:100 - len(kept)]
            targets = kept + filler
        return position[np.asarray(targets)]
    out = na.score(prep, fake_run(tmp_path, prep, rankings), sample)
    assert out['variants']['ngfix']['newest_excess_pp'] == 0 and out['variants']['ngfix']['recall_at_100']['200'] == 1.0
    assert out['variants']['base']['within_query_ratio'] > 5 and out['variants']['base']['newest_excess_pp'] > 20
    assert out['paired_against_base']['ngfix_rfix']['excess_change_pp'] < -20
