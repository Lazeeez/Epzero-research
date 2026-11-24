import json

import numpy as np
import pytest

from research.longeval import bigann_stream, insertion_pilot

BLOCKS = [(0, 500), (500, 900), (900, 1500), (1500, 2100), (2100, 2500), (2500, 3000)]


def runbook(blocks, dataset=bigann_stream.DATASET, extra=''):
    lines = ['random-xs-clustered:', '  max_pts: 10', '  1:', '    operation: "insert"', '    start: 0', '    end: 10',
             f'{dataset}:', f'  max_pts: {blocks[-1][1]}']
    step = 1
    for start, end in blocks:
        lines += [f'  {step}:', '    operation: "insert"', f'    start: {start}', f'    end: {end}',
                  f'  {step + 1}:', '    operation: "search"']
        step += 2
    lines += [extra, f'  gt_url: "https://example.invalid/{dataset}"'] if extra else [f'  gt_url: "https://example.invalid/{dataset}"']
    return '\n'.join(lines) + '\n'


def write_bin(path, array):
    with open(path, 'wb') as f:
        np.array(array.shape, dtype='<u4').tofile(f)
        np.ascontiguousarray(array, dtype='<f4').tofile(f)


def write_gt(path, ids):
    with open(path, 'wb') as f:
        np.array(ids.shape, dtype='<u4').tofile(f)
        ids.astype('<u4').tofile(f)
        np.zeros(ids.shape, dtype='<f4').tofile(f)


def brute(data, queries):
    d = ((queries[:, None, :].astype(np.float64) - data[None].astype(np.float64)) ** 2).sum(-1)
    return np.argsort(d, axis=1, kind='stable')[:, :100]


@pytest.fixture
def stream(tmp_path):
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(len(BLOCKS), 8)) * 4
    data = np.concatenate([centres[b] + rng.normal(size=(e - s, 8)) for b, (s, e) in enumerate(BLOCKS)]).astype(np.float32)
    queries = (centres[rng.integers(0, len(BLOCKS), 40)] + rng.normal(size=(40, 8))).astype(np.float32)
    write_bin(tmp_path / 'data.fbin', data)
    write_bin(tmp_path / 'queries.fbin', queries)
    write_gt(tmp_path / 'gt100', brute(data, queries))
    (tmp_path / 'runbook.yaml').write_text(runbook(BLOCKS))
    return tmp_path, data, queries


def test_the_runbook_parser_reads_one_dataset_and_rejects_deletes_and_gaps():
    blocks, max_pts = bigann_stream.parse_runbook(runbook(BLOCKS))
    assert blocks == BLOCKS and max_pts == 3000
    with pytest.raises(ValueError):
        bigann_stream.parse_runbook(runbook(BLOCKS, extra='  99:\n    operation: "delete"\n    start: 0\n    end: 5'))
    with pytest.raises(ValueError):
        bigann_stream.check_blocks([(0, 500), (600, 3000)], 3000)
    with pytest.raises(ValueError):
        bigann_stream.check_blocks(BLOCKS[:-1], 3000)


def test_cohorts_follow_the_third_in_which_each_block_starts():
    cohort, block, tenth = bigann_stream.cohorts(BLOCKS, 3000)
    # thirds start at 1000 and 2000: blocks starting at 0, 500, 900 -> 1; 1500 -> 2; 2100, 2500 -> 3
    assert [int(cohort[s]) for s, _ in BLOCKS] == [1, 1, 1, 2, 3, 3]
    assert block[499] == 0 and block[500] == 1 and tenth[0] == 0 and tenth[499] == 9 and tenth[500] == 0


def test_prepare_writes_the_study_format_with_checked_exact_targets(stream, tmp_path):
    folder, data, queries = stream
    summary = bigann_stream.prepare(folder / 'data.fbin', folder / 'queries.fbin', folder / 'gt100', folder / 'runbook.yaml',
                                    tmp_path / 'sample')
    sample = tmp_path / 'sample'
    assert np.array_equal(np.load(sample / 'vectors.npy'), data)
    assert np.array_equal(np.load(sample / 'exact-top100.npy'), brute(data, queries))
    assert summary['ground_truth_check']['passed'] and summary['metric'] == 'l2'
    assert summary['cohort_sizes'] == {'1': 1500, '2': 600, '3': 900}
    assert json.loads((sample / 'sample.json').read_text())['files']['data.fbin']['bytes'] == 8 + data.nbytes
    # the study's appended order is file order, and reversed puts cohort 3's blocks first
    rank, cohort = np.load(sample / 'lexical-rank.npy'), np.load(sample / 'cohort.npy')
    assert np.array_equal(insertion_pilot.insertion_order(rank, cohort, 'appended', 0), np.arange(3000))
    assert insertion_pilot.insertion_order(rank, cohort, 'reversed', 0)[0] == 2100


def test_prepare_stops_when_the_track_ground_truth_disagrees(stream, tmp_path):
    folder, data, queries = stream
    wrong = brute(data, queries)
    wrong[0] = np.arange(100)                       # one query's targets replaced
    write_gt(folder / 'gt100', wrong)
    with pytest.raises(ValueError, match='disagree'):
        bigann_stream.prepare(folder / 'data.fbin', folder / 'queries.fbin', folder / 'gt100', folder / 'runbook.yaml',
                              tmp_path / 'sample')


def test_the_tenth_table_counts_misses_by_position_in_own_block():
    targets = np.array([[0, 499] + list(range(1, 99))])
    tenth = bigann_stream.cohorts(BLOCKS, 3000)[2]
    labels = targets.copy()
    labels[0, 1] = -1                               # rank 2 (target 499, last tenth of block 0) missed
    table = bigann_stream.tenth_table(labels, targets, tenth)
    assert table[0, 9, 0].tolist() == [1, 1] and table[..., 0].sum() == 100 and table[..., 1].sum() == 1


def test_analyze_reads_order_and_engine_builds(stream, tmp_path):
    folder, _, _ = stream
    bigann_stream.prepare(folder / 'data.fbin', folder / 'queries.fbin', folder / 'gt100', folder / 'runbook.yaml', tmp_path / 's')
    sample = tmp_path / 's'
    for order in ('appended', 'reversed', 'shuffled'):
        for seed in (11, 29):
            insertion_pilot.run(sample, tmp_path / 'faiss' / f'{order}-{seed}', order=order, seed=seed, backend='faiss_hnsw',
                                M=4, ef_construction=8, ef=10, efs=(10, 200))
    targets, cohort = np.load(sample / 'exact-top100.npy'), np.load(sample / 'cohort.npy')
    rng = np.random.default_rng(1)
    for arm, newest_loss in (('grown', .3), ('repaired', .1), ('repaired8', .05), ('repaired8w', .05), ('rebuilt', .05)):
        for seed in (11, 29):
            out = tmp_path / 'engine' / f'{arm}-{seed}'
            out.mkdir(parents=True)
            keep = rng.random(targets.shape) > np.where(cohort[targets] == 3, newest_loss, .05)
            labels = np.where(keep, targets, -1)
            np.save(out / 'labels-ef200.npy', labels.astype(np.int32))
            table = np.zeros((len(targets), 3, 3, 2), dtype=np.int64)
            for q in range(len(targets)):
                for r, doc in enumerate(targets[q], 1):
                    b = next(i for i, (lo, hi) in enumerate(insertion_pilot.BANDS) if lo <= r <= hi)
                    table[q, cohort[doc] - 1, b] += (1, not keep[q, r - 1])
            np.save(out / 'per-query-ef200.npy', table)
    out = bigann_stream.analyze(sample, {'faiss_hnsw': tmp_path / 'faiss'}, tmp_path / 'engine')
    assert out['order_faiss_hnsw']['paired_appended_to_reversed']['seeds'] == [11, 29]
    assert set(out['order_faiss_hnsw']['within_block']) == {'appended', 'reversed'}
    assert out['order_faiss_hnsw']['global_position_shuffled']['seeds'] == [11, 29]
    assert out['engine']['repaired8']['decision']['effect'] is True
    assert out['engine']['repaired8']['effect_gap_change']['gap_change_pp']['value'] < 0
    assert out['engine']['repaired8w']['decision']['effect'] is True
    assert set(out['engine']['within_block']) == {'grown', 'repaired', 'repaired8', 'repaired8w', 'rebuilt'}
