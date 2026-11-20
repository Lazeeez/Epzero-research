"""Exhaustive, bounded-memory training-query ranking sensitivity to encoder precision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .dense import ARTIFACTS
from .evaluation import rank, validate_run, write_run
from .mapped import MappedSnapshot
from .oracle import exact_mapped
from .query_diagnostic import load_pair, validate_inputs
from .release import checksum
from .storage import atomic_json


def ranking_changes(left, right, *, depth=100):
    if set(left) != set(right) or not left or type(depth) is not int or depth < 10:
        raise ValueError('invalid ranking populations or diagnostic depth')
    rows = {}
    for qid in left:
        a, b = rank(left[qid]), rank(right[qid])
        for pairs in (a, b):
            if (len(pairs) != depth or len({key for key, _ in pairs}) != depth
                    or not all(np.isfinite(score) for _, score in pairs)):
                raise ValueError('invalid or incomplete precision ranking')
        a, b = [key for key, _ in a], [key for key, _ in b]
        rows[qid] = {'top1_same': a[0] == b[0], 'ordered_top10_same': a[:10] == b[:10],
                     f'ordered_top{depth}_same': a == b,
                     'overlap_at_10': len(set(a[:10]) & set(b[:10])) / 10,
                     f'overlap_at_{depth}': len(set(a) & set(b)) / depth}
    return rows


def sensitivity(folder, scan_path, inputs, bf16, fp32, output, *, snapshot=1, block_rows=2048):
    folder, scan_path, inputs, bf16, fp32, output = map(Path, (folder, scan_path, inputs, bf16, fp32, output))
    if snapshot not in (1, 2, 3) or type(block_rows) is not int or block_rows <= 0:
        raise ValueError('invalid snapshot or block size')
    ids, a, b = load_pair(inputs, bf16, fp32)
    contract, _ = validate_inputs(inputs)
    scan = json.loads(scan_path.read_text())
    if scan.get('status') != 'completed' or not scan.get('numeric_vectors_valid'):
        raise ValueError('requires a completed valid vector audit')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'purpose': 'training-query precision ranking sensitivity; not ANN or relevance results',
              'snapshot': snapshot, 'query_count': len(ids), 'depth': 100, 'metric': 'ip',
              'accumulation': 'float64 over original float32 inputs', 'tie_rule': 'score descending then external ID descending',
              'block_rows': block_rows, 'source_sha256': {name: checksum(Path(__file__).with_name(name))
                  for name in ('query_sensitivity.py', 'query_diagnostic.py', 'oracle.py', 'mapped.py')},
              'query_inputs_sha256': checksum(inputs / 'query-inputs.json'),
              'bf16_receipt_sha256': checksum(bf16 / 'query-encoding.json'),
              'fp32_receipt_sha256': checksum(fp32 / 'query-encoding.json'), 'scan_sha256': checksum(scan_path),
              'limits': ['No relevance judgments are loaded', 'No representation acceptance threshold assigned',
                         'Concurrent workloads may affect runtime; not a latency benchmark']}
    atomic_json(output / 'report.json', report)
    try:
        with MappedSnapshot(folder, expected_archive_sha256=ARTIFACTS[snapshot-1]['archive_sha256']) as mapped:
            if (scan['vector_sha256'] != mapped.receipt['members']['vecs.f4']['sha256']
                    or checksum(folder / 'vecs.f4') != scan['vector_sha256']):
                raise ValueError('reference vectors differ from audit/archive')
            if len(mapped.ids) < 100:
                raise ValueError('requires at least 100 live documents')
            report['vector_sha256'] = scan['vector_sha256']
            combined_ids = ['bf16:' + q for q in ids] + ['fp32:' + q for q in ids]
            started = time.perf_counter()
            run = exact_mapped(mapped.ids, mapped.vectors, combined_ids,
                               np.concatenate((a, b)), k=100, metric='ip', block_rows=block_rows,
                               progress=lambda value: atomic_json(output / 'progress.json', value))
            report['exact_seconds'] = time.perf_counter() - started
            left = {q: run['bf16:' + q] for q in ids}
            right = {q: run['fp32:' + q] for q in ids}
            for name, values in (('bfloat16', left), ('float32', right)):
                validate_run(values, ids, mapped.ids, 100)
                write_run(output / (name + '.run'), values)
        rows = ranking_changes(left, right)
        cohorts = {'all_training': ids, **contract['roles']}
        report['cohorts'] = {name: {'count': len(keys), 'mean': {
            metric: float(np.mean([rows[q][metric] for q in keys])) for metric in next(iter(rows.values()))}}
            for name, keys in cohorts.items()}
        report['per_query'] = rows
        report['artifacts'] = {n: checksum(output / n) for n in ('bfloat16.run', 'float32.run')}
        report['status'] = 'completed'
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        atomic_json(output / 'report.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'scan', 'inputs', 'bf16', 'fp32', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--snapshot', type=int, default=1)
    parser.add_argument('--block-rows', type=int, default=2048)
    args = parser.parse_args()
    report = sensitivity(args.folder, args.scan, args.inputs, args.bf16, args.fp32, args.output,
                         snapshot=args.snapshot, block_rows=args.block_rows)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_query'}, indent=2))


if __name__ == '__main__':
    main()
