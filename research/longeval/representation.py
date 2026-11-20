"""Portable, audited document samples and measured encoder agreement.

No observed agreement automatically approves a query representation or sets a threshold.
Run as ``python -m research.longeval.representation --help``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .encoder import read_records
from .release import checksum
from .storage import atomic_json


def compare_arrays(reference_ids, reference, candidate_ids, candidate):
    """Compare original float32 values by ID with float64 norm/dot accumulation."""
    for ids, values in ((reference_ids, reference), (candidate_ids, candidate)):
        if (not ids or any(not isinstance(x, str) or not x or any(c.isspace() for c in x) for x in ids)
                or len(set(ids)) != len(ids) or values.ndim != 2 or values.shape[0] != len(ids)
                or values.shape[1] == 0 or values.dtype != np.dtype('<f4')
                or not np.isfinite(values).all()):
            raise ValueError('invalid comparison IDs or vectors')
        if not np.allclose(np.linalg.norm(values.astype(np.float64), axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError('comparison requires unit-normalized original vectors')
    if set(reference_ids) != set(candidate_ids) or reference.shape[1] != candidate.shape[1]:
        raise ValueError('comparison membership or dimension mismatch')
    lookup = {key: i for i, key in enumerate(candidate_ids)}
    rows = []
    for key, stored in zip(reference_ids, reference):
        encoded = candidate[lookup[key]]
        x, y = stored.astype(np.float64), encoded.astype(np.float64)
        cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        rows.append({'id': key, 'cosine': cosine, 'l2': float(np.linalg.norm(x-y)),
                     'max_absolute_error': float(np.max(np.abs(x-y))),
                     'bitwise_equal': stored.tobytes() == encoded.tobytes()})
    cosines = np.array([r['cosine'] for r in rows])
    return {'count': len(rows), 'cosine_min': float(cosines.min()),
            'cosine_median': float(np.median(cosines)), 'cosine_max': float(cosines.max()),
            'bitwise_equal_rows': sum(r['bitwise_equal'] for r in rows),
            'max_coordinate_error': max(r['max_absolute_error'] for r in rows), 'rows': rows}


def prepare_sample(folder, text_audit, output, *, snapshot=1):
    from .dense import ARTIFACTS
    from .mapped import MappedSnapshot

    folder, text_audit, output = Path(folder), Path(text_audit), Path(output)
    audit = json.loads(text_audit.read_text())
    if audit.get('status') != 'completed' or snapshot not in (1, 2, 3):
        raise ValueError('requires a completed text audit and valid snapshot')
    item = next((x for x in audit['snapshots'] if x['snapshot'] == snapshot), None)
    if item is None or item['conflicting_duplicate_rows'] or item.get('dense_coverage', {}).get('missing_text_ids') != 0:
        raise ValueError('ambiguous or incomplete audited text coverage')
    sample = text_audit.parent / f'snapshot-{snapshot}-encoder-sample.jsonl'
    if checksum(sample) != item['encoder_sample_sha256']:
        raise ValueError('sample differs from audited selection')
    records = read_records(sample)
    ids = [x['id'] for x in records]
    with MappedSnapshot(folder, expected_archive_sha256=ARTIFACTS[snapshot-1]['archive_sha256']) as mapped:
        vector_hash = checksum(folder / 'vecs.f4')
        if vector_hash != mapped.receipt['members']['vecs.f4']['sha256']:
            raise ValueError('reference vector payload changed')
        if checksum(folder / 'docnos.npids') != item['dense_coverage']['id_map_sha256']:
            raise ValueError('text audit uses a different identifier map')
        wanted = set(ids)
        positions = {key: pos for pos, key in enumerate(mapped.ids) if key in wanted}
        if positions.keys() != wanted:
            raise ValueError('sample includes unindexed IDs')
        vectors = np.array(mapped.vectors[[positions[key] for key in ids]], dtype='<f4', copy=True)
    compare_arrays(ids, vectors, ids, vectors)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(sample, output / 'records.jsonl')
    atomic_json(output / 'ids.json', ids)
    np.save(output / 'vectors.npy', vectors, allow_pickle=False)
    report = {'status': 'completed', 'kind': 'audited document reference sample', 'snapshot': snapshot,
              'count': len(ids), 'selection': 'bottom SHA256(longeval-encoder-check-v1:ID) across eligible snapshot IDs',
              'text_audit_sha256': checksum(text_audit), 'archive_sha256': item['archive_sha256'],
              'reference_vector_sha256': vector_hash, 'source_sha256': checksum(__file__),
              'files': {n: checksum(output / n) for n in ('records.jsonl', 'ids.json', 'vectors.npy')}}
    atomic_json(output / 'reference.json', report)
    return report


def compare_encoding(reference_folder, encoded_folder, output):
    reference_folder, encoded_folder, output = Path(reference_folder), Path(encoded_folder), Path(output)
    if output.exists():
        raise ValueError('comparison output already exists')
    ref = json.loads((reference_folder / 'reference.json').read_text())
    enc = json.loads((encoded_folder / 'encoding.json').read_text())
    if ref.get('status') != 'completed' or enc.get('status') != 'completed':
        raise ValueError('requires completed reference and encoding receipts')
    for name in ('records.jsonl', 'ids.json', 'vectors.npy'):
        if checksum(reference_folder / name) != ref['files'][name]:
            raise ValueError('reference sample checksum mismatch')
    for name, key in (('vectors.npy', 'vectors_sha256'), ('ids.json', 'ids_sha256')):
        if checksum(encoded_folder / name) != enc[key]:
            raise ValueError('encoded output checksum mismatch')
    if enc['input_sha256'] != ref['files']['records.jsonl']:
        raise ValueError('encoder used different input text')
    ids = json.loads((reference_folder / 'ids.json').read_text())
    if ids != [r['id'] for r in read_records(reference_folder / 'records.jsonl')]:
        raise ValueError('reference records and vector IDs disagree')
    measured = compare_arrays(ids, np.load(reference_folder / 'vectors.npy', allow_pickle=False),
                              json.loads((encoded_folder / 'ids.json').read_text()),
                              np.load(encoded_folder / 'vectors.npy', allow_pickle=False))
    result = {'status': 'completed', 'purpose': 'document agreement diagnostic; not query compatibility certification',
              'reference_sha256': checksum(reference_folder / 'reference.json'),
              'encoding_sha256': checksum(encoded_folder / 'encoding.json'), 'source_sha256': checksum(__file__),
              'dtype': enc['dtype'], 'device': enc['device'], 'acceptance_threshold': None, **measured}
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare-sample')
    prepare.add_argument('folder', type=Path)
    prepare.add_argument('text_audit', type=Path)
    prepare.add_argument('output', type=Path)
    prepare.add_argument('--snapshot', type=int, default=1)
    compare = sub.add_parser('compare')
    compare.add_argument('reference', type=Path)
    compare.add_argument('encoded', type=Path)
    compare.add_argument('output', type=Path)
    args = parser.parse_args()
    result = (prepare_sample(args.folder, args.text_audit, args.output, snapshot=args.snapshot)
              if args.command == 'prepare-sample' else compare_encoding(args.reference, args.encoded, args.output))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
