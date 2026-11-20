"""Training-only query precision diagnostic; not an approved research representation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import encoder
from .release import MANIFEST_SHA256, TRAIN, checksum, manifest_entries, read_queries
from .representation import compare_arrays
from .storage import atomic_json

BASELINE_REVISION = 'a488f50f2132f38c07631f484eec0b67f3a18926'
BASELINE_SHA256 = 'a85554638a09418362fa0d6f78b43d1b89f466868952d691875c9a693680e883'
PROMPT = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'


def query_records(queries):
    if not queries or any(not isinstance(k, str) or not k or any(c.isspace() for c in k)
                          or not isinstance(v, str) or not v.strip() for k, v in queries.items()):
        raise ValueError('invalid query IDs or text')
    ordered = sorted(queries, key=lambda key: hashlib.sha256(('longeval-role-v1:' + key).encode()).digest())
    roles = {'development': sorted(ordered[:len(ordered)//2]), 'calibration': sorted(ordered[len(ordered)//2:])}
    records = []
    for key in sorted(queries):
        text = PROMPT + queries[key]
        records.append({'id': key, 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    return records, roles


def prepare_queries(release_folder, baseline, output):
    release_folder, baseline, output = Path(release_folder), Path(baseline), Path(output)
    if checksum(baseline) != BASELINE_SHA256:
        raise ValueError('baseline source identity mismatch')
    item = manifest_entries(release_folder)[TRAIN]
    source = release_folder / item['cache_path']
    if checksum(source, 'md5') != item['expected_md5']:
        raise ValueError('training query release identity mismatch')
    queries = read_queries(source)
    if len(queries) != 100:
        raise ValueError('expected the pinned 100-query training release')
    records, roles = query_records(queries)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / 'queries.tsv')
    shutil.copyfile(baseline, output / 'baseline.py')
    with (output / 'records.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'status': 'prepared', 'purpose': 'training-query precision sensitivity only',
              'approved_for_primary_experiments': False, 'release_key': TRAIN, 'count': len(records),
              'release_manifest_sha256': MANIFEST_SHA256, 'query_source_md5': item['expected_md5'],
              'baseline_revision': BASELINE_REVISION, 'prompt': PROMPT, 'roles': roles,
              'source_sha256': checksum(__file__), 'files': {n: checksum(output / n)
                  for n in ('queries.tsv', 'baseline.py', 'records.jsonl')}}
    atomic_json(output / 'query-inputs.json', report)
    return report


def validate_inputs(folder):
    folder = Path(folder)
    report = json.loads((folder / 'query-inputs.json').read_text())
    if (report.get('status') != 'prepared' or report.get('approved_for_primary_experiments') is not False
            or report.get('release_key') != TRAIN or report.get('release_manifest_sha256') != MANIFEST_SHA256
            or report.get('baseline_revision') != BASELINE_REVISION or report.get('prompt') != PROMPT):
        raise ValueError('invalid training-query diagnostic contract')
    for name in ('queries.tsv', 'baseline.py', 'records.jsonl'):
        if checksum(folder / name) != report['files'][name]:
            raise ValueError('query input checksum mismatch')
    if checksum(folder / 'baseline.py') != BASELINE_SHA256 or checksum(folder / 'queries.tsv', 'md5') != report['query_source_md5']:
        raise ValueError('query source identity mismatch')
    expected, roles = query_records(read_queries(folder / 'queries.tsv'))
    rows = encoder.read_records(folder / 'records.jsonl')
    if rows != expected or len(rows) != 100 or report['count'] != 100 or report['roles'] != roles:
        raise ValueError('query text, ordering or role contract changed')
    return report, rows


def encode_queries(model, inputs, output, *, dtype='float32', device='cuda', threads=4):
    inputs, output = Path(inputs), Path(output)
    contract, rows = validate_inputs(inputs)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'kind': 'training-query-precision-diagnostic',
              'purpose': 'candidate query reconstruction; no primary-experiment approval',
              'approved_for_primary_experiments': False, 'prompt': PROMPT, 'count': len(rows),
              'query_inputs_sha256': checksum(inputs / 'query-inputs.json'), 'roles': contract['roles'],
              'source_sha256': checksum(__file__)}
    atomic_json(output / 'query-encoding.json', report)
    try:
        # Reuse the tested text encoder on already-prompted inputs. Preserve its original report
        # separately; this outer receipt declares the actual query semantics and prompt provenance.
        raw = encoder.encode_documents(model, inputs / 'records.jsonl', output / 'encoder',
                                       dtype=dtype, device=device, threads=threads, max_length=8192)
        validate_inputs(inputs)
        ids = json.loads((output / 'encoder/ids.json').read_text())
        if ids != [r['id'] for r in rows] or raw['input_sha256'] != contract['files']['records.jsonl']:
            raise ValueError('query encoder input/output identity mismatch')
        values = np.load(output / 'encoder/vectors.npy', allow_pickle=False)
        compare_arrays(ids, values, ids, values)
        if values.shape != (100, 2560):
            raise ValueError('query embedding dimensions changed')
        report.update(status='completed', dtype=raw['dtype'], device=raw['device'],
                      files={name: checksum(output / name) for name in
                             ('encoder/encoding.json', 'encoder/ids.json', 'encoder/vectors.npy')})
        atomic_json(output / 'query-encoding.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        atomic_json(output / 'query-encoding.json', report)
        raise


def load_queries(inputs, folder):
    inputs, folder = Path(inputs), Path(folder)
    contract, records = validate_inputs(inputs)
    report = json.loads((folder / 'query-encoding.json').read_text())
    if (report.get('status') != 'completed' or report.get('kind') != 'training-query-precision-diagnostic'
            or report.get('approved_for_primary_experiments') is not False or report.get('prompt') != PROMPT
            or report.get('query_inputs_sha256') != checksum(inputs / 'query-inputs.json')
            or report.get('roles') != contract['roles']):
        raise ValueError('query encoding contract mismatch')
    for name in ('encoder/encoding.json', 'encoder/ids.json', 'encoder/vectors.npy'):
        if checksum(folder / name) != report['files'][name]:
            raise ValueError('query encoding checksum mismatch')
    raw = json.loads((folder / 'encoder/encoding.json').read_text())
    if (raw.get('status') != 'completed' or raw['input_sha256'] != contract['files']['records.jsonl']
            or raw['dtype'] != report['dtype'] or raw['device'] != report['device']
            or raw['model_identity']['model'] != encoder.MODEL or raw['model_identity']['revision'] != encoder.REVISION
            or raw['vectors_sha256'] != report['files']['encoder/vectors.npy']
            or raw['ids_sha256'] != report['files']['encoder/ids.json']):
        raise ValueError('query model/output identity mismatch')
    ids = json.loads((folder / 'encoder/ids.json').read_text())
    values = np.load(folder / 'encoder/vectors.npy', allow_pickle=False)
    if ids != [r['id'] for r in records] or values.shape != (100, 2560):
        raise ValueError('query population/shape mismatch')
    compare_arrays(ids, values, ids, values)
    return ids, values, raw


def load_pair(inputs, bf16, fp32):
    ids, a, left = load_queries(inputs, bf16)
    other, b, right = load_queries(inputs, fp32)
    if ids != other or left['dtype'] != 'bfloat16' or right['dtype'] != 'float32':
        raise ValueError('requires matching BF16 and float32 query populations')
    for key in ('device', 'model_identity', 'packages', 'pooling', 'normalization', 'attention',
                'truncation', 'max_length', 'batch_size', 'threads', 'gpu', 'tf32', 'source_sha256'):
        if left.get(key) != right.get(key):
            raise ValueError(f'query precision pair changes another setting: {key}')
    if left['device'] != 'cuda' or left.get('tf32') is not False:
        raise ValueError('requires CUDA pair with TF32 disabled')
    if [(r['id'], r['tokens']) for r in left['rows']] != [(r['id'], r['tokens']) for r in right['rows']]:
        raise ValueError('query tokenization differs between precisions')
    return ids, a, b


def compare_precisions(inputs, bf16, fp32, output):
    output = Path(output)
    if output.exists():
        raise ValueError('comparison output already exists')
    ids, a, b = load_pair(inputs, bf16, fp32)
    report = {'status': 'completed', 'purpose': 'training-query precision sensitivity; no relevance claim',
              'acceptance_threshold': None, 'query_inputs_sha256': checksum(Path(inputs) / 'query-inputs.json'),
              'bf16_receipt_sha256': checksum(Path(bf16) / 'query-encoding.json'),
              'fp32_receipt_sha256': checksum(Path(fp32) / 'query-encoding.json'),
              'source_sha256': checksum(__file__), **compare_arrays(ids, a, ids, b)}
    atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('release', type=Path); p.add_argument('baseline', type=Path); p.add_argument('output', type=Path)
    p = sub.add_parser('encode'); p.add_argument('model', type=Path); p.add_argument('inputs', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='float32'); p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p = sub.add_parser('compare'); p.add_argument('inputs', type=Path); p.add_argument('bf16', type=Path); p.add_argument('fp32', type=Path); p.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare_queries(args.release, args.baseline, args.output)
    elif args.command == 'encode': result = encode_queries(args.model, args.inputs, args.output, dtype=args.dtype, device=args.device, threads=args.threads)
    else: result = compare_precisions(args.inputs, args.bf16, args.fp32, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'roles')}, indent=2))


if __name__ == '__main__':
    main()
