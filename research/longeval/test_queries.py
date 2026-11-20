"""Held-out test-query encoding (FINAL_PROTOCOL_V1_CANDIDATE.md, step 1).

Produces query vectors only: no exact search, no ANN search and no judgments are read.
Uses the validated batch-1 path in ``encoder.encode_documents`` (CUDA FP32, TF32 off),
identical to the training-query encoding of job 7107422. Before test queries are encoded,
the training queries are re-encoded and must match the verified vectors (cosine gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import encoder
from .query_diagnostic import PROMPT
from .release import checksum, read_queries
from .storage import atomic_json

TEST_FILE = 'longeval_adhoc-queries-snapshot-test.tsv'
TEST_SHA256 = 'c7ea5fed1aa1dd37b647edfd686acd9cb26891b1cde52c43e3dc361f4942e254'
COSINE_GATE = 0.99999


def records(queries):
    rows = []
    for key in sorted(queries):
        text = PROMPT + queries[key]
        rows.append({'id': key, 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    return rows


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def prepare(release, reference, output):
    """``reference``: verified job-7107422 folder (inputs/ and float32/encoder/)."""
    release, reference, output = map(Path, (release, reference, output))
    source = release / 'files' / TEST_FILE
    if checksum(source) != TEST_SHA256:
        raise ValueError('test query release identity mismatch')
    test = read_queries(source)
    if len(test) != 381:
        raise ValueError('expected the 381 released test queries')
    ref_encoding = json.loads((reference / 'float32/encoder/encoding.json').read_text())
    if (ref_encoding['dtype'] != 'float32' or ref_encoding['device'] != 'cuda' or ref_encoding.get('tf32') is not False
            or ref_encoding['vectors_sha256'] != checksum(reference / 'float32/encoder/vectors.npy')):
        raise ValueError('reference training encoding identity mismatch')
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'test-records.jsonl', records(test))
    shutil.copyfile(reference / 'inputs/records.jsonl', output / 'training-records.jsonl')
    shutil.copyfile(reference / 'float32/encoder/vectors.npy', output / 'training-vectors.npy')
    shutil.copyfile(reference / 'float32/encoder/ids.json', output / 'training-ids.json')
    names = ('test-records.jsonl', 'training-records.jsonl', 'training-vectors.npy', 'training-ids.json')
    contract = {'status': 'prepared', 'purpose': 'held-out test-query vectors; no retrieval outcome',
                'prompt': PROMPT, 'test_count': len(test), 'test_source_sha256': TEST_SHA256,
                'reference_encoding_sha256': checksum(reference / 'float32/encoder/encoding.json'),
                'source_sha256': checksum(__file__), 'held_out_retrieval_evaluated': False,
                'files': {n: checksum(output / n) for n in names}}
    atomic_json(output / 'test-inputs.json', contract)
    return contract


def validate_inputs(folder):
    folder = Path(folder)
    contract = json.loads((folder / 'test-inputs.json').read_text())
    if contract.get('status') != 'prepared' or contract.get('prompt') != PROMPT:
        raise ValueError('invalid test-query input contract')
    for name, expected in contract['files'].items():
        if checksum(folder / name) != expected:
            raise ValueError(f'test-query input checksum mismatch: {name}')
    rows = encoder.read_records(folder / 'test-records.jsonl')
    if len(rows) != contract['test_count'] or [r['id'] for r in rows] != sorted(r['id'] for r in rows):
        raise ValueError('test-query records changed')
    if json.loads((folder / 'training-ids.json').read_text()) != [r['id'] for r in encoder.read_records(folder / 'training-records.jsonl')]:
        raise ValueError('training reference order changed')
    return contract


def cosine_check(left, right):
    a, b = left.astype(np.float64), right.astype(np.float64)
    cos = np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return {'min_cosine': float(cos.min()), 'mean_cosine': float(cos.mean()),
            'max_abs_difference': float(np.abs(a - b).max()), 'gate': COSINE_GATE,
            'passed': bool(cos.min() >= COSINE_GATE)}


def encode(model, inputs, output, *, threads=4):
    inputs, output = Path(inputs), Path(output)
    contract = validate_inputs(inputs)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'inputs_sha256': checksum(inputs / 'test-inputs.json'), 'source_sha256': checksum(__file__)}
    atomic_json(output / 'test-encoding.json', report)
    try:
        training = encoder.encode_documents(model, inputs / 'training-records.jsonl', output / 'training',
                                            dtype='float32', device='cuda', threads=threads, max_length=8192)
        check = cosine_check(np.load(output / 'training/vectors.npy'), np.load(inputs / 'training-vectors.npy'))
        report['training_check'] = check
        atomic_json(output / 'test-encoding.json', report)
        if not check['passed']:
            raise ValueError('training re-encoding differs from the verified vectors')
        raw = encoder.encode_documents(model, inputs / 'test-records.jsonl', output / 'test',
                                       dtype='float32', device='cuda', threads=threads, max_length=8192)
        ids = json.loads((output / 'test/ids.json').read_text())
        values = np.load(output / 'test/vectors.npy', allow_pickle=False)
        if ids != [r['id'] for r in encoder.read_records(inputs / 'test-records.jsonl')] or values.shape != (contract['test_count'], 2560):
            raise ValueError('test encoding population or shape mismatch')
        report.update(status='completed', dtype=raw['dtype'], device=raw['device'], tf32=raw.get('tf32'),
                      gpu=raw.get('gpu'), files={n: checksum(output / n) for n in
                      ('test/encoding.json', 'test/ids.json', 'test/vectors.npy')})
        atomic_json(output / 'test-encoding.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)[:500])
        atomic_json(output / 'test-encoding.json', report)
        raise


def load_test_queries(inputs, encoded):
    """Validated test IDs and vectors for downstream jobs."""
    inputs, encoded = Path(inputs), Path(encoded)
    validate_inputs(inputs)
    report = json.loads((encoded / 'test-encoding.json').read_text())
    if (report.get('status') != 'completed' or report['inputs_sha256'] != checksum(inputs / 'test-inputs.json')
            or not report['training_check']['passed'] or report['dtype'] != 'float32'
            or report['device'] != 'cuda' or report.get('tf32') is not False):
        raise ValueError('test encoding contract mismatch')
    for name, expected in report['files'].items():
        if checksum(encoded / name) != expected:
            raise ValueError(f'test encoding checksum mismatch: {name}')
    ids = json.loads((encoded / 'test/ids.json').read_text())
    values = np.load(encoded / 'test/vectors.npy', allow_pickle=False)
    if (values.dtype != np.float32 or values.shape != (len(ids), 2560) or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values.astype(np.float64), axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError('invalid test query vectors')
    return ids, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('prepare'); s.add_argument('release', type=Path); s.add_argument('reference', type=Path); s.add_argument('output', type=Path)
    s = sub.add_parser('validate'); s.add_argument('inputs', type=Path)
    s = sub.add_parser('encode'); s.add_argument('model', type=Path); s.add_argument('inputs', type=Path); s.add_argument('output', type=Path)
    s.add_argument('--threads', type=int, default=4)
    s = sub.add_parser('verify'); s.add_argument('inputs', type=Path); s.add_argument('encoded', type=Path)
    a = p.parse_args()
    if a.command == 'prepare':
        r = prepare(a.release, a.reference, a.output)
    elif a.command == 'validate':
        r = {'status': 'valid', 'test_count': validate_inputs(a.inputs)['test_count']}
    elif a.command == 'encode':
        r = encode(a.model, a.inputs, a.output, threads=a.threads)
    else:
        ids, _ = load_test_queries(a.inputs, a.encoded)
        r = {'status': 'verified', 'test_count': len(ids)}
    print(json.dumps({k: v for k, v in r.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
