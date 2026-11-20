"""Prepare paired, checked sample inputs for the pinned native NGFix competitor."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .query_diagnostic import load_queries, validate_inputs
from .release import checksum
from .storage import atomic_json


def write_matrix(path, values, dtype):
    values = np.asarray(values)
    if values.ndim != 2 or not all(values.shape) or max(values.shape) >= 2**32:
        raise ValueError('matrix requires positive uint32 dimensions')
    if np.dtype(dtype) not in (np.dtype('<f4'), np.dtype('<i4')):
        raise ValueError('unsupported matrix dtype')
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite matrix')
    if np.dtype(dtype).kind == 'i' and (np.any(values != np.floor(values))
            or np.any(values < 0) or np.any(values > np.iinfo(np.int32).max)):
        raise ValueError('invalid neighbor IDs')
    converted = np.ascontiguousarray(values, dtype=dtype)
    if not np.all(np.isfinite(converted)):
        raise ValueError('matrix conversion overflow')
    with Path(path).open('xb') as stream:
        stream.write(struct.pack('<II', *converted.shape))
        stream.write(converted.tobytes())


def read_matrix(path, dtype):
    with Path(path).open('rb') as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError('short matrix header')
        n, d = struct.unpack('<II', header)
        if not n or not d or Path(path).stat().st_size != 8 + n*d*4:
            raise ValueError('matrix length mismatch')
        if np.dtype(dtype) not in (np.dtype('<f4'), np.dtype('<i4')):
            raise ValueError('unsupported matrix dtype')
        return np.frombuffer(stream.read(), dtype=dtype).reshape(n, d).copy()


def prepare_sample(sample, inputs, encoding, output):
    import epzero
    sample, inputs, encoding, output = map(Path, (sample, inputs, encoding, output))
    original = json.loads((sample / 'report.json').read_text())
    for name in ('state-before.index', 'sample-rows.npy', 'exact.json'):
        if checksum(sample / name) != original['artifacts'][name]:
            raise ValueError('sample artifact checksum mismatch')
    labels = np.load(sample / 'sample-rows.npy', allow_pickle=False)
    index = epzero.Index.load(str(sample / 'state-before.index'))
    data = np.ascontiguousarray([index.get_vector(int(label)) for label in labels], dtype=np.float32)
    del index
    if hashlib.sha256(data.tobytes()).hexdigest() != original['sample_vectors_sha256']:
        raise ValueError('sample vector identity mismatch')
    ids, queries, _ = load_queries(inputs, encoding)
    contract, _ = validate_inputs(inputs)
    roles = contract['roles']
    ordered_ids = roles['calibration'] + roles['development']
    ordered_queries = np.ascontiguousarray([queries[ids.index(q)] for q in ordered_ids])
    calibration = ordered_queries[:len(roles['calibration'])]
    similarities = calibration.astype(np.float64) @ data.astype(np.float64).T
    exact200 = np.asarray([np.lexsort((labels, -row))[:200] for row in similarities], dtype='<i4')
    frozen = json.loads((sample / 'exact.json').read_text())
    for q, neighbors in zip(roles['calibration'], exact200):
        if frozen[q] != list(map(int, labels[neighbors[:100]])):
            raise ValueError('recomputed calibration reference differs')
    positions = {int(v): i for i, v in enumerate(labels)}
    truth100 = {q: [positions[v] for v in frozen[q]] for q in ordered_ids}
    output.mkdir(exist_ok=False)
    write_matrix(output / 'base.fbin', data, '<f4')
    write_matrix(output / 'queries.fbin', ordered_queries, '<f4')
    write_matrix(output / 'calibration_top200.ibin', exact200, '<i4')
    for name, expected, dtype in [('base.fbin', data, '<f4'), ('queries.fbin', ordered_queries, '<f4'),
                                   ('calibration_top200.ibin', exact200, '<i4')]:
        if not np.array_equal(read_matrix(output / name, dtype), expected):
            raise ValueError('binary roundtrip mismatch')
    atomic_json(output / 'queries.json', {'order': ordered_ids, 'roles': roles, 'exact100': truth100})
    atomic_json(output / 'labels.json', list(map(int, labels)))
    atomic_json(output / 'report.json', {'status': 'prepared',
        'upstream_commit': '9acd987cf1c420069f8fd602c0d4e4aa81053558',
        'source_sha256': checksum(__file__), 'driver_sha256': checksum(Path(__file__).with_name('ngfix_driver.cpp')),
        'sample_report_sha256': checksum(sample / 'report.json'),
        'query_input_sha256': checksum(inputs / 'query-inputs.json'),
        'query_encoding_sha256': checksum(encoding / 'query-encoding.json'),
        'base_shape': list(data.shape), 'query_count': len(ordered_ids),
        'calibration_count': len(calibration), 'calibration_exact_depth': 200,
        'calibration_reference_vector_pairs': int(len(calibration)*len(data)),
        'held_out_retrieval_evaluated': False, 'native_execution_completed': False,
        'limits': ['Small exposed development sample; not full-data comparison',
                   '50 fitting queries are an integration screen, not a fair training budget conclusion',
                   'Original native AVX-512 kernels require compatible hardware'],
        'artifacts': {name: checksum(output / name) for name in
            ('base.fbin', 'queries.fbin', 'calibration_top200.ibin', 'queries.json', 'labels.json')}})
    print(json.dumps({'status': 'prepared', 'output': str(output), 'calibration_count': len(calibration)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sample', 'inputs', 'encoding', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    prepare_sample(**vars(parser.parse_args()))
