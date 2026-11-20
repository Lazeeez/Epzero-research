"""Package the held-out test-query GPU encoding job with its prepared inputs and checksums."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from research.longeval.test_queries import validate_inputs

ROOT = Path.cwd()
NAME = 'longeval-test-queries-03'
INPUTS = ROOT / 'build/longeval-test-query-inputs-01'
EXTRA = ('research/__init__.py', 'research/longeval/encoder-requirements.txt',
         'research/longeval/docs/protocol/FINAL_PROTOCOL_V1_CANDIDATE.md', 'research/longeval/tests/test_test_queries.py',
         'scripts/pace/longeval-node-env.sh', 'scripts/pace/longeval-test-queries.sbatch',
         'scripts/pace/LONGEVAL_TEST_QUERIES.md', 'scripts/pace/package-longeval-test-queries.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    contract = validate_inputs(INPUTS)
    payload = {path.relative_to(ROOT).as_posix(): path.read_bytes()
               for path in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for path in sorted(INPUTS.iterdir()):
        payload['inputs/' + path.name] = path.read_bytes()
    # Package 01 named the wrong package directory; package 02 failed on a second interop-thread call.
    script = payload['scripts/pace/longeval-test-queries.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script or 'build/pace-longeval-test-queries/jobs/' not in script
            or f'sbatch {NAME}/' not in payload['scripts/pace/LONGEVAL_TEST_QUERIES.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'held-out test-query FP32 encoding; no retrieval outcome',
                'test_count': contract['test_count'], 'inputs_sha256': sha(payload['inputs/test-inputs.json']),
                'held_out_retrieval_evaluated': False,
                'files': {name: sha(raw) for name, raw in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    payload['SHA256SUMS'] = ''.join(f'{sha(raw)}  {name}\n' for name, raw in sorted(payload.items())).encode()
    unpacked = ROOT / 'build' / (NAME + '-check')
    unpacked.mkdir(exist_ok=False)
    for name, raw in payload.items():
        target = unpacked / name
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(raw)
    output = ROOT / 'build' / (NAME + '.tar.gz')
    with output.open('xb') as target, gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for name, raw in sorted(payload.items()):
                entry = tarfile.TarInfo(NAME + '/' + name)
                entry.size, entry.mode = len(raw), 0o755 if name.endswith(('.sh', '.sbatch')) else 0o644
                tar.addfile(entry, io.BytesIO(raw))
    with tarfile.open(output) as tar:
        if {m.name for m in tar} != {NAME + '/' + n for n in payload}:
            raise ValueError('package member inventory differs')
        for name, raw in payload.items():
            if tar.extractfile(NAME + '/' + name).read() != raw:
                raise ValueError('archive roundtrip differs')
    receipt = {'archive': output.name, 'sha256': sha(output.read_bytes()), 'bytes': output.stat().st_size,
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json']),
               'inputs_sha256': manifest['inputs_sha256'], 'gpu_runtime_completed': False}
    (ROOT / 'build' / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
