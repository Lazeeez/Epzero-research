"""Package the pseudo-query GPU encoding job with its prepared inputs and checksums."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from research.longeval.pseudo_queries import validate_inputs

ROOT = Path.cwd()
NAME = 'longeval-pseudoquery-02'
INPUTS = ROOT / 'build/longeval-pseudoquery-inputs-01'
EXTRA = ('research/__init__.py', 'research/longeval/encoder-requirements.txt',
         'research/longeval/docs/repair/PSEUDO_QUERY_PLAN.md', 'research/longeval/tests/test_pseudo_queries.py',
         'scripts/pace/longeval-node-env.sh', 'scripts/pace/longeval-pseudoquery.sbatch',
         'scripts/pace/LONGEVAL_PSEUDOQUERY.md', 'scripts/pace/package-longeval-pseudoquery.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    contract, _ = validate_inputs(INPUTS)
    payload = {path.relative_to(ROOT).as_posix(): path.read_bytes()
               for path in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for path in sorted(INPUTS.iterdir()):
        payload['inputs/' + path.name] = path.read_bytes()
    manifest = {'package': NAME, 'purpose': 'pseudo-query FP32 encoding, stage 1 of PSEUDO_QUERY_PLAN.md',
                'input_counts': contract['counts'], 'inputs_sha256': sha(payload['inputs/pseudo-inputs.json']),
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
