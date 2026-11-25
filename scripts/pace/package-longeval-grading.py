"""Package the E1 grading job (LONGEVAL_GRADING.md, amendment 5) with its two grading folders.

Usage, from the repository root:
    python scripts/pace/package-longeval-grading.py BUILD_DIR

BUILD_DIR/longeval-grading-01 holds `primary` (coverage.prepare) and `substitutions` (grading_inputs),
written beforehand. The archive holds LongEval texts and is never committed.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.longeval.coverage import INPUT_FILES, read_inputs  # noqa: E402
from research.longeval.judge import MODEL_ID, MODEL_REVISION  # noqa: E402

NAME = 'longeval-grading-01'
FOLDERS = ('substitutions', 'primary')
EXTRA = ('research/__init__.py', 'research/longeval/judge-requirements.txt', 'research/longeval/judge-kernels.txt',
         'research/longeval/judge-model.json', 'research/longeval/requirements.txt',
         'research/longeval/docs/protocol/JUDGE_VALIDATION_THRESHOLDS.md',
         'research/longeval/tests/test_judge.py', 'research/longeval/tests/test_coverage.py',
         'research/longeval/tests/test_grading_inputs.py',
         'scripts/pace/longeval-node-env.sh', 'scripts/pace/longeval-grading.sbatch',
         'scripts/pace/longeval-judge-server.yaml', 'scripts/pace/LONGEVAL_GRADING.md',
         'scripts/pace/package-longeval-grading.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    source = Path(sys.argv[1]).resolve() / NAME
    contracts = {name: read_inputs(source / name) for name in FOLDERS}
    if contracts['primary']['pool_pairs'] != 6298 or contracts['primary']['validation_pairs'] != 1000:
        raise ValueError('the primary folder is not decision 3\'s pool and validation sample')
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for folder in FOLDERS:
        for name in (*INPUT_FILES, 'inputs.json'):
            payload[f'inputs/grading/{folder}/{name}'] = (source / folder / name).read_bytes()
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch', '.yaml')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/longeval-grading.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script or 'build/pace-longeval-grading/jobs/' not in script
            or f'sbatch {NAME}/' not in payload['scripts/pace/LONGEVAL_GRADING.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'E1 grading of the judging pool (decisions 1-3, amendment 5)',
                'model': MODEL_ID, 'model_revision': MODEL_REVISION,
                'folders': {name: {'pairs': c['pairs'], 'validation_pairs': c['validation_pairs'], 'pool_pairs': c['pool_pairs'],
                                   'inputs_sha256': sha(payload[f'inputs/grading/{name}/inputs.json'])}
                            for name, c in contracts.items()},
                'files': {n: sha(r) for n, r in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    payload['SHA256SUMS'] = ''.join(f'{sha(r)}  {n}\n' for n, r in sorted(payload.items())).encode()
    output = source.parent / (NAME + '.tar.gz')
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
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json'])}
    (source.parent / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
