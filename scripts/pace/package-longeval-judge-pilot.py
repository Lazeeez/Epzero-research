"""Package the judge-pilot GPU job with its prepared inputs and checksums.

Usage, from the repository root:
    python scripts/pace/package-longeval-judge-pilot.py INPUTS OUTPUT_DIR

INPUTS is the folder written by `python -m research.longeval.judge prepare`. The archive and its
receipt are written to OUTPUT_DIR. The archive holds LongEval query and document text, so it
belongs under an ignored build directory and is never committed.
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
from research.longeval.judge import MODEL_ID, MODEL_REVISION, validate_inputs  # noqa: E402

NAME = 'longeval-judge-pilot-02'
EXTRA = ('research/__init__.py', 'research/longeval/judge-requirements.txt', 'research/longeval/judge-kernels.txt',
         'research/longeval/judge-model.json',
         'research/longeval/docs/protocol/JUDGE_VALIDATION_THRESHOLDS.md', 'research/longeval/tests/test_judge.py',
         'scripts/pace/longeval-node-env.sh', 'scripts/pace/longeval-judge-pilot.sbatch',
         'scripts/pace/longeval-judge-server.yaml', 'scripts/pace/LONGEVAL_JUDGE_PILOT.md',
         'scripts/pace/package-longeval-judge-pilot.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    inputs, target = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    contract = validate_inputs(inputs)
    payload = {path.relative_to(ROOT).as_posix(): path.read_bytes()
               for path in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for path in sorted(inputs.iterdir()):
        payload['inputs/' + path.name] = path.read_bytes()
    # Shell files must reach the node with LF line endings, whatever the checkout did. (Two frozen
    # Python files are stored with CRLF on purpose; Python reads them either way.)
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch', '.yaml')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/longeval-judge-pilot.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script or 'build/pace-longeval-judge-pilot/jobs/' not in script
            or f'sbatch {NAME}/' not in payload['scripts/pace/LONGEVAL_JUDGE_PILOT.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'judge pilot on prompt-development pairs; no study outcome',
                'model': MODEL_ID, 'model_revision': MODEL_REVISION, 'pairs': contract['pairs'],
                'inputs_sha256': sha(payload['inputs/inputs.json']), 'prompt_sha256': contract['prompt_sha256'],
                'judge_output_exists': False,
                'files': {name: sha(raw) for name, raw in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    payload['SHA256SUMS'] = ''.join(f'{sha(raw)}  {name}\n' for name, raw in sorted(payload.items())).encode()
    target.mkdir(parents=True, exist_ok=True)
    output = target / (NAME + '.tar.gz')
    with output.open('xb') as stream, gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for name, raw in sorted(payload.items()):
                entry = tarfile.TarInfo(NAME + '/' + name)
                entry.size, entry.mode = len(raw), 0o755 if name.endswith(('.sh', '.sbatch')) else 0o644
                tar.addfile(entry, io.BytesIO(raw))
    with tarfile.open(output) as tar:
        if {member.name for member in tar} != {NAME + '/' + name for name in payload}:
            raise ValueError('package member inventory differs')
        for name, raw in payload.items():
            if tar.extractfile(NAME + '/' + name).read() != raw:
                raise ValueError('archive roundtrip differs')
    receipt = {'archive': output.name, 'sha256': sha(output.read_bytes()), 'bytes': output.stat().st_size,
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json']),
               'inputs_sha256': manifest['inputs_sha256'], 'gpu_runtime_completed': False}
    (target / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
