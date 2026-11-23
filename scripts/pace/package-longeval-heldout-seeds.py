"""Package the further-seeds follow-up of the held-out primary arms (exploratory).

Usage, from the repository root:
    python scripts/pace/package-longeval-heldout-seeds.py BUILD_DIR

BUILD_DIR holds `longeval-heldout-inputs-01` (encoded queries and exact references, already used by the
frozen run). The package ships the research code (the frozen files unchanged; the freeze receipt is
checked), the new `heldout_seeds.py`, the inputs, the configurations and the job script. No judgments.
The archive is derived from LongEval data: never commit it.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'longeval-heldout-seeds-01'
EXTRA = ('research/__init__.py', 'research/longeval/requirements.txt', 'research/longeval/trajectory-development.json',
         'research/longeval/ivf-development.json', 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json',
         'research/longeval/tests/test_heldout_seeds.py', 'research/longeval/tests/test_heldout.py',
         'research/longeval/tests/test_trajectory.py', 'research/longeval/tests/test_ivf_trajectory.py',
         'research/longeval/tests/test_mapped.py', 'scripts/pace/longeval-node-env.sh',
         'scripts/pace/longeval-heldout-seeds.sbatch', 'scripts/pace/LONGEVAL_HELDOUT_SEEDS.md',
         'scripts/pace/package-longeval-heldout-seeds.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    build = Path(sys.argv[1]).resolve()
    inputs = build / 'longeval-heldout-inputs-01'
    sys.path.insert(0, str(ROOT))
    from research.longeval.heldout import check
    from research.longeval.heldout_seeds import EXTRA_SEEDS
    freeze = json.loads((ROOT / 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json').read_text())
    for name, expected in freeze['files'].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError(f'frozen file changed after the freeze: {name}')
    preflight = check(inputs, ROOT / 'research/longeval/trajectory-development.json',
                      ROOT / 'research/longeval/ivf-development.json')
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for path in sorted(inputs.rglob('*')):
        if path.is_file():
            payload['inputs/' + path.relative_to(inputs).as_posix()] = path.read_bytes()
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/longeval-heldout-seeds.sbatch'].decode()
    allowed = '|'.join(map(str, EXTRA_SEEDS))
    if (f'PACKAGE="$ROOT/{NAME}"' not in script or f'case "$SEED" in {allowed})' not in script
            or f'{NAME}/scripts/pace/longeval-heldout-seeds.sbatch' not in payload['scripts/pace/LONGEVAL_HELDOUT_SEEDS.md'].decode()):
        raise ValueError('job script or runbook does not match this package or its seeds')
    manifest = {'package': NAME, 'purpose': 'EXPLORATORY further construction seeds for the held-out primary arms; no judgments',
                'seeds': list(EXTRA_SEEDS), 'freeze_sha256': sha(payload['research/longeval/FINAL_PROTOCOL_V1_FREEZE.json']),
                'preflight': preflight, 'files': {n: sha(r) for n, r in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    payload['SHA256SUMS'] = ''.join(f'{sha(r)}  {n}\n' for n, r in sorted(payload.items())).encode()
    output = build / (NAME + '.tar.gz')
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
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json']), 'seeds': list(EXTRA_SEEDS)}
    (build / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
