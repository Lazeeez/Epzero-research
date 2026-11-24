"""Package the repaired-Epzero arm at full scale (exploratory follow-up; LONGEVAL_ENGINE_ARM.md).
Package -01 held grown, repaired and rebuilt; -02 holds grown and repaired8 (8 links, screen D13).

Usage, from the repository root:
    python scripts/pace/package-longeval-engine-arm.py BUILD_DIR ORYX_DIR

BUILD_DIR holds `longeval-heldout-inputs-01` (encoded test and training queries and exact
references). ORYX_DIR is the Epzero checkout: the `engine` folder of its `min-backlinks` branch is
packaged (tracked files only, via git archive) and its commit recorded. No judgments. The archive is
derived from LongEval data and is never committed.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'longeval-engine-arm-02'
ARMS = ['grown', 'repaired8']
BRANCH = 'min-backlinks'
EXTRA = ('research/__init__.py', 'research/longeval/requirements.txt', 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json',
         'research/longeval/EXTENSION_PROTOCOL_V1_FREEZE.json', 'research/longeval/tests/test_engine_arm_analysis.py',
         'research/longeval/tests/test_post_freeze_analyses.py', 'scripts/pace/longeval-node-env.sh',
         'scripts/pace/longeval-engine-arm.sbatch', 'scripts/pace/LONGEVAL_ENGINE_ARM.md',
         'scripts/pace/package-longeval-engine-arm.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def engine_source(oryx):
    commit = subprocess.check_output(['git', 'rev-parse', BRANCH], cwd=oryx, text=True).strip()
    # core.autocrlf off: package the committed bytes, not a Windows checkout's line endings.
    raw = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'archive', '--format=tar', commit, 'engine'], cwd=oryx)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                files[member.name] = tar.extractfile(member).read()
    if 'engine/pyproject.toml' not in files or not any(b'extra_backlinks' in raw for raw in files.values()):
        raise ValueError(f'the {BRANCH} engine source lacks the repair options')
    return commit, files


def main():
    build, oryx = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    inputs = build / 'longeval-heldout-inputs-01'
    sys.path.insert(0, str(ROOT))
    from research.longeval.heldout import check
    for receipt in ('FINAL_PROTOCOL_V1_FREEZE.json', 'EXTENSION_PROTOCOL_V1_FREEZE.json'):
        freeze = json.loads((ROOT / 'research/longeval' / receipt).read_text())
        for name, expected in freeze['files'].items():
            if sha((ROOT / name).read_bytes()) != expected:
                raise ValueError(f'frozen file changed after the freeze: {name}')
    preflight = check(inputs, ROOT / 'research/longeval/trajectory-development.json', ROOT / 'research/longeval/ivf-development.json')
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    commit, engine = engine_source(oryx)
    payload.update(engine)
    for path in sorted(inputs.rglob('*')):
        if path.is_file():
            payload['inputs/' + path.relative_to(inputs).as_posix()] = path.read_bytes()
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/longeval-engine-arm.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script
            or f'{NAME}/scripts/pace/longeval-engine-arm.sbatch' not in payload['scripts/pace/LONGEVAL_ENGINE_ARM.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    if f'RUN_ARMS="{" ".join(ARMS)}"' not in script:
        raise ValueError('job script runs other arms than the manifest records')
    manifest = {'package': NAME, 'purpose': 'repaired-Epzero arm at full scale; post-freeze; no judgments',
                'engine': {'repository': 'Epzero', 'branch': BRANCH, 'commit': commit, 'files': len(engine)},
                'arms': ARMS, 'seeds': [11, 29, 47], 'preflight': preflight,
                'files': {n: sha(r) for n, r in sorted(payload.items())}}
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
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json']), 'engine_commit': commit}
    (build / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
