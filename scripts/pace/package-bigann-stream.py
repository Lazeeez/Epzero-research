"""Package the public streaming workload (exploratory follow-up; BIGANN_STREAM.md).

Usage, from the repository root:
    python scripts/pace/package-bigann-stream.py BUILD_DIR ORYX_DIR

Holds the research code, the job script and the Epzero `engine` folder of the `min-backlinks` branch
(tracked files only, via git archive; its commit recorded). No data: every job downloads the public
stream itself. The archive is written to BUILD_DIR and never committed.
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
NAME = 'bigann-stream-02'                  # -01 had four engine arms; -02 adds repaired8w (screen D15)
BRANCH = 'min-backlinks'
EXTRA = ('research/__init__.py', 'research/longeval/requirements.txt', 'research/longeval/tests/test_bigann_stream.py',
         'research/longeval/tests/test_order_pairs.py',
         'research/longeval/tests/test_engine_arm_analysis.py', 'scripts/pace/longeval-node-env.sh',
         'scripts/pace/bigann-stream.sbatch', 'scripts/pace/BIGANN_STREAM.md', 'scripts/pace/package-bigann-stream.py')


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
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    commit, engine = engine_source(oryx)
    payload.update(engine)
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:      # shell scripts only; Python reads either
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/bigann-stream.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script
            or f'{NAME}/scripts/pace/bigann-stream.sbatch' not in payload['scripts/pace/BIGANN_STREAM.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'public streaming workload (Big-ANN clustered MSTuring-10M); post-freeze; no LongEval data',
                'engine': {'repository': 'Epzero', 'branch': BRANCH, 'commit': commit, 'files': len(engine)},
                'parts': {'order': {'backends': ['epzero', 'faiss_hnsw'], 'orders': ['appended', 'reversed', 'shuffled']},
                          'engine': {'arms': ['grown', 'repaired', 'repaired8', 'repaired8w', 'rebuilt']}},
                'seeds': [11, 29, 47], 'files': {n: sha(r) for n, r in sorted(payload.items())}}
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
