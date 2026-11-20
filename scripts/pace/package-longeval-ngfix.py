"""Produce an isolated NGFix screen package with no corpus vectors or binary indexes."""
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from research.longeval.release import checksum

ROOT = Path.cwd()
# Local extracted dense snapshots (see research/longeval/docs/data/INGESTION.md).
DENSE = Path(os.environ.get('LONGEVAL_DENSE', Path.home() / 'epzero-longeval/dense'))
NAME = 'longeval-ngfix-screen-02'
UPSTREAM = ROOT / 'build/third-party/ngfix'
COMMIT = '9acd987cf1c420069f8fd602c0d4e4aa81053558'


def main():
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-c', 'core.autocrlf=true', '-c', 'core.filemode=false',
        '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip()
    prepared = ROOT / 'build/longeval-ngfix-inputs-01'
    prior = json.loads((ROOT / 'build/longeval-mechanism-audit-01-manifest.json').read_text())
    report = json.loads((prepared / 'report.json').read_text())
    for name, expected in report['artifacts'].items(): assert checksum(prepared / name) == expected
    assert checksum(prepared / 'report.json') == prior['files']['build/longeval-ngfix-inputs-01/report.json']
    assert checksum(ROOT / 'research/longeval/ngfix_driver.cpp') == report['driver_sha256']
    payload = {}
    def add(name, path):
        assert name not in payload
        payload[name] = Path(path).read_bytes()
    for name in ('__init__.py', 'release.py', 'storage.py', 'dense.py', 'query_diagnostic.py',
                 'encoder.py', 'representation.py', 'ngfix_inputs.py', 'ngfix_screen.py', 'ngfix_driver.cpp'):
        add('research/longeval/' + name, ROOT / 'research/longeval' / name)
    for name in ('tests/test_ngfix_screen.py', 'tests/test_ngfix_inputs.py'):
        add('research/longeval/' + name, ROOT / 'research/longeval' / name)
    for name in ('longeval-ngfix.sbatch', 'longeval-node-env.sh', 'package-longeval-ngfix.py', 'LONGEVAL_NGFIX.md'):
        add('scripts/pace/' + name, ROOT / 'scripts/pace' / name)
    tracked = subprocess.check_output(['git', '-C', str(UPSTREAM), 'ls-files'], text=True).splitlines()
    for name in tracked:
        if name.startswith(('ngfixlib/', 'rabitqlib/')) or name in ('LICENSE', 'README.md'):
            add('vendor/ngfix/' + name, UPSTREAM / name)
    boost = ROOT / 'build/ngfix-dependencies/extracted/usr/include'
    # Use -M, not -MM: Boost marks some of its own includes as system headers.
    dependencies = (ROOT / 'build/ngfix-header-dependencies-02.txt').read_text().replace('\\\n', ' ').split()
    boost_paths = {Path(n).resolve() for n in dependencies if n.startswith('build/ngfix-dependencies/extracted/usr/include/')}
    assert boost_paths
    for path in sorted(boost_paths): add('vendor/boost/' + path.relative_to(boost).as_posix(), path)
    add('vendor/boost/COPYRIGHT', ROOT / 'build/ngfix-dependencies/extracted/usr/share/doc/libboost1.83-dev/copyright')
    for name in ('queries.fbin', 'calibration_top200.ibin', 'labels.json', 'queries.json', 'report.json'):
        add('inputs/' + name, prepared / name)
    add('inputs/snapshot-1-ingestion.json', DENSE / 'snapshot-1/ingestion.json')
    import hashlib
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    manifest = {'purpose': 'Isolated NGFix correctness and sample integration, no held-out queries',
        'upstream_commit': COMMIT, 'dataset_vectors_included': False,
        'required_compiler': 'gcc/13.3.0', 'boost_headers': len(boost_paths),
        'parent_evidence_manifest_sha256': checksum(ROOT / 'build/longeval-mechanism-audit-01-manifest.json'),
        'files': {name: sha(raw) for name, raw in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    check = ROOT / 'build' / (NAME + '-check')
    check.mkdir(exist_ok=False)
    for name, raw in payload.items():
        path = check / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    destination = ROOT / 'build' / (NAME + '.tar.gz')
    with destination.open('xb') as target, gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, raw in sorted(payload.items()):
                info = tarfile.TarInfo(NAME + '/' + name)
                info.size = len(raw); info.mode = 0o755 if name.endswith(('.sh', '.sbatch')) else 0o644
                archive.addfile(info, io.BytesIO(raw))
    with tarfile.open(destination) as archive:
        assert {m.name for m in archive} == {NAME+'/'+n for n in payload}
        for name, raw in payload.items(): assert archive.extractfile(NAME+'/'+name).read() == raw
    receipt = {'archive': destination.name, 'sha256': checksum(destination), 'bytes': destination.stat().st_size,
        'files': len(manifest['files']), 'boost_headers': len(boost_paths), 'upstream_commit': COMMIT,
        'native_runtime_validated_locally': False}
    (ROOT / 'build' / (NAME+'-receipt.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
