"""Package disjoint query roles without altering the verified upstream adapter."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from research.longeval.ngfix_disjoint import protocol

ROOT = Path.cwd()
NAME = 'longeval-ngfix-disjoint-01'
PARENT = 'longeval-ngfix-screen-03'
PARENT_SHA = '9be895484ad89c6246fc49346254609328aabadbb51e754b9c91c6e2c26e90d3'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    archive = ROOT / 'build' / (PARENT + '.tar.gz')
    if sha(archive.read_bytes()) != PARENT_SHA:
        raise ValueError('parent package changed')
    payload = {}
    with tarfile.open(archive) as tar:
        for item in tar:
            if not item.isfile() or not item.name.startswith(PARENT + '/') or '..' in Path(item.name).parts:
                raise ValueError('invalid parent member')
            name = item.name[len(PARENT)+1:]
            if name in payload: raise ValueError('duplicate parent member')
            payload[name] = tar.extractfile(item).read()
    parent = json.loads(payload.pop('manifest.json'))
    if {name: sha(raw) for name, raw in payload.items()} != parent['files']:
        raise ValueError('parent payload changed')
    for name in ('research/longeval/ngfix_disjoint.py', 'research/longeval/tests/test_ngfix_disjoint.py',
                 'scripts/pace/package-longeval-ngfix-disjoint.py',
                 'scripts/pace/longeval-ngfix-disjoint.sbatch', 'scripts/pace/LONGEVAL_NGFIX_DISJOINT.md'):
        payload[name] = (ROOT / name).read_bytes()
    contract = protocol(ROOT / 'build/longeval-ngfix-inputs-01')
    payload['protocol.json'] = (json.dumps(contract, indent=2, sort_keys=True) + '\n').encode()
    changed = [name for name, digest in parent['files'].items() if sha(payload[name]) != digest]
    if changed:
        raise ValueError(f'parent files changed: {changed}')
    manifest = {**parent, 'parent_archive_sha256': PARENT_SHA,
        'purpose': 'Independent fitting/tuning roles; native algorithm and compiler unchanged',
        'correction': 'Three disjoint 30/20/50 roles; explicit /tmp step CWD and absolute test paths',
        'changed_parent_files': [], 'files': {name: sha(raw) for name, raw in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
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
        'files': len(manifest['files']), 'parent_sha256': PARENT_SHA, 'changed_parent_files': [],
        'protocol_sha256': sha(payload['protocol.json']), 'native_runtime_completed': False}
    (ROOT / 'build' / (NAME + '-receipt.json')).write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
