"""Immutable IVF overlay on the sealed full-cohort work-budget supplement."""
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build'
PARENT_SHA = '1462a0003513ce290d6aaa6806b5b6acc5739b4c643f8921e8b40a9e30d6c7fd'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def archive_files(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError('parent archive identity mismatch')
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            if not item.isfile() or name.is_absolute() or '..' in name.parts or '\\' in item.name or item.name in files:
                raise ValueError('unsafe or duplicate archive member')
            files[item.name] = archive.extractfile(item).read()
    return files


def main():
    payload = archive_files(BUILD / 'longeval-pace-work-budget-01.tar.gz', PARENT_SHA)
    prior = json.loads(payload.pop('longeval-gpu-manifest.json'))
    if {name: sha(raw) for name, raw in payload.items()} != prior['files']:
        raise ValueError('parent supplemental manifest mismatch')
    additions = (
        'research/longeval/ivf_trajectory.py', 'research/longeval/ivf_verification.py',
        'research/longeval/ivf-development.json', 'research/longeval/docs/development/IVF_DEVELOPMENT_PLAN.md',
        'research/longeval/tests/test_ivf_trajectory.py', 'research/longeval/tests/test_ivf_verification.py',
        'scripts/pace/longeval-ivf.sbatch', 'scripts/pace/LONGEVAL_IVF.md',
        'scripts/pace/package-longeval-ivf.py',
    )
    for name in additions:
        if name in payload:
            raise ValueError('new supplement would overwrite prior source')
        payload[name] = (ROOT / name).read_bytes()
    base = archive_files(BUILD / 'longeval-pace-source.tar.gz',
        'ecadaca9c01ea377cd9b95ff3d3237b66b295e146f82bbf71413a1cbe5477d9c')
    base.update(archive_files(BUILD / 'longeval-pace-bootstrap-01.tar.gz',
        '0e7008ab50afe47749d97d2a4aa21cc6ec413baf2774765560bdd4658c2a9b4f'))
    count = 0
    for name in ('longeval-source-manifest.json', 'longeval-bootstrap-manifest.json'):
        for path, expected in json.loads(base[name])['files'].items():
            if sha((ROOT / path).read_bytes()) != expected or path in payload and sha(payload[path]) != expected:
                raise ValueError(f'original source changed: {path}')
            count += 1
    for path, expected in prior['files'].items():
        if not path.startswith('build/') and sha((ROOT / path).read_bytes()) != expected:
            raise ValueError(f'prior supplement source changed: {path}')
    manifest = {**prior, 'purpose': 'Three full-data development IVF maintenance controls at three seeds',
                'parent_archive_sha256': PARENT_SHA,
                'files': {name: sha(raw) for name, raw in sorted(payload.items())}}
    payload['longeval-gpu-manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    base.update(payload)
    check = BUILD / 'longeval-ivf-package-check-02'
    check.mkdir(exist_ok=True)
    for name, raw in base.items():
        target = (check / name).resolve()
        if check.resolve() not in target.parents:
            raise ValueError('unsafe package-check path')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f'existing package-check file changed: {name}')
        if not target.exists():
            target.write_bytes(raw)
    subprocess.run([sys.executable, 'scripts/pace/longeval-gpu-verify.py'], cwd=check, check=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, raw in sorted(payload.items()):
                item = tarfile.TarInfo(name)
                item.size = len(raw)
                item.mode = 0o755 if name.endswith(('.sh', '.sbatch')) else 0o644
                archive.addfile(item, io.BytesIO(raw))
    raw = buffer.getvalue()
    target = BUILD / 'longeval-pace-ivf-02.tar.gz'
    if target.exists() and target.read_bytes() != raw:
        raise ValueError('release already exists with different bytes; use a new version')
    if not target.exists():
        target.write_bytes(raw)
    receipt = {'archive': str(target), 'sha256': sha(raw), 'bytes': len(raw),
               'original_source_files_verified': count, 'preserved_prior_files': len(prior['files']),
               'files': len(manifest['files']), 'parent_archive_sha256': PARENT_SHA}
    (BUILD / 'longeval-pace-ivf-02-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
