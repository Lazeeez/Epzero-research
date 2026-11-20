"""Build and verify the additive retention supplement from the sealed prior bundle."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_archive(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f'archive identity changed: {path}')
    payload = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if (not member.isfile() or member.name.startswith('/') or '\\' in member.name
                    or '..' in Path(member.name).parts or member.name in payload):
                raise ValueError('invalid archive member')
            payload[member.name] = archive.extractfile(member).read()
    return payload


def main():
    payload = load_archive(BUILD / 'longeval-pace-diagnostic-01.tar.gz',
        '778877d75f76ee100698078012933e8ac4ec8f98e0b23c4fa5a69672523f0461')
    prior = json.loads(payload.pop('longeval-gpu-manifest.json'))
    if {name: sha(raw) for name, raw in payload.items()} != prior['files']:
        raise ValueError('prior supplemental manifest mismatch')
    additions = (
        'research/longeval/retention_search.py',
        'research/longeval/retention_diagnostic.py',
        'research/longeval/tests/test_retention_search.py',
        'research/longeval/tests/test_retention_diagnostic.py',
        'research/longeval/docs/development/HNSW_RETENTION_EXPERIMENT.md',
        'research/longeval/docs/development/HNSW_FAILURE_DIAGNOSTIC_RESULTS.md',
        'scripts/pace/longeval-retention.sbatch',
        'scripts/pace/LONGEVAL_RETENTION.md',
        'scripts/pace/package-longeval-retention.py',
        'build/longeval-diagnostic-comparison/comparison.json',
        'build/longeval-diagnostic-comparison/beam-retention-analysis.json',
    )
    for name in additions:
        if name in payload:
            raise ValueError(f'new experiment must not modify prior supplement: {name}')
        payload[name] = (ROOT / name).read_bytes()
    manifest = {**prior, 'purpose': 'Post-hoc deferred-candidate expansion; full native parity; seeds 11 and 29',
                'parent_archive_sha256': '778877d75f76ee100698078012933e8ac4ec8f98e0b23c4fa5a69672523f0461',
                'files': {name: sha(raw) for name, raw in sorted(payload.items())}}
    payload['longeval-gpu-manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()

    base = load_archive(BUILD / 'longeval-pace-source.tar.gz',
        'ecadaca9c01ea377cd9b95ff3d3237b66b295e146f82bbf71413a1cbe5477d9c')
    base.update(load_archive(BUILD / 'longeval-pace-bootstrap-01.tar.gz',
        '0e7008ab50afe47749d97d2a4aa21cc6ec413baf2774765560bdd4658c2a9b4f'))
    for name in ('longeval-source-manifest.json', 'longeval-bootstrap-manifest.json'):
        for path, expected in json.loads(base[name])['files'].items():
            if sha((ROOT / path).read_bytes()) != expected:
                raise ValueError(f'original validated source changed: {path}')
            if path in payload and sha(payload[path]) != expected:
                raise ValueError(f'overlay changes original source: {path}')
    for path, expected in prior['files'].items():
        if not path.startswith('build/') and sha((ROOT / path).read_bytes()) != expected:
            raise ValueError(f'prior diagnostic source changed: {path}')
    base.update(payload)
    check = BUILD / 'longeval-retention-package-check-01'
    check.mkdir(exist_ok=True)
    for name, raw in base.items():
        path = (check / name).resolve()
        if check.resolve() not in path.parents:
            raise ValueError('invalid package-check path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f'existing package-check file differs: {name}')
        if not path.exists():
            path.write_bytes(raw)
    subprocess.run([sys.executable, 'scripts/pace/longeval-gpu-verify.py'], cwd=check, check=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, raw in sorted(payload.items()):
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.mode = 0o755 if name.endswith(('.sh', '.sbatch')) else 0o644
                archive.addfile(info, io.BytesIO(raw))
    raw = buffer.getvalue()
    target = BUILD / 'longeval-pace-retention-01.tar.gz'
    if target.exists() and target.read_bytes() != raw:
        raise ValueError('existing release differs; use a new package version')
    if not target.exists():
        target.write_bytes(raw)
    receipt = {'archive': str(target), 'sha256': sha(raw), 'bytes': len(raw),
               'files': len(manifest['files']), 'preserved_prior_files': len(prior['files']),
               'original_source_files_verified': 127}
    (BUILD / 'longeval-pace-retention-01-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
