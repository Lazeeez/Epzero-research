"""Add native HNSW timings to the immutable IVF supplement, with verified baselines."""
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build'
PARENT_SHA = '00b37d9e216cac5f2563e070ad3a0ba363ca1b1e50fa8287c887f7d5baffcf2f'


def main():
    spec = importlib.util.spec_from_file_location('ivf_packaging', ROOT / 'scripts/pace/package-longeval-ivf.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    sha, load = helpers.sha, helpers.archive_files
    payload = load(BUILD / 'longeval-pace-ivf-02.tar.gz', PARENT_SHA)
    prior = json.loads(payload.pop('longeval-gpu-manifest.json'))
    if {n: sha(raw) for n, raw in payload.items()} != prior['files']:
        raise ValueError('prior package manifest mismatch')
    additions = ('research/longeval/native_cost.py', 'research/longeval/tests/test_native_cost.py',
                 'research/longeval/docs/development/NATIVE_COST_PLAN.md', 'scripts/pace/longeval-native-cost.sbatch',
                 'scripts/pace/LONGEVAL_NATIVE_COST.md', 'scripts/pace/package-longeval-native-cost.py')
    for name in additions:
        if name in payload:
            raise ValueError('new supplement would overwrite prior file')
        payload[name] = (ROOT / name).read_bytes()
    baseline_map = {(11, 'epzero'): 7108471, (29, 'epzero'): 7122361, (47, 'epzero'): 7122367,
                    (11, 'faiss_hnsw'): 7114741, (29, 'faiss_hnsw'): 7122364, (47, 'faiss_hnsw'): 7122370}
    for (seed, kind), job in baseline_map.items():
        parent = BUILD / f'longeval-pace-trajectory-evidence-{job}'
        report_path = parent / f'{job}/trajectory/report.json'
        verification = parent / 'local-verification.json'
        record = json.loads(verification.read_text())
        report = json.loads(report_path.read_text())
        if (record['status'] != 'verified' or record['report_sha256'] != sha(report_path.read_bytes())
                or (report['seed'], report['backend'], report['mode']) != (seed, kind, 'incremental')):
            raise ValueError('unverified native baseline')
        prefix = f'build/longeval-native-cost-inputs/seed-{seed}/{kind}'
        payload[prefix + '.report.json'] = report_path.read_bytes()
        payload[prefix + '.verification.json'] = verification.read_bytes()
    base = load(BUILD / 'longeval-pace-source.tar.gz',
                'ecadaca9c01ea377cd9b95ff3d3237b66b295e146f82bbf71413a1cbe5477d9c')
    base.update(load(BUILD / 'longeval-pace-bootstrap-01.tar.gz',
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
    manifest = {**prior, 'purpose': 'Native HNSW timings under the IVF query protocol; original state reproduction',
                'parent_archive_sha256': PARENT_SHA, 'files': {n: sha(raw) for n, raw in sorted(payload.items())}}
    payload['longeval-gpu-manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    base.update(payload)
    check = BUILD / 'longeval-native-cost-package-check-02'
    check.mkdir(exist_ok=True)
    for name, raw in base.items():
        path = (check / name).resolve()
        if check.resolve() not in path.parents:
            raise ValueError('unsafe package path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('package-check file already differs')
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
    destination = BUILD / 'longeval-pace-native-cost-02.tar.gz'
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError('existing release differs; create a new version')
    if not destination.exists():
        destination.write_bytes(raw)
    receipt = {'sha256': sha(raw), 'bytes': len(raw), 'files': len(manifest['files']),
               'preserved_prior_files': len(prior['files']), 'original_source_files_verified': count,
               'parent_archive_sha256': PARENT_SHA}
    (BUILD / 'longeval-pace-native-cost-02-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
