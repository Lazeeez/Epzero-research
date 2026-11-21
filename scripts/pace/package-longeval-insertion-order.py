"""Package the EXPLORATORY full-scale insertion-order experiment.

Usage, from the repository root:
    python scripts/pace/package-longeval-insertion-order.py INPUTS HELDOUT_JOBS OUTPUT_DIR

INPUTS is the held-out inputs folder (encoded queries and exact references). HELDOUT_JOBS is the
`jobs` folder of the verified held-out evidence; its six incremental HNSW reports go into the
package as baselines for the recorded comparison. No judgments are included. The archive is
derived from LongEval data and is never committed.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'longeval-insertion-order-01'
EXTRA = ('research/__init__.py', 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json',
         'research/longeval/tests/test_post_freeze_analyses.py', 'scripts/pace/longeval-node-env.sh',
         'scripts/pace/longeval-insertion-order.sbatch', 'scripts/pace/LONGEVAL_INSERTION_ORDER.md',
         'scripts/pace/package-longeval-insertion-order.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    inputs, jobs, target = (Path(arg).resolve() for arg in sys.argv[1:4])
    sys.path.insert(0, str(ROOT))
    from research.longeval.heldout import load_heldout
    freeze = json.loads((ROOT / 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json').read_text())
    for name, expected in freeze['files'].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError(f'frozen file changed after the freeze: {name}')
    load_heldout(inputs)
    payload = {path.relative_to(ROOT).as_posix(): path.read_bytes()
               for path in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for path in sorted(inputs.rglob('*')):
        if path.is_file():
            payload['inputs/' + path.relative_to(inputs).as_posix()] = path.read_bytes()
    baselines = {}
    for path in sorted(jobs.glob('*/trajectory/report.json')):
        report = json.loads(path.read_text())
        if report['mode'] != 'incremental':
            continue
        name = f"{report['backend']}-{report['seed']}.json"
        if name in baselines or report['status'] != 'completed' or report.get('evaluation_role') != 'evaluation':
            raise ValueError(f'unexpected baseline: {path}')
        baselines[name] = path.parent.parent.name
        payload['inputs/baselines/' + name] = path.read_bytes()
    if len(baselines) != 6:
        raise ValueError('requires the six incremental HNSW baselines of the held-out run')
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/longeval-insertion-order.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script
            or f'{NAME}/scripts/pace/longeval-insertion-order.sbatch' not in payload['scripts/pace/LONGEVAL_INSERTION_ORDER.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'EXPLORATORY full-scale insertion-order builds; post-freeze; no judgments',
                'freeze_sha256': sha(payload['research/longeval/FINAL_PROTOCOL_V1_FREEZE.json']),
                'baseline_jobs': baselines, 'datasets': ['full', 'draw1', 'draw2', 'draw3'],
                'orders': ['appended', 'reversed', 'shuffled'], 'seeds': [11, 29, 47],
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
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json']), 'baseline_jobs': baselines}
    (target / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
