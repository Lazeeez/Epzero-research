"""Package the held-out native timing jobs (FINAL_PROTOCOL_V1, clarification 2).

Includes the frozen research code (unchanged since the freeze), the new timing module, the
held-out inputs with exact references, and the six analysed incremental HNSW reports as
baselines. No judgments. LongEval-derived: never commit the package.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build'
NAME = 'longeval-heldout-cost-01'
INPUTS = BUILD / 'longeval-heldout-inputs-01'
JOBS = BUILD / 'longeval-heldout-01-evidence/build/pace-longeval-heldout/jobs'
EVIDENCE_SHA256 = '886047e02cdabb1d73e914b019f6cd80f0d34acf0bcd0dd6d57326cd95fabf1f'
EXTRA = ('research/__init__.py', 'research/longeval/trajectory-development.json',
         'research/longeval/FINAL_PROTOCOL_V1.md', 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json',
         'research/longeval/tests/test_heldout.py', 'research/longeval/tests/test_heldout_cost.py',
         'research/longeval/tests/test_trajectory.py', 'research/longeval/tests/test_ivf_trajectory.py',
         'research/longeval/tests/test_mapped.py', 'scripts/pace/longeval-node-env.sh',
         'scripts/pace/longeval-heldout-cost.sbatch', 'scripts/pace/LONGEVAL_HELDOUT_COST.md',
         'scripts/pace/package-longeval-heldout-cost.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    sys.path.insert(0, str(ROOT))
    from research.longeval.heldout import load_heldout
    freeze = json.loads((ROOT / 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json').read_text())
    for name, expected in freeze['files'].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError(f'frozen file changed after the freeze: {name}')
    if sha((BUILD / 'longeval-heldout-01-evidence.tar.gz').read_bytes()) != EVIDENCE_SHA256:
        raise ValueError('held-out evidence archive identity changed')
    load_heldout(INPUTS)
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for path in sorted(INPUTS.rglob('*')):
        if path.is_file():
            payload['inputs/' + path.relative_to(INPUTS).as_posix()] = path.read_bytes()
    baselines = {}
    for path in sorted(JOBS.glob('*/trajectory/report.json')):
        report = json.loads(path.read_text())
        if report['mode'] != 'incremental':
            continue
        name = f"{report['backend']}-{report['seed']}.json"
        if name in baselines or report['status'] != 'completed' or report.get('evaluation_role') != 'evaluation':
            raise ValueError(f'unexpected baseline: {path}')
        baselines[name] = path.parent.parent.name
        payload['inputs/baselines/' + name] = path.read_bytes()
    if len(baselines) != 6:
        raise ValueError('requires six incremental HNSW baselines')
    script = payload['scripts/pace/longeval-heldout-cost.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script
            or f'{NAME}/scripts/pace/longeval-heldout-cost.sbatch' not in payload['scripts/pace/LONGEVAL_HELDOUT_COST.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'held-out native HNSW timing and work counts; secondary; no judgments',
                'freeze_sha256': sha(payload['research/longeval/FINAL_PROTOCOL_V1_FREEZE.json']),
                'heldout_evidence_sha256': EVIDENCE_SHA256, 'baseline_jobs': baselines,
                'files': {n: sha(r) for n, r in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    payload['SHA256SUMS'] = ''.join(f'{sha(r)}  {n}\n' for n, r in sorted(payload.items())).encode()
    output = BUILD / (NAME + '.tar.gz')
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
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json']), 'baseline_jobs': baselines}
    (BUILD / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
