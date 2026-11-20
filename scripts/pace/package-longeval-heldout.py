"""Assemble held-out inputs (phase 'inputs') and the self-contained PACE package (phase 'package').

Phase 'inputs' copies verified local sources into build/longeval-heldout-inputs-01 without any
test outcome. Phase 'package' requires the frozen protocol receipt and the three held-out exact
references (computed after the freeze), then writes build/longeval-heldout-01.tar.gz.
Test-query vectors and runs are LongEval-derived data: never commit the package or its inputs.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build'
NAME = 'longeval-heldout-01'
INPUTS = BUILD / 'longeval-heldout-inputs-01'
FREEZE = ROOT / 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json'
SOURCES = {
    'training/queries': (BUILD / 'longeval-training-query-inputs-01', ('queries.tsv', 'baseline.py', 'records.jsonl', 'query-inputs.json')),
    'training/float32': (BUILD / 'longeval-pace-query-evidence-7107422/7107422/float32',
                         ('query-encoding.json', 'encoder/encoding.json', 'encoder/ids.json', 'encoder/vectors.npy')),
    'test/inputs': (BUILD / 'longeval-test-query-inputs-01', None),
    'test/encoding': (BUILD / 'longeval-testq-7164678/7164678/encoding', None),
}
EXTRA = ('research/__init__.py', 'research/longeval/requirements.txt', 'research/longeval/trajectory-development.json',
         'research/longeval/ivf-development.json', 'research/longeval/FINAL_PROTOCOL_V1.md',
         'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json',
         'research/longeval/tests/test_heldout.py', 'research/longeval/tests/test_trajectory.py',
         'research/longeval/tests/test_ivf_trajectory.py', 'research/longeval/tests/test_mapped.py',
         'scripts/pace/longeval-node-env.sh', 'scripts/pace/longeval-heldout.sbatch',
         'scripts/pace/LONGEVAL_HELDOUT.md', 'scripts/pace/package-longeval-heldout.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def assemble_inputs():
    INPUTS.mkdir(exist_ok=False)
    for target, (source, names) in SOURCES.items():
        names = names or sorted(p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file())
        for name in names:
            path = INPUTS / target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
    for step in (1, 2, 3):
        folder = INPUTS / f'training/exact/snapshot-{step}'
        folder.mkdir(parents=True)
        for name in ('report.json', 'float32.run'):
            shutil.copyfile(BUILD / f'longeval-query-sensitivity-01/snapshot-{step}/{name}', folder / name)
    shutil.copyfile(BUILD / 'longeval-text-full-audit/audit.json', INPUTS / 'training/text-audit.json')
    sys.path.insert(0, str(ROOT))
    from research.longeval.heldout import load_heldout
    queries, _, _, evidence = load_heldout(INPUTS, references=False)
    print(json.dumps(evidence, indent=2))


def build_package():
    sys.path.insert(0, str(ROOT))
    from research.longeval.heldout import check
    freeze = json.loads(FREEZE.read_text())
    for name, expected in freeze['files'].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError(f'frozen file changed after the freeze: {name}')
    for step in (1, 2, 3):
        if not (INPUTS / f'test/exact/snapshot-{step}/report.json').exists():
            raise ValueError('held-out exact references are missing')
    preflight = check(INPUTS, ROOT / 'research/longeval/trajectory-development.json',
                      ROOT / 'research/longeval/ivf-development.json')
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for path in sorted(INPUTS.rglob('*')):
        if path.is_file():
            payload['inputs/' + path.relative_to(INPUTS).as_posix()] = path.read_bytes()
    script = payload['scripts/pace/longeval-heldout.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script
            or f'{NAME}/scripts/pace/longeval-heldout.sbatch' not in payload['scripts/pace/LONGEVAL_HELDOUT.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'held-out ANN arms under FINAL_PROTOCOL_V1; no judgments included',
                'freeze_sha256': sha(payload['research/longeval/FINAL_PROTOCOL_V1_FREEZE.json']),
                'preflight': preflight, 'files': {n: sha(r) for n, r in sorted(payload.items())}}
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
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json']),
               'freeze_sha256': manifest['freeze_sha256']}
    (BUILD / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    {'inputs': assemble_inputs, 'package': build_package}[sys.argv[1]]()
