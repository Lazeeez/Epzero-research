"""Package the E4 BGE encoding job (second encoder).

Usage, from the repository root:
    python scripts/pace/package-longeval-bge.py BUILD_DIR

BUILD_DIR holds `longeval-bge-ids-snapshot-3.json.gz` (written by `bge_encoder ids` from the verified
released snapshot 3) and `longeval-release/files` (the document manifest and the two query files are
copied; no judgments). The archive is derived from LongEval data and is never committed.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'longeval-bge-02'
EXTRA = ('research/__init__.py', 'research/longeval/encoder-requirements.txt', 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json',
         'research/longeval/EXTENSION_PROTOCOL_V1_FREEZE.json', 'research/longeval/docs/protocol/E4_SECOND_ENCODER.md',
         'research/longeval/tests/test_bge_encoder.py', 'scripts/pace/longeval-node-env.sh', 'scripts/pace/longeval-bge.sbatch',
         'scripts/pace/LONGEVAL_BGE.md', 'scripts/pace/package-longeval-bge.py')
RELEASE = ('downloads.json', 'longeval_adhoc-queries-snapshot-test.tsv', 'task1_longeval_adhoc-queries-snapshot-train.tsv')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    build = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(ROOT))
    from research.longeval.bge_encoder import read_ids
    for receipt_name in ('FINAL_PROTOCOL_V1_FREEZE.json', 'EXTENSION_PROTOCOL_V1_FREEZE.json'):
        freeze = json.loads((ROOT / 'research/longeval' / receipt_name).read_text())
        for name, expected in freeze['files'].items():
            if sha((ROOT / name).read_bytes()) != expected:
                raise ValueError(f'frozen file changed after the freeze: {name}')
    ids_path = build / 'longeval-bge-ids-snapshot-3.json.gz'
    if len(read_ids(ids_path)) != 1_661_900:
        raise ValueError('the ID list does not hold the 1,661,900 documents of snapshot 3')
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    payload['inputs/bge-ids-snapshot-3.json.gz'] = ids_path.read_bytes()
    for name in RELEASE:
        payload['release/files/' + name] = (build / 'longeval-release/files' / name).read_bytes()
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/longeval-bge.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script
            or f'{NAME}/scripts/pace/longeval-bge.sbatch' not in payload['scripts/pace/LONGEVAL_BGE.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'E4 second-encoder vectors (BGE-base-en-v1.5); no judgments',
                'ids_sha256': sha(payload['inputs/bge-ids-snapshot-3.json.gz']),
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
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json'])}
    (build / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
