"""Package screen A1's encoding job (docs/screen-a1-arxiv-stream.md).

Usage, from the repository root:
    python scripts/pace/package-arxiv-stream.py PREPARED BUILD_DIR

PREPARED is `research/arxiv_stream/prepare.py`'s output folder (CC0 arXiv metadata, never committed). Writes
`arxiv-stream-01.tar.gz` and its receipt to BUILD_DIR.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'arxiv-stream-01'
# Every LongEval module, as the BGE packager includes them: the encoder imports storage, which imports dense.
CODE = ('research/__init__.py', 'research/longeval/encoder-requirements.txt',
        'research/longeval/tests/test_bge_encoder.py', 'research/arxiv_stream/__init__.py',
        'research/arxiv_stream/prepare.py', 'research/arxiv_stream/encode.py', 'research/arxiv_stream/tests/test_prepare.py',
        'docs/screen-a1-arxiv-stream.md', 'scripts/pace/longeval-node-env.sh', 'scripts/pace/arxiv-bge.sbatch',
        'scripts/pace/ARXIV_STREAM.md', 'scripts/pace/package-arxiv-stream.py')
INPUTS = ('manifest.json', 'corpus.jsonl.gz', 'queries-new.jsonl.gz', 'queries-uniform.jsonl.gz')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    prepared, build = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if not manifest['harvest']['complete']:
        raise ValueError('prepared from an incomplete harvest')
    for name in INPUTS[1:]:
        digest = hashlib.sha256()
        with gzip.open(prepared / name, 'rb') as f:
            for line in f:
                digest.update(line)
        if digest.hexdigest() != manifest['content_sha256'][name[:-len('.jsonl.gz')]]:
            raise ValueError(f'{name}: content differs from its manifest')
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in CODE})
    payload.update({'inputs/' + name: (prepared / name).read_bytes() for name in INPUTS})
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    if f'PACKAGE="$ROOT/{NAME}"' not in payload['scripts/pace/arxiv-bge.sbatch'].decode():
        raise ValueError('job script does not reference this package')
    package_manifest = {'package': NAME, 'purpose': 'screen A1: BGE vectors of arXiv abstracts in submission order',
                        'prepared_manifest_sha256': sha(payload['inputs/manifest.json']),
                        'files': {n: sha(r) for n, r in sorted(payload.items())}}
    payload['manifest.json'] = (json.dumps(package_manifest, indent=2, sort_keys=True) + '\n').encode()
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
    receipt = {'archive': output.name, 'sha256': sha(output.read_bytes()), 'bytes': output.stat().st_size,
               'files': len(payload), 'corpus': manifest['corpus']}
    (build / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
