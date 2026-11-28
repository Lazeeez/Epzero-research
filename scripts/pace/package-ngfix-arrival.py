"""Package screen A2b (docs/screen-a2b-ngfix.md).

Usage, from the repository root:
    python scripts/pace/package-ngfix-arrival.py PILOT BUILD_DIR

PILOT holds the samples `sample` (LongEval, never committed) and `public-glove`; BUILD_DIR holds the verified
`longeval-ngfix-screen-02.tar.gz`, whose pinned NGFix source and Boost header closure are reused unchanged.
Writes `ngfix-arrival-01.tar.gz` and its receipt to BUILD_DIR. Large inputs are streamed, not held in memory.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'ngfix-arrival-01'
PARENT, PARENT_SHA = 'longeval-ngfix-screen-02', '5532d5b8a7954efe33a291e1a583cfccaa743a9831f6aaaa859670fed66661f6'
CODE = ('research/__init__.py', 'research/longeval/__init__.py', 'research/longeval/ngfix_driver.cpp',
        'research/longeval/ngfix_arrival.py', 'research/longeval/accumulation.py', 'research/longeval/insertion_pilot.py',
        'research/longeval/insertion_strata.py', 'research/longeval/tests/test_ngfix_arrival.py',
        'docs/screen-a2b-ngfix.md', 'scripts/pace/longeval-node-env.sh', 'scripts/pace/ngfix-arrival.sbatch',
        'scripts/pace/NGFIX_ARRIVAL.md', 'scripts/pace/package-ngfix-arrival.py')
SAMPLES = {'longeval': 'sample', 'glove': 'public-glove'}
SAMPLE_FILES = ('sample.json', 'vectors.npy', 'queries.npy', 'exact-top100.npy', 'cohort.npy', 'lexical-rank.npy')


def file_sha(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main():
    pilot, build = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    parent = build / (PARENT + '.tar.gz')
    if file_sha(parent) != PARENT_SHA:
        raise ValueError('the parent NGFix package differs from its verified checksum')
    small = {}
    with tarfile.open(parent) as archive:
        for member in archive:
            name = member.name[len(PARENT) + 1:]
            if member.isfile() and name.startswith(('vendor/ngfix/', 'vendor/boost/')):
                small[name] = archive.extractfile(member).read()
        driver = archive.extractfile(PARENT + '/research/longeval/ngfix_driver.cpp').read()
    if (ROOT / 'research/longeval/ngfix_driver.cpp').read_bytes() != driver:
        raise ValueError('the adapter differs from the verified package\'s')
    small.update({name: (ROOT / name).read_bytes() for name in CODE})
    for name, raw in small.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    if f'PACKAGE="$ROOT/{NAME}"' not in small['scripts/pace/ngfix-arrival.sbatch'].decode():
        raise ValueError('job script does not reference this package')
    large = {f'inputs/{key}/{f}': pilot / folder / f for key, folder in SAMPLES.items() for f in SAMPLE_FILES}
    files = {n: hashlib.sha256(r).hexdigest() for n, r in small.items()}
    files.update({n: file_sha(p) for n, p in large.items()})
    manifest = {'package': NAME, 'purpose': 'screen A2b: NGFix against the arrival bias', 'parent': PARENT,
                'parent_sha256': PARENT_SHA, 'files': dict(sorted(files.items()))}
    small['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    files['manifest.json'] = hashlib.sha256(small['manifest.json']).hexdigest()
    small['SHA256SUMS'] = ''.join(f'{files[n]}  {n}\n' for n in sorted(files)).encode()
    output = build / (NAME + '.tar.gz')
    with output.open('xb') as target, gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for name in sorted(set(small) | set(large)):
                entry = tarfile.TarInfo(NAME + '/' + name)
                entry.mode = 0o755 if name.endswith(('.sh', '.sbatch')) else 0o644
                if name in small:
                    entry.size = len(small[name])
                    tar.addfile(entry, io.BytesIO(small[name]))
                else:
                    entry.size = large[name].stat().st_size
                    with open(large[name], 'rb') as f:
                        tar.addfile(entry, f)
    receipt = {'archive': output.name, 'sha256': file_sha(output), 'bytes': output.stat().st_size,
               'files': len(small) + len(large), 'manifest_sha256': files['manifest.json']}
    (build / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
