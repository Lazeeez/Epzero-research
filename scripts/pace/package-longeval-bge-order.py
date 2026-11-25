"""Package E4 on the BGE vectors: the arrival-order reversal and the engine arm (LONGEVAL_BGE_ORDER.md).

Usage, from the repository root:
    python scripts/pace/package-longeval-bge-order.py BUILD_DIR ORYX_DIR

Holds the research code, the job script, the Epzero `engine` folder of `main` (tracked files only, via
git archive; its commit recorded) and the Qwen order rerun's sample metadata (document order, arrival
snapshots, query order) from job 7253605 in BUILD_DIR's order-rerun evidence. No vectors: the jobs read
the verified BGE vectors on PACE. The archive is derived from LongEval data and is never committed.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'longeval-bge-order-01'
BRANCH = 'main'
METADATA = ('doc-ids.txt.gz', 'cohort.npy', 'query-ids.json', 'confirm-query-ids.json')
ORDER_JOBS = 'longeval-insertion-order-03-evidence/build/pace-longeval-insertion-order-ids/jobs'
EXTRA = ('research/__init__.py', 'research/longeval/requirements.txt', 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json',
         'research/longeval/tests/test_bge_sample.py', 'research/longeval/tests/test_post_freeze_analyses.py',
         'research/longeval/tests/test_order_pairs.py', 'research/longeval/tests/test_engine_arm_analysis.py',
         'scripts/pace/longeval-node-env.sh', 'scripts/pace/longeval-bge-order.sbatch',
         'scripts/pace/LONGEVAL_BGE_ORDER.md', 'scripts/pace/package-longeval-bge-order.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def engine_source(oryx):
    commit = subprocess.check_output(['git', 'rev-parse', BRANCH], cwd=oryx, text=True).strip()
    raw = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'archive', '--format=tar', commit, 'engine'], cwd=oryx)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                files[member.name] = tar.extractfile(member).read()
    if 'engine/pyproject.toml' not in files or not any(b'set_extra_backlinks' in raw for raw in files.values()):
        raise ValueError(f'the {BRANCH} engine source lacks the repair options')
    return commit, files


def main():
    build, oryx = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    # The sample metadata of the order rerun; every one of its six jobs wrote the same files.
    jobs = sorted(p for p in (build / ORDER_JOBS).iterdir() if p.is_dir())
    for name in METADATA:
        # Content, not bytes: each job's gzip header carries its own timestamp.
        read = (lambda p: gzip.decompress(p.read_bytes())) if name.endswith('.gz') else (lambda p: p.read_bytes())
        copies = {sha(read(job / name)) for job in jobs}
        if len(copies) != 1:
            raise ValueError(f'the order rerun jobs disagree on {name}')
        payload[f'inputs/qwen-sample/{name}'] = (build / ORDER_JOBS / '7253605' / name).read_bytes()
    with gzip.open(io.BytesIO(payload['inputs/qwen-sample/doc-ids.txt.gz']), 'rt', encoding='utf-8') as handle:
        if len(handle.read().split('\n')[:-1]) != 1661900:
            raise ValueError('the order rerun metadata is not the full snapshot 3')
    commit, engine = engine_source(oryx)
    payload.update(engine)
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/longeval-bge-order.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script
            or f'{NAME}/scripts/pace/longeval-bge-order.sbatch' not in payload['scripts/pace/LONGEVAL_BGE_ORDER.md'].decode()):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'E4 on the BGE vectors: arrival-order reversal and engine arm; post-freeze; no judgments',
                'engine': {'repository': 'Epzero', 'branch': BRANCH, 'commit': commit, 'files': len(engine)},
                'bge_job': 7290861, 'qwen_sample_metadata_job': 7253605,
                'parts': {'order': {'backends': ['epzero', 'faiss_hnsw'], 'orders': ['appended', 'reversed', 'shuffled']},
                          'engine': {'arms': ['grown', 'repaired8w', 'rebuilt']}},
                'seeds': [11, 29, 47], 'files': {n: sha(r) for n, r in sorted(payload.items())}}
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
               'files': len(payload), 'manifest_sha256': sha(payload['manifest.json']), 'engine_commit': commit}
    (build / (NAME + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
