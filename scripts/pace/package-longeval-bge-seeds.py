"""Package the BGE ten-seed follow-up of E4 (LONGEVAL_BGE_SEEDS.md, amendment 7).

Usage, from the repository root:
    python scripts/pace/package-longeval-bge-seeds.py BUILD_DIR

Holds the research code (the frozen cores among it), the frozen configurations, protocol and freeze
receipt, the tests, the job script, and the inputs: each released snapshot's `docnos.npids` and
`pt_meta.json` (BUILD_DIR/bge-qwen-docnos, checked against the release receipts when copied), the
training-query contract (BUILD_DIR/longeval-training-query-inputs-01) and the Qwen order rerun's sample
metadata (job 7253605). No vectors: the jobs read the verified BGE vectors on PACE. The archive is derived
from LongEval data and is never committed.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'longeval-bge-seeds-01'
METADATA = ('doc-ids.txt.gz', 'cohort.npy', 'query-ids.json', 'confirm-query-ids.json')
ORDER_JOBS = 'longeval-insertion-order-03-evidence/build/pace-longeval-insertion-order-ids/jobs'
TRAINING = ('queries.tsv', 'baseline.py', 'records.jsonl', 'query-inputs.json')
# The released snapshots' document maps (their ingestion receipts' docnos.npids SHA-256).
RELEASE_DOCNOS_SHA256 = {1: '811ce5b2ca9da194635a8dd4193b437c71fc4f6ab44c879e850948e7d3c7a3a7',
                         2: '17ffc3090dd91d91a49eacd1e0cade53d76389f7d909e3e7fe7c03817af4861d',
                         3: 'b686dab9e43a7a7cd0faa3e70842ddfb37aeae55d0fe697a4adeff330045b773'}
EXTRA = ('research/__init__.py', 'research/longeval/requirements.txt', 'research/longeval/trajectory-development.json',
         'research/longeval/ivf-development.json', 'research/longeval/FINAL_PROTOCOL_V1.md',
         'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json',
         'research/longeval/tests/test_bge_heldout.py', 'research/longeval/tests/test_bge_sample.py',
         'research/longeval/tests/test_heldout.py', 'research/longeval/tests/test_trajectory.py',
         'research/longeval/tests/test_ivf_trajectory.py', 'research/longeval/tests/test_mapped.py',
         'research/longeval/tests/test_post_freeze_analyses.py',
         'scripts/pace/longeval-node-env.sh', 'scripts/pace/longeval-bge-seeds.sbatch',
         'scripts/pace/LONGEVAL_BGE_SEEDS.md', 'scripts/pace/package-longeval-bge-seeds.py',
         'research/longeval/docs/protocol/EXTENSION_AMENDMENTS.md')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    build = Path(sys.argv[1]).resolve()
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / 'research/longeval').glob('*.py'))}
    payload.update({name: (ROOT / name).read_bytes() for name in EXTRA})
    for step in (1, 2, 3):
        for name in ('docnos.npids', 'pt_meta.json'):
            payload[f'inputs/qwen-docnos/snapshot-{step}/{name}'] = (build / 'bge-qwen-docnos' / f'snapshot-{step}' / name).read_bytes()
        meta = json.loads(payload[f'inputs/qwen-docnos/snapshot-{step}/pt_meta.json'])
        if meta.get('type') != 'dense_index' or meta.get('vec_size') != 2560:
            raise ValueError(f'snapshot {step}: not the released metadata')
        if sha(payload[f'inputs/qwen-docnos/snapshot-{step}/docnos.npids']) != RELEASE_DOCNOS_SHA256[step]:
            raise ValueError(f'snapshot {step} document map is not the released one')
    for name in TRAINING:
        payload[f'inputs/training/queries/{name}'] = (build / 'longeval-training-query-inputs-01' / name).read_bytes()
    jobs = sorted(p for p in (build / ORDER_JOBS).iterdir() if p.is_dir())
    for name in METADATA:
        read = (lambda p: gzip.decompress(p.read_bytes())) if name.endswith('.gz') else (lambda p: p.read_bytes())
        if len({sha(read(job / name)) for job in jobs}) != 1:
            raise ValueError(f'the order rerun jobs disagree on {name}')
        payload[f'inputs/qwen-sample/{name}'] = (build / ORDER_JOBS / '7253605' / name).read_bytes()
    for name, raw in payload.items():
        if name.endswith(('.sh', '.sbatch')) and b'\r' in raw:
            raise ValueError(f'CRLF line endings in {name}')
    script = payload['scripts/pace/longeval-bge-seeds.sbatch'].decode()
    if (f'PACKAGE="$ROOT/{NAME}"' not in script
            or f'{NAME}/scripts/pace/longeval-bge-seeds.sbatch' not in payload['scripts/pace/LONGEVAL_BGE_SEEDS.md'].decode()
            or 'case "$SEED" in 101|211|307|401|503|601|701)' not in script):
        raise ValueError('job script or runbook does not reference this package')
    manifest = {'package': NAME, 'purpose': 'EXPLORATORY seven further seeds for the E4 held-out primary arms on the BGE vectors (amendment 7); no judgments',
                'bge_job': 7290861, 'qwen_sample_metadata_job': 7253605,
                'docnos_sha256': {str(s): sha(payload[f'inputs/qwen-docnos/snapshot-{s}/docnos.npids']) for s in (1, 2, 3)},
                'freeze_sha256': sha(payload['research/longeval/FINAL_PROTOCOL_V1_FREEZE.json']),
                'arms': {'hnsw': {'backends': ['epzero', 'faiss_hnsw'], 'modes': ['incremental']},
                         'ivf': {'modes': ['frozen_incremental']}},
                'seeds': [101, 211, 307, 401, 503, 601, 701], 'files': {n: sha(r) for n, r in sorted(payload.items())}}
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
