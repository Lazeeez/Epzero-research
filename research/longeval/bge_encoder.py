"""E4: encode the LongEval-Sci collection and queries with BGE-base-en-v1.5 (design, pinned revision and
predictions: docs/protocol/E4_SECOND_ENCODER.md, frozen with the extension protocol).

    python -m research.longeval.bge_encoder fetch MODEL_DIR
    python -m research.longeval.bge_encoder ids DENSE IDS.json.gz          # locally, from the verified snapshot
    python -m research.longeval.bge_encoder encode MODEL_DIR IDS.json.gz DOCUMENTS RELEASE OUTPUT --device cuda

IDS lists the document IDs of the released Qwen snapshot 3 in its order (written by `ids` from the
verified snapshot, so the BGE collection is exactly the study's); DOCUMENTS holds the pinned text
archives; RELEASE the released files (query texts). OUTPUT receives the snapshot-3 document vectors (float32, L2-normalised, 768 columns, in the
order of the released snapshot-3 IDs), the test and training query vectors, and a report. Everything
written is derived from LongEval data and is never committed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np

from .release import checksum
from .storage import atomic_json, fetch_archive

MODEL = 'BAAI/bge-base-en-v1.5'
REVISION = 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
DIM = 768
MAX_LENGTH = 512
INSTRUCTION = 'Represent this sentence for searching relevant passages: '
# Git blob identities for small files; the LFS SHA-256 for the weights (E4_SECOND_ENCODER.md).
FILES = {
    'config.json': (777, '09a8200d8a549a3b9dba53b67156ca26810cc720'),
    'tokenizer.json': (711396, '688882a79f44442ddc1f60d70334a7ff5df0fb47'),
    'tokenizer_config.json': (366, '37fca74771bc76a8e01178ce3a6055a0995f8093'),
    'vocab.txt': (231508, 'fb140275c155a9c7c5a3b3e0e77a9e839594a938'),
    'special_tokens_map.json': (125, 'a8b3208c2884c4efb86e49300fdd3dc877220cdf'),
    '1_Pooling/config.json': (190, 'da5bfd57e34ca45582e4bdbaa3e6deb9efffa08d'),
    'sentence_bert_config.json': (52, 'ea85692bff64b0d1917833c31ddbca8ab10f5455'),
    'modules.json': (349, '952a9b81c0bfd99800fabf352f69c7ccd46c5e43'),
    'model.safetensors': (437955512, 'c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7'),
}
VERIFY_SAMPLE = 1000
VERIFY_MIN_COSINE = 0.9999
# Job 7254809 spent its 4 hours in the encode phase without finishing the first 50,000 documents
# and without saying why. On an H200 the model encodes thousands of documents per second, so a
# GPU run slower than this after the probe is not running as designed and stops early.
MIN_GPU_DOCUMENTS_PER_SECOND = 200
PROBE_DOCUMENTS = 5_000
PROGRESS_SECONDS = 60


def throughput_check(device, done, seconds):
    """The documents-per-second rate so far; refuses a GPU run that is far too slow after the probe."""
    rate = done / max(seconds, 1e-9)
    if device == 'cuda' and done >= PROBE_DOCUMENTS and rate < MIN_GPU_DOCUMENTS_PER_SECOND:
        raise ValueError(f'{rate:.0f} documents per second after {done} documents on the GPU, below '
                         f'{MIN_GPU_DOCUMENTS_PER_SECOND}: the model is not running on the GPU as expected')
    return rate


def git_blob_hash(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def verify_model(folder, files=FILES):
    folder = Path(folder)
    result = {}
    for name, (size, expected) in files.items():
        path = folder / name
        if path.stat().st_size != size:
            raise ValueError(f'model file size mismatch: {name}')
        sha256 = checksum(path)
        actual = sha256 if len(expected) == 64 else git_blob_hash(path.read_bytes())
        if actual != expected:
            raise ValueError(f'model file identity mismatch: {name}')
        result[name] = {'bytes': size, 'sha256': sha256}
    return {'model': MODEL, 'revision': REVISION, 'files': result}


def fetch_model(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, (size, expected) in FILES.items():
        path = folder / name
        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        if len(expected) == 64:
            fetch_archive(url, path, expected, size, progress=lambda value: print(json.dumps(value), flush=True))
        elif not path.exists():
            with urlopen(url, timeout=60) as response:
                raw = response.read(size + 1)
            if len(raw) != size or git_blob_hash(raw) != expected:
                raise ValueError(f'model file identity mismatch: {name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as target:
                target.write(raw)
    result = verify_model(folder)
    atomic_json(folder / 'model-identity.json', result)
    return result


def query_text(query):
    """BGE's published retrieval instruction is prefixed to queries, never to documents."""
    return INSTRUCTION + query


class Encoder:
    """BERT encoder, CLS pooling, float32 L2 normalisation, right truncation at MAX_LENGTH tokens."""

    def __init__(self, model_folder, device):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; refusing device fallback')
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_folder), local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(model_folder), local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype=torch.float32).to(device).eval()

    def token_counts(self, texts):
        return np.array([len(ids) for ids in self.tokenizer(list(texts), truncation=False, add_special_tokens=True)['input_ids']])

    def encode(self, texts, batch_size=128, progress=None):
        """Vectors in the order of `texts`; batches are formed by length to limit padding.
        `progress(done)` is called after each batch with the number of texts encoded so far."""
        texts = list(texts)
        order = np.argsort([len(t) for t in texts], kind='stable')
        out = np.empty((len(texts), DIM), dtype=np.float32)
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                rows = order[start:start + batch_size]
                batch = self.tokenizer([texts[i] for i in rows], padding=True, truncation=True, max_length=MAX_LENGTH,
                                       return_tensors='pt').to(self.device)
                cls = self.model(**batch).last_hidden_state[:, 0].float()
                vectors = self.torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy()
                if not np.isfinite(vectors).all():
                    raise ValueError('non-finite embedding')
                out[rows] = vectors
                if progress is not None:
                    progress(min(start + batch_size, len(texts)))
        return out


def verification_rows(ids, count=VERIFY_SAMPLE):
    """A deterministic sample of row positions, by a keyed hash of the document IDs."""
    keyed = sorted(range(len(ids)), key=lambda i: hashlib.sha256(('bge-verify-v1:' + ids[i]).encode()).digest())
    return np.sort(np.array(keyed[:count]))


def min_cosine(a, b):
    return float(np.min(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))


def write_ids(dense, path):
    """The document IDs of the verified released snapshot 3, in its order."""
    from .dense import ARTIFACTS
    from .mapped import MappedSnapshot
    with MappedSnapshot(Path(dense) / 'snapshot-3', expected_archive_sha256=ARTIFACTS[2]['archive_sha256']) as snap:
        ids = list(snap.ids)
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(ids, handle)
    return {'documents': len(ids), 'sha256': checksum(path)}


def read_ids(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        ids = json.load(handle)
    if len(ids) != len(set(ids)) or not all(isinstance(d, str) and d for d in ids):
        raise ValueError('document ID list is not a list of unique, non-empty strings')
    return ids


def encode_collection(model_folder, ids_path, documents, release, output, *, device='cuda', batch_size=128):
    from .judge import document_texts
    from .release import read_queries
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'design': 'docs/protocol/E4_SECOND_ENCODER.md', 'model': MODEL, 'revision': REVISION,
              'pooling': 'CLS', 'normalisation': 'float32 L2', 'max_length': MAX_LENGTH, 'truncation': 'right',
              'query_instruction': INSTRUCTION, 'device': device, 'batch_size': batch_size,
              'source_sha256': checksum(__file__)}
    atomic_json(output / 'report.json', report)
    try:
        report['model_identity'] = verify_model(model_folder)
        report['packages'] = {name: importlib.metadata.version(name)
                              for name in ('torch', 'transformers', 'numpy', 'safetensors', 'tokenizers')}
        ids = read_ids(ids_path)
        report['ids_sha256'] = checksum(ids_path)
        started = time.perf_counter()
        found = document_texts(documents, ids)                    # refuses if any study document has no text
        texts = [found[d] for d in ids]
        del found
        report['documents'] = len(ids)
        report['text_seconds'] = time.perf_counter() - started
        atomic_json(output / 'report.json', report)
        print(json.dumps({'stage': 'texts', 'documents': len(ids), 'seconds': round(report['text_seconds'], 1)}), flush=True)
        encoder = Encoder(model_folder, device)
        report['model_device'] = str(next(encoder.model.parameters()).device)
        report['torch_threads'] = encoder.torch.get_num_threads()
        if device == 'cuda':
            report['gpu'] = encoder.torch.cuda.get_device_name(0)
        atomic_json(output / 'report.json', report)
        print(json.dumps({'stage': 'model', 'device': report['model_device'], 'gpu': report.get('gpu')}), flush=True)
        started = time.perf_counter()
        counts = encoder.token_counts(texts)
        report['token_seconds'] = time.perf_counter() - started
        report['tokens'] = {'mean': float(counts.mean()), 'p50': float(np.median(counts)), 'p99': float(np.quantile(counts, .99)),
                            'max': int(counts.max()), 'truncated': int((counts > MAX_LENGTH).sum()),
                            'truncated_share': float((counts > MAX_LENGTH).mean())}
        atomic_json(output / 'report.json', report)
        started = time.perf_counter()
        vectors = np.lib.format.open_memmap(output / 'documents-snapshot-3.npy', mode='w+', dtype=np.float32,
                                            shape=(len(ids), DIM))
        step = 50_000
        shown = [0.0]

        def progress(done):
            elapsed = time.perf_counter() - started
            rate = throughput_check(device, done, elapsed)
            if elapsed - shown[0] >= PROGRESS_SECONDS:
                shown[0] = elapsed
                print(json.dumps({'encoding': done, 'of': len(ids), 'seconds': round(elapsed, 1),
                                  'documents_per_second': round(rate, 1)}), flush=True)
        for start in range(0, len(ids), step):
            vectors[start:start + step] = encoder.encode(texts[start:start + step], batch_size,
                                                         progress=lambda done, base=start: progress(base + done))
            print(json.dumps({'encoded': min(start + step, len(ids)), 'of': len(ids)}), flush=True)
        vectors.flush()
        report['document_seconds'] = time.perf_counter() - started
        atomic_json(output / 'documents-snapshot-3-ids.json', ids)
        # Verification (design): the sample encoded again on a different device (CPU) or, if this run
        # is on the CPU, with a different thread count, in batches of one (no padding).
        rows = verification_rows(ids)
        if device == 'cuda':
            checker, setting = Encoder(model_folder, 'cpu'), 'cpu'
        else:
            threads = encoder.torch.get_num_threads()
            encoder.torch.set_num_threads(1 if threads > 1 else 2)
            checker, setting = encoder, f'cpu, {encoder.torch.get_num_threads()} threads instead of {threads}'
        again = np.stack([checker.encode([texts[i]], 1)[0] for i in rows])
        report['verification'] = {'documents': len(rows), 'batch_size': 1, 'second_setting': setting,
                                  'min_cosine': min_cosine(np.asarray(vectors[rows]), again),
                                  'required': VERIFY_MIN_COSINE}
        if report['verification']['min_cosine'] < VERIFY_MIN_COSINE:
            raise ValueError('re-encoded documents differ beyond the declared tolerance')
        files = Path(release) / 'files'
        for name, path in (('test', files / 'longeval_adhoc-queries-snapshot-test.tsv'),
                           ('training', files / 'task1_longeval_adhoc-queries-snapshot-train.tsv')):
            queries = read_queries(path)
            keys = sorted(queries)
            np.save(output / f'queries-{name}.npy', encoder.encode([query_text(queries[k]) for k in keys], batch_size))
            atomic_json(output / f'queries-{name}-ids.json', keys)
            report[f'queries_{name}'] = len(keys)
        report['files'] = {p.name: checksum(p) for p in sorted(output.glob('*')) if p.name != 'report.json'}
        report['status'] = 'completed'
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)[:500])
        atomic_json(output / 'report.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fetch', help='download and verify the pinned model files')
    p.add_argument('model', type=Path)
    p = sub.add_parser('ids', help='write the document IDs of the verified released snapshot 3')
    p.add_argument('dense', type=Path)
    p.add_argument('output', type=Path)
    p = sub.add_parser('encode', help='encode the snapshot-3 collection and the queries')
    for name in ('model', 'ids', 'documents', 'release', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    a = parser.parse_args()
    if a.command == 'fetch':
        print(json.dumps(fetch_model(a.model), indent=2))
    elif a.command == 'ids':
        print(json.dumps(write_ids(a.dense, a.output), indent=2))
    else:
        print(json.dumps(encode_collection(a.model, a.ids, a.documents, a.release, a.output, device=a.device,
                                           batch_size=a.batch_size), indent=2))


if __name__ == '__main__':
    main()
