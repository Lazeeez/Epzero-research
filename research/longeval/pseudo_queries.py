"""Encoder-synthesized pseudo-queries from document titles; see PSEUDO_QUERY_PLAN.md.

Preparation runs locally on released files. Encoding runs on a CUDA node with the pinned
model. Pseudo-queries are build-time index inputs, not research query vectors. Held-out and
training query texts are read only to drop exact normalized-text collisions (count recorded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np

from .query_diagnostic import PROMPT
from .release import checksum, read_queries
from .storage import atomic_json

SALT = 'longeval-pseudoquery-v1|'
SET_B_COUNT = 131072
MAX_WORDS = 64
COSINE_GATE = 0.99999
LABEL_MAP_SHA256 = '78509abe8277f17fb964e3ca099552bbc80745f1d16aed1b5780cb7523bca5f9'
DOCUMENT_PREFIX = ('data/processed/doc_collection_09032026_abstract_2/snapshot-3/'
                   'longeval_sci_test-09-11_2026_abstract/documents/')
DOCUMENT_ZIP = 'longeval_sci_test-09-11_2026_abstract.zip'
QUERY_FILES = ('task1_longeval_adhoc-queries-snapshot-train.tsv', 'longeval_adhoc-queries-snapshot-test.tsv')
# Fixed before any pseudo-query was encoded; small and conventional on purpose.
STOPWORDS = frozenset('''a an and are as at be by for from has have in into is it its of on or
over that the their this to under using via was were what when where which while with within
without towards toward through between among based new study analysis case approach'''.split())
WORD = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")


def collapse(text):
    return ' '.join(text.split())


def keywords(title, count=4):
    tokens = [t for t in WORD.findall(title.lower()) if t not in STOPWORDS]
    return ' '.join(tokens[:count]) or None


def normalized(text):
    return ' '.join(WORD.findall(text.lower()))


def forms(title):
    return {'title': collapse(title) or None, 'keywords4': keywords(title)}


def hash_order(ids):
    return sorted(ids, key=lambda key: hashlib.sha256((SALT + key).encode()).digest())


def build_records(sets, titles, real_texts):
    """Return ordered prompted records per set and exclusion counts.

    ``sets`` maps a set name to (document IDs, form names). Records are ordered by the
    supplied document order, then form order. IDs are ``set:form:document``.
    """
    records, excluded = {}, {}
    for name, (doc_ids, form_names) in sets.items():
        rows, counts = [], {'missing_title': 0, 'empty_form': 0, 'too_long': 0, 'real_query_collision': 0}
        for doc in doc_ids:
            if doc not in titles:
                counts['missing_title'] += 1
                continue
            values = forms(titles[doc])
            for form in form_names:
                value = values[form]
                if not value:
                    counts['empty_form'] += 1
                elif len(value.split()) > MAX_WORDS:
                    counts['too_long'] += 1
                elif normalized(value) in real_texts:
                    counts['real_query_collision'] += 1
                else:
                    text = PROMPT + value
                    rows.append({'id': f'{name}:{form}:{doc}', 'document': doc, 'form': form, 'text': text,
                                 'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
        records[name], excluded[name] = rows, counts
    return records, excluded


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def read_titles(archive, wanted):
    titles = {}
    with zipfile.ZipFile(archive) as source:
        names = sorted(n for n in source.namelist() if n.startswith(DOCUMENT_PREFIX) and n.endswith('.jsonl'))
        if not names:
            raise ValueError('document archive has no snapshot-3 members')
        for name in names:
            with source.open(name) as stream:
                for line in stream:
                    record = json.loads(line)
                    doc, title = record.get('id'), record.get('title')
                    if doc in wanted:
                        if not isinstance(title, str):
                            raise ValueError('invalid document title')
                        if titles.setdefault(doc, title) != title:
                            raise ValueError('conflicting duplicate document title')
    return titles


def row_ids(npids_path):
    """External IDs in the snapshot file's own row order."""
    from npids import Lookup
    with Lookup(npids_path) as lookup:
        count = len(lookup)
        return [str(key) for start in range(0, count, 65536)
                for key in lookup.fwd[np.arange(start, min(count, start + 65536), dtype=np.int64)]]


def sample_ids(snapshot1_npids, rows):
    """Sample labels are snapshot-1 row positions (frontier pilot), not final-map indices."""
    ids = row_ids(snapshot1_npids)
    if len(ids) != 869902 or any(type(r) is not int or not 0 <= r < len(ids) for r in rows):
        raise ValueError('sample rows must index the 869,902-row snapshot-1 file')
    return [ids[r] for r in rows]


def final_ids(npids_path):
    ids = row_ids(npids_path)
    labels = tuple(sorted(ids))
    digest = hashlib.sha256(json.dumps(labels, ensure_ascii=False, allow_nan=False,
                                       separators=(',', ':')).encode()).hexdigest()
    if len(set(ids)) != len(ids) or digest != LABEL_MAP_SHA256:
        raise ValueError('final-snapshot label map differs from the audited map')
    return labels


def prepare(release, sample_inputs, reference, output, *, set_b=SET_B_COUNT):
    """``reference`` is a verified job-7107422 float32 query encoding folder."""
    release, sample_inputs, reference, output = map(Path, (release, sample_inputs, reference, output))
    sample_report = json.loads((sample_inputs / 'report.json').read_text())
    if checksum(sample_inputs / 'labels.json') != sample_report['artifacts']['labels.json']:
        raise ValueError('sample label checksum mismatch')
    labels = final_ids(release / 'dense/snapshot-3.npids')
    # Corrected 2026-11-19: package 01 indexed the sorted final map with snapshot-1 row
    # positions, so its Set A titles came from other documents (see PSEUDO_QUERY_RESULTS.md).
    sample = sample_ids(release / 'dense/snapshot-1.npids',
                        json.loads((sample_inputs / 'labels.json').read_text()))
    if len(sample) != 8192 or len(set(sample)) != 8192 or not set(sample) <= set(labels):
        raise ValueError('expected 8,192 distinct live sample documents')
    full = hash_order(labels)[:set_b]
    titles = read_titles(release / 'documents' / DOCUMENT_ZIP, set(sample) | set(full))
    real = {normalized(text) for name in QUERY_FILES for text in read_queries(release / 'files' / name).values()}
    records, excluded = build_records({'A': (sample, ('title', 'keywords4')), 'B': (full, ('keywords4',))},
                                      titles, real)
    ref_rows = read_jsonl(reference / 'inputs/records.jsonl')
    ref_ids = json.loads((reference / 'float32/encoder/ids.json').read_text())
    ref_vectors = np.load(reference / 'float32/encoder/vectors.npy', allow_pickle=False)
    encoding = json.loads((reference / 'float32/encoder/encoding.json').read_text())
    if (ref_ids != [r['id'] for r in ref_rows] or ref_vectors.shape != (100, 2560)
            or encoding['dtype'] != 'float32' or encoding['device'] != 'cuda' or encoding.get('tf32') is not False
            or encoding['vectors_sha256'] != checksum(reference / 'float32/encoder/vectors.npy')
            or any(not r['text'].startswith(PROMPT) for r in ref_rows)):
        raise ValueError('reference query encoding identity mismatch')
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in records.items():
        write_jsonl(output / f'records-{name}.jsonl', rows)
    write_jsonl(output / 'reference-records.jsonl', [{'id': r['id'], 'text': r['text']} for r in ref_rows])
    shutil.copyfile(reference / 'float32/encoder/vectors.npy', output / 'reference-vectors.npy')
    files = ('records-A.jsonl', 'records-B.jsonl', 'reference-records.jsonl', 'reference-vectors.npy')
    contract = {'status': 'prepared', 'purpose': 'build-time pseudo-query inputs; not research query vectors',
                'prompt': PROMPT, 'salt': SALT, 'set_b_count': set_b, 'max_words': MAX_WORDS,
                'stopwords_sha256': hashlib.sha256(' '.join(sorted(STOPWORDS)).encode()).hexdigest(),
                'counts': {name: len(rows) for name, rows in records.items()}, 'excluded': excluded,
                'sample_report_sha256': checksum(sample_inputs / 'report.json'),
                'reference_encoding_sha256': checksum(reference / 'float32/encoder/encoding.json'),
                'label_map_sha256': LABEL_MAP_SHA256, 'source_sha256': checksum(__file__),
                'held_out_retrieval_evaluated': False,
                'files': {name: checksum(output / name) for name in files}}
    atomic_json(output / 'pseudo-inputs.json', contract)
    return contract


def validate_inputs(folder):
    folder = Path(folder)
    contract = json.loads((folder / 'pseudo-inputs.json').read_text())
    if contract.get('status') != 'prepared' or contract.get('prompt') != PROMPT:
        raise ValueError('invalid pseudo-query input contract')
    for name, expected in contract['files'].items():
        if checksum(folder / name) != expected:
            raise ValueError(f'pseudo-query input checksum mismatch: {name}')
    sets = {name: read_jsonl(folder / f'records-{name}.jsonl') for name in contract['counts']}
    for name, rows in sets.items():
        ids = [r['id'] for r in rows]
        if (len(rows) != contract['counts'][name] or len(set(ids)) != len(ids)
                or any(hashlib.sha256(r['text'].encode()).hexdigest() != r['text_sha256']
                       or not r['text'].startswith(PROMPT) for r in rows)):
            raise ValueError('pseudo-query records changed')
    return contract, sets


def length_batches(lengths, batch):
    """Equal-length groups in first-occurrence order; each index appears exactly once."""
    if type(batch) is not int or batch <= 0 or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError('invalid lengths or batch size')
    groups = {}
    for index, length in enumerate(lengths):
        groups.setdefault(length, []).append(index)
    return [members[i:i + batch] for members in groups.values() for i in range(0, len(members), batch)]


def encode_rows(texts, tokenize, forward, batch, *, dim=2560, progress=None):
    """Tokenize without padding, run equal-length batches, restore input order.

    ``forward`` receives a list of equal-length token-ID lists and returns float32
    L2-normalized last-token embeddings, one row per sequence.
    """
    tokens = tokenize(texts)
    if len(tokens) != len(texts):
        raise ValueError('tokenizer population mismatch')
    out = np.empty((len(texts), dim), dtype=np.float32)
    filled = np.zeros(len(texts), dtype=bool)
    groups = length_batches([len(t) for t in tokens], batch)
    for number, members in enumerate(groups):
        values = np.asarray(forward([tokens[i] for i in members]), dtype=np.float32)
        if values.shape != (len(members), dim) or not np.isfinite(values).all():
            raise ValueError('invalid embedding batch')
        if not np.allclose(np.linalg.norm(values.astype(np.float64), axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError('embeddings are not unit normalized')
        out[members] = values
        filled[members] = True
        if progress:
            progress(number + 1, len(groups))
    if not filled.all():
        raise ValueError('some rows were not encoded')
    return out, [len(t) for t in tokens]


def cosine_check(left, right):
    a, b = left.astype(np.float64), right.astype(np.float64)
    cos = np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return {'min_cosine': float(cos.min()), 'mean_cosine': float(cos.mean()),
            'max_abs_difference': float(np.abs(a - b).max()), 'gate': COSINE_GATE,
            'passed': bool(cos.min() >= COSINE_GATE)}


def encode(model_folder, inputs, output, *, batch=64, device='cuda', threads=4, dim=2560):
    import importlib.metadata
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer
    from . import encoder

    inputs, output = Path(inputs), Path(output)
    contract, sets = validate_inputs(inputs)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'dtype': 'float32', 'device': device, 'batch': batch,
              'batching': 'equal token length only; no padding', 'pooling': 'last token',
              'normalization': 'float32 L2', 'attention': 'eager', 'truncation': 'none',
              'inputs_sha256': checksum(inputs / 'pseudo-inputs.json'), 'source_sha256': checksum(__file__),
              'sets': {}}
    atomic_json(output / 'encoding.json', report)
    try:
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA requested but unavailable; refusing device fallback')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report.update(gpu=torch.cuda.get_device_name(0), tf32=False)
        report['model_identity'] = encoder.verify_model(model_folder)
        report['packages'] = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'tokenizers')}
        torch.set_num_threads(threads)
        torch.manual_seed(11)
        tokenizer = AutoTokenizer.from_pretrained(str(model_folder), local_files_only=True, trust_remote_code=False)
        started = time.perf_counter()
        model = AutoModel.from_pretrained(str(model_folder), local_files_only=True, trust_remote_code=False,
                                          use_safetensors=True, dtype=torch.float32, low_cpu_mem_usage=True,
                                          attn_implementation='eager').to(device).eval()
        report['model_load_seconds'] = time.perf_counter() - started

        def tokenize(texts):
            return [list(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]

        def forward(rows):
            ids = torch.tensor(rows, dtype=torch.long, device=device)
            with torch.inference_mode():
                hidden = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
                return functional.normalize(hidden[:, -1, :].float(), p=2, dim=1).cpu().numpy()

        def run(name, texts):
            began = time.perf_counter()
            vectors, lengths = encode_rows(texts, tokenize, forward, batch, dim=dim,
                progress=lambda done, total: print(json.dumps({'set': name, 'batches': done, 'of': total}), flush=True)
                if done % 50 == 0 or done == total else None)
            return vectors, {'count': len(texts), 'seconds': time.perf_counter() - began,
                             'tokens': {'min': min(lengths), 'max': max(lengths), 'mean': float(np.mean(lengths))}}

        references = read_jsonl(inputs / 'reference-records.jsonl')
        vectors, stats = run('reference', [r['text'] for r in references])
        check = {**stats, **cosine_check(vectors, np.load(inputs / 'reference-vectors.npy', allow_pickle=False))}
        report['reference_check'] = check
        atomic_json(output / 'encoding.json', report)
        if not check['passed']:
            raise ValueError('batched encoder differs from the verified query encoding')
        for name, rows in sets.items():
            vectors, stats = run(name, [r['text'] for r in rows])
            np.save(output / f'vectors-{name}.npy', vectors, allow_pickle=False)
            atomic_json(output / f'ids-{name}.json', [r['id'] for r in rows])
            report['sets'][name] = {**stats, 'vectors_sha256': checksum(output / f'vectors-{name}.npy'),
                                    'ids_sha256': checksum(output / f'ids-{name}.json')}
            atomic_json(output / 'encoding.json', report)
        report['status'] = 'completed'
        atomic_json(output / 'encoding.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)[:500])
        atomic_json(output / 'encoding.json', report)
        raise


def verify_output(inputs, output):
    """Check a returned encoding against its inputs without the model."""
    contract, sets = validate_inputs(inputs)
    report = json.loads((Path(output) / 'encoding.json').read_text())
    if report.get('status') != 'completed' or report['inputs_sha256'] != checksum(Path(inputs) / 'pseudo-inputs.json'):
        raise ValueError('encoding incomplete or for different inputs')
    if not report['reference_check']['passed'] or report.get('tf32') is not False:
        raise ValueError('encoder gate failed or TF32 not disabled')
    for name, rows in sets.items():
        entry = report['sets'][name]
        ids = json.loads((Path(output) / f'ids-{name}.json').read_text())
        values = np.load(Path(output) / f'vectors-{name}.npy', allow_pickle=False)
        if (checksum(Path(output) / f'vectors-{name}.npy') != entry['vectors_sha256'] or ids != [r['id'] for r in rows]
                or values.shape != (len(rows), 2560) or values.dtype != np.float32 or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values.astype(np.float64), axis=1), 1, atol=1e-5, rtol=0)):
            raise ValueError(f'returned pseudo-query set {name} is invalid')
    return {'status': 'verified', 'sets': {n: report['sets'][n]['count'] for n in sets},
            'reference_min_cosine': report['reference_check']['min_cosine']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('release', 'sample_inputs', 'reference', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--set-b', type=int, default=SET_B_COUNT)
    p = sub.add_parser('validate'); p.add_argument('inputs', type=Path)
    p = sub.add_parser('encode'); p.add_argument('model', type=Path); p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda'); p.add_argument('--threads', type=int, default=4)
    p = sub.add_parser('verify'); p.add_argument('inputs', type=Path); p.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.release, args.sample_inputs, args.reference, args.output, set_b=args.set_b)
    elif args.command == 'validate':
        contract, _ = validate_inputs(args.inputs)
        result = {'status': 'valid', 'counts': contract['counts']}
    elif args.command == 'encode':
        result = encode(args.model, args.inputs, args.output, batch=args.batch, device=args.device, threads=args.threads)
    else:
        result = verify_output(args.inputs, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'model_identity')}, indent=2))


if __name__ == '__main__':
    main()
