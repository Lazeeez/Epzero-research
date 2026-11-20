"""EXPLORATORY follow-up: a locally run relevance judge that follows the organisers' procedure.

Not part of FINAL_PROTOCOL_V1. The validation thresholds are in docs/protocol/JUDGE_VALIDATION_THRESHOLDS.md.

The lab's released LLM labels were made by `clef26/topic-extraction/evaluation` of
https://github.com/clef-longeval/longeval-code (commit b5f11fd2d58e, 2026-06-07):

  input    query = the query text; document = title + " " + abstract (ir_datasets_longeval default_text)
  prompt   `umbrella_zeroshot_no_desc_no_narrative`, adapted there from UMBRELA's zero-shot basic prompt
  call     one user message to an OpenAI-compatible chat endpoint, with no system message and no
           sampling parameter, so the provider's defaults applied; one call per query-document pair
  grade    UMBRELA's answer parser applied to the message content; 0 when nothing matches

`prompt` and `parse_grade` reproduce that prompt and that parser exactly (UMBRELA is
https://github.com/castorini/umbrela, Apache-2.0); tests pin both. Everything else here is the
plumbing around them: drawing prompt-development pairs, calling an endpoint, and summarising.

This module prepares and runs the *pilot*: a small set of officially graded pairs, outside the
study's own pool and disjoint from the later validation sample, used to check that the judge
runs, that answers parse, and how stable the grades are. It decides no threshold.

LongEval texts and identifiers stay in the prepared inputs and the returned evidence; neither is committed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import http.server
import json
from pathlib import Path
import random
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

from .documents import DOCUMENTS, read_document
from .release import checksum, read_queries
from .storage import atomic_json, fetch_archive

# Module-level imports stay light: the job runs this file with a Python that has only numpy.
SNAPSHOTS = (1, 2, 3)
TEST_FILE = 'longeval_adhoc-queries-snapshot-test.tsv'
TEST_SHA256 = 'c7ea5fed1aa1dd37b647edfd686acd9cb26891b1cde52c43e3dc361f4942e254'

FAMILY = 'gpt-oss-120b'
MODEL_ID = 'openai/gpt-oss-120b'
# https://huggingface.co/openai/gpt-oss-120b at this revision; MXFP4 weights. `metal/` and `original/` are not needed.
MODEL_REVISION = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
MODEL_FILES = {
    'chat_template.jinja': (16738, 'a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146'),
    'config.json': (2089, '933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566'),
    'generation_config.json': (177, 'f9970ada892d2d1f72e3ed0a6535ccebadd11897318794ca671d8c7014c957da'),
    'model.safetensors.index.json': (54511, 'ede2655fdc05008561983b6e0829c600727c28d591e071077377059f03a6c00e'),
    'special_tokens_map.json': (98, 'dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293'),
    'tokenizer_config.json': (4200, '9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070'),
}
# The weight shards and the tokenizer are pinned by size and SHA-256 in judge-model.json (written from the Hub's metadata).
SEED = 20261121
PER_STRATUM = 30                      # pairs per (official grade 0-2) x (first graded snapshot 1-3)
MAX_TOKENS = 4096                     # a cap against runaway generations; the organisers set none
# Sampling settings under test. `provider_default` sends nothing, as the organisers did.
CONDITIONS = {'provider_default': {}, 'temperature_0': {'temperature': 0}}
REPEATS = 2


def prompt(query: str, document: str) -> str:
    """The organisers' `umbrella_zeroshot_no_desc_no_narrative` prompt, character for character."""
    return ("""You are an expert judge of a content. Using your internal knowledge and simple commonsense reasoning, try to verify if the passage is relevance category to the query.
Here, "0" represent that the passage has nothing to do with the query, "1" represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and "2" represents that the passage is dedicated to the query and contains the exact answer.

Provide explanation for the relevance and give your answer with from one of the categories 0, 1, or 2 only. One of the categorical values if compulsory in answer.

Instructions: Think about the question. Read the TREC-style query. After explaining your reasoning, provide your answer in terms of 0, 1, or 2 category. Only provide the relevance category on the last line. Do not provide any further details on the last line.

###

Query: """ + query + """
Document: """ + document + """

Explanation:
    """)


_PATTERNS = (
    r'"o"\s*[:-=]?\s*(0|1|2|3)',
    r"\'o\'\s*[:-=]?\s*(0|1|2|3)",
    r"o\s*[:-=]?\s*(0|1|2|3)",
    r'"overall_score"\s*[:-=]?\s*(0|1|2|3)',
    r'"overall"\s*[:-=]?\s*(0|1|2|3)',
    r'"overall score"\s*[:-=]?\s*(0|1|2|3)',
    r'"final score"\s*[:-=]?\s*(0|1|2|3)',
    r'final score\s*[:-=]?\s*(0|1|2|3)',
    r"final score is (0|1|2|3)",
    r'"final_score"\s*[:-=]?\s*(0|1|2|3)',
    r'"score"\s*[:-=]?\s*(0|1|2|3)',
    r'"o_score"\s*[:-=]?\s*(0|1|2|3)',
    r"output score is (0|1|2|3)",
    r"score is (0|1|2|3)",
    r"[a-zA-Z]+\s+is\s+(0|1|2|3)\s",
    r"relevance category\s*[:-=]?\s*(0|1|2|3)",
    r"relevance category\s*[:-=]?\s*(0|1|2|3)",
    r"relevance category is (0|1|2|3)",
    r"it falls into the category (0|1|2|3)",
    r"category\s*(0|1|2|3)",
    r"relevance category (0|1|2|3)",
    r"relevance category for this passage would be (0|1|2|3)",
    r"the relevance category would be (0|1|2|3)",
    r"\n*(0|1|2|3)",
)


def parse_grade(response: str):
    """UMBRELA's answer parser as the organisers used it: (grade, valid). The first pattern that
    matches decides, by its last match; a response with no match is graded 0 and marked invalid."""
    response = response.strip().lower()
    for pattern in _PATTERNS:
        matched = None
        for match in re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            matched = match
        if matched:
            return int(matched.group(1)), 1
    return 0, 0


def last_line_grade(response: str):
    """The grade on the last non-empty line, which is where the prompt asks for it; None if that
    line is not a single category. A diagnostic only: the organisers' parser is `parse_grade`."""
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    match = re.fullmatch(r'\W*(?:(?:relevance\s+)?category\W*)?([0-3])\W*', lines[-1], re.IGNORECASE) if lines else None
    return int(match.group(1)) if match else None


# ------------------------------------------------------------------------------------------ inputs
def _write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def _read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def official_pairs(judged):
    """{(query, document): (grade, first snapshot graded)}; the grade of a pair is the same at every snapshot."""
    pairs = {}
    for step in SNAPSHOTS:
        for query, docs in judged[step].items():
            for doc, grade in docs.items():
                if pairs.setdefault((query, doc), (grade, step))[0] != grade:
                    raise ValueError('conflicting official grades for one pair')
    return pairs


def select_pairs(pairs, excluded, *, per_stratum=PER_STRATUM, seed=SEED):
    """A fixed number of pairs per (grade 0-2, first graded snapshot), chosen by a keyed hash.

    Grade-3 pairs are left for the validation sample, which takes all four of them.
    """
    strata = {}
    for (query, doc), (grade, step) in pairs.items():
        if grade in (0, 1, 2) and (query, doc) not in excluded:
            key = hashlib.sha256(f'judge-pilot-v1:{seed}:{query}:{doc}'.encode()).digest()
            strata.setdefault((grade, step), []).append((key, query, doc))
    chosen = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        if len(members) < per_stratum:
            raise ValueError(f'stratum {stratum} has only {len(members)} eligible pairs')
        chosen.extend((query, doc, stratum) for _, query, doc in members[:per_stratum])
    return chosen


def document_texts(folder, wanted):
    """Title-and-abstract text of the wanted documents, read from the released archives.

    The newest snapshot is read first: the collection only grows and shared documents keep their text.
    """
    wanted, found = set(wanted), {}
    folder = Path(folder)
    for _, _, _, subfolder in reversed(DOCUMENTS):
        if len(found) == len(wanted):
            break
        with zipfile.ZipFile(folder / (subfolder + '.zip')) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith('.jsonl'):
                    continue
                with archive.open(name) as stream:
                    for line in stream:
                        record = json.loads(line)
                        if record.get('id') in wanted and record['id'] not in found:
                            docid, text = read_document(record)
                            if text:
                                found[docid] = text
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f'{len(missing)} documents have no text in the release')
    return found


def prepare(release, ledger, output, *, per_stratum=PER_STRATUM, seed=SEED):
    """Draw the prompt-development pairs and write the judge's inputs.

    `pairs.jsonl` is all the judge sees: an opaque pair number, the query and the document text, in
    a shuffled order. The official grades go to `reference.jsonl`, which only the analysis reads.
    """
    from .heldout_analysis import load_test_qrels
    release, ledger, output = Path(release), Path(ledger), Path(output)
    queries_file = release / 'files' / TEST_FILE
    if checksum(queries_file) != TEST_SHA256:
        raise ValueError('test query release identity mismatch')
    queries = read_queries(queries_file)
    qrels, identity = load_test_qrels(release / 'files')
    pairs = {key: value for key, value in official_pairs(qrels[FAMILY]).items() if key[0] in queries}
    with ledger.open(newline='') as handle:
        pool = {(row['query'], row['document']) for row in csv.DictReader(handle)}
    chosen = select_pairs(pairs, pool, per_stratum=per_stratum, seed=seed)
    texts = document_texts(release / 'documents', {doc for _, doc, _ in chosen})
    random.Random(seed).shuffle(chosen)
    rows, reference = [], []
    for number, (query, doc, (grade, step)) in enumerate(chosen, 1):
        rows.append({'pair': number, 'query_id': query, 'doc_id': doc, 'query': queries[query], 'text': texts[doc]})
        reference.append({'pair': number, 'official_grade': grade, 'first_graded_snapshot': step})
    return write_inputs(output, rows, reference, {
        'family': FAMILY, 'per_stratum': per_stratum, 'strata': 'official grade 0-2 x first graded snapshot 1-3',
        'seed': seed, 'excluded': 'pairs in the study pool ledger; official grade 3',
        'official_pairs': len(pairs), 'pool_pairs_excluded': len(pool & pairs.keys()),
        'judgment_sha256': {f'{FAMILY}/snapshot-{s}': identity[f'{FAMILY}/snapshot-{s}'] for s in SNAPSHOTS},
        'test_queries_sha256': TEST_SHA256, 'ledger_sha256': checksum(ledger)})


def write_inputs(output, rows, reference, provenance):
    """Write the input folder: the judge's pairs, the separate reference grades, and the contract."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / 'pairs.jsonl', rows)
    _write_jsonl(output / 'reference.jsonl', reference)
    lengths = [len(prompt(row['query'], row['text'])) for row in rows]
    contract = {'status': 'prepared', 'purpose': 'judge pilot on prompt-development pairs; no study outcome',
                'pairs': len(rows), 'queries': len({row['query_id'] for row in rows}),
                'prompt_characters': {'min': min(lengths), 'median': int(np.median(lengths)), 'max': max(lengths)},
                'prompt_sha256': hashlib.sha256(prompt('{query}', '{document}').encode()).hexdigest(),
                **provenance, 'source_sha256': checksum(__file__),
                'files': {name: checksum(output / name) for name in ('pairs.jsonl', 'reference.jsonl')}}
    atomic_json(output / 'inputs.json', contract)
    return contract


def validate_inputs(folder):
    folder = Path(folder)
    contract = json.loads((folder / 'inputs.json').read_text())
    if contract.get('status') != 'prepared' or set(contract['files']) != {'pairs.jsonl', 'reference.jsonl'}:
        raise ValueError('not a prepared judge-pilot input folder')
    for name, digest in contract['files'].items():
        if checksum(folder / name) != digest:
            raise ValueError(f'input checksum mismatch: {name}')
    if contract['prompt_sha256'] != hashlib.sha256(prompt('{query}', '{document}').encode()).hexdigest():
        raise ValueError('the inputs were prepared for a different prompt')
    rows, reference = _read_jsonl(folder / 'pairs.jsonl'), _read_jsonl(folder / 'reference.jsonl')
    numbers = [row['pair'] for row in rows]
    if numbers != list(range(1, len(rows) + 1)) or numbers != [row['pair'] for row in reference] or len(rows) != contract['pairs']:
        raise ValueError('pair numbering differs between the inputs')
    if any(set(row) != {'pair', 'query_id', 'doc_id', 'query', 'text'} or not row['query'].strip() or not row['text'].strip()
           for row in rows):
        raise ValueError('a pair lacks its query or document text')
    if len({(row['query_id'], row['doc_id']) for row in rows}) != len(rows):
        raise ValueError('a pair occurs twice')
    if any(row['official_grade'] not in (0, 1, 2) for row in reference):
        raise ValueError('unexpected official grade in the pilot reference')
    return contract


# ------------------------------------------------------------------------------------------ model files
def model_manifest(path):
    """The pinned file list: small files from this module, weights and tokenizer from judge-model.json."""
    pinned = json.loads(Path(path).read_text())
    if pinned['model'] != MODEL_ID or pinned['revision'] != MODEL_REVISION:
        raise ValueError('judge-model.json is for another model revision')
    files = {name: tuple(value) for name, value in MODEL_FILES.items()}
    files.update({name: (item['bytes'], item['sha256']) for name, item in pinned['files'].items()})
    return files


def fetch_model(manifest, target, *, workers=4):
    """Download the pinned revision into `target` and verify every file by size and SHA-256.

    Interrupted transfers resume; a file that is already complete is only verified.
    """
    files = model_manifest(manifest)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, (size, digest) = item
        fetch_archive(f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}', target / name, digest, size)
        print(json.dumps({'file': name, 'bytes': size, 'status': 'verified'}), flush=True)
        return name, {'bytes': size, 'sha256': digest}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        identity = dict(pool.map(fetch, sorted(files.items())))
    record = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'files': identity,
              'bytes': sum(item['bytes'] for item in identity.values())}
    atomic_json(target / 'model-identity.json', record)
    return record


# ------------------------------------------------------------------------------------------ calling the endpoint
def _post(url, payload, timeout):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def ask(base_url, model, row, params, *, timeout=900, attempts=3):
    """One judgment: the organisers' request (model and one user message) plus the condition's parameters."""
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt(row['query'], row['text'])}],
               'max_tokens': MAX_TOKENS, **params}
    started = time.perf_counter()
    error = None
    for attempt in range(1, attempts + 1):
        try:
            body = _post(base_url.rstrip('/') + '/chat/completions', payload, timeout)
            choice = body['choices'][0]
            return {'pair': row['pair'], 'status': 'ok', 'attempts': attempt, 'seconds': time.perf_counter() - started,
                    'message': choice['message'], 'finish_reason': choice.get('finish_reason'), 'usage': body.get('usage')}
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, IndexError, ValueError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            time.sleep(min(30, 2 ** attempt))
    return {'pair': row['pair'], 'status': 'failed', 'attempts': attempts, 'seconds': time.perf_counter() - started,
            'error': error}


def wait_for_server(base_url, *, pid=None, timeout=3600, interval=10):
    """Block until the endpoint lists a model; fail early if the server process `pid` has exited."""
    import os
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid is not None:
            try:
                os.kill(pid, 0)
            except OSError:
                raise RuntimeError('the model server exited before it became ready') from None
        try:
            with urllib.request.urlopen(base_url.rstrip('/') + '/models', timeout=10) as response:
                names = [item['id'] for item in json.loads(response.read())['data']]
            if names:
                return {'ready': True, 'models': names}
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError):
            pass
        time.sleep(interval)
    raise RuntimeError(f'the model server was not ready after {timeout} seconds')


def probe(inputs, output, *, base_url, model=MODEL_ID):
    """One judgment before the run, kept in full. It must come back with a final answer in the
    message content, separate from the reasoning, because that is the text the parser reads."""
    inputs, output = Path(inputs), Path(output)
    validate_inputs(inputs)
    row = _read_jsonl(inputs / 'pairs.jsonl')[0]
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt(row['query'], row['text'])}],
               'max_tokens': MAX_TOKENS, 'temperature': 0}
    body = _post(base_url.rstrip('/') + '/chat/completions', payload, 900)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'probe.json', {'pair': row['pair'], 'response': body})
    message = body['choices'][0]['message']
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('the probe returned no final answer in the message content')
    if any(marker in content for marker in ('<|channel|>', '<|start|>', '<|message|>')):
        raise RuntimeError('the probe content still holds channel markers: reasoning and answer were not separated')
    grade, valid = parse_grade(content)
    return {'pair': row['pair'], 'grade': grade, 'parsed': bool(valid), 'last_line_grade': last_line_grade(content),
            'finish_reason': body['choices'][0].get('finish_reason'), 'usage': body.get('usage'),
            'message_fields': sorted(key for key, value in message.items() if value), 'content_characters': len(content)}


def judge(inputs, output, *, base_url, model=MODEL_ID, conditions=None, repeats=REPEATS, workers=16, seed=SEED):
    """Grade every pair under each condition, `repeats` times, in a fresh shuffled order each time.

    Responses are appended as they arrive, so an interrupted run resumes where it stopped.
    """
    inputs, output = Path(inputs), Path(output)
    contract = validate_inputs(inputs)
    rows = _read_jsonl(inputs / 'pairs.jsonl')
    conditions = dict(CONDITIONS if conditions is None else conditions)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for name, params in conditions.items():
        for repeat in range(1, repeats + 1):
            path = output / f'responses-{name}-r{repeat}.jsonl'
            done = {row['pair'] for row in _read_jsonl(path) if row['status'] == 'ok'} if path.exists() else set()
            todo = [row for row in rows if row['pair'] not in done]
            random.Random(f'{seed}:{name}:{repeat}').shuffle(todo)
            started = time.perf_counter()
            lock = threading.Lock()
            with path.open('a', encoding='utf-8', newline='\n') as stream, \
                    concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(lambda row: ask(base_url, model, row, params), todo):
                    with lock:
                        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                        stream.flush()
            runs.append({'condition': name, 'repeat': repeat, 'file': path.name, 'requested': len(todo),
                         'already_done': len(done), 'wall_seconds': time.perf_counter() - started})
            print(json.dumps(runs[-1]), flush=True)
    record = {'status': 'completed', 'model': model, 'conditions': conditions, 'repeats': repeats, 'workers': workers,
              'max_tokens': MAX_TOKENS, 'seed': seed, 'inputs_sha256': checksum(inputs / 'inputs.json'),
              'prompt_sha256': contract['prompt_sha256'], 'source_sha256': checksum(__file__), 'runs': runs}
    atomic_json(output / 'judging.json', record)
    return record


# ------------------------------------------------------------------------------------------ summary
def _content(row):
    """The text the organisers' parser saw: the message content (the final answer, not the reasoning)."""
    content = row.get('message', {}).get('content') if row.get('status') == 'ok' else None
    return content if isinstance(content, str) else None


def _agreement(left, right, clusters, *, draws=2000, seed=SEED):
    """Exact and binary agreement and linear-weighted kappa of two grade vectors, with query-clustered intervals."""
    from .heldout_pool import weighted_kappa
    left, right = np.asarray(left), np.asarray(right)

    def measures(rows):
        confusion = np.zeros((4, 4), dtype=np.int64)
        np.add.at(confusion, (left[rows], right[rows]), 1)
        return (float(np.mean(left[rows] == right[rows])), float(np.mean((left[rows] > 0) == (right[rows] > 0))),
                weighted_kappa(confusion))

    members = {}
    for index, cluster in enumerate(clusters):
        members.setdefault(cluster, []).append(index)
    groups = [np.array(rows) for rows in members.values()]
    rng = np.random.default_rng(seed)
    boot = np.array([measures(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]))
                     for _ in range(draws)])
    point = measures(np.arange(len(left)))
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (left, right), 1)
    out = {'pairs': int(len(left)), 'confusion_rows_first_columns_second': confusion.tolist()}
    for i, name in enumerate(('exact_agreement', 'binary_agreement', 'linear_weighted_kappa')):
        column = boot[:, i][np.isfinite(boot[:, i])]
        out[name] = point[i] if np.isfinite(point[i]) else None          # kappa is undefined for a single cell
        out[name + '_interval_95'] = np.quantile(column, [.025, .975]).tolist() if len(column) else None
    return out


def analyze(inputs, output):
    """Summarise a pilot run: does the judge answer, do answers parse, how stable are the grades, and
    how do they compare with the official grades of these prompt-development pairs."""
    inputs, output = Path(inputs), Path(output)
    validate_inputs(inputs)
    record = json.loads((output / 'judging.json').read_text())
    rows = {row['pair']: row for row in _read_jsonl(inputs / 'pairs.jsonl')}
    reference = {row['pair']: row for row in _read_jsonl(inputs / 'reference.jsonl')}
    numbers = sorted(rows)
    clusters = [rows[n]['query_id'] for n in numbers]
    official = [reference[n]['official_grade'] for n in numbers]
    grades, contents, runs = {}, {}, []
    for run in record['runs']:
        latest = {}
        for row in _read_jsonl(output / run['file']):
            if row['status'] == 'ok' or row['pair'] not in latest:
                latest[row['pair']] = row
        key = (run['condition'], run['repeat'])
        parsed = {n: parse_grade(_content(latest[n])) if n in latest and _content(latest[n]) is not None else (0, 0)
                  for n in numbers}
        last = {n: last_line_grade(_content(latest[n])) if n in latest and _content(latest[n]) else None for n in numbers}
        grades[key] = [parsed[n][0] for n in numbers]
        contents[key] = [_content(latest[n]) if n in latest else None for n in numbers]
        ok = [latest[n] for n in numbers if n in latest and latest[n]['status'] == 'ok']
        usage = [row['usage'] for row in ok if row.get('usage')]
        completion = sum(u.get('completion_tokens', 0) for u in usage)
        item = {'condition': run['condition'], 'repeat': run['repeat'], 'pairs': len(numbers),
                'answered': len(ok), 'failed_requests': len(numbers) - len(ok),
                'without_final_content': sum(_content(row) is None or not _content(row).strip() for row in ok),
                'unparsed': sum(parsed[n][1] == 0 for n in numbers),
                # How often the organisers' parser returns something other than the category on the last line.
                'last_line_is_a_category': sum(last[n] is not None for n in numbers),
                'parser_differs_from_last_line': sum(last[n] is not None and last[n] != parsed[n][0] for n in numbers),
                'finish_reasons': {reason: sum(row.get('finish_reason') == reason for row in ok)
                                   for reason in sorted({str(row.get('finish_reason')) for row in ok})},
                'grade_counts': {str(g): grades[key].count(g) for g in range(4)},
                'wall_seconds': run['wall_seconds'], 'requests_in_this_invocation': run['requested'],
                'mean_prompt_tokens': float(np.mean([u.get('prompt_tokens', 0) for u in usage])) if usage else None,
                'mean_completion_tokens': completion / len(usage) if usage else None,
                'completion_tokens_per_second': completion / run['wall_seconds'] if run['wall_seconds'] and run['requested'] == len(numbers) else None,
                'pairs_per_hour': 3600 * run['requested'] / run['wall_seconds'] if run['wall_seconds'] and run['requested'] else None,
                'against_official': _agreement(official, grades[key], clusters),
                'agreement_by_official_grade': {str(g): float(np.mean([a == b for a, b in zip(official, grades[key]) if a == g]))
                                                for g in sorted(set(official))}}
        runs.append(item)
    repeats = []
    for name in record['conditions']:
        for a in range(1, record['repeats'] + 1):
            for b in range(a + 1, record['repeats'] + 1):
                same_text = [x is not None and x == y for x, y in zip(contents[(name, a)], contents[(name, b)])]
                repeats.append({'condition': name, 'repeats': [a, b], 'identical_answers': float(np.mean(same_text)),
                                **_agreement(grades[(name, a)], grades[(name, b)], clusters)})
    between = []
    names = list(record['conditions'])
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            between.append({'conditions': [first, second],
                            **_agreement(grades[(first, 1)], grades[(second, 1)], clusters)})
    report = {'status': 'completed',
              'kind': 'PILOT on prompt-development pairs: not the validation sample, and no threshold is decided here',
              'model': record['model'], 'max_tokens': record['max_tokens'], 'conditions': record['conditions'],
              'official_grade_counts': {str(g): official.count(g) for g in sorted(set(official))},
              'runs': runs, 'between_repeats': repeats, 'between_conditions_first_repeat': between,
              'inputs_sha256': record['inputs_sha256'], 'prompt_sha256': record['prompt_sha256'],
              'source_sha256': checksum(__file__)}
    atomic_json(output / 'report.json', report)
    return report


# ------------------------------------------------------------------------------------------ a stand-in endpoint
class _Mock(http.server.BaseHTTPRequestHandler):
    """A stand-in for the model server, for tests and dry runs: it answers from a hash of the prompt."""

    def log_message(self, *args):
        pass

    def _send(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model'}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        text = payload['messages'][0]['content']
        digest = hashlib.sha256(text.encode()).digest()
        grade = digest[0] % 3
        # Without a temperature the stand-in varies its answer between calls, like a sampling model.
        if 'temperature' not in payload and next(self.server.counter) % 5 == 0:
            grade = (grade + 1) % 3
        content = f'The document is graded on its match with the query.\n{grade}'
        self._send({'id': 'mock', 'object': 'chat.completion', 'model': payload['model'],
                    'choices': [{'index': 0, 'finish_reason': 'stop',
                                 'message': {'role': 'assistant', 'content': content, 'reasoning_content': 'mock reasoning'}}],
                    'usage': {'prompt_tokens': len(text) // 4, 'completion_tokens': 12, 'total_tokens': len(text) // 4 + 12}})


def mock_server(port=0):
    import itertools
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), _Mock)
    server.counter = itertools.count(1)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='draw the pilot pairs and write the inputs (local, needs the release)')
    for name in ('release', 'ledger', 'output'):
        p.add_argument(name, type=Path)
    p = sub.add_parser('validate')
    p.add_argument('inputs', type=Path)
    p = sub.add_parser('fetch-model', help='download and verify the pinned model revision')
    p.add_argument('manifest', type=Path)
    p.add_argument('target', type=Path)
    p = sub.add_parser('wait-server', help='wait until the endpoint is ready')
    p.add_argument('--base-url', required=True)
    p.add_argument('--pid', type=int)
    p.add_argument('--timeout', type=int, default=3600)
    p = sub.add_parser('probe', help='send one judgment and check the shape of the answer')
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', default=MODEL_ID)
    p = sub.add_parser('judge')
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--workers', type=int, default=16)
    p = sub.add_parser('analyze')
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p = sub.add_parser('mock-server', help='serve stand-in answers on a local port (for dry runs)')
    p.add_argument('--port', type=int, default=8765)
    a = parser.parse_args()
    if a.command == 'prepare':
        r = prepare(a.release, a.ledger, a.output)
    elif a.command == 'validate':
        r = validate_inputs(a.inputs)
        r = {key: r[key] for key in ('pairs', 'queries', 'prompt_sha256', 'files')}
    elif a.command == 'fetch-model':
        r = fetch_model(a.manifest, a.target)
        r = {key: r[key] for key in ('model', 'revision', 'bytes')}
    elif a.command == 'wait-server':
        r = wait_for_server(a.base_url, pid=a.pid, timeout=a.timeout)
    elif a.command == 'probe':
        r = probe(a.inputs, a.output, base_url=a.base_url, model=a.model)
    elif a.command == 'judge':
        r = judge(a.inputs, a.output, base_url=a.base_url, model=a.model, workers=a.workers)
        r = {'status': r['status'], 'runs': len(r['runs'])}
    elif a.command == 'analyze':
        r = analyze(a.inputs, a.output)
        r = {'runs': [{k: run[k] for k in ('condition', 'repeat', 'answered', 'unparsed', 'without_final_content', 'pairs_per_hour')}
                      for run in r['runs']]}
    else:
        server = mock_server(a.port)
        print(json.dumps({'mock_server': f'http://127.0.0.1:{server.server_address[1]}/v1'}), flush=True)
        server.serve_forever()
        return
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
