"""EXPLORATORY follow-up: a blinded human audit of what approximate search drops and what it puts in its place.

Not part of FINAL_PROTOCOL_V1. The held-out analysis found that the documents an approximate index
substitutes into a top 10 are mostly ungraded, so their relevance is unknown. This module draws a
sample of query-document pairs for a person to grade without knowing which is which:

  dropped      in the exact top 10 of a query at a snapshot, but not in the approximate top 10
  replacement  in the approximate top 10, but not in the exact top 10

over the three primary arms, their three seeds and the three snapshots, on the primary panel. A pair
that is dropped in one run and a replacement in another is left out, and counted.

`prepare` writes the annotation pages (one for everyone, one 50-pair overlap for a second annotator)
and a separate key. The pages show only the query and the document; the key holds the roles and the
official grades and is read by `analyze` alone. Grades use the scale of the organisers' prompt (0, 1, 2).

Everything written here holds LongEval text or identifiers and stays in the build folder.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import random

import numpy as np

from .release import checksum, read_queries
from .storage import atomic_json

SEED = 20261121
PER_ROLE = 100
OVERLAP_PER_ROLE = 25
ROLES = ('dropped', 'replacement')
DEPTH = 10
TEST_FILE = 'longeval_adhoc-queries-snapshot-test.tsv'
SCALE = {0: 'The document has nothing to do with the query.',
         1: 'The document has some answer for the query, but the answer may be a bit unclear, or hidden amongst '
            'extraneous information.',
         2: 'The document is dedicated to the query and contains the exact answer.'}


def roles(primary, exact, arms, population):
    """{(query, document): {'dropped': n, 'replacement': n}}: occurrences of each role over arms, seeds and snapshots."""
    out = {}
    wanted = set(arms.values())
    for (backend, mode, seed, step, regime), run in primary.items():
        if (backend, mode, regime) not in wanted:
            continue
        for query in population:
            reference = {doc for doc, _ in exact[step][query][:DEPTH]}
            returned = {doc for doc, _ in run[query][:DEPTH]}
            for doc in reference - returned:
                out.setdefault((query, doc), {'dropped': 0, 'replacement': 0})['dropped'] += 1
            for doc in returned - reference:
                out.setdefault((query, doc), {'dropped': 0, 'replacement': 0})['replacement'] += 1
    return out


def select(occurrences, *, per_role=PER_ROLE, overlap=OVERLAP_PER_ROLE, seed=SEED):
    """`per_role` pairs of each role, by a keyed hash; the first `overlap` of each go to the second annotator."""
    chosen, counts = [], {}
    for role in ROLES:
        members = sorted((hashlib.sha256(f'human-audit-v1:{seed}:{query}:{doc}'.encode()).digest(), query, doc)
                         for (query, doc), n in occurrences.items() if n[role] and not n[ROLES[1 - ROLES.index(role)]])
        counts[role] = len(members)
        if len(members) < per_role:
            raise ValueError(f'only {len(members)} pairs have the role {role}')
        for rank, (_, query, doc) in enumerate(members[:per_role]):
            chosen.append({'query_id': query, 'doc_id': doc, 'role': role, 'occurrences': occurrences[(query, doc)][role],
                           'selection_probability': per_role / len(members), 'overlap': rank < overlap})
    counts['both_roles_excluded'] = sum(bool(n['dropped']) and bool(n['replacement']) for n in occurrences.values())
    return chosen, counts


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f7f6f3; --card:#ffffff; --ink:#14130f; --muted:#5d5b55; --line:#dedcd6; --accent:#1f5fbf; --done:#1a7f55; }
@media (prefers-color-scheme: dark) { :root { --bg:#141412; --card:#1e1d1a; --ink:#f2f0ea; --muted:#a8a59c; --line:#34322d; --accent:#7fb0f5; --done:#5fd0a0; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 system-ui, sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 20px 16px 64px; }
h1 { font-size: 20px; margin: 0 0 4px; } .muted { color: var(--muted); font-size: 14px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; margin-top: 14px; }
.query { font-size: 20px; font-weight: 650; margin: 2px 0 0; }
.doc { white-space: pre-wrap; margin: 6px 0 0; }
.label { text-transform: uppercase; letter-spacing: .06em; font-size: 12px; color: var(--muted); }
.grades { display: grid; gap: 8px; margin-top: 10px; }
.grades button { text-align: left; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--line); background: transparent;
  color: var(--ink); font: inherit; cursor: pointer; }
.grades button b { display: inline-block; width: 1.6em; }
.grades button.on { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
.row button, .row input { font: inherit; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--ink); }
.row button { cursor: pointer; } .row button.primary { border-color: var(--accent); color: var(--accent); }
textarea { width: 100%; min-height: 54px; margin-top: 10px; font: inherit; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--card); color: var(--ink); }
progress { width: 100%; height: 8px; margin-top: 10px; accent-color: var(--done); }
details { margin-top: 10px; } summary { cursor: pointer; color: var(--accent); }
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="muted">Grade how well each document answers the query. Your grades are saved in this browser as you go.</div>
<details open><summary>How to grade</summary>
<div class="card" style="margin-top:8px">
<div><b>0</b> &nbsp; __S0__</div><div><b>1</b> &nbsp; __S1__</div><div><b>2</b> &nbsp; __S2__</div>
<div class="muted" style="margin-top:8px">Judge the document against the query as written. Do not look anything up, and do not go back to
change grades because of later documents. Keys: <b>0</b>, <b>1</b>, <b>2</b> to grade; <b>&larr;</b> and <b>&rarr;</b> to move.</div>
</div></details>
<div class="row"><label class="muted" for="who">Annotator</label><input id="who" placeholder="your initials"></div>
<progress id="bar" value="0" max="1"></progress><div class="muted" id="count"></div>
<div class="card"><div class="label">Query</div><div class="query" id="query"></div></div>
<div class="card"><div class="label">Document</div><div class="doc" id="doc"></div></div>
<div class="card"><div class="label">Your grade</div>
<div class="grades" id="grades"></div>
<textarea id="note" placeholder="Optional note"></textarea>
<div class="row"><button id="prev">&larr; Previous</button><button id="next">Next &rarr;</button>
<button id="todo">Next ungraded</button><button id="save" class="primary">Download grades</button></div></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const set = JSON.parse(document.getElementById('data').textContent);
const key = 'audit-' + set.id;
const scale = __SCALE__;
let state = {who: '', grades: {}, notes: {}, at: 0};
try { state = Object.assign(state, JSON.parse(localStorage.getItem(key) || '{}')); } catch (e) {}
const el = id => document.getElementById(id);
function store() { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} }
function show() {
  const p = set.pairs[state.at];
  el('query').textContent = p.query; el('doc').textContent = p.text;
  el('note').value = state.notes[p.pair] || ''; el('who').value = state.who || '';
  const done = Object.keys(state.grades).length;
  el('bar').value = done / set.pairs.length;
  el('count').textContent = 'Pair ' + (state.at + 1) + ' of ' + set.pairs.length + ' \\u00b7 ' + done + ' graded';
  const box = el('grades'); box.innerHTML = '';
  [0, 1, 2].forEach(g => {
    const b = document.createElement('button');
    b.innerHTML = '<b>' + g + '</b>'; b.appendChild(document.createTextNode(scale[g]));
    if (state.grades[p.pair] === g) b.className = 'on';
    b.onclick = () => grade(g); box.appendChild(b);
  });
}
function grade(g) { state.grades[set.pairs[state.at].pair] = g; store(); show(); }
function move(step) { state.at = Math.max(0, Math.min(set.pairs.length - 1, state.at + step)); store(); show(); window.scrollTo(0, 0); }
el('prev').onclick = () => move(-1); el('next').onclick = () => move(1);
el('todo').onclick = () => { const i = set.pairs.findIndex(p => !(p.pair in state.grades)); if (i >= 0) { state.at = i; store(); show(); window.scrollTo(0, 0); } };
el('note').oninput = e => { state.notes[set.pairs[state.at].pair] = e.target.value; store(); };
el('who').oninput = e => { state.who = e.target.value; store(); };
el('save').onclick = () => {
  const out = {set: set.id, pairs_sha256: set.pairs_sha256, annotator: state.who, saved: new Date().toISOString(),
               grades: state.grades, notes: state.notes};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'grades-' + set.id + '-' + (state.who || 'anonymous') + '.json'; a.click();
};
document.addEventListener('keydown', e => {
  if (['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) return;
  if (['0', '1', '2'].includes(e.key)) grade(Number(e.key));
  if (e.key === 'ArrowRight') move(1); if (e.key === 'ArrowLeft') move(-1);
});
show();
</script></body></html>
"""


def page(title, set_id, pairs, pairs_sha256):
    data = json.dumps({'id': set_id, 'pairs_sha256': pairs_sha256, 'pairs': pairs}, ensure_ascii=False).replace('</', '<\\/')
    out = PAGE.replace('__TITLE__', html.escape(title)).replace('__SCALE__', json.dumps({str(k): v for k, v in SCALE.items()}))
    for grade, text in SCALE.items():
        out = out.replace(f'__S{grade}__', html.escape(text))
    return out.replace('__DATA__', data)


def write_kit(output, chosen, queries, texts, official, provenance, *, seed=SEED):
    """Write the blinded pages and the key for the chosen pairs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    chosen = list(chosen)
    random.Random(seed).shuffle(chosen)
    pairs, key = [], []
    for number, item in enumerate(chosen, 1):
        pairs.append({'pair': number, 'query': queries[item['query_id']], 'text': texts[item['doc_id']]})
        key.append({'pair': number, **item, 'official_grade': official.get((item['query_id'], item['doc_id']))})
    with (output / 'key.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in key:
            stream.write(json.dumps(row) + '\n')
    digest = hashlib.sha256(json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    (output / 'annotate.html').write_text(page('Relevance audit', 'all', pairs, digest), encoding='utf-8', newline='\n')
    second = [row for row, item in zip(pairs, key) if item['overlap']]
    (output / 'annotate-overlap.html').write_text(page('Relevance audit (second annotator)', 'overlap', second, digest),
                                                  encoding='utf-8', newline='\n')
    contract = {'status': 'prepared', 'purpose': 'blinded human audit of dropped and replacement documents',
                'pairs': len(pairs), 'overlap_pairs': len(second), 'pairs_sha256': digest, 'seed': seed,
                'scale': {str(k): v for k, v in SCALE.items()}, **provenance, 'source_sha256': checksum(__file__),
                'files': {name: checksum(output / name) for name in ('key.jsonl', 'annotate.html', 'annotate-overlap.html')}}
    atomic_json(output / 'audit.json', contract)
    return contract


def prepare(inputs, release, evidence, output, *, per_role=PER_ROLE, overlap=OVERLAP_PER_ROLE, seed=SEED):
    """Draw the audit pairs from the verified held-out runs and write the kit."""
    from .heldout_analysis import PRIMARY_ARMS, load_test_qrels, panel
    from .heldout_pool import FAMILY, load
    from .judge import document_texts, official_pairs
    release = Path(release)
    ids, exact, primary, _, loaded = load(inputs, evidence, None)
    qrels, identity = load_test_qrels(release / 'files')
    population = panel(qrels[FAMILY], ids)
    occurrences = roles(primary, exact, PRIMARY_ARMS, population)
    chosen, counts = select(occurrences, per_role=per_role, overlap=overlap, seed=seed)
    official = {pair: grade for pair, (grade, _) in official_pairs(qrels[FAMILY]).items()}
    queries = read_queries(release / 'files' / TEST_FILE)
    texts = document_texts(release / 'documents', {item['doc_id'] for item in chosen})
    return write_kit(output, chosen, queries, texts, official, {
        'arms': sorted(PRIMARY_ARMS), 'panel_queries': len(population), 'depth': DEPTH, 'per_role': per_role,
        'eligible_pairs': counts, 'inputs': loaded,
        'judgment_sha256': {f'{FAMILY}/snapshot-{s}': identity[f'{FAMILY}/snapshot-{s}'] for s in (1, 2, 3)}}, seed=seed)


def _kappa(a, b, size=3):
    from .heldout_pool import weighted_kappa
    confusion = np.zeros((size, size), dtype=np.int64)
    np.add.at(confusion, (np.asarray(a), np.asarray(b)), 1)
    value = weighted_kappa(confusion)
    return (value if np.isfinite(value) else None), confusion.tolist()


def analyze(kit, grades, *, second=None, draws=10_000, seed=SEED):
    """Human grades by role, their difference with a query-clustered bootstrap, the agreement with the
    official grades where one exists, and the agreement between two annotators on the overlap."""
    kit = Path(kit)
    contract = json.loads((kit / 'audit.json').read_text())
    key = {row['pair']: row for row in map(json.loads, (kit / 'key.jsonl').read_text().splitlines())}

    def read(path):
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        if record.get('pairs_sha256') != contract['pairs_sha256']:
            raise ValueError('these grades belong to another set of pairs')
        return {int(pair): int(grade) for pair, grade in record['grades'].items()}, record.get('annotator')

    first, who = read(grades)
    graded = sorted(pair for pair in first if pair in key)
    if any(first[pair] not in (0, 1, 2) for pair in graded):
        raise ValueError('grades must be 0, 1 or 2')
    by_role = {role: [pair for pair in graded if key[pair]['role'] == role] for role in ROLES}
    out = {'status': 'completed', 'kind': 'EXPLORATORY human audit', 'annotator': who, 'pairs': len(key), 'graded': len(graded),
           'complete': len(graded) == len(key), 'roles': {}}
    for role, pairs in by_role.items():
        values = np.array([first[pair] for pair in pairs])
        out['roles'][role] = {'graded': len(pairs), 'mean_grade': float(values.mean()) if len(values) else None,
                              'relevant_share': float((values > 0).mean()) if len(values) else None,
                              'grade_counts': {str(g): int((values == g).sum()) for g in (0, 1, 2)},
                              'officially_graded': sum(key[pair]['official_grade'] is not None for pair in pairs)}
    if all(by_role.values()):
        clusters = {}
        for pair in graded:
            clusters.setdefault(key[pair]['query_id'], []).append(pair)
        groups = list(clusters.values())
        rng = np.random.default_rng(seed)

        def difference(pairs):
            a = [first[p] for p in pairs if key[p]['role'] == 'dropped']
            b = [first[p] for p in pairs if key[p]['role'] == 'replacement']
            return np.mean(a) - np.mean(b) if a and b else np.nan

        boot = np.array([difference([p for j in rng.integers(0, len(groups), len(groups)) for p in groups[j]])
                         for _ in range(draws)])
        boot = boot[np.isfinite(boot)]
        out['dropped_minus_replacement'] = {'mean_grade_difference': float(difference(graded)),
                                            'interval_95': np.quantile(boot, [.025, .975]).tolist(),
                                            'bootstrap': {'unit': 'query', 'draws': draws, 'seed': seed}}
    both = [pair for pair in graded if key[pair]['official_grade'] is not None and key[pair]['official_grade'] <= 2]
    if both:
        kappa, confusion = _kappa([first[p] for p in both], [key[p]['official_grade'] for p in both])
        out['against_official_grades'] = {
            'pairs': len(both), 'binary_agreement': float(np.mean([(first[p] > 0) == (key[p]['official_grade'] > 0) for p in both])),
            'linear_weighted_kappa': kappa, 'confusion_human_rows_official_columns': confusion}
    if second is not None:
        other, other_who = read(second)
        shared = sorted(pair for pair in other if pair in first and key.get(pair, {}).get('overlap'))
        if shared:
            kappa, confusion = _kappa([first[p] for p in shared], [other[p] for p in shared])
            out['between_annotators'] = {'second_annotator': other_who, 'pairs': len(shared),
                                         'exact_agreement': float(np.mean([first[p] == other[p] for p in shared])),
                                         'linear_weighted_kappa': kappa, 'confusion_first_rows_second_columns': confusion}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='draw the pairs and write the annotation pages and the key')
    for name in ('inputs', 'release', 'evidence', 'output'):
        p.add_argument(name, type=Path)
    p = sub.add_parser('analyze', help='summarise downloaded grades against the key')
    p.add_argument('kit', type=Path)
    p.add_argument('grades', type=Path)
    p.add_argument('--second', type=Path, help="the second annotator's grades for the overlap")
    p.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.command == 'prepare':
        r = prepare(a.inputs, a.release, a.evidence, a.output)
        r = {key: r[key] for key in ('pairs', 'overlap_pairs', 'eligible_pairs', 'panel_queries')}
    else:
        r = analyze(a.kit, a.grades, second=a.second)
        if a.output:
            atomic_json(a.output, r)
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
