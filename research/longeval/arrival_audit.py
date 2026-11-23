"""The arrival-order human audit (analysis D of scripts/pace/LONGEVAL_INSERTION_ORDER_IDS.md).

Exploratory follow-up, not part of FINAL_PROTOCOL_V1. From the as-arrived and reversed full-scale
builds of seed 11 (both implementations pooled) at ef 200, on the primary panel at snapshot 3:

  dropped      a document in the exact top 10 that the build leaves out of its top 10, from the
               newest arrival group (snapshot 3) or the oldest (snapshot 1);
  replacement  for each chosen dropped pair, one document the same build put in its top 10 that is
               not in the exact top 10, for the same query.

`per_group` dropped pairs are drawn per arrival group by a keyed hash, then one replacement each. Pairs
that are both dropped (somewhere) and a replacement (somewhere) are left out, and counted. The kit is
written by the frozen `audit.write_kit` (blinded pages, separate key), so the grader sees neither the
role, nor the arrival group, nor the order. The sampler lives here rather than in the frozen
`audit.py` (amendment recorded in docs/protocol/EXTENSION_AMENDMENTS.md).

    python -m research.longeval.arrival_audit JOBS INPUTS RELEASE OUTPUT

Everything written holds LongEval text or identifiers and stays in the build folder.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 20261123
PER_GROUP = 50
OVERLAP_PER_GROUP = 13      # dropped pairs per group shown to a second annotator, with their replacements: 52 pairs
DEPTH = 10
ORDERS = ('appended', 'reversed')


def build_roles(builds, exact, cohort, doc_ids, query_ids, population):
    """Dropped pairs by arrival group, and the replacements each build offers, over the given builds.

    `builds` maps a build name to its returned sample positions [query, >= DEPTH]; `exact` holds the
    exact top-100 sample positions [query, 100]. Returns (dropped, replacements, both): dropped maps
    (query, doc) -> {'group': 1 or 3, 'builds': [names]}; replacements maps (build, query) -> [docs];
    both is the set of pairs seen in both roles."""
    dropped, replacements, offered = {}, {}, set()
    wanted = set(population)
    for name, labels in sorted(builds.items()):
        for q, query in enumerate(query_ids):
            if query not in wanted:
                continue
            reference = [int(x) for x in exact[q, :DEPTH]]
            returned = [int(x) for x in labels[q, :DEPTH]]
            for doc in reference:
                if doc not in returned and cohort[doc] in (1, 3):
                    item = dropped.setdefault((query, doc_ids[doc]), {'group': int(cohort[doc]), 'builds': []})
                    item['builds'].append(name)
            extra = [doc_ids[doc] for doc in returned if doc not in reference]
            replacements[(name, query)] = extra
            offered.update((query, d) for d in extra)
    both = set(dropped) & offered
    return dropped, replacements, both


def _key(seed, *parts):
    return hashlib.sha256(':'.join(['arrival-audit-v1', str(seed), *map(str, parts)]).encode()).digest()


def select(dropped, replacements, both, *, per_group=PER_GROUP, overlap=OVERLAP_PER_GROUP, seed=SEED):
    """`per_group` dropped pairs per arrival group and one replacement for each, by keyed hashes."""
    chosen, counts, used = [], {'both_roles_excluded': len(both)}, set()
    for group in (3, 1):
        members = sorted((_key(seed, query, doc), query, doc) for (query, doc), item in dropped.items()
                         if item['group'] == group and (query, doc) not in both)
        counts[f'dropped_group_{group}'] = len(members)
        if len(members) < per_group:
            raise ValueError(f'only {len(members)} dropped pairs in arrival group {group}')
        taken = 0
        for _, query, doc in members:
            if taken == per_group:
                break
            build = sorted(dropped[(query, doc)]['builds'], key=lambda b: _key(seed, query, doc, b))[0]
            options = sorted((_key(seed, query, d, build), d) for d in replacements[(build, query)]
                             if (query, d) not in both and (query, d) not in used)
            if not options:
                continue
            substitute = options[0][1]
            used.update({(query, doc), (query, substitute)})
            share = per_group / len(members)
            chosen.append({'query_id': query, 'doc_id': doc, 'role': 'dropped', 'arrival_group': group, 'build': build,
                           'occurrences': len(dropped[(query, doc)]['builds']), 'selection_probability': share,
                           'overlap': taken < overlap})
            chosen.append({'query_id': query, 'doc_id': substitute, 'role': 'replacement', 'arrival_group': group,
                           'build': build, 'occurrences': 1, 'selection_probability': share, 'overlap': taken < overlap})
            taken += 1
        if taken < per_group:
            raise ValueError(f'only {taken} dropped pairs in arrival group {group} have a usable replacement')
    return chosen, counts


def load_builds(jobs):
    """Seed-11 as-arrived and reversed builds at ef 200 from the rerun's job folders, with the shared sample files."""
    builds, shared = {}, None
    for job in sorted(Path(jobs).iterdir()):
        for folder in sorted(job.glob('full/*/*-11')):
            order = folder.name.rsplit('-', 1)[0]
            if order in ORDERS:
                builds[f'{folder.parent.name}/{order}-11'] = np.load(folder / 'labels-ef200.npy')
                shared = shared or job
    if shared is None or len(builds) != 2 * len(ORDERS):
        raise ValueError(f'expected the as-arrived and reversed seed-11 builds of both implementations, found {sorted(builds)}')
    with gzip.open(shared / 'doc-ids.txt.gz', 'rt', encoding='utf-8') as handle:
        doc_ids = handle.read().split()
    query_ids = json.loads((shared / 'query-ids.json').read_text())
    return builds, np.load(shared / 'exact-top100.npy'), np.load(shared / 'cohort.npy'), doc_ids, query_ids


def prepare(jobs, inputs, release, output, *, per_group=PER_GROUP, overlap=OVERLAP_PER_GROUP, seed=SEED):
    from .audit import TEST_FILE, write_kit
    from .heldout_analysis import load_test_qrels, panel
    from .heldout_pool import FAMILY
    from .judge import document_texts, official_pairs
    from .release import read_queries
    release = Path(release)
    builds, exact, cohort, doc_ids, query_ids = load_builds(jobs)
    qrels, identity = load_test_qrels(release / 'files')
    population = panel(qrels[FAMILY], query_ids)
    dropped, replacements, both = build_roles(builds, exact, cohort, doc_ids, query_ids, population)
    chosen, counts = select(dropped, replacements, both, per_group=per_group, overlap=overlap, seed=seed)
    official = {pair: grade for pair, (grade, _) in official_pairs(qrels[FAMILY]).items()}
    queries = read_queries(release / 'files' / TEST_FILE)
    texts = document_texts(release / 'documents', {item['doc_id'] for item in chosen})
    return write_kit(output, chosen, queries, texts, official, {
        'design': 'arrival-order audit, analysis D of LONGEVAL_INSERTION_ORDER_IDS.md', 'builds': sorted(builds),
        'ef': 200, 'snapshot': 3, 'depth': DEPTH, 'per_group': per_group, 'panel_queries': len(population),
        'eligible': counts, 'judgment_sha256': {f'{FAMILY}/snapshot-3': identity[f'{FAMILY}/snapshot-3']}}, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('jobs', 'inputs', 'release', 'output'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    print(json.dumps(prepare(a.jobs, a.inputs, a.release, a.output), indent=2))


if __name__ == '__main__':
    main()
