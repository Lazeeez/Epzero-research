"""E4: the held-out design on the BGE vectors (amendment 4 of EXTENSION_AMENDMENTS.md).

    python -m research.longeval.bge_heldout prepare BGE_VECTORS QWEN_DOCNOS TRAINING_QUERIES SAMPLE OUTPUT
    python -m research.longeval.bge_heldout hnsw PREPARED OUTPUT CONFIG --backend epzero --mode incremental --seed 11 \
        --package-root PACKAGE --native-origin checkout-build
    python -m research.longeval.bge_heldout ivf PREPARED OUTPUT CONFIG --mode frozen_incremental --seed 11 ...
    python -m research.longeval.bge_heldout analyze PREPARED RELEASE EVIDENCE OUTPUT --hnsw-config C --ivf-config C

The frozen loaders accept only the Qwen inputs, so this module supplies the inputs and leaves the work to
the frozen code, unchanged:

- `prepare` writes snapshots 1-3 in the frozen reader's layout (`mapped.MappedSnapshot`), each with the
  verified BGE vectors of the released snapshot's documents in its row order (QWEN_DOCNOS holds each
  released snapshot's `docnos.npids` and `pt_meta.json`); the query cohorts (calibration: the frozen
  contract's role in TRAINING_QUERIES; evaluation: the test queries in SAMPLE's order); and the exact
  references by the frozen `oracle.exact_mapped`. SAMPLE is a sample folder written by `bge_sample` (the
  same query order, with BGE exact neighbours): its snapshot-3 neighbours, scored again here, must agree
  with the references. Never pass a Qwen sample with its own `exact-top100.npy`.
- `hnsw` and `ivf` run the frozen `trajectory.trajectory_core` and `ivf_trajectory.ivf_core`.
- `analyze` runs the frozen `heldout_analysis.collect` and `analyze_core` with the frozen labels.

Everything written is derived from LongEval data and is never committed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from .bge_sample import PINNED, sha256
from .heldout import DEPTH, ROLE

IDENTITY = 'BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
SNAPSHOTS = (1, 2, 3)


def _snapshot_ids(folder):
    from .dense import inspect_docnos
    meta = json.loads((Path(folder) / 'pt_meta.json').read_text())
    membership, ids = inspect_docnos(Path(folder) / 'docnos.npids', meta['doc_count'])
    if membership['duplicate_rows']:
        raise ValueError('duplicate document identifiers')
    return meta, list(ids)


def write_snapshots(bge, qwen_docnos, output, *, pinned=PINNED):
    """Snapshots 1-3 as the frozen reader expects them, holding BGE vectors; returns their specs."""
    from .mapped import MappedSnapshot
    bge, qwen_docnos, output = Path(bge), Path(qwen_docnos), Path(output)
    if pinned is not None and sha256(bge / 'documents-snapshot-3.npy') != pinned['documents-snapshot-3.npy']:
        raise ValueError('documents-snapshot-3.npy is not the verified BGE file')
    documents = np.load(bge / 'documents-snapshot-3.npy', mmap_mode='r')
    position = {doc: row for row, doc in enumerate(json.loads((bge / 'documents-snapshot-3-ids.json').read_text()))}
    source = pinned['documents-snapshot-3.npy'] if pinned is not None else sha256(bge / 'documents-snapshot-3.npy')
    specs = []
    for step in SNAPSHOTS:
        meta, ids = _snapshot_ids(qwen_docnos / f'snapshot-{step}')
        missing = [doc for doc in ids if doc not in position]
        if missing:
            raise ValueError(f'snapshot {step}: {len(missing)} documents have no BGE vector')
        rows = np.fromiter((position[doc] for doc in ids), dtype=np.int64, count=len(ids))
        folder = output / f'snapshot-{step}'
        folder.mkdir(parents=True, exist_ok=False)
        with open(folder / 'vecs.f4', 'wb') as handle:
            for start in range(0, len(rows), 50_000):
                np.ascontiguousarray(documents[rows[start:start + 50_000]], dtype='<f4').tofile(handle)
        shutil.copyfile(qwen_docnos / f'snapshot-{step}' / 'docnos.npids', folder / 'docnos.npids')
        (folder / 'pt_meta.json').write_text(json.dumps({**meta, 'vec_size': int(documents.shape[1])}))
        members = {name: {'bytes': (folder / name).stat().st_size, 'sha256': sha256(folder / name)}
                   for name in ('pt_meta.json', 'docnos.npids', 'vecs.f4')}
        archive = hashlib.sha256(f'{IDENTITY}:{source}:snapshot-{step}'.encode()).hexdigest()
        receipt = {'status': 'verified', 'archive_sha256': archive, 'members': members, 'encoder': IDENTITY,
                   'source_documents_sha256': source, 'note': 'BGE vectors of the released snapshot documents (amendment 4)'}
        (folder / 'ingestion.json').write_text(json.dumps(receipt, indent=2) + '\n')
        with MappedSnapshot(folder, expected_archive_sha256=archive) as mapped:   # the frozen reader accepts it
            if list(mapped.ids) != ids or mapped.vectors.shape != (len(ids), documents.shape[1]):
                raise ValueError(f'snapshot {step} does not read back as written')
        specs.append({'snapshot': step, 'folder': str(folder), 'archive_sha256': archive,
                      'vector_sha256': members['vecs.f4']['sha256']})
    return specs


def _select(ids, vectors, wanted, what):
    position = {q: row for row, q in enumerate(ids)}
    if len(position) != len(ids) or any(q not in position for q in wanted):
        raise ValueError(f'BGE {what} queries do not hold the study\'s {what} cohort')
    return np.ascontiguousarray(vectors[[position[q] for q in wanted]], dtype=np.float32)


def cohorts(bge, training_queries, sample):
    """The calibration and evaluation cohorts, with the study's IDs in its order and BGE vectors."""
    from .query_diagnostic import validate_inputs
    bge = Path(bge)
    contract, _ = validate_inputs(Path(training_queries))
    calibration = list(contract['roles']['calibration'])
    evaluation = json.loads((Path(sample) / 'query-ids.json').read_text())
    if set(calibration) & set(evaluation):
        raise ValueError('calibration and evaluation query IDs overlap')
    load = lambda name: (json.loads((bge / f'queries-{name}-ids.json').read_text()), np.load(bge / f'queries-{name}.npy'))
    return {'calibration': SimpleNamespace(ids=calibration, vectors=_select(*load('training'), calibration, 'calibration')),
            ROLE: SimpleNamespace(ids=evaluation, vectors=_select(*load('test'), evaluation, 'evaluation'))}


def references(specs, queries):
    """Exact top-100 per snapshot for both cohorts, by the frozen exhaustive search."""
    from .mapped import MappedSnapshot
    from .oracle import exact_mapped
    ids = list(queries['calibration'].ids) + list(queries[ROLE].ids)
    vectors = np.ascontiguousarray(np.vstack([queries['calibration'].vectors, queries[ROLE].vectors]), dtype='<f4')
    refs = []
    for spec in specs:
        with MappedSnapshot(spec['folder'], expected_archive_sha256=spec['archive_sha256']) as mapped:
            refs.append(exact_mapped(mapped.ids, mapped.vectors, ids, vectors, k=DEPTH, metric='ip'))
    return refs


def sample_agreement(refs, queries, sample, documents):
    """Snapshot-3 references against bge_sample's exact neighbours (a second code path), under the study's rule.
    The sample's neighbours are scored here, in float64, from the same BGE vectors."""
    from .heldout import reference_agreement
    sample = Path(sample)
    with gzip.open(sample / 'doc-ids.txt.gz', 'rt', encoding='utf-8') as handle:
        docs = handle.read().split('\n')[:-1]
    top = np.load(sample / 'exact-top100.npy')
    order = json.loads((sample / 'query-ids.json').read_text())
    if order != list(queries[ROLE].ids):
        raise ValueError('the BGE sample holds another query order')
    q64 = queries[ROLE].vectors.astype(np.float64)
    theirs = {}
    for row, q in enumerate(order):
        scores = np.asarray(documents[top[row]], dtype=np.float64) @ q64[row]
        theirs[q] = [(docs[i], float(s)) for i, s in zip(top[row], scores)]
    mine = {q: refs[2][q] for q in queries[ROLE].ids}
    same_ids = all([doc for doc, _ in theirs[q]] == [doc for doc, _ in mine[q]] for q in order)
    return {**reference_agreement(mine, theirs), 'identical_id_order': same_ids}


def _dump(refs, path):
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(refs, handle)


def _load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return {q: [(doc, float(score)) for doc, score in pairs] for q, pairs in json.load(handle).items()}


def prepare(bge, qwen_docnos, training_queries, sample, output, *, pinned=PINNED):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specs = write_snapshots(bge, qwen_docnos, output, pinned=pinned)
    queries = cohorts(bge, training_queries, sample)
    refs = references(specs, queries)
    for step, ref in zip(SNAPSHOTS, refs):
        _dump(ref, output / f'references-snapshot-{step}.json.gz')
    for role, cohort in queries.items():
        np.save(output / f'{role}-queries.npy', cohort.vectors)
    check = None
    if (Path(sample) / 'exact-top100.npy').exists():
        check = sample_agreement(refs, queries, sample, np.load(Path(bge) / 'documents-snapshot-3.npy', mmap_mode='r'))
    evidence = {'encoder': IDENTITY, 'specs': specs,
                'calibration_ids': queries['calibration'].ids, 'evaluation_ids': queries[ROLE].ids,
                'calibration_count': len(queries['calibration'].ids), 'evaluation_count': len(queries[ROLE].ids),
                'references_sha256': {str(s): sha256(output / f'references-snapshot-{s}.json.gz') for s in SNAPSHOTS},
                'bge_files_sha256': ({n: h for n, h in pinned.items()} if pinned is not None else None),
                'sample_agreement': check}
    (output / 'prepared.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if check is not None and not check['passed']:
        raise ValueError('snapshot-3 references differ from the BGE sample beyond rounding near ties')
    return evidence


def load_prepared(prepared):
    """Specs, queries, references and the inputs identity recorded with every run."""
    prepared = Path(prepared)
    evidence = json.loads((prepared / 'prepared.json').read_text())
    for step in SNAPSHOTS:
        if sha256(prepared / f'references-snapshot-{step}.json.gz') != evidence['references_sha256'][str(step)]:
            raise ValueError('prepared references changed')
    queries = {'calibration': SimpleNamespace(ids=evidence['calibration_ids'], vectors=np.load(prepared / 'calibration-queries.npy')),
               ROLE: SimpleNamespace(ids=evidence['evaluation_ids'], vectors=np.load(prepared / f'{ROLE}-queries.npy'))}
    refs = [_load(prepared / f'references-snapshot-{step}.json.gz') for step in SNAPSHOTS]
    identity = {'bge_encoder': IDENTITY, 'bge_prepared_sha256': sha256(prepared / 'prepared.json'),
                'bge_references_sha256': json.dumps(evidence['references_sha256'], sort_keys=True),
                'calibration_count': evidence['calibration_count'], 'evaluation_count': evidence['evaluation_count']}
    return evidence['specs'], queries, refs, identity


def run_hnsw(prepared, output, config_path, *, kind, mode, seed, package_root, native_origin):
    from .heldout import _require_single_thread, heldout_provenance
    from .release import checksum
    from .trajectory import trajectory_core, validate_settings
    config = json.loads(Path(config_path).read_text())
    validate_settings(config, kind, mode, seed)
    if mode not in ('incremental', 'replay') or config['k'] != DEPTH:
        raise ValueError('held-out HNSW arms are incremental and identical-history replay at depth 100')
    _require_single_thread()
    specs, queries, refs, identity = load_prepared(prepared)
    record = heldout_provenance(package_root, native_origin)
    record['heldout_inputs'] = {**identity, 'config_sha256': checksum(config_path)}
    return trajectory_core(specs, queries, refs, output, config, kind=kind, mode=mode, seed=seed,
                           provenance_record=record, evaluation_role=ROLE)


def run_ivf(prepared, output, config_path, *, mode, seed, package_root, native_origin):
    from .heldout import _require_single_thread, heldout_provenance
    from .ivf_trajectory import ivf_core, validate_settings
    from .release import checksum
    config = json.loads(Path(config_path).read_text())
    validate_settings(config, mode, seed)
    if config['k'] != DEPTH:
        raise ValueError('held-out IVF arm requires depth 100')
    _require_single_thread()
    specs, queries, refs, identity = load_prepared(prepared)
    record = heldout_provenance(package_root, native_origin)
    record['heldout_inputs'] = {**identity, 'config_sha256': checksum(config_path)}
    return ivf_core(specs, queries, refs, output, config, mode=mode, seed=seed,
                    provenance_record=record, evaluation_role=ROLE)


def analyze(prepared, release, evidence, output, *, hnsw_config, ivf_config):
    """As heldout_analysis.analyze_heldout, with the BGE cohorts and references."""
    from . import heldout_analysis as A
    from .release import checksum
    _, queries, refs, identity = load_prepared(prepared)
    query_ids = queries[ROLE].ids
    exact = {step: {q: refs[step - 1][q] for q in query_ids} for step in A.SNAPSHOTS}
    reports, runs, excluded = A.collect(sorted(Path(evidence).rglob('report.json')), ROLE, query_ids, exact,
                                        config_sha256=(checksum(hnsw_config), checksum(ivf_config)),
                                        inputs_sha256={k: identity[k] for k in ('bge_prepared_sha256', 'bge_references_sha256')})
    binaries = {r['provenance']['epzero_binary_sha256'] for r in reports.values()}
    qrels, judgments = A.load_test_qrels(release)
    result, rows = A.analyze_core(runs, exact, qrels, query_ids, primary_family=A.PRIMARY_FAMILY, families=A.FAMILIES,
                                  equality=A.equality_checks(reports, runs),
                                  context={'kind': 'held-out evaluation under FINAL_PROTOCOL_V1 on the BGE vectors (E4, amendment 4)',
                                           'query_count': len(query_ids)})
    extra = {'judgment_sha256': judgments, 'inputs': identity, 'epzero_binaries': sorted(binaries),
             'conditions': len(reports), 'runs': len(runs), 'excluded_attempts': excluded}
    return A.write_outputs(Path(output), result, rows, extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('bge', 'qwen_docnos', 'training_queries', 'sample', 'output'):
        p.add_argument(name, type=Path)
    for arm in ('hnsw', 'ivf'):
        p = sub.add_parser(arm)
        for name in ('prepared', 'output', 'config'):
            p.add_argument(name, type=Path)
        if arm == 'hnsw':
            p.add_argument('--backend', choices=('epzero', 'faiss_hnsw'), required=True)
            p.add_argument('--mode', choices=('incremental', 'replay'), required=True)
        else:
            p.add_argument('--mode', choices=('frozen_incremental', 'frozen_rebuild', 'refreshed_rebuild'), required=True)
        p.add_argument('--seed', type=int, required=True)
        p.add_argument('--package-root', type=Path, required=True)
        p.add_argument('--native-origin', required=True)
    p = sub.add_parser('analyze')
    for name in ('prepared', 'release', 'evidence', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--hnsw-config', type=Path, required=True)
    p.add_argument('--ivf-config', type=Path, required=True)
    a = parser.parse_args()
    if a.command == 'prepare':
        r = prepare(a.bge, a.qwen_docnos, a.training_queries, a.sample, a.output)
        r = {k: r[k] for k in ('calibration_count', 'evaluation_count', 'references_sha256', 'sample_agreement')}
    elif a.command == 'hnsw':
        r = run_hnsw(a.prepared, a.output, a.config, kind=a.backend, mode=a.mode, seed=a.seed,
                     package_root=a.package_root, native_origin=a.native_origin)
    elif a.command == 'ivf':
        r = run_ivf(a.prepared, a.output, a.config, mode=a.mode, seed=a.seed,
                    package_root=a.package_root, native_origin=a.native_origin)
    else:
        r = analyze(a.prepared, a.release, a.evidence, a.output, hnsw_config=a.hnsw_config, ivf_config=a.ivf_config)
    print(json.dumps(r if isinstance(r, dict) else {'result': str(r)}, indent=2, default=str)[:4000])


if __name__ == '__main__':
    main()
