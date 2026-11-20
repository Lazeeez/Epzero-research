"""Three-seed native competitor harness check; small-sample, local timing only."""
from collections import defaultdict
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from .query_diagnostic import load_queries, validate_inputs
from .release import checksum
from .standard_graphs import StandardGraph
from .storage import atomic_json


def main():
    import epzero
    import epzero._core
    import faiss
    import hnswlib
    root = Path.cwd()
    base = root / 'build/longeval-frontier-pilot-01'
    output = root / 'build/longeval-standard-graphs-01'
    original = json.loads((base / 'report.json').read_text())
    for name in ('state-before.index', 'sample-rows.npy', 'exact.json'):
        if checksum(base / name) != original['artifacts'][name]:
            raise ValueError('sample artifact identity mismatch')
    labels = np.load(base / 'sample-rows.npy', allow_pickle=False)
    original_index = epzero.Index.load(str(base / 'state-before.index'))
    data = np.ascontiguousarray([original_index.get_vector(int(label)) for label in labels], dtype=np.float32)
    del original_index
    if hashlib.sha256(data.tobytes()).hexdigest() != original['sample_vectors_sha256']:
        raise ValueError('sample vector identity mismatch')
    inputs = root / 'build/longeval-training-query-inputs-01'
    encoding = root / 'build/longeval-pace-query-evidence-7107422/7107422/float32'
    ids, queries, _ = load_queries(inputs, encoding)
    contract, _ = validate_inputs(inputs)
    exact = json.loads((base / 'exact.json').read_text())
    similarities = queries.astype(np.float64) @ data.astype(np.float64).T
    positions = {int(v): i for i, v in enumerate(labels)}
    for qi, qid in enumerate(ids):
        if exact[qid] != list(map(int, labels[np.lexsort((labels, -similarities[qi]))[:100]])):
            raise ValueError('independent exact reference mismatch')
    affinity_before = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity_before[0]})
    kinds, grid = ('epzero', 'faiss_hnsw', 'hnswlib'), (100, 200, 400, 800)
    faiss_native = importlib.import_module('faiss._' + faiss.IndexHNSWFlat.__module__.split('.')[-1])
    output.mkdir(exist_ok=False)
    report = {'status': 'running', 'purpose': __doc__,
        'sources': {p.name: checksum(p) for p in (Path(__file__), Path(__file__).with_name('standard_graphs.py'))},
        'sample_report_sha256': checksum(base / 'report.json'),
        'query_encoding_sha256': checksum(encoding / 'query-encoding.json'),
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'faiss-cpu', 'hnswlib')},
        'binaries': {name: {'path': module.__file__, 'sha256': checksum(module.__file__)}
                     for name, module in [('epzero', epzero._core), ('faiss', faiss_native), ('hnswlib', hnswlib)]},
        'platform': platform.platform(), 'cpuinfo': Path('/proc/cpuinfo').read_text().split('\n\n')[0],
        'affinity_before': affinity_before, 'affinity_used': sorted(os.sched_getaffinity(0)),
        'config': {'seeds': [11, 29, 47], 'M': 16, 'ef_construction': 100, 'k': 100,
                   'ef_grid': grid, 'targets': [.95, .99], 'threads': 1,
                   'warmups': 1, 'timed_repeats': 3, 'query_batch': 1},
        'held_out_retrieval_evaluated': False,
        'limits': ['8,192-vector subset, local WSL CPU, not exclusive PACE hardware',
                   'Binding calls timed; serialization and validation excluded',
                   'Fixed construction parameters; not memory/build-budget Pareto tuning',
                   'Each library has its own graph and seed semantics',
                   'hnswlib0.8.0 is a pinned baseline, not a claim of latest-version coverage',
                   'Cached local hnswlib wheel; compiler options are not matched between libraries',
                   'No repaired graph native implementation is included']}
    atomic_json(output / 'report.json', report)
    groups, build_records, selected = defaultdict(list), [], []
    def state(backend):
        backend.native_call(100, 100)  # Normalize mutable search settings before serialization.
        with tempfile.TemporaryDirectory(dir=output) as temp:
            path = Path(temp) / 'index.bin'
            if backend.kind == 'epzero':
                backend.index.save(str(path))
            elif backend.kind == 'hnswlib':
                backend.index.save_index(str(path))
            else:
                faiss.write_index(backend.index, str(path))
            return {'sha256': checksum(path), 'bytes': path.stat().st_size}
    with (output / 'queries.jsonl').open('x') as stream:
        for si, seed in enumerate((11, 29, 47)):
            build_order = kinds[si:] + kinds[:si]
            backends = {}
            before = {}
            for kind in build_order:
                print(json.dumps({'stage': 'build', 'seed': seed, 'backend': kind}), flush=True)
                backend = StandardGraph(kind, data, labels, seed=seed)
                backends[kind] = backend
                before[kind] = state(backend)
                build_records.append({'seed': seed, 'backend': kind,
                    'build_nanoseconds': backend.build_nanoseconds, 'serialized_state': before[kind]})
            for role in ('calibration', 'development'):
                for qi, qid in enumerate(contract['roles'][role]):
                    query_at = ids.index(qid)
                    for ei, ef in enumerate(grid):
                        rotate = (si + qi + ei) % len(kinds)
                        order = kinds[rotate:] + kinds[:rotate]
                        for kind in order:
                            row = backends[kind].measure(queries[query_at:query_at + 1], k=100, ef=ef)[0]
                            got = [v for v, _ in row['results']]
                            expected_scores = [similarities[query_at, positions[v]] for v in got]
                            if not np.allclose(expected_scores, [s for _, s in row['results']], atol=2e-6, rtol=0):
                                raise ValueError('native score mismatch')
                            record = {'seed': seed, 'backend': kind, 'role': role, 'query_id': qid, 'ef': ef,
                                'recall': len(set(got) & set(exact[qid])) / 100, **row}
                            groups[(seed, kind, role, ef)].append(record)
                            stream.write(json.dumps(record, allow_nan=False) + '\n')
                if role == 'calibration':
                    for kind in kinds:
                        for target in (.95, .99):
                            feasible = [ef for ef in grid if np.mean([r['recall'] for r in groups[(seed, kind, role, ef)]]) >= target]
                            chosen = min(feasible) if feasible else None
                            selected.append({'seed': seed, 'backend': kind, 'target': target,
                                             'selected_ef': chosen, 'calibration_target_met': bool(feasible)})
                    atomic_json(output / f'calibration-seed-{seed}.json', selected[-6:])
                print(json.dumps({'stage': 'measured', 'seed': seed, 'role': role}), flush=True)
            for kind, backend in backends.items():
                if state(backend) != before[kind]:
                    raise ValueError('native search changed serialized graph')
            del backends, backend
    summaries = []
    for (seed, kind, role, ef), rows in sorted(groups.items()):
        medians = np.array([np.median(r['native_call_nanoseconds']) / 1e6 for r in rows])
        summaries.append({'seed': seed, 'backend': kind, 'role': role, 'ef': ef, 'count': len(rows),
            'mean_recall': float(np.mean([r['recall'] for r in rows])),
            'min_recall': min(r['recall'] for r in rows), 'below_90': sum(r['recall'] < .9 for r in rows),
            'mean_query_median_ms': float(medians.mean()), 'p50_query_median_ms': float(np.quantile(medians, .5)),
            'p95_query_median_ms': float(np.quantile(medians, .95))})
    for selection in selected:
        selection['development'] = next((s for s in summaries if
            (s['seed'], s['backend'], s['role'], s['ef']) ==
            (selection['seed'], selection['backend'], 'development', selection['selected_ef'])), None)
    report.update(status='completed', query_conditions=3600, graphs=9,
        graph_states_unchanged=True, build_records=build_records, summaries=summaries,
        selected_operating_points=selected,
        artifacts={p.name: checksum(p) for p in output.iterdir() if p.is_file() and p.name != 'report.json'})
    atomic_json(output / 'report.json', report)
    print(json.dumps({'status': 'completed', 'graphs': 9, 'query_conditions': 3600}), flush=True)


if __name__ == '__main__':
    main()
