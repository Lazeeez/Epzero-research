"""Independent validation of native HNSW timing cohorts; no native execution."""
import math
import statistics


def verify_cohort(point, run, references, query_ids, backend):
    if (backend not in ('epzero', 'faiss_hnsw') or set(run) != set(query_ids)
            or len(query_ids) != len(set(query_ids))
            or [r['query_id'] for r in point['queries']] != list(query_ids)
            or set(point['ann_recall_per_query']) != set(query_ids)):
        raise ValueError('cohort membership or query order differs')
    rows = []
    for query in point['queries']:
        qid = query['query_id']
        pairs = run[qid]
        if (len(pairs) != 100 or len({d for d, _ in pairs}) != 100
                or any(not math.isfinite(s) for _, s in pairs)
                or pairs != sorted(pairs, key=lambda p: (p[1], p[0]), reverse=True)):
            raise ValueError('invalid ranking')
        exact = {d for d, _ in references[qid]}
        if len(exact) != 100:
            raise ValueError('invalid exact reference')
        recall = len({d for d, _ in pairs} & exact) / 100
        if abs(recall - point['ann_recall_per_query'][qid]) > 1e-12:
            raise ValueError('reported recall differs')
        times = query['native_call_nanoseconds']
        if len(times) != 3 or any(type(t) is not int or t <= 0 for t in times):
            raise ValueError('invalid timing repeats')
        counters = ('epzero_distance_computations', 'epzero_nodes_expanded') if backend == 'epzero' else ('faiss_native_ndis', 'faiss_native_nhops')
        if set(query) != {'query_id', 'native_call_nanoseconds', *counters}:
            raise ValueError('incorrect counter convention')
        if any(type(query[k]) is not int or query[k] <= 0 for k in counters):
            raise ValueError('invalid native counters')
        rows.append({'query_id': qid, 'recall': recall, 'median_nanoseconds': statistics.median(times),
                     'native_call_nanoseconds': times, **{k: query[k] for k in counters}})
    if abs(statistics.mean(r['recall'] for r in rows) - point['ann_recall_mean']) > 1e-12:
        raise ValueError('reported mean recall differs')
    return rows
