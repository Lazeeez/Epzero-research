"""Small explicit native IP adapters for method-screen competitor validation.

Separate from the sealed trajectory adapters. No quantization, automatic tuning,
hidden normalization or backend substitution. Returned scores are inner products.
"""
import time

import numpy as np


class StandardGraph:
    def __init__(self, kind, vectors, labels, *, seed=11, M=16, ef_construction=100):
        values = np.asarray(vectors)
        labels = np.asarray(labels)
        if (values.ndim != 2 or values.dtype != np.float32 or not values.flags.c_contiguous
                or not np.isfinite(values).all() or labels.shape != (len(values),)
                or labels.dtype != np.int64 or np.any(labels < 0)
                or len(set(map(int, labels))) != len(labels) or len(values) < 1
                or type(seed) is not int or not 0 <= seed < 2**31
                or type(M) is not int or M < 2 or ef_construction < M):
            raise ValueError('invalid explicit float32 graph inputs')
        self.kind, self.labels, self.dimension = kind, labels.copy(), values.shape[1]
        self.live = set(map(int, labels))
        started = time.perf_counter_ns()
        if kind == 'epzero':
            import epzero
            self.index = epzero.Index(dim=self.dimension, metric='ip', max_elements=len(values),
                M=M, ef_construction=ef_construction, seed=seed, quantization='none',
                keep_vectors=True, nav_dim=0, fill_links=False, fast_build=False, half_vectors=False)
            self.index.add(values, labels=labels, threads=1)
        elif kind == 'faiss_hnsw':
            import faiss
            faiss.omp_set_num_threads(1)
            self.index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.rng = faiss.RandomGenerator(seed)
            self.index.add(values)
        elif kind == 'hnswlib':
            import hnswlib
            self.index = hnswlib.Index(space='ip', dim=self.dimension)
            self.index.init_index(max_elements=len(values), ef_construction=ef_construction,
                                  M=M, random_seed=seed)
            self.index.set_num_threads(1)
            self.index.add_items(values, labels, num_threads=1)
        else:
            raise ValueError('unsupported native competitor')
        self.build_nanoseconds = time.perf_counter_ns() - started

    def native_call(self, k, ef):
        if type(k) is not int or type(ef) is not int or not 0 < k <= len(self.labels) or ef < k:
            raise ValueError('invalid k or ef')
        if self.kind == 'epzero':
            return self.index.search, dict(k=k, ef=ef, rerank=0, rerank_z=0, patience=0, threads=1)
        if self.kind == 'hnswlib':
            self.index.set_ef(ef)
            return self.index.knn_query, dict(k=k, num_threads=1)
        self.index.hnsw.efSearch = ef
        return self.index.search, dict(k=k)

    def decode(self, raw, k):
        if self.kind == 'faiss_hnsw':
            scores, internal = raw
            if np.any(internal < 0) or np.any(internal >= len(self.labels)):
                raise ValueError('invalid internal labels or padding')
            labels = self.labels[internal]
        else:
            labels, distances = raw
            scores = np.float32(1) - distances
        if (labels.shape != (1, k) or scores.shape != (1, k) or not np.isfinite(scores).all()
                or len(set(map(int, labels[0]))) != k or not set(map(int, labels[0])) <= self.live):
            raise ValueError('invalid native ranking')
        return list(zip(map(int, labels[0]), map(float, scores[0])))

    def measure(self, queries, *, k, ef):
        values = np.asarray(queries)
        if (values.ndim != 2 or values.shape[1] != self.dimension or values.dtype != np.float32
                or not np.isfinite(values).all()):
            raise ValueError('invalid query matrix')
        call, kwargs = self.native_call(k, ef)
        rows = []
        for vector in values:
            query = np.ascontiguousarray(vector.reshape(1, -1))
            times, first = [], None
            for repeat in range(4):  # One immediately preceding warmup, three timed calls.
                started = time.perf_counter_ns()
                raw = call(query, **kwargs)
                elapsed = time.perf_counter_ns() - started
                ranking = self.decode(raw, k)  # Outside timed native/binding call.
                if first is None:
                    first = ranking
                elif first != ranking:
                    raise ValueError('native repeat changed ranking')
                if repeat:
                    times.append(elapsed)
            rows.append({'results': first, 'native_call_nanoseconds': times})
        return rows
