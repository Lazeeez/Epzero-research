"""Explicit single-threaded float32 adapters. No automatic tuning or hidden fallback."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .contracts import Table, digest
from .evaluation import rank, validate_run


def streamed_faiss_hash(index):
    """Hash the canonical serialization without allocating a second full index in RAM."""
    import faiss

    digest = hashlib.sha256()
    def write(block):
        digest.update(block)
        return len(block)
    faiss.write_index(index, faiss.PyCallbackIOWriter(write, 1 << 20))
    return digest.hexdigest()


def streamed_graph_hash(rows):
    """Match contracts.digest(list(rows)) while materializing at most one node's neighbors."""
    result = hashlib.sha256(b"[")
    for number, row in enumerate(rows):
        if number:
            result.update(b",")
        result.update(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":")).encode())
    result.update(b"]")
    return result.hexdigest()


class Backend:
    def __init__(self, kind, config, representation, ordered_ids, seed):
        import epzero
        import faiss

        self.kind, self.config = kind, config
        self.ids = tuple(ordered_ids)
        self.labels = {key: i for i, key in enumerate(self.ids)}
        self.live = set()
        self.inserted = []
        self.metric = representation["metric"]
        dim = representation["dim"]
        self.training_ids = ()
        self.seed = seed
        faiss.omp_set_num_threads(1)
        if kind == "epzero":
            self.index = epzero.Index(dim=dim, metric=self.metric, max_elements=len(ordered_ids),
                                      M=config["M"], ef_construction=config["ef_construction"], seed=seed,
                                      quantization="none", keep_vectors=True, nav_dim=0, fill_links=False,
                                      fast_build=False, half_vectors=False)
        elif kind == "faiss_hnsw":
            metric = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
            self.index = faiss.IndexHNSWFlat(dim, config["M"], metric)
            self.index.hnsw.efConstruction = config["ef_construction"]
            self.index.hnsw.rng = faiss.RandomGenerator(seed)
        elif kind == "faiss_ivf":
            metric = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
            quantizer = faiss.IndexFlat(dim, metric)
            self.index = faiss.IndexIVFFlat(quantizer, dim, config["nlist"], metric)
            self.index.cp.seed = seed
            self.index.cp.niter = config["ivf_iterations"]
        else:
            raise ValueError(f"unknown backend: {kind}")

    def train(self, table: Table):
        if self.kind != "faiss_ivf":
            return
        if self.live or self.training_ids:
            raise ValueError("IVF training requires a fresh index")
        # Sort training IDs independently of source-file row order and preserve them as artifacts.
        order = np.argsort(table.ids)
        self.index.train(np.ascontiguousarray(table.vectors[order]))
        self.training_ids = tuple(table.ids[i] for i in order)

    def add(self, table: Table, ids, *, batch_rows=4096):
        if type(batch_rows) is not int or batch_rows <= 0:
            raise ValueError("insertion batch rows must be a positive integer")
        ids = tuple(ids)
        if len(ids) != len(set(ids)) or self.live.intersection(ids):
            raise ValueError("delta-only adapter rejects duplicate/replacement IDs")
        if set(ids) - set(table.ids) or set(ids) - self.labels.keys():
            raise ValueError("unknown insertion ID")
        if not ids:
            return
        positions = {key: i for i, key in enumerate(table.ids)}
        for start in range(0, len(ids), batch_rows):
            chunk = ids[start:start + batch_rows]
            values = np.ascontiguousarray(table.vectors[[positions[key] for key in chunk]])
            labels = np.array([self.labels[key] for key in chunk], dtype=np.int64)
            if self.kind == "epzero":
                self.index.add(values, labels=labels, threads=1)
            elif self.kind == "faiss_ivf":
                self.index.add_with_ids(values, labels)
            else:
                # IndexHNSWFlat uses row positions as labels; map through recorded insertion order.
                # One row per call keeps FAISS's batch shuffling out of the order experiment.
                for row in values:
                    self.index.add(row.reshape(1, -1))
            self.live.update(chunk)
            self.inserted.extend(chunk)

    def search(self, queries, k, effort):
        if effort <= 0:
            raise ValueError("search effort must be positive")
        if self.kind == "epzero":
            labels, distances = self.index.search(queries.vectors, k=k, ef=max(k, effort),
                                                   rerank=0, rerank_z=0, patience=0, threads=1)
            scores = -distances
            if self.metric == "ip":
                scores = scores + 1  # Epzero's IP distance is 1-dot.
        else:
            if self.kind == "faiss_hnsw":
                self.index.hnsw.efSearch = max(k, effort)
            else:
                if effort > self.config["nlist"]:
                    raise ValueError("nprobe exceeds nlist")
                self.index.nprobe = effort
            distances, labels = self.index.search(queries.vectors, k)
            scores = -distances if self.metric == "l2" else distances - (self.metric == "cosine")
        lookup = self.inserted if self.kind == "faiss_hnsw" else self.ids
        run = {}
        for q, row, values in zip(queries.ids, labels, scores):
            if any(int(label) < 0 or int(label) >= len(lookup) for label in row):
                raise ValueError("backend returned padding/invalid labels; search depth or effort is infeasible")
            run[q] = rank([(lookup[int(label)], float(value)) for label, value in zip(row, values)])
        validate_run(run, queries.ids, self.live, k)
        return run

    def state(self):
        result = {"backend": self.kind, "count": len(self.live), "seed": self.seed,
                  "live_set_sha256": digest(sorted(self.live)),
                  "insertion_order_sha256": digest(self.inserted),
                  "training_ids": self.training_ids, "training_ids_sha256": digest(self.training_ids)}
        if self.kind == "epzero":
            graph = ((key, int(self.index.level(self.labels[key])),
                      [list(map(int, self.index.neighbors(self.labels[key], layer)))
                       for layer in range(self.index.level(self.labels[key]) + 1)])
                     for key in sorted(self.live))
            result.update(params=self.index.params, deleted=self.index.deleted_count,
                          topology_sha256=streamed_graph_hash(graph))
            if self.index.deleted_count or len(self.index) != len(self.live):
                raise ValueError("unexpected tombstones or index membership count")
        else:
            if self.index.ntotal != len(self.live):
                raise ValueError("index membership count mismatch")
            result["serialized_index_sha256"] = streamed_faiss_hash(self.index)
        return result
