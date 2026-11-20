import hashlib
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from research.longeval.backends import Backend, streamed_faiss_hash, streamed_graph_hash
from research.longeval.contracts import digest


def test_streamed_fingerprints_match_original_serialized_bytes():
    index = faiss.IndexHNSWFlat(4, 8)
    index.add(np.eye(4, dtype=np.float32))
    assert streamed_faiss_hash(index) == hashlib.sha256(faiss.serialize_index(index).tobytes()).hexdigest()
    rows = [("01", 2, [[1, 2], [], [0]]), ("文書", 0, [[0]])]
    assert streamed_graph_hash(iter(rows)) == digest(rows)
    assert streamed_graph_hash(iter([])) == digest([])


@pytest.mark.parametrize("kind", ["epzero", "faiss_hnsw", "faiss_ivf"])
def test_insertion_chunk_size_preserves_index_state_and_ranking(kind):
    config = {"M": 8, "ef_construction": 64, "nlist": 2, "ivf_iterations": 10}
    rng = np.random.default_rng(91)
    table = SimpleNamespace(ids=[f"d{i}" for i in range(160)], vectors=rng.normal(size=(160, 8)).astype(np.float32))
    queries = SimpleNamespace(ids=[f"q{i}" for i in range(5)], vectors=rng.normal(size=(5, 8)).astype(np.float32))
    whole = Backend(kind, config, {"dim": 8, "metric": "l2"}, table.ids, 11)
    chunked = Backend(kind, config, {"dim": 8, "metric": "l2"}, table.ids, 11)
    whole.train(table)
    chunked.train(table)
    whole.add(table, table.ids, batch_rows=1000)
    chunked.add(table, table.ids, batch_rows=13)
    assert whole.state() == chunked.state()
    assert whole.search(queries, 10, 2 if kind == "faiss_ivf" else 64) == chunked.search(queries, 10, 2 if kind == "faiss_ivf" else 64)
