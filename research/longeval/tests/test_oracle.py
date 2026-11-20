from types import SimpleNamespace

import numpy as np
import pytest

from research.longeval.evaluation import exact
from research.longeval.oracle import exact_mapped


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_memory_mapped_oracle_matches_independent_reference_and_faiss(tmp_path, metric):
    import faiss

    rng = np.random.default_rng(51)
    docs = rng.normal(size=(173, 19)).astype(np.float32)
    queries = rng.normal(size=(9, 19)).astype(np.float32)
    if metric == "cosine":
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    path = tmp_path / "vectors.f4"
    docs.tofile(path)
    vectors = np.memmap(path, mode="r", dtype="<f4", shape=docs.shape)
    ids, qids = [str(i) for i in range(len(docs))], [f"q{i}" for i in range(len(queries))]
    actual = exact_mapped(ids, vectors, qids, queries, k=100, metric=metric, block_rows=31, query_batch=4)
    reference = exact(SimpleNamespace(ids=ids, vectors=docs), SimpleNamespace(ids=qids, vectors=queries), 100, metric, block_size=37)
    index = faiss.IndexFlatL2(19) if metric == "l2" else faiss.IndexFlatIP(19)
    index.add(docs)
    distances, rows = index.search(queries, 100)
    for i, qid in enumerate(qids):
        assert [key for key, score in actual[qid]] == [key for key, score in reference[qid]]
        np.testing.assert_allclose([score for key, score in actual[qid]], [score for key, score in reference[qid]], rtol=1e-13, atol=1e-13)
        assert [key for key, score in actual[qid]] == [ids[j] for j in rows[i]]
        expected = -distances[i] if metric == "l2" else distances[i] - (metric == "cosine")
        np.testing.assert_allclose([score for key, score in actual[qid]], expected, rtol=1e-5, atol=1e-5)
    vectors._mmap.close()


def test_boundary_ties_across_blocks_use_ids_not_partition_order():
    ids = ["002", "2", "10", "z", "a", "x"]
    vectors = np.ones((len(ids), 2), dtype=np.float32)
    for block in (1, 2, 4, 100):
        result = exact_mapped(ids, vectors, ["q"], np.ones((1, 2), dtype=np.float32), k=3, block_rows=block)
        assert result == {"q": [("z", 2.0), ("x", 2.0), ("a", 2.0)]}


def test_l2_does_not_lose_small_distances_by_norm_expansion():
    vectors = np.array([[1e7, 1], [1e7, 0]], dtype=np.float32)
    query = np.array([[1e7, 0]], dtype=np.float32)
    assert exact_mapped(["a", "b"], vectors, ["q"], query, metric="l2") == {"q": [("b", 0.0), ("a", -1.0)]}


def test_reject_nonfinite_and_unnormalized_inputs():
    with pytest.raises(ValueError, match="nonfinite document"):
        exact_mapped(["a"], np.array([[np.nan, 0]], dtype=np.float32), ["q"], np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="not unit normalized"):
        exact_mapped(["a"], np.ones((1, 2), dtype=np.float32), ["q"], np.ones((1, 2), dtype=np.float32), metric="cosine")
