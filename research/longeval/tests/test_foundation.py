import json
import math
from pathlib import Path

import faiss
import numpy as np
import pytest
import pytrec_eval

from research.longeval.backends import Backend
from research.longeval.contracts import Table, delta, load_dataset, read_qrels, validate_roles
from research.longeval.evaluation import (ann_recall, evaluate, exact, rank, read_run, tie_aware_recall,
                                         validate_run, write_run)
from research.longeval.fixture import make_fixture
from research.longeval.runner import calibrate, run_pilot, validate_config, verify_artifacts

PILOT = Path(__file__).resolve().parents[1] / "pilot.json"


def table(ids=("a", "b"), values=((1, 0), (0, 1)), texts=None):
    return Table(tuple(ids), tuple(texts or ids), np.array(values, dtype=np.float32))


def test_delta_membership_text_and_vectors():
    before = table()
    reordered = table(("b", "c", "a"), ((0, 1), (1, 1), (1, 0)))
    assert delta(before, reordered) == ("c",)
    with pytest.raises(ValueError, match="removed"):
        delta(before, table(("a",), ((1, 0),)))
    with pytest.raises(ValueError, match="changed"):
        delta(before, table(values=((1.001, 0), (0, 1))))
    with pytest.raises(ValueError, match="changed"):
        delta(before, table(texts=("changed text", "b")))
    assert delta(before, before) == ()


@pytest.mark.parametrize("ids,values", [(("a", "a"), ((1, 0), (0, 1))),
                                        (("a", "b"), ((float("nan"), 0), (0, 1))),
                                        (("white space", "b"), ((1, 0), (0, 1)))])
def test_invalid_tables(ids, values):
    with pytest.raises(ValueError):
        table(ids, values)


def test_table_owns_canonical_vectors():
    values = np.ones((1, 2), dtype=np.float32)
    snapshot = Table(("0001",), ("text",), values)
    values[0, 0] = 3
    assert snapshot.vectors[0, 0] == 1
    assert not snapshot.vectors.flags.writeable
    with pytest.raises(ValueError, match="float32"):
        Table(("0001",), ("text",), values.astype(np.float64))


@pytest.mark.parametrize("collision", ["id", "text", "vector"])
def test_query_leakage(collision):
    roles = {"development": table(("d",), ((1, 0),)),
             "calibration": table(("c",), ((0, 1),)),
             "evaluation": table(("e",), ((1, 1),))}
    validate_roles(roles)
    roles["evaluation"] = table(("d" if collision == "id" else "e",),
                                ((1, 0),) if collision == "vector" else ((1, 1),),
                                ("d" if collision == "text" else "e",))
    with pytest.raises(ValueError, match="leakage"):
        validate_roles(roles)


@pytest.mark.parametrize("raw", ["0.5", "nan", "inf", "1\nq 0 d 2"])
def test_invalid_judgments(tmp_path, raw):
    path = tmp_path / "qrels"
    path.write_text(f"q 0 d {raw}\n")
    with pytest.raises(ValueError):
        read_qrels(path)


def test_metric_denominators_and_missing_queries():
    qrels = {"q": {"a": 2, "b": 1}, "zero": {"a": 0}}
    run = {"q": [("b", 2), ("a", 1)]}
    explicit = evaluate(run, qrels, ["q", "zero", "unjudged"], population="explicit", ndcg_k=2, recall_k=2)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert explicit["count"] == 3
    assert explicit["mean"]["nDCG@2"] == pytest.approx(expected / 3)
    assert explicit["mean"]["R@2"] == pytest.approx(1 / 3)
    assert evaluate(run, qrels, ["q", "zero"], population="judged")["count"] == 2
    assert evaluate(run, qrels, ["q", "zero"], population="positive")["count"] == 1
    assert evaluate({}, qrels, ["q"], population="explicit")["mean"]["nDCG@10"] == 0


def test_independent_trec_evaluator_with_zero_and_missing_topics():
    rng = np.random.default_rng(100)
    docs = [f"d{i}" for i in range(40)]
    qrels = {f"q{i}": {d: int(rng.integers(0, 4)) for d in docs[:20]} for i in range(8)}
    qrels["zero"] = {"d0": 0}
    qrels["missing"] = {"d1": 1}
    run = {q: [(d, float(int(rng.integers(0, 5)))) for d in rng.choice(docs, 30, replace=False)]
           for q in qrels if q != "missing"}
    ours = evaluate(run, qrels, list(qrels) + ["unjudged"], population="explicit", recall_k=10)
    external = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10"}).evaluate(
        {q: dict(pairs) for q, pairs in run.items()})
    for q, values in ours["per_query"].items():
        assert values["nDCG@10"] == pytest.approx(external.get(q, {}).get("ndcg_cut_10", 0))
        assert values["R@10"] == pytest.approx(external.get(q, {}).get("recall_10", 0))


def test_round_trip_and_ties(tmp_path):
    run = {"q": rank([("a", .12345644), ("z", .12345641), ("x", -1e-100), ("y", -1e-100)])}
    path = tmp_path / "scores.run"
    write_run(path, run)
    assert read_run(path) == run
    result = evaluate(read_run(path), {"q": {"a": 1}}, ["q"], population="explicit", ndcg_k=1)
    assert result["mean"]["nDCG@1"] == 1
    assert rank([("a", 1), ("z", 1)])[0][0] == "z"


@pytest.mark.parametrize("pairs", [[("a", 1)], [("a", 1), ("a", 0)],
                                    [("a", 1), ("-1", 0)], [("a", float("nan")), ("b", 0)]])
def test_output_contract(pairs):
    with pytest.raises(ValueError):
        validate_run({"q": pairs}, ["q"], ["a", "b"], 2)


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_independent_exact_agrees_with_faiss_flat(metric):
    rng = np.random.default_rng(9)
    values = rng.normal(size=(73, 8)).astype(np.float32)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    if metric == "cosine":
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    corpus = table([f"d{i}" for i in range(len(values))], values)
    topics = table([f"q{i}" for i in range(len(queries))], queries)
    ours = exact(corpus, topics, 7, metric, block_size=9)
    index = faiss.IndexFlatL2(8) if metric == "l2" else faiss.IndexFlatIP(8)
    index.add(values)
    scores, ids = index.search(queries, 7)
    for row, q in enumerate(topics.ids):
        assert [d for d, _ in ours[q]] == [corpus.ids[i] for i in ids[row]]
        expected = -scores[row] if metric == "l2" else scores[row] - (metric == "cosine")
        np.testing.assert_allclose([s for _, s in ours[q]], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_epzero_exact_reference_and_adapter_scores(metric):
    rng = np.random.default_rng(18)
    values = rng.normal(size=(73, 8)).astype(np.float32)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    if metric == "cosine":
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    corpus = table([f"d{i}" for i in range(len(values))], values)
    topics = table([f"q{i}" for i in range(len(queries))], queries)
    reference = exact(corpus, topics, 7, metric)
    config = json.loads(PILOT.read_text())
    backend = Backend("epzero", config, {"dim": 8, "metric": metric}, corpus.ids, 11)
    backend.add(corpus, corpus.ids)
    labels, distances = backend.index.exact_search(topics.vectors, k=7, threads=1)
    approx = backend.search(topics, 7, 128)
    for i, q in enumerate(topics.ids):
        assert [corpus.ids[label] for label in labels[i]] == [d for d, _ in reference[q]]
        expected = -distances[i] + (metric == "ip")
        np.testing.assert_allclose([s for _, s in reference[q]], expected, atol=1e-5, rtol=1e-5)
        assert [d for d, _ in approx[q]] == [d for d, _ in reference[q]]
        np.testing.assert_allclose([s for _, s in approx[q]], expected, atol=1e-5, rtol=1e-5)


def test_exact_l2_avoids_norm_cancellation_and_tie_diagnostic():
    corpus = table(("a", "b", "z"), ((10000, 0), (10001, 0), (10001, 0)))
    queries = table(("q",), ((10000, 0),))
    reference = exact(corpus, queries, 2, "l2", block_size=1)
    assert reference["q"] == [("a", 0), ("z", -1)]
    run = {"q": [("a", 0), ("b", -1)]}
    assert ann_recall(run, reference)["q"] == .5
    assert tie_aware_recall(run, reference, corpus, queries, "l2")["q"] == 1
    missing_strict = {"q": [("z", -1), ("b", -1)]}
    assert tie_aware_recall(missing_strict, reference, corpus, queries, "l2")["q"] == .5


def test_nonmonotone_calibration_and_target_miss():
    class Fake:
        def search(self, queries, k, effort):
            return {"q": [("a" if effort == 16 else "b", 1)]}
    queries = table(("q",), ((1, 0),))
    reference = {"q": [("a", 1)]}
    result = calibrate(Fake(), queries, reference, 1, [8, 16, 32], .95)
    assert result["effort"] == 16 and result["target_met"]
    missed = calibrate(Fake(), queries, reference, 1, [8, 32], .95)
    assert not missed["target_met"] and missed["effort"] == 8


@pytest.fixture
def fixture_path(tmp_path):
    return make_fixture(tmp_path / "inputs")


def test_dataset_and_config(fixture_path):
    dataset = load_dataset(fixture_path)
    assert list(map(len, dataset.additions)) == [192, 96, 96]
    assert len(dataset.ordered_ids) == 384
    config = json.loads(PILOT.read_text())
    validate_config(config)
    config["threads"] = 4
    with pytest.raises(ValueError, match="one"):
        validate_config(config)


def test_changed_input_identity_and_zero_cosine_rejected(fixture_path):
    before = load_dataset(fixture_path).manifest()
    path = fixture_path.parent / "evaluation.npy"
    vectors = np.load(path)
    vectors[0, 0] += .01
    np.save(path, vectors)
    assert load_dataset(fixture_path).manifest() != before
    spec = json.loads(fixture_path.read_text())
    spec["representation"].update(metric="cosine", normalization="unit")
    fixture_path.write_text(json.dumps(spec))
    vectors[:] = 0
    np.save(path, vectors)
    with pytest.raises(ValueError, match="unit normalized"):
        load_dataset(fixture_path)


@pytest.mark.parametrize("kind", ["epzero", "faiss_hnsw", "faiss_ivf"])
def test_adapters_reject_replacements(fixture_path, kind):
    dataset = load_dataset(fixture_path)
    config = json.loads(PILOT.read_text())
    backend = Backend(kind, config, dataset.spec["representation"], dataset.ordered_ids, 11)
    backend.train(dataset.snapshots[0])
    backend.add(dataset.snapshots[0], dataset.additions[0])
    with pytest.raises(ValueError, match="replacement"):
        backend.add(dataset.snapshots[0], [dataset.additions[0][0]])
    assert backend.state()["count"] == 192


def test_three_snapshot_pilot_and_artifact_tampering(fixture_path, tmp_path):
    config = json.loads(PILOT.read_text())
    config["seeds"] = [11]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "pilot"
    manifest = run_pilot(fixture_path, path, output, "unverified")
    assert manifest["status"] == "completed"
    assert manifest["result_rows"] == 51
    assert verify_artifacts(output, fixture_path)["synthetic"]
    with pytest.raises(FileExistsError):
        run_pilot(fixture_path, path, output)
    results = json.loads((output / "results.json").read_text())
    assert all(row["metrics"]["count"] == 24 for row in results)
    states = json.loads((output / "states.json").read_text())
    frozen = [s for s in states if s["arm"] == "faiss_ivf_frozen"]
    refreshed = [s for s in states if s["arm"] == "faiss_ivf_refreshed"]
    assert [len(s["training_ids"]) for s in frozen] == [192] * 3
    assert [len(s["training_ids"]) for s in refreshed] == [192, 288, 384]
    assert frozen[1]["parent_state_sha256"] == frozen[0]["state_sha256"]
    (output / results[0]["run_file"]).write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_artifacts(output)


def test_failed_stage_is_retained(fixture_path, tmp_path):
    config = json.loads(PILOT.read_text())
    config["k"] = 10000
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        run_pilot(fixture_path, path, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="completed"):
        verify_artifacts(output)
