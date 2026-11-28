import gzip
import json

import numpy as np
import pytest

from research.arxiv_stream import stream


def fake_encoding(tmp_path, n=3000, dim=32, queries=40, seed=0):
    """An encoder output and a prepared corpus for `n` abstracts over 30 months, without a model."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(20, dim))
    def draw(count):
        x = centers[rng.integers(0, 20, count)] + 0.3 * rng.normal(size=(count, dim))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    vectors, prepared = tmp_path / "vectors", tmp_path / "prepared"
    vectors.mkdir()
    prepared.mkdir()
    ids = [f"{i:07d}" for i in range(n)]
    np.save(vectors / "corpus.npy", draw(n))
    np.save(vectors / "queries-new.npy", draw(queries))
    np.save(vectors / "queries-uniform.npy", draw(queries))
    with gzip.open(vectors / "corpus-ids.json.gz", "wt") as f:
        json.dump(ids, f)
    (vectors / "report.json").write_text(json.dumps({"status": "completed"}))
    with gzip.open(prepared / "corpus.jsonl.gz", "wt") as f:
        for i, d in enumerate(ids):
            month = i * 30 // n
            f.write(json.dumps({"id": d, "created": f"{2000 + month // 12}-{month % 12 + 1:02d}-15", "abstract": "x"}) + "\n")
    return vectors, prepared


def test_cohorts_are_thirds_in_order():
    c = stream.cohorts(10)
    assert c.tolist() == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_sample_refuses_vectors_out_of_order(tmp_path):
    vectors, prepared = fake_encoding(tmp_path, n=300)
    with gzip.open(vectors / "corpus-ids.json.gz", "wt") as f:
        json.dump([f"{i:07d}" for i in reversed(range(300))], f)
    with pytest.raises(ValueError, match="order"):
        stream.sample(vectors, prepared, tmp_path / "sample")


def test_scoring_matches_the_frozen_runner_and_the_repair_runs(tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("epzero")
    from research.longeval import insertion_pilot
    vectors, prepared = fake_encoding(tmp_path)
    summary = stream.sample(vectors, prepared, tmp_path / "sample")
    assert summary["count"] == 3000 and summary["months"] == 30 and summary["metric"] == "ip"
    exact = np.load(tmp_path / "sample/exact-top100.npy")
    q, x = np.load(tmp_path / "sample/queries.npy"), np.load(tmp_path / "sample/vectors.npy")
    assert set(exact[0].tolist()) == set(np.argsort(-(x @ q[0]), kind="stable")[:100].tolist())

    report = insertion_pilot.run(tmp_path / "sample", tmp_path / "plain", order="appended", seed=11, backend="epzero",
                                 efs=(200,))
    labels = np.load(tmp_path / "plain/labels-ef200.npy").astype(np.int64)
    cohort = np.load(tmp_path / "sample/cohort.npy")
    table, place, hits = stream.score(labels, exact, cohort, np.arange(3000) * 10 // 3000)
    assert np.array_equal(table, np.load(tmp_path / "plain/per-query.npy"))
    assert np.array_equal(place, np.load(tmp_path / "plain/per-query-position.npy"))
    assert hits == pytest.approx(report["recall_at_100"])

    fixed = stream.repaired(tmp_path / "sample", tmp_path / "repaired", seed=11, threads=2)
    assert fixed["months"] == 30 and fixed["refined_nodes"] > 0
    assert set(fixed["widths"]) == {"100", "200", "400", "800"} and set(fixed["confirm"]["widths"]) == set(fixed["widths"])
    assert fixed["widths"]["800"]["recall_at_100"] >= fixed["widths"]["100"]["recall_at_100"]


def test_analysis_reads_jobs_and_scores_every_prediction(tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("epzero")
    from research.arxiv_stream import analyze
    from research.longeval import insertion_pilot
    vectors, prepared = fake_encoding(tmp_path)
    stream.sample(vectors, prepared, tmp_path / "sample")
    jobs = tmp_path / "jobs"
    job = 0
    for backend in ("epzero", "faiss_hnsw"):
        for order in ("appended", "reversed", "shuffled"):
            job += 1
            (jobs / str(job)).mkdir(parents=True)
            (jobs / str(job) / "arm.txt").write_text(f"arm={order}-{backend}-11\n")
            insertion_pilot.run(tmp_path / "sample", jobs / str(job) / "build", order=order, seed=11, backend=backend,
                                efs=(100, 200))
    (jobs / "99").mkdir()
    (jobs / "99" / "arm.txt").write_text("arm=repaired-epzero-11\n")
    stream.repaired(tmp_path / "sample", jobs / "99" / "build", seed=11, threads=2, widths=(100, 200))
    out = analyze.analyze(tmp_path / "sample", jobs)
    assert len(out["builds"]) == 7
    for name in ("new-content", "uniform"):
        e = out["by_set"][name]["epzero"]
        assert {"appended_excess_pp", "reversed_minus_appended_pp", "repaired_minus_appended_pp",
                "shuffled_last_minus_first_tenth_pp"} <= set(e)
    p = out["predictions"]
    assert set(p["1_appended_excess_above_1pp_interval_above_0"]) == {"new-content/epzero", "new-content/faiss_hnsw",
                                                                      "uniform/epzero", "uniform/faiss_hnsw"}
    assert set(p["4_repair"]) == {"new-content", "uniform"} and set(p["5_exposure_r2_at_least_0.9"]) == {"epzero-11", "faiss_hnsw-11"}
    assert sum(v["targets"] for v in out["age_curve"]["epzero"].values()) == 40 * 100
