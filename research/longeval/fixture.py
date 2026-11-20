"""Deterministic synthetic data for pipeline validation, never LongEval evidence."""
import json
from pathlib import Path

import numpy as np


def make_fixture(folder: Path):
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261116)
    dim = 16
    centers = rng.normal(size=(8, dim)) * 2
    clusters = rng.integers(0, len(centers), size=384)
    vectors = (centers[clusters] + rng.normal(size=(384, dim))).astype(np.float32)
    ids = tuple(f"doc-{i:04d}" for i in range(384))

    def table(name, keys, texts, values):
        record_path = folder / f"{name}.jsonl"
        record_path.write_text("".join(json.dumps({"id": key, "text": text}) + "\n"
                                       for key, text in zip(keys, texts)), encoding="utf-8")
        np.save(folder / f"{name}.npy", values)
        return {"records": record_path.name, "vectors": f"{name}.npy"}

    query_specs, query_clusters = {}, {}
    for role, count in (("development", 8), ("calibration", 12), ("evaluation", 24)):
        groups = rng.integers(0, len(centers), size=count)
        values = (centers[groups] + rng.normal(size=(count, dim))).astype(np.float32)
        keys = tuple(f"{role}-{i:03d}" for i in range(count))
        query_specs[role] = table(role, keys, [f"synthetic query {key}" for key in keys], values)
        query_clusters.update(zip(keys, map(int, groups)))
    snapshots = []
    for i, count in enumerate((192, 288, 384), 1):
        name = f"snapshot-{i}"
        entry = table(name, ids[:count], [f"synthetic document {key}" for key in ids[:count]], vectors[:count])
        entry.update(name=name, qrels=f"{name}.qrels")
        # Complete artificial judgments based on cluster identity, independently of ANN output.
        (folder / entry["qrels"]).write_text("".join(
            f"{q} 0 {key} {int(clusters[j] == group)}\n"
            for q, group in query_clusters.items() for j, key in enumerate(ids[:count])), encoding="utf-8")
        snapshots.append(entry)
    spec = {"schema": 1, "synthetic": True, "release": "synthetic-pipeline-fixture-v1",
            "representation": {"model": "numpy-synthetic-clusters", "revision": "20261116-v1",
                               "preprocessing": "8 Gaussian centers plus independent noise",
                               "normalization": "none", "dtype": "float32", "dim": dim, "metric": "l2"},
            "queries": query_specs, "snapshots": snapshots}
    path = folder / "dataset.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path
