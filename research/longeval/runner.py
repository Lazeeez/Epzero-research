"""Synchronous three-snapshot pilot with explicit barriers and an artifact ledger."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

from .backends import Backend
from .contracts import digest, file_hash, load_dataset
from .evaluation import ann_recall, evaluate, exact, read_run, tie_aware_recall, validate_run, write_run

ARMS = ("exact", "epzero_incremental", "epzero_replay", "epzero_shuffled",
        "faiss_hnsw_incremental", "faiss_hnsw_replay", "faiss_hnsw_shuffled",
        "faiss_ivf_frozen", "faiss_ivf_refreshed")


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_config(config):
    expected = {"schema", "seeds", "k", "M", "ef_construction", "ef", "ef_grid", "nlist", "nprobe",
                "nprobe_grid", "ivf_iterations", "recall_target", "population", "threads", "regimes", "arms"}
    if set(config) != expected or config["schema"] != 1:
        raise ValueError("unknown or missing configuration fields")
    if config["threads"] != 1:
        raise ValueError("deterministic pilot requires one construction/search thread")
    for key in ("k", "M", "ef_construction", "ef", "nlist", "nprobe", "ivf_iterations"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["M"] < 2:
        raise ValueError("M must be at least 2")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]) or any(
            type(s) is not int or not 0 <= s < 2**31 for s in config["seeds"]):
        raise ValueError("unique nonnegative 31-bit seeds required")
    for key, choices in (("arms", ARMS), ("regimes", ("frozen", "matched_recall"))):
        if not config[key] or len(set(config[key])) != len(config[key]) or set(config[key]) - set(choices):
            raise ValueError(f"unsupported/duplicate {key}")
    if "exact" not in config["arms"]:
        raise ValueError("exact reference arm is mandatory")
    if config["population"] not in ("explicit", "positive", "judged") or not 0 < config["recall_target"] <= 1:
        raise ValueError("invalid population or target")
    for key in ("ef_grid", "nprobe_grid"):
        values = config[key]
        if not values or any(type(v) is not int or v <= 0 for v in values) or values != sorted(set(values)):
            raise ValueError(f"{key} must be sorted unique positive integers")
    if min(config["ef_grid"] + [config["ef"]]) < config["k"]:
        raise ValueError("ef must be at least the search depth")
    if max(config["nprobe_grid"] + [config["nprobe"]]) > config["nlist"]:
        raise ValueError("nprobe exceeds nlist")


def measured(ledger, stage, identity, action):
    """Instrument development costs, not isolated publication-quality latency benchmarks.

    Peak RSS is sampled for the entire harness process, including all resident input tables and
    live arms. It is a lower bound, not per-index memory. Sampling overhead is included.
    """
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    wall, cpu = time.perf_counter(), time.process_time()
    sampler.start()
    status = "failed"
    try:
        value = action()
        status = "completed"
        return value
    finally:
        stop.set()
        sampler.join()
        peak[0] = max(peak[0], process.memory_info().rss)
        ledger.append({**identity, "stage": stage, "status": status,
                       "wall_seconds": time.perf_counter() - wall,
                       "process_cpu_seconds": time.process_time() - cpu,
                       "sampled_process_peak_rss_bytes": peak[0], "rss_sample_interval_ms": 2})


def calibrate(backend, queries, reference, k, grid, target):
    points = []
    for effort in grid:
        start = time.perf_counter()
        try:
            run = backend.search(queries, k, effort)
        except ValueError as error:
            # Incomplete IVF outputs are infeasible points, never silently padded or dropped.
            points.append({"effort": effort, "valid": False, "reason": str(error),
                           "wall_seconds": time.perf_counter() - start})
            continue
        values = ann_recall(run, reference)
        points.append({"effort": effort, "valid": True, "recall": float(np.mean(list(values.values()))),
                       "wall_seconds": time.perf_counter() - start})
    valid = [p for p in points if p["valid"]]
    if not valid:
        raise ValueError("no feasible calibration point")
    feasible = [p for p in valid if p["recall"] >= target]
    chosen = min(feasible, key=lambda p: p["effort"]) if feasible else max(valid, key=lambda p: (p["recall"], -p["effort"]))
    repeat = ann_recall(backend.search(queries, k, chosen["effort"]), reference)
    repeated = float(np.mean(list(repeat.values())))
    if repeated != chosen["recall"]:
        raise ValueError("calibration selection failed repeat validation")
    return {"effort": chosen["effort"], "calibration_recall": repeated,
            "target": target, "target_met": repeated >= target, "k": k,
            "query_ids": list(queries.ids), "points": points}


def provenance(native_origin):
    import epzero
    import epzero._core
    root = Path(__file__).resolve().parents[2]
    code = {}
    for folder in (root / "engine", root / "research" / "longeval"):
        for path in sorted(folder.rglob("*")):
            if any(p in ("build", "__pycache__", ".pytest_cache") for p in path.relative_to(folder).parts):
                continue
            if path.is_file() and (path.suffix in (".py", ".cpp", ".h", ".toml") or path.name == "CMakeLists.txt"):
                code[path.relative_to(root).as_posix()] = file_hash(path)
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_files": code, "source_sha256": digest(code),
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "faiss-cpu", "epzero", "psutil")},
            "native_origin_declared": native_origin, "epzero_binary_sha256": file_hash(Path(epzero._core.__file__)),
            "epzero_machine": epzero.machine(),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def run_pilot(dataset_path: Path, config_path: Path, output: Path, native_origin="unverified"):
    output.mkdir(parents=True, exist_ok=False)
    ledger, results, states = [], [], []
    manifest = {"schema": 1, "status": "running", "purpose": "pipeline validation; not research evidence",
                "checkpoint_policy": "fresh-process deterministic replay; native reload continuation disabled",
                "timing_scope": "end-to-end adapter batch including validation; warmup plus three repetitions",
                "memory_scope": "sampled whole-process RSS lower bound, shared inputs/arms, not per-index peak"}
    write_json(output / "manifest.json", manifest)
    try:
        dataset = load_dataset(dataset_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_config(config)
        if config["k"] > min(len(t.ids) for t in dataset.snapshots):
            raise ValueError("k exceeds the smallest snapshot")
        representation = dataset.spec["representation"]
        manifest.update(config=config, config_sha256=file_hash(config_path), dataset=dataset.manifest(),
                        synthetic=dataset.spec["synthetic"], provenance=provenance(native_origin))
        write_json(output / "manifest.json", manifest)
        write_json(output / "labels.json", dict(enumerate(dataset.ordered_ids)))
        evaluation = dataset.queries["evaluation"]
        calibration = dataset.queries["calibration"]
        for seed in config["seeds"]:
            persistent = {}
            chronological = []
            parents = {}
            for step, (snapshot, additions, qrels) in enumerate(zip(dataset.snapshots, dataset.additions, dataset.qrels)):
                chronological.extend(additions)
                identity = {"seed": seed, "snapshot": step + 1}
                reference = measured(ledger, "oracle_evaluation", identity,
                                     lambda: exact(snapshot, evaluation, config["k"], representation["metric"]))
                cal_reference = None
                if "matched_recall" in config["regimes"]:
                    cal_reference = measured(ledger, "oracle_calibration", identity,
                                             lambda: exact(snapshot, calibration, config["k"], representation["metric"]))
                runs_at_snapshot, graph_at_snapshot = {}, {}
                for arm in config["arms"]:
                    context = {**identity, "arm": arm}
                    backend = None
                    if arm != "exact":
                        kind = "epzero" if arm.startswith("epzero") else "faiss_hnsw" if arm.startswith("faiss_hnsw") else "faiss_ivf"
                        continuing = arm.endswith("incremental") or arm == "faiss_ivf_frozen"
                        if continuing and arm in persistent:
                            backend = persistent[arm]
                            measured(ledger, "insert_delta", context, lambda: backend.add(snapshot, additions))
                        else:
                            backend = measured(ledger, "allocate", context,
                                               lambda: Backend(kind, config, representation, dataset.ordered_ids, seed))
                            if kind == "faiss_ivf":
                                measured(ledger, "train", context, lambda: backend.train(snapshot))
                            order = list(chronological)
                            if arm.endswith("shuffled"):
                                order = np.random.default_rng(seed + step).permutation(order).tolist()
                            measured(ledger, "build", context, lambda: backend.add(snapshot, order))
                            if continuing:
                                persistent[arm] = backend
                        state = backend.state()
                        if state["live_set_sha256"] != snapshot.manifest()["live_set_sha256"]:
                            raise ValueError("backend live set differs from snapshot")
                        state.update(context, parent_state_sha256=parents.get(arm) if continuing else None,
                                     construction="delta continuation" if continuing and step else "fresh replay/build")
                        state["state_sha256"] = digest(state)
                        parents[arm] = state["state_sha256"]
                        states.append(state)
                        if "topology_sha256" in state:
                            graph_at_snapshot[arm] = state["topology_sha256"]
                    regimes = ["reference"] if arm == "exact" else config["regimes"]
                    for regime in regimes:
                        tuning = None
                        if backend is None:
                            run, effort = reference, None
                        else:
                            ivf = backend.kind == "faiss_ivf"
                            effort = config["nprobe"] if ivf else config["ef"]
                            if regime == "matched_recall":
                                grid = config["nprobe_grid"] if ivf else config["ef_grid"]
                                tuning = measured(ledger, "calibration", {**context, "regime": regime},
                                                  lambda: calibrate(backend, calibration, cal_reference, config["k"], grid,
                                                                    config["recall_target"]))
                                effort = tuning["effort"]
                            measured(ledger, "warmup", {**context, "regime": regime},
                                     lambda: backend.search(evaluation, config["k"], effort))
                            repeats = [measured(ledger, "search_batch", {**context, "regime": regime, "repeat": r},
                                                lambda: backend.search(evaluation, config["k"], effort)) for r in range(3)]
                            if any(repeated != repeats[0] for repeated in repeats[1:]):
                                raise ValueError("no-update search repeat changed")
                            run = repeats[0]
                        validate_run(run, evaluation.ids, snapshot.ids, config["k"])
                        stem = f"seed-{seed}.snapshot-{step + 1}.{arm}.{regime}"
                        write_run(output / f"{stem}.run", run)
                        if read_run(output / f"{stem}.run") != run:
                            raise ValueError("run round trip changed scores/order")
                        metrics = evaluate(run, qrels, evaluation.ids, population=config["population"],
                                           ndcg_k=min(10, config["k"]), recall_k=config["k"])
                        overlaps = ann_recall(run, reference)
                        tie_aware = tie_aware_recall(run, reference, snapshot, evaluation, representation["metric"])
                        row = {**context, "regime": regime, "effort": effort, "calibration": tuning,
                               "metrics": metrics, "ann_recall_per_query": overlaps,
                               "ann_recall_mean": float(np.mean(list(overlaps.values()))),
                               "tie_aware_ann_recall_per_query": tie_aware, "tie_atol": 1e-5,
                               "run_file": f"{stem}.run"}
                        if tuning:
                            row["evaluation_target_met"] = row["ann_recall_mean"] >= tuning["target"]
                        results.append(row)
                        runs_at_snapshot[(arm, regime)] = run
                    # Snapshot measurement is read-only: fingerprint state around searches.
                    if backend is not None and backend.kind == "epzero":
                        after = backend.state()
                        if after["topology_sha256"] != state["topology_sha256"]:
                            raise ValueError("measurement changed graph topology")
                for prefix in ("epzero", "faiss_hnsw"):
                    for regime in config["regimes"]:
                        a, b = (f"{prefix}_incremental", regime), (f"{prefix}_replay", regime)
                        if a in runs_at_snapshot and b in runs_at_snapshot and runs_at_snapshot[a] != runs_at_snapshot[b]:
                            raise ValueError(f"{prefix} incremental/replay rankings differ")
                if {"epzero_incremental", "epzero_replay"} <= graph_at_snapshot.keys() and (
                        graph_at_snapshot["epzero_incremental"] != graph_at_snapshot["epzero_replay"]):
                    raise ValueError("Epzero incremental/replay graph differs")
        # Catch inputs modified while a run was in progress; there is no cache or implicit resume.
        if load_dataset(dataset_path).manifest() != manifest["dataset"] or file_hash(config_path) != manifest["config_sha256"]:
            raise ValueError("inputs changed during experiment")
        write_json(output / "results.json", results)
        write_json(output / "states.json", states)
        write_json(output / "costs.json", ledger)
        manifest["status"] = "completed"
        manifest["result_rows"] = len(results)
        manifest["artifacts"] = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.name != "manifest.json"}
        manifest["artifact_bytes"] = sum(p.stat().st_size for p in output.iterdir() if p.name != "manifest.json")
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        write_json(output / "costs.json", ledger)
        write_json(output / "manifest.json", manifest)
        raise


def verify_artifacts(output: Path, dataset_path: Path | None = None):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("only completed experiments may enter analysis")
    for name, expected in manifest["artifacts"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"artifact checksum mismatch: {name}")
    if dataset_path is not None:
        dataset = load_dataset(dataset_path)
        if dataset.manifest() != manifest["dataset"]:
            raise ValueError("dataset identity changed")
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        config = manifest["config"]
        for row in results:
            run = read_run(output / row["run_file"])
            step = row["snapshot"] - 1
            queries = dataset.queries["evaluation"]
            validate_run(run, queries.ids, dataset.snapshots[step].ids, config["k"])
            metrics = evaluate(run, dataset.qrels[step], queries.ids, population=config["population"],
                               ndcg_k=min(10, config["k"]), recall_k=config["k"])
            if metrics != row["metrics"]:
                raise ValueError("re-scored relevance metrics differ")
    return manifest
