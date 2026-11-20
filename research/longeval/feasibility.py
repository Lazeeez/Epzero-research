"""One-backend-per-process resource probes on audited real document vectors.

Document vectors used as probes are held out from the index. They are not user queries and
cannot support relevance, calibration, or final research-quality latency claims.
"""
from __future__ import annotations

import gc
import hashlib
import heapq
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .backends import Backend
from .dense import ARTIFACTS
from .evaluation import ann_recall, write_run
from .mapped import MappedSnapshot
from .oracle import exact_mapped
from .release import checksum
from .runner import measured, provenance
from .storage import atomic_json


def select_ids(ids, count, probes, seed):
    if count <= 0 or probes <= 0 or count + probes > len(ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid feasibility sample size or duplicate IDs")
    def key(docid):
        return hashlib.sha256(f"longeval-resource-v1:{seed}:{docid}".encode()).digest(), docid
    selected = heapq.nsmallest(count + probes, ids, key=key)
    return sorted(selected[:count]), selected[count:]


def run_feasibility(folder, audit_path, output, *, snapshot=1, kind="epzero", count=8000,
                    probes=16, seed=11, k=100, native_origin="unverified"):
    if snapshot not in (1, 2, 3) or kind not in ("epzero", "faiss_hnsw", "faiss_ivf") or not 0 < k <= count:
        raise ValueError("invalid feasibility configuration")
    output, folder, audit_path = Path(output), Path(folder), Path(audit_path)
    output.mkdir(parents=True, exist_ok=False)
    config = {"M": 16, "ef_construction": 100, "nlist": min(64, max(1, count // 40)), "ivf_iterations": 10}
    report = {"status": "running", "purpose": "resource feasibility only; held-out document-vector probes",
              "snapshot": snapshot, "backend": kind, "count": count, "probes": probes, "seed": seed,
              "k": k, "metric": "ip", "config": config,
              "sampling_rule": "bottom SHA256(longeval-resource-v1:seed:ID), then lexical corpus insertion",
              "memory_scope": "sampled single-worker process RSS, including Python and inputs; not index-only memory",
              "limits": ["Document probes are not research queries", "No relevance measurements",
                         "Subset costs do not establish full-corpus feasibility", "Concurrent system workloads may affect timing"],
              "costs": []}
    atomic_json(output / "report.json", report)
    try:
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "completed" or not audit.get("numeric_vectors_valid"):
            raise ValueError("requires a completed valid vector scan")
        if checksum(folder / "vecs.f4") != audit["vector_sha256"]:
            raise ValueError("vector bytes differ from the audited input")
        report.update(audit_sha256=checksum(audit_path), vector_sha256=audit["vector_sha256"],
                      provenance=provenance(native_origin))
        with MappedSnapshot(folder, expected_archive_sha256=ARTIFACTS[snapshot - 1]["archive_sha256"]) as mapped:
            if mapped.receipt["members"]["vecs.f4"]["sha256"] != audit["vector_sha256"]:
                raise ValueError("audit and archive identities differ")
            ids, query_ids = select_ids(mapped.ids, count, probes, seed)
            positions = {docid: i for i, docid in enumerate(mapped.ids)}
            corpus = SimpleNamespace(ids=ids, vectors=np.ascontiguousarray(mapped.vectors[[positions[x] for x in ids]]))
            queries = SimpleNamespace(ids=query_ids, vectors=np.ascontiguousarray(mapped.vectors[[positions[x] for x in query_ids]]))
        del mapped, positions
        gc.collect()
        atomic_json(output / "selected_ids.json", {"indexed": ids, "held_out_document_probes": query_ids})
        report["selected_ids_sha256"] = checksum(output / "selected_ids.json")
        report["input_vector_bytes"] = corpus.vectors.nbytes + queries.vectors.nbytes
        ledger = report["costs"]
        measure = lambda stage, action: measured(ledger, stage, {"backend": kind}, action)
        backend = measure("allocate", lambda: Backend(kind, config, {"metric": "ip", "dim": corpus.vectors.shape[1]}, ids, seed))
        # This cheap engineering probe deliberately trains on its full small corpus. A larger
        # scientific IVF experiment must declare and freeze its own training sample contract.
        measure("train", lambda: backend.train(corpus))
        measure("insert", lambda: backend.add(corpus, ids, batch_rows=512))
        report["state"] = measure("fingerprint", backend.state)
        reference = measure("exact", lambda: exact_mapped(ids, corpus.vectors, query_ids, queries.vectors, k=k, metric="ip"))
        write_run(output / "exact.run", reference)
        report["searches"] = []
        efforts = sorted({1, min(4, config["nlist"]), config["nlist"]}) if kind == "faiss_ivf" else [max(k, 100), max(k, 200)]
        for effort in sorted(set(efforts)):
            try:
                measure(f"warmup-{effort}", lambda: backend.search(queries, k, effort))
                run = measure(f"search-{effort}", lambda: backend.search(queries, k, effort))
            except ValueError as error:
                if kind != "faiss_ivf" or "padding/invalid labels" not in str(error):
                    raise
                report["searches"].append({"effort": effort, "valid": False, "reason": "fewer than k candidates"})
                continue
            write_run(output / f"effort-{effort}.run", run)
            values = ann_recall(run, reference)
            report["searches"].append({"effort": effort, "valid": True, "document_probe_ann_recall_mean": float(np.mean(list(values.values()))),
                                       "per_probe": values})
        report.update(status="completed", artifacts={p.name: checksum(p) for p in output.iterdir() if p.name != "report.json"})
        atomic_json(output / "report.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        atomic_json(output / "report.json", report)
        raise
