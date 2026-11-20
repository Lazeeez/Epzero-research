"""Read and audit full dense snapshots without copying their vector matrices.

The row digest join is by external document ID, never by row position. This checks the
fixed-vector assumption only; unchanged source text and compatible queries need separate evidence.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import mmap
from pathlib import Path
import sys
import time

import numpy as np

from .dense import ARTIFACTS, inspect_docnos
from .release import checksum
from .storage import atomic_json


class MappedSnapshot:
    def __init__(self, folder, *, expected_archive_sha256):
        self.folder = Path(folder)
        receipt = json.loads((self.folder / "ingestion.json").read_text())
        if receipt.get("status") != "verified" or receipt.get("archive_sha256") != expected_archive_sha256:
            raise ValueError("snapshot lacks a verified ingestion identity")
        self.receipt = receipt
        required = {"pt_meta.json", "docnos.npids", "vecs.f4"}
        if not required <= receipt["members"].keys():
            raise ValueError("incomplete ingestion receipt")
        for name in required:
            path, expected = self.folder / name, receipt["members"][name]
            if path.stat().st_size != expected["bytes"]:
                raise ValueError("extracted member size mismatch")
            if name != "vecs.f4" and checksum(path) != expected["sha256"]:
                raise ValueError("extracted metadata/identifier checksum mismatch")
        meta = json.loads((self.folder / "pt_meta.json").read_text())
        n, dim = meta.get("doc_count"), meta.get("vec_size")
        if (meta.get("type") != "dense_index" or meta.get("format") != "flex"
                or type(n) is not int or type(dim) is not int or n <= 0 or dim <= 0):
            raise ValueError("invalid dense metadata")
        if receipt["members"]["vecs.f4"]["bytes"] != n * dim * 4:
            raise ValueError("vector file does not match its declared float32 shape")
        self.membership, self.ids = inspect_docnos(self.folder / "docnos.npids", n)
        if self.membership["duplicate_rows"]:
            raise ValueError("duplicate document identifiers")
        # The release was produced on little-endian x86_64. No dtype conversion, normalization,
        # canonicalization, or full-matrix copy occurs here.
        self.vectors = np.memmap(self.folder / "vecs.f4", mode="r", dtype="<f4", shape=(n, dim))

    def close(self):
        self.vectors._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def scan_vectors(snapshot, output, *, chunk_rows=2048, unit_atol=1e-5):
    """Hash every original byte and row, and measure norms with float64 accumulation.

    Numeric scratch storage scales with chunk_rows * dimension. mmap pages are discardable,
    and Linux is advised to release them after each chunk. Fingerprints are disk-backed.
    Nonfinite/zero rows produce a completed but failed representation gate, not silent repairs.
    """
    if type(chunk_rows) is not int or chunk_rows <= 0 or not np.isfinite(unit_atol) or unit_atol < 0:
        raise ValueError("invalid vector scan parameters")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    vectors = snapshot.vectors
    n, dim = vectors.shape
    source = snapshot.folder / "vecs.f4"
    before = source.stat()
    temporary = output / "row_sha256.partial.npy"
    fingerprints = np.lib.format.open_memmap(temporary, mode="w+", dtype="V32", shape=(n,))
    digest = hashlib.sha256()
    finite_rows = zero_rows = unit_rows = 0
    norm_min, norm_max = float("inf"), float("-inf")
    norm_sum = norm_sumsq = 0.0
    bad_ids = []
    started = time.perf_counter()
    receipt = {"status": "scanning", "rows": n, "dimension": dim}
    atomic_json(output / "scan.json", receipt)
    try:
        for start in range(0, n, chunk_rows):
            end = min(n, start + chunk_rows)
            block = vectors[start:end]
            digest.update(memoryview(block).cast("B"))
            for offset, row in enumerate(block):
                fingerprints[start + offset] = np.void(hashlib.sha256(row).digest())
            values = np.array(block, dtype=np.float64, copy=True)
            finite = np.isfinite(values).all(axis=1)
            norms = np.linalg.norm(values[finite], axis=1)
            finite_rows += int(finite.sum())
            zero_rows += int(np.count_nonzero(norms == 0))
            unit_rows += int(np.count_nonzero(np.abs(norms - 1) <= unit_atol))
            if norms.size:
                norm_min, norm_max = min(norm_min, float(norms.min())), max(norm_max, float(norms.max()))
                norm_sum += float(norms.sum())
                norm_sumsq += float(np.square(norms).sum())
            bad_ids.extend(snapshot.ids[start + int(i)] for i in np.flatnonzero(~finite)[:max(0, 20 - len(bad_ids))])
            del block, values
            if hasattr(vectors._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                vectors._mmap.madvise(mmap.MADV_DONTNEED)
        fingerprints.flush()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("vector source changed during audit")
        if digest.hexdigest() != snapshot.receipt["members"]["vecs.f4"]["sha256"]:
            raise ValueError("vector payload checksum mismatch")
        # Close before renaming for Windows compatibility.
        fingerprints._mmap.close()
        final = output / "row_sha256.npy"
        temporary.rename(final)
        mean = norm_sum / finite_rows if finite_rows else None
        receipt.update(status="completed", vector_sha256=digest.hexdigest(),
                       fingerprints_sha256=checksum(final), finite_rows=finite_rows,
                       nonfinite_rows=n - finite_rows, nonfinite_id_examples=bad_ids,
                       zero_norm_rows=zero_rows, unit_norm_rows=unit_rows, unit_atol=unit_atol,
                       norm_min=norm_min if finite_rows else None,
                       norm_max=norm_max if finite_rows else None, norm_mean=mean,
                       norm_std=(max(0, norm_sumsq / finite_rows - mean**2)**0.5) if finite_rows else None,
                       numeric_vectors_valid=finite_rows == n and zero_rows == 0,
                       chunk_rows=chunk_rows, seconds=time.perf_counter() - started)
        atomic_json(output / "scan.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        atomic_json(output / "scan.json", receipt)
        raise
    finally:
        if not fingerprints._mmap.closed:
            fingerprints._mmap.close()


def compare_shared(previous_ids, previous_hashes, current_ids, current_hashes):
    """Compare SHA-256 row digests after joining reordered snapshots by document ID."""
    if len(previous_ids) != len(previous_hashes) or len(current_ids) != len(current_hashes):
        raise ValueError("identifier/fingerprint count mismatch")
    previous = {key: row for row, key in enumerate(previous_ids)}
    if len(previous) != len(previous_ids) or len(set(current_ids)) != len(current_ids):
        raise ValueError("duplicate IDs in vector join")
    shared = changed = reordered = 0
    examples = []
    for row, key in enumerate(current_ids):
        old = previous.get(key)
        if old is None:
            continue
        shared += 1
        reordered += old != row
        if previous_hashes[old] != current_hashes[row]:
            changed += 1
            if len(examples) < 20:
                examples.append(key)
    removed = len(previous) - shared
    return {"shared_ids": shared, "added_ids": len(current_ids) - shared,
            "removed_ids": removed, "shared_ids_at_different_rows": reordered,
            "bitwise_changed_shared_vectors": changed, "changed_id_examples": examples,
            "fixed_vector_insert_only_valid": removed == 0 and changed == 0}


def audit_vectors(folder, output, *, snapshots=(1, 2, 3), chunk_rows=2048):
    folder, output = Path(folder), Path(output)
    if not snapshots or list(snapshots) != sorted(set(snapshots)) or set(snapshots) - {1, 2, 3}:
        raise ValueError("select increasing unique snapshots from 1, 2, 3")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "status": "running", "kind": "full dense vector audit",
              "python": sys.version, "source_sha256": {name: checksum(Path(__file__).with_name(name))
                  for name in ("mapped.py", "storage.py", "dense.py", "release.py")},
              "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "npids", "lz4")},
              "snapshots": [], "limits": ["Source text consistency not established",
                  "Historical encoder weights and query compatibility not established",
                  "Not a retrieval experiment; norms are measured without altering stored vectors"]}
    atomic_json(output / "audit.json", report)
    previous_ids, previous_hashes = None, None
    try:
        for entry in ARTIFACTS:
            step = entry["snapshot"]
            if step not in snapshots:
                continue
            print(json.dumps({"stage": "scan", "snapshot": step}), flush=True)
            with MappedSnapshot(folder / f"snapshot-{step}", expected_archive_sha256=entry["archive_sha256"]) as snapshot:
                destination = output / f"snapshot-{step}"
                scan = scan_vectors(snapshot, destination, chunk_rows=chunk_rows)
                hashes = np.load(destination / "row_sha256.npy", mmap_mode="r", allow_pickle=False)
                item = {**entry, "membership": snapshot.membership, "scan": scan,
                        "ingestion_receipt_sha256": checksum(snapshot.folder / "ingestion.json"),
                        "members": snapshot.receipt["members"]}
                if previous_ids is not None:
                    item["transition"] = compare_shared(previous_ids, previous_hashes, snapshot.ids, hashes)
                    previous_hashes._mmap.close()
                previous_ids, previous_hashes = snapshot.ids, hashes
                report["snapshots"].append(item)
                atomic_json(output / "audit.json", report)
                print(json.dumps({"stage": "scanned", "snapshot": step, "scan": scan,
                                  "transition": item.get("transition")}), flush=True)
        report.update(status="completed", fixed_vector_gate_passed=list(snapshots) == [1, 2, 3] and all(
            item["scan"]["numeric_vectors_valid"] and item.get("transition", {}).get("fixed_vector_insert_only_valid", True)
            for item in report["snapshots"]))
        atomic_json(output / "audit.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        atomic_json(output / "audit.json", report)
        raise
    finally:
        if previous_hashes is not None:
            previous_hashes._mmap.close()
