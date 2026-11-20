"""Resumable full-archive ingestion and bounded-memory vector validation.

Downloaded archives are promoted only after their pinned full SHA-256 matches. Extracted members
are written to a staging directory without tar path extraction. Neither operation loads a matrix.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from .dense import ARTIFACTS
from .release import checksum, download


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch_archive(url, path, expected_sha256, size, *, request_bytes=64 << 20, progress=None):
    """Resume byte ranges from an identity-bound partial file; final checksum is mandatory.

    Partial data is never a completed archive. A failed transfer can be restarted with the same
    command; a checksum mismatch is retained for diagnosis and is not silently retried/replaced.
    """
    path = Path(path)
    if size <= 0 or request_bytes <= 0 or not url.startswith("https://"):
        raise ValueError("invalid archive download contract")
    if path.exists():
        if path.stat().st_size != size or checksum(path) != expected_sha256:
            raise ValueError("existing archive identity mismatch")
        return {"bytes": size, "sha256": expected_sha256, "status": "verified"}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    identity_path = path.with_name(path.name + ".download.json")
    identity = {"url": url, "sha256": expected_sha256, "bytes": size}
    if identity_path.exists():
        saved = json.loads(identity_path.read_text())
        if saved["identity"] != identity:
            raise ValueError("partial download belongs to another artifact")
    elif partial.exists():
        raise ValueError("partial download has no identity record")
    atomic_json(identity_path, {"identity": identity, "status": "downloading"})
    received = partial.stat().st_size if partial.exists() else 0
    if received > size:
        raise ValueError("partial file exceeds archive size")
    if shutil.disk_usage(path.parent).free < size - received + (1 << 30):
        raise ValueError("insufficient disk space for archive plus safety margin")
    failures = 0
    last_report = 0.0
    while received < size:
        end = min(size, received + request_bytes) - 1
        request = Request(url, headers={"Range": f"bytes={received}-{end}", "Accept-Encoding": "identity"})
        try:
            with urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {received}-{end}/{size}":
                    raise ValueError("server did not honor the requested archive range")
                with partial.open("ab") as target:
                    remaining = end - received + 1
                    while remaining:
                        block = response.read(min(1 << 20, remaining))
                        if not block:
                            raise URLError("incomplete archive range")
                        target.write(block)
                        remaining -= len(block)
                    target.flush()
                    os.fsync(target.fileno())
            received = partial.stat().st_size
            failures = 0
        except (URLError, TimeoutError, ConnectionError) as error:
            # HTTP errors have an error code, not an access URL, in the persisted record.
            received = partial.stat().st_size if partial.exists() else 0
            failures += 1
            if failures >= 4:
                atomic_json(identity_path, {"identity": identity, "status": "download_failed",
                                           "received_bytes": received, "error_type": type(error).__name__})
                raise RuntimeError(f"archive transfer interrupted at {received} bytes; resume with the same command") from None
            time.sleep(min(8, 2**failures))
        if progress and time.monotonic() - last_report >= 10:
            progress({"stage": "download", "file": path.name, "received_bytes": received, "total_bytes": size})
            last_report = time.monotonic()
    atomic_json(identity_path, {"identity": identity, "status": "verifying"})
    if checksum(partial) != expected_sha256:
        atomic_json(identity_path, {"identity": identity, "status": "checksum_failed"})
        raise ValueError("full archive checksum mismatch; partial file retained")
    partial.rename(path)
    atomic_json(identity_path, {"identity": identity, "status": "verified"})
    return {"bytes": size, "sha256": expected_sha256, "status": "verified"}


def extract_archive(archive_path, sidecar, output):
    import lz4.frame

    archive_path, output = Path(archive_path), Path(output)
    if checksum(archive_path) != sidecar["expected_sha256"]:
        raise ValueError("refusing extraction of an unverified archive")
    expected = {item["path"]: item["size"] for item in sidecar["contents"]}
    if len(expected) != len(sidecar["contents"]) or set(expected) - {
            "docnos.npids", "pt_meta.json", "vecs.f4", "index-ir-metadata.yml"}:
        raise ValueError("unexpected/duplicate archive member contract")
    if not {"docnos.npids", "pt_meta.json", "vecs.f4"} <= expected.keys():
        raise ValueError("missing required dense members")
    if output.exists():
        receipt = json.loads((output / "ingestion.json").read_text())
        if receipt["archive_sha256"] != sidecar["expected_sha256"] or receipt["status"] != "verified":
            raise ValueError("existing extraction identity mismatch")
        for name, item in receipt["members"].items():
            if (name not in expected or item["bytes"] != expected[name]
                    or (output / name).stat().st_size != expected[name]
                    or checksum(output / name) != item["sha256"]):
                raise ValueError("existing extracted member corrupted")
        if set(receipt["members"]) != set(expected):
            raise ValueError("incomplete existing extraction")
        return receipt
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < sum(expected.values()) + (1 << 30):
        raise ValueError("insufficient disk space for extraction")
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(exist_ok=False)
    receipt = {"status": "extracting", "archive_sha256": sidecar["expected_sha256"], "members": {}}
    atomic_json(staging / "ingestion.json", receipt)
    try:
        with lz4.frame.open(archive_path, "rb") as decoded, tarfile.open(fileobj=decoded, mode="r|") as archive:
            for member in archive:
                if member.name not in expected or not member.isfile() or member.name in receipt["members"]:
                    raise ValueError("unexpected archive member")
                if member.size != expected[member.name]:
                    raise ValueError("archive member size mismatch")
                digest = hashlib.sha256()
                count = 0
                with archive.extractfile(member) as source, (staging / member.name).open("xb") as target:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        target.write(block)
                        digest.update(block)
                        count += len(block)
                if count != member.size:
                    raise ValueError("truncated archive member")
                receipt["members"][member.name] = {"bytes": count, "sha256": digest.hexdigest()}
                atomic_json(staging / "ingestion.json", receipt)
        if set(receipt["members"]) != set(expected):
            raise ValueError("missing archive member")
        receipt["status"] = "verified"
        atomic_json(staging / "ingestion.json", receipt)
        staging.rename(output)
        return receipt
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        atomic_json(staging / "ingestion.json", receipt)
        raise


def ingest_dense(folder, snapshots=(1, 2, 3)):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if len(set(snapshots)) != len(snapshots) or set(snapshots) - {1, 2, 3}:
        raise ValueError("select unique snapshots 1, 2, 3")
    results = []
    for entry in ARTIFACTS:
        step = entry["snapshot"]
        if step not in snapshots:
            continue
        base = (f"https://huggingface.co/datasets/jueri/longeval-2026-snapshot-{step}-index-dense/"
                f"resolve/{entry['revision']}/artifact.tar.lz4")
        sidecar_path = folder / f"snapshot-{step}.sidecar.json"
        download(base + ".json", sidecar_path, entry["sidecar_sha256"], limit=65536)
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar["expected_sha256"] != entry["archive_sha256"]:
            raise ValueError("archive contract mismatch")
        print(json.dumps({"stage": "start", "snapshot": step, "archive_bytes": entry["archive_bytes"]}), flush=True)
        archive = folder / f"snapshot-{step}.tar.lz4"
        fetch_archive(base, archive, entry["archive_sha256"], entry["archive_bytes"],
                      progress=lambda event: print(json.dumps({"snapshot": step, **event}), flush=True))
        print(json.dumps({"stage": "extract", "snapshot": step}), flush=True)
        result = extract_archive(archive, sidecar, folder / f"snapshot-{step}")
        results.append({"snapshot": step, **result})
        print(json.dumps({"stage": "verified", "snapshot": step}), flush=True)
    return results


def prepare_dense(folder, output):
    """Run ingestion followed by all-vector validation, with a persistent job status."""
    from .mapped import audit_vectors

    folder, output = Path(folder).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "stage": "ingestion", "pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "data_folder": str(folder),
              "source_sha256": {name: checksum(Path(__file__).with_name(name)) for name in
                                ("storage.py", "mapped.py", "dense.py", "release.py")}}
    atomic_json(output / "job.json", status)
    try:
        status["ingestion"] = ingest_dense(folder)
        status["stage"] = "vector_audit"
        atomic_json(output / "job.json", status)
        audit = audit_vectors(folder, output / "vectors")
        status.update(status="completed", stage="complete", fixed_vector_gate_passed=audit["fixed_vector_gate_passed"],
                      finished_utc=datetime.now(timezone.utc).isoformat(), audit_sha256=checksum(output / "vectors/audit.json"))
        atomic_json(output / "job.json", status)
        return status
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output / "job.json", status)
        raise
