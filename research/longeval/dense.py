"""Inspect bounded prefixes of revision-pinned dense artifacts, never full vectors.

Archive sidecars are SHA-256 verified. Prefixes have local hashes, but do NOT verify the full
archive SHA-256. Their metadata is provisional until full corpus/vector validation.
"""
from __future__ import annotations

import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen

from .release import checksum, download, judgment_coverage

# Resolved from the organizer-linked dense collection on 2026-11-16. The similarly named
# repositories without '-dense' are sparse indexes and must not be substituted.
ARTIFACTS = (
    {"snapshot": 1, "revision": "c6a54e3969887357ed2828714281923260bb3864",
     "archive_bytes": 8886222549,
     "archive_sha256": "5f9649a12f958701bced1b02d05d959bb111d58ef7cff84bfa2d151a7b3166e2",
     "sidecar_sha256": "0791246f6f604d2a5afb04646298f0a172e212b53aaa0b9d6b4b13a8efcdf8f2"},
    {"snapshot": 2, "revision": "f75720b44e4bea83c5795ffbe615184e68d6e64b",
     "archive_bytes": 13515203935,
     "archive_sha256": "56e468dc96d116803f61069152875d32c9933d8a69bc1e64458e8b51727a43c1",
     "sidecar_sha256": "312cbae585f038e11396d98035d7e3e1b46c3634c37f2fb9ff2be205accd5d5a"},
    {"snapshot": 3, "revision": "b742f1709ad59361454432b2db9d86726ed1b1f4",
     "archive_bytes": 16984335708,
     "archive_sha256": "9f691c45ec9d5567f1588d93c2f9b5a91dab4774f33f4534dfec36d05be3f661",
     "sidecar_sha256": "a271305f81d2062a4aae9c5e7033ee5c4cb5888cdb2232b517b2fb6741e3dd6e"},
)


def fetch_prefix(url, path, length, total):
    """Require a real, bounded byte-range response; never fall back to a full download."""
    path = Path(path)
    receipt = path.with_suffix(".receipt.json")
    if not 0 < length <= 32 << 20 or length >= total:
        raise ValueError("invalid prefix budget")
    if path.exists():
        record = json.loads(receipt.read_text())
        if record["sha256"] != checksum(path) or path.stat().st_size != length or record["url"] != url:
            raise ValueError("cached prefix identity mismatch")
        return record
    temporary = path.with_name(path.name + ".part")
    request = Request(url, headers={"Range": f"bytes=0-{length - 1}", "Accept-Encoding": "identity"})
    created = False
    try:
        with urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-{length - 1}/{total}":
                raise ValueError("server did not honor the exact bounded range")
            with temporary.open("xb") as stream:
                created = True
                remaining = length
                while remaining:
                    block = response.read(min(1 << 20, remaining))
                    if not block:
                        raise ValueError("truncated prefix download")
                    stream.write(block)
                    remaining -= len(block)
        record = {"url": url, "bytes": length, "sha256": checksum(temporary),
                  "full_archive_checksum_verified": False}
        temporary.rename(path)
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
        return record
    finally:
        if created and temporary.exists():
            temporary.unlink()


def inspect_prefix(path, sidecar, docnos_path=None):
    import lz4.frame

    expected = {item["path"]: item for item in sidecar["contents"]}
    if len(expected) != len(sidecar["contents"]):
        raise ValueError("duplicate sidecar members")
    seen, metadata = {}, None
    total = 0
    with lz4.frame.open(path, "rb") as decoded, tarfile.open(fileobj=decoded, mode="r|") as archive:
        for member in archive:
            if member.name not in expected or not member.isfile() or member.name in seen:
                raise ValueError("unexpected archive member")
            if member.size != expected[member.name]["size"] or member.size > 32 << 20:
                raise ValueError("member size disagrees or exceeds prefix budget")
            total += member.size
            if total > 32 << 20:
                raise ValueError("decoded prefix exceeds budget")
            stream = archive.extractfile(member)
            raw = stream.read()
            if len(raw) != member.size:
                raise ValueError("truncated archive member")
            seen[member.name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            if member.name == "docnos.npids" and docnos_path is not None:
                docnos_path = Path(docnos_path)
                if docnos_path.exists():
                    if checksum(docnos_path) != seen[member.name]["sha256"]:
                        raise ValueError("cached document map differs from archive")
                else:
                    with docnos_path.open("xb") as target:
                        target.write(raw)
            if member.name == "pt_meta.json":
                metadata = json.loads(raw)
                break  # Do not request or decompress the vector member.
    if metadata is None:
        raise ValueError("index metadata not present in prefix")
    if metadata.get("type") != "dense_index" or metadata.get("format") != "flex":
        raise ValueError("archive is not a dense FlexIndex")
    n, dim = metadata.get("doc_count"), metadata.get("vec_size")
    if type(n) is not int or type(dim) is not int or n <= 0 or dim <= 0:
        raise ValueError("invalid vector shape in metadata")
    if expected["vecs.f4"]["size"] != 4 * n * dim:
        raise ValueError("declared vector bytes disagree with float32 matrix shape")
    return {"metadata": metadata, "prefix_members": seen,
            "vector_bytes_declared": expected["vecs.f4"]["size"],
            "vector_payload_verified": False, "document_IDs_enumerated": False}


def inspect_docnos(path, expected_count, previous=()):
    from npids import Lookup
    import numpy as np

    with Lookup(path) as lookup:
        count = len(lookup)
        if count != expected_count:
            raise ValueError("document-map count differs from metadata")
        # Chunked decoding avoids millions of scalar mmap lookups. IDs alone fit this audit's
        # modest memory budget; this is not the full-corpus vector loader.
        ids = [str(key) for start in range(0, count, 65536)
               for key in lookup.fwd[np.arange(start, min(count, start + 65536), dtype=np.int64)]]
    if any(not key or any(c.isspace() for c in key) for key in ids):
        raise ValueError("invalid stored document IDs")
    current, old = set(ids), set(previous)
    def id_hash(values):
        h = hashlib.sha256()
        for key in values:
            h.update(key.encode() + b"\n")
        return h.hexdigest()
    return {"rows": count, "distinct_ids": len(current), "duplicate_rows": count - len(current),
            "added_ids": len(current - old), "removed_ids": len(old - current),
            "shared_ids": len(old & current), "sorted_ids_sha256": id_hash(sorted(current)),
            "row_order_sha256": id_hash(ids),
            "previous_rows_form_prefix": ids[:len(previous)] == list(previous),
            "insert_only_membership_valid": len(current) == count and not old - current}, ids


def inspect_dense(folder, output, *, offline=False, release_files=None):
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError(output)
    folder.mkdir(parents=True, exist_ok=True)
    records, previous = [], []
    for entry in ARTIFACTS:
        number = entry["snapshot"]
        base = (f"https://huggingface.co/datasets/jueri/longeval-2026-snapshot-{number}-index-dense/"
                f"resolve/{entry['revision']}/artifact.tar.lz4")
        sidecar_path = folder / f"snapshot-{number}.json"
        if not offline:
            download(base + ".json", sidecar_path, entry["sidecar_sha256"], limit=65536)
        if checksum(sidecar_path) != entry["sidecar_sha256"]:
            raise ValueError("dense sidecar checksum mismatch")
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar["expected_sha256"] != entry["archive_sha256"]:
            raise ValueError("archive identity differs from pinned manifest")
        contents = {item["path"]: item for item in sidecar["contents"]}
        # Flush offsets are compressed-byte offsets. Include a small amount of the next block so
        # streaming tar can finish the preceding metadata record without reading the full archive.
        length = contents["vecs.f4"]["offset"] + 65536
        prefix = folder / f"snapshot-{number}.prefix.lz4"
        if offline:
            receipt = json.loads(prefix.with_suffix(".receipt.json").read_text())
            if receipt["sha256"] != checksum(prefix) or prefix.stat().st_size != length or receipt["url"] != base:
                raise ValueError("offline prefix identity mismatch")
        else:
            receipt = fetch_prefix(base, prefix, length, entry["archive_bytes"])
        docnos = folder / f"snapshot-{number}.npids"
        inspected = inspect_prefix(prefix, sidecar, docnos)
        membership, previous = inspect_docnos(docnos, inspected["metadata"]["doc_count"], previous)
        inspected.update(document_IDs_enumerated=True, membership=membership)
        if release_files is not None:
            inspected["judgment_coverage"] = judgment_coverage(release_files, number, previous)
        records.append({**entry, "prefix": receipt, **inspected})
    result = {"schema": 1, "status": "completed", "kind": "bounded archive-prefix inspection",
              "python": sys.version, "audit_source_sha256": checksum(__file__),
              "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "npids", "lz4")},
              "artifacts": records, "limits": ["Full archive checksums not verified", "Vectors not inspected",
                                                   "Encoder revision/normalization/query compatibility remain unverified"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result
