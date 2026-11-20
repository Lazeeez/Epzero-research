"""Pinned document-text downloads and streaming title/abstract consistency audits.

Only identifier/digest maps are retained across snapshots, never full document text. ZIP members
are read as streams without extracting paths. Access query strings are not persisted or logged.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

from .release import checksum, MANIFEST_SHA256
from .storage import atomic_json


DOCUMENTS = (
    (1, "longeval_sci_training_2026_documents", 612477131, "longeval_sci_training_2026_abstract"),
    (2, "longeval_sci_06_08_2026_documents", 944968484, "longeval_sci_test-06-08_2026_abstract"),
    (3, "longeval_sci_09_11_2026_documents", 1196642551, "longeval_sci_test-09-11_2026_abstract"),
)


def fetch_zip(url, path, md5, size, *, request_bytes=16 << 20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {"source_url_sha256": hashlib.sha256(url.encode()).hexdigest(), "md5": md5, "bytes": size}
    state_path = path.with_suffix(".download.json")
    partial = path.with_name(path.name + ".part")
    if path.exists():
        if path.stat().st_size != size or checksum(path, "md5") != md5:
            raise ValueError("existing document archive checksum mismatch")
        return
    if state_path.exists():
        if json.loads(state_path.read_text())["identity"] != identity:
            raise ValueError("document partial identity mismatch")
    elif partial.exists():
        raise ValueError("document partial lacks an identity record")
    received = partial.stat().st_size if partial.exists() else 0
    if received > size or request_bytes <= 0 or not url.startswith("https://"):
        raise ValueError("invalid document download contract")
    if shutil.disk_usage(path.parent).free < size - received + (1 << 30):
        raise ValueError("insufficient document download space")
    atomic_json(state_path, {"identity": identity, "status": "downloading", "received_bytes": received})
    errors = 0
    while received < size:
        end = min(size, received + request_bytes) - 1
        try:
            request = Request(url, headers={"Range": f"bytes={received}-{end}", "Accept-Encoding": "identity"})
            with urlopen(request, timeout=30) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {received}-{end}/{size}":
                    raise ValueError("document server did not honor requested range")
                with partial.open("ab") as stream:
                    remaining = end - received + 1
                    while remaining:
                        block = response.read(min(65536, remaining))
                        if not block:
                            raise URLError("incomplete document range")
                        stream.write(block)
                        remaining -= len(block)
            received = partial.stat().st_size
            errors = 0
            atomic_json(state_path, {"identity": identity, "status": "downloading", "received_bytes": received})
            print(json.dumps({"file": path.name, "received_bytes": received, "total_bytes": size}), flush=True)
        except (URLError, TimeoutError, ConnectionError) as error:
            received = partial.stat().st_size if partial.exists() else 0
            errors += 1
            if errors >= 4:
                atomic_json(state_path, {"identity": identity, "status": "failed", "received_bytes": received,
                                         "error_type": type(error).__name__})
                raise RuntimeError("document transfer interrupted; same command can resume") from None
            time.sleep(2**errors)
    if checksum(partial, "md5") != md5:
        atomic_json(state_path, {"identity": identity, "status": "checksum_failed"})
        raise ValueError("document archive checksum mismatch; partial retained")
    partial.rename(path)
    atomic_json(state_path, {"identity": identity, "status": "verified", "sha256": checksum(path)})


def document_entries(manifest):
    manifest = Path(manifest)
    if checksum(manifest) != MANIFEST_SHA256:
        raise ValueError("document release manifest checksum mismatch")
    raw = json.loads(manifest.read_text())["longeval-sci-2026"]
    for step, key, size, subfolder in DOCUMENTS:
        entry = raw[key]
        name = entry["cache_path"]
        if Path(name).name != name or "\\" in name:
            raise ValueError("invalid document cache filename")
        url = entry.get("url") or entry["urls"][0]
        if isinstance(url, dict):
            url = url["url"]
        yield {"snapshot": step, "key": key, "bytes": size, "subfolder": subfolder,
               "name": name, "url": url, "md5": entry["expected_md5"]}


def fetch_documents(manifest, folder):
    folder = Path(folder)
    results = []
    for entry in document_entries(manifest):
        path = folder / entry["name"]
        fetch_zip(entry["url"], path, entry["md5"], entry["bytes"])
        with zipfile.ZipFile(path) as archive:
            members = [{"name": item.filename, "bytes": item.file_size} for item in archive.infolist()]
        receipt = {key: value for key, value in entry.items() if key != "url"}
        receipt.update(sha256=checksum(path), members=members)
        atomic_json(path.with_suffix(".receipt.json"), receipt)
        results.append(receipt)
        print(json.dumps({"stage": "verified", "snapshot": entry["snapshot"], "members": len(members)}), flush=True)
    return results


def read_document(record):
    # Follow the pinned LongEvalSciDoc.default_text exactly; whitespace is not normalized.
    docid, title, abstract = record.get("id"), record.get("title"), record.get("abstract")
    if not isinstance(docid, str) or not docid or any(c.isspace() for c in docid):
        raise ValueError("invalid document identifier")
    if not isinstance(title, str) or (abstract is not None and not isinstance(abstract, str)):
        raise ValueError("invalid document title/abstract")
    return docid, title + (" " + abstract if abstract else "")


def audit_document_zip(path, prefix, *, sample_size=32):
    """Return exact title-plus-abstract hashes and a deterministic encoder-validation sample.

    Empty default_text is excluded as in the upstream iterator. Conflicting duplicate IDs are
    reported as ambiguous: archive/file iteration order must not silently choose a representation.
    """
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    hashes, sample = {}, []
    rows = empty = duplicates = conflicts = 0
    conflict_examples = []
    members = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member names")
        for name in sorted(names):
            if not name.startswith(prefix) or not name.endswith(".jsonl"):
                continue
            relative = PurePosixPath(name).relative_to(PurePosixPath(prefix))
            if len(relative.parts) != 1 or ".." in relative.parts or "\\" in name:
                raise ValueError("unexpected document member path")
            members.append(name)
            with archive.open(name) as stream:
                for line_number, line in enumerate(stream, 1):
                    record = json.loads(line)
                    docid, text = read_document(record)
                    rows += 1
                    if not text:
                        empty += 1
                        continue
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if docid in hashes:
                        duplicates += 1
                        if hashes[docid] != digest:
                            conflicts += 1
                            if len(conflict_examples) < 20:
                                conflict_examples.append(docid)
                        continue
                    hashes[docid] = digest
                    priority = int.from_bytes(hashlib.sha256(("longeval-encoder-check-v1:" + docid).encode()).digest(), "big")
                    item = (-priority, docid, {"id": docid, "text": text, "text_sha256": digest.hex(),
                                              "member": name, "line": line_number})
                    if len(sample) < sample_size:
                        heapq.heappush(sample, item)
                    elif item[:2] > sample[0][:2]:
                        heapq.heapreplace(sample, item)
    if not members:
        raise ValueError("document ZIP contains no expected JSONL members")
    return {"rows": rows, "eligible_unique_ids": len(hashes), "empty_text_rows": empty,
            "duplicate_eligible_rows": duplicates, "conflicting_duplicate_rows": conflicts,
            "conflicting_id_examples": conflict_examples, "members": members}, hashes, [x[2] for x in sorted(sample, reverse=True)]


def compare_texts(previous, current):
    shared = changed = 0
    examples = []
    for docid, value in current.items():
        if docid in previous:
            shared += 1
            if value != previous[docid]:
                changed += 1
                if len(examples) < 20:
                    examples.append(docid)
    return {"shared_ids": shared, "added_ids": len(current) - shared, "removed_ids": len(previous) - shared,
            "changed_shared_texts": changed, "changed_id_examples": examples,
            "fixed_text_insert_only_valid": changed == 0 and shared == len(previous)}


def audit_documents(manifest, folder, output, *, snapshots=(1, 2, 3), dense_maps=None):
    from .dense import inspect_docnos

    if not snapshots or list(snapshots) != sorted(set(snapshots)) or set(snapshots) - {1, 2, 3}:
        raise ValueError("select increasing unique document snapshots")
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "source_sha256": checksum(__file__), "manifest_sha256": checksum(manifest),
              "text_rule": "title + (' ' + abstract if abstract else '')", "snapshots": [],
              "limits": ["No encoder/vector consistency is implied by text consistency",
                         "Conflicting duplicate IDs are ambiguous; first-file order is not silently accepted"]}
    atomic_json(output / "audit.json", report)
    previous = None
    try:
        for entry in document_entries(manifest):
            step = entry["snapshot"]
            if step not in snapshots:
                continue
            archive = folder / entry["name"]
            if archive.stat().st_size != entry["bytes"] or checksum(archive, "md5") != entry["md5"]:
                raise ValueError("document archive identity mismatch")
            prefix = f"data/processed/doc_collection_09032026_abstract_2/snapshot-{step}/{entry['subfolder']}/documents/"
            stats, current, sample = audit_document_zip(archive, prefix)
            item = {"snapshot": step, "archive_sha256": checksum(archive), "archive_md5": entry["md5"], **stats}
            if previous is not None:
                item["transition"] = compare_texts(previous, current)
            if dense_maps is not None:
                maps = Path(dense_maps)
                sidecar = json.loads((maps / f"snapshot-{step}.json").read_text())
                # ID-map provenance is separately established by the dense prefix/full audits.
                # This comparison rehashes that input and reports its scope; it is not a vector audit.
                from npids import Lookup
                with Lookup(maps / f"snapshot-{step}.npids") as lookup:
                    count = len(lookup)
                _, dense_ids = inspect_docnos(maps / f"snapshot-{step}.npids", count)
                dense_ids = set(dense_ids)
                item["dense_coverage"] = {"id_map_sha256": checksum(maps / f"snapshot-{step}.npids"),
                    "archive_sha256_declared": sidecar["expected_sha256"], "dense_ids": len(dense_ids),
                    "missing_text_ids": len(dense_ids - current.keys()), "unindexed_text_ids": len(current.keys() - dense_ids)}
            sample_path = output / f"snapshot-{step}-encoder-sample.jsonl"
            with sample_path.open("x", encoding="utf-8", newline="\n") as stream:
                for row in sample:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            item["encoder_sample_sha256"] = checksum(sample_path)
            previous = current
            report["snapshots"].append(item)
            atomic_json(output / "audit.json", report)
            print(json.dumps(item), flush=True)
        report.update(status="completed", fixed_text_gate_passed=list(snapshots) == [1, 2, 3] and all(
            item["conflicting_duplicate_rows"] == 0 and item.get("transition", {}).get("fixed_text_insert_only_valid", True)
            for item in report["snapshots"]))
        atomic_json(output / "audit.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        atomic_json(output / "audit.json", report)
        raise
