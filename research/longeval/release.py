"""Bounded, checksum-pinned download and offline audit of real LongEval small files.

No corpus/vector archives are downloaded. Upstream URLs may contain public access query strings;
only sanitized URLs are written to audit output. No remote Python code is executed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
import zipfile

REVISION = "63e73bdf54138ffed4246ce679627d7b50bd623a"
MANIFEST_URL = (f"https://raw.githubusercontent.com/clef-longeval/ir-datasets-longeval/{REVISION}/"
                "ir_datasets_longeval/etc/downloads.json")
MANIFEST_SHA256 = "25d669a013a9db9dfc084a4c0468a12567e1c0d09077b150ff58447019eaa85e"
MODELS = {"gpt-oss-120b": "openai-gpt-oss-120b", "gpt-oss-20b": "openai-gpt-oss-20b",
          "llama3-1-8b": "llama-3.1-8b-instant", "qwen3-32b": "qwen-qwen3-32b"}
TRAIN = "longeval_sci_training_2026_queries"
TEST = "longeval_sci_testing_2026_queries"
SMALL_KEYS = (TRAIN, TEST, "longeval_sci_training_2026_qrels_dctr", "longeval_sci_training_2026_qrels_raw",
              *(f"test_qrels_snapshot_{step}_{kind}" for step in (1, 2, 3) for kind in ("dctr", "raw")),
              "test_qrels_snapshot_3_human", "llm-synthetic-qrels")


def checksum(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sanitized_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def download(url, path, expected, algorithm="sha256", limit=32 << 20):
    """Resume only already-complete, verified files. Never replace a corrupt existing file."""
    path = Path(path)
    if path.exists():
        if checksum(path, algorithm) != expected:
            raise ValueError(f"existing file checksum mismatch: {path.name}")
        return
    if urlsplit(url).scheme != "https":
        raise ValueError("downloads require HTTPS")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    created = False
    try:
        request = Request(url, headers={"User-Agent": "Epzero-LongEval-research-audit/1"})
        with urlopen(request, timeout=60) as response:
            if int(response.headers.get("Content-Length", "0")) > limit:
                raise ValueError("download exceeds the small-file budget")
            with temporary.open("xb") as stream:
                created = True
                size = 0
                for block in iter(lambda: response.read(1 << 20), b""):
                    size += len(block)
                    if size > limit:
                        raise ValueError("download exceeds the small-file budget")
                    stream.write(block)
        if checksum(temporary, algorithm) != expected:
            raise ValueError(f"download checksum mismatch: {path.name}")
        temporary.rename(path)
    except (HTTPError, URLError) as error:
        # Do not copy an access query string into an exception/log.
        raise RuntimeError(f"download failed for {path.name}: {getattr(error, 'code', type(error).__name__)}") from None
    finally:
        if created and temporary.exists():
            temporary.unlink()


def manifest_entries(folder):
    path = Path(folder) / "downloads.json"
    if checksum(path) != MANIFEST_SHA256:
        raise ValueError("upstream manifest identity changed")
    entries = json.loads(path.read_text(encoding="utf-8"))["longeval-sci-2026"]
    selected = {}
    for key in SMALL_KEYS:
        item = entries[key]
        if Path(item["cache_path"]).name != item["cache_path"] or "\\" in item["cache_path"]:
            raise ValueError("invalid release filename")
        if urlsplit(item["url"]).hostname != "researchdata.tuwien.ac.at":
            raise ValueError("unexpected release host")
        selected[key] = item
    return selected


def fetch_release(folder):
    folder = Path(folder)
    download(MANIFEST_URL, folder / "downloads.json", MANIFEST_SHA256, limit=1 << 20)
    entries = manifest_entries(folder)
    for key, item in entries.items():
        download(item["url"], folder / item["cache_path"], item["expected_md5"], "md5")
    return len(entries)


def read_queries(path):
    result = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2 or not parts[0] or not parts[1].strip():
            raise ValueError(f"invalid query row {line_number}")
        qid, text = parts
        if qid in result or any(c.isspace() for c in qid):
            raise ValueError("duplicate/invalid query ID")
        result[qid] = text
    return result


def parse_qrels(text):
    result = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid qrels row {line_number}")
        query, _, document, raw = fields
        value = float(raw)
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("relevance must be a finite integer")
        if document in result.get(query, {}):
            raise ValueError("duplicate qrel pair")
        result.setdefault(query, {})[document] = int(value)
    return result


def qrel_summary(qrels, allowed_queries):
    unknown = set(qrels) - set(allowed_queries)
    if unknown:
        raise ValueError(f"qrels contain {len(unknown)} unknown query IDs")
    positive = sorted(q for q, judgments in qrels.items() if any(r > 0 for r in judgments.values()))
    return {"rows": sum(map(len, qrels.values())), "judged_queries": len(qrels),
            "positive_queries": len(positive), "positive_query_ids": positive,
            "judged_query_ids": sorted(qrels),
            "label_histogram": dict(sorted(Counter(str(r) for j in qrels.values() for r in j.values()).items())),
            "unique_documents": len({d for j in qrels.values() for d in j})}


def judgment_coverage(folder, snapshot, live_ids):
    """Check released judgment document IDs against an enumerated dense-index live set."""
    folder = Path(folder)
    if snapshot not in (1, 2, 3):
        raise ValueError("snapshot must be 1, 2 or 3")
    entries = manifest_entries(folder)
    live = set(live_ids)
    selected = {key: item for key, item in entries.items() if key.startswith(f"test_qrels_snapshot_{snapshot}_")
                or (snapshot == 1 and "training_2026_qrels" in key)}
    sources = {}
    for key, item in selected.items():
        path = folder / item["cache_path"]
        if checksum(path, "md5") != item["expected_md5"]:
            raise ValueError("qrels checksum mismatch during membership audit")
        sources[key] = (path.read_bytes(), checksum(path))
    item = entries["llm-synthetic-qrels"]
    archive = folder / item["cache_path"]
    if checksum(archive, "md5") != item["expected_md5"]:
        raise ValueError("LLM archive checksum mismatch during membership audit")
    with zipfile.ZipFile(archive) as z:
        for model, prefix in MODELS.items():
            name = f"llm_synthetic_qrels/{prefix}-umbrella_zeroshot_no_desc_no_narrative-snapshot-{snapshot}.qrels.txt"
            if z.getinfo(name).file_size > 16 << 20:
                raise ValueError("selected qrels member exceeds size budget")
            raw = z.read(name)
            sources[model] = raw, hashlib.sha256(raw).hexdigest()
    result = {}
    for key, (raw, sha) in sources.items():
        qrels = parse_qrels(raw.decode("utf-8-sig"))
        missing = [(q, d, rel) for q, rows in qrels.items() for d, rel in rows.items() if d not in live]
        result[key] = {"qrels_sha256": sha, "judgment_rows": sum(map(len, qrels.values())),
                       "missing_document_ids": sorted({d for _, d, _ in missing}),
                       "missing_pairs": len(missing), "missing_positive_pairs": sum(rel > 0 for _, _, rel in missing)}
    return result


def audit_release(folder, output):
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError(output)
    entries = manifest_entries(folder)
    files = {}
    for key, item in entries.items():
        path = folder / item["cache_path"]
        observed = checksum(path, "md5")
        if observed != item["expected_md5"]:
            raise ValueError(f"release checksum mismatch: {path.name}")
        files[key] = {"file": path.name, "bytes": path.stat().st_size, "md5": observed,
                      "sha256": checksum(path), "source": sanitized_url(item["url"])}
    train = read_queries(folder / entries[TRAIN]["cache_path"])
    test = read_queries(folder / entries[TEST]["cache_path"])
    if set(train) & set(test):
        raise ValueError("training/test query IDs overlap")
    # Freeze a provisional development/calibration split without consulting any qrels.
    ordered_train = sorted(train, key=lambda q: hashlib.sha256(("longeval-role-v1:" + q).encode()).digest())
    roles = {"development": sorted(ordered_train[:len(train) // 2]),
             "calibration": sorted(ordered_train[len(train) // 2:]), "evaluation": sorted(test)}
    normalized = lambda text: " ".join(text.casefold().split())
    role_texts = {role: {normalized((test if role == "evaluation" else train)[q]) for q in ids}
                  for role, ids in roles.items()}
    text_overlap = {f"{a}/{b}": len(role_texts[a] & role_texts[b]) for a, b in
                    (("development", "calibration"), ("development", "evaluation"), ("calibration", "evaluation"))}
    if any(text_overlap.values()):
        raise ValueError("normalized query texts overlap across proposed roles")
    judgments = {}
    for key, item in entries.items():
        if "qrels" in key and key != "llm-synthetic-qrels":
            parsed = parse_qrels((folder / item["cache_path"]).read_text(encoding="utf-8-sig"))
            judgments[key] = qrel_summary(parsed, train if "training" in key else test)
    members = {}
    archive = folder / entries["llm-synthetic-qrels"]["cache_path"]
    with zipfile.ZipFile(archive) as z:
        # Read only exact selected members in memory. Never extract remote paths to disk.
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member names")
        for model, prefix in MODELS.items():
            for snapshot in (1, 2, 3):
                name = f"llm_synthetic_qrels/{prefix}-umbrella_zeroshot_no_desc_no_narrative-snapshot-{snapshot}.qrels.txt"
                info = z.getinfo(name)
                if info.file_size > 16 << 20:
                    raise ValueError("selected qrels member exceeds size budget")
                raw = z.read(name)
                key = f"{model}/snapshot-{snapshot}"
                members[key] = {"member": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                judgments[key] = qrel_summary(parse_qrels(raw.decode("utf-8-sig")), test)
    panels = {}
    for family in ("dctr", "raw", *MODELS):
        keys = ([f"test_qrels_snapshot_{s}_{family}" for s in (1, 2, 3)] if family in ("dctr", "raw") else
                [f"{family}/snapshot-{s}" for s in (1, 2, 3)])
        ids = sorted(set.intersection(*(set(judgments[k]["positive_query_ids"]) for k in keys)))
        panels[family] = {"count": len(ids), "query_ids": ids,
                          "selection": "retrospective common-positive; not available for online tuning"}
    report = {"schema": 1, "status": "completed", "kind": "real released query/qrel audit; no retrieval",
              "python": sys.version, "audit_source_sha256": checksum(__file__),
              "integration_revision": REVISION, "manifest_sha256": MANIFEST_SHA256,
              "files": files, "selected_archive_members": members,
              "queries": {"train": len(train), "test": len(test), "id_overlap": 0,
                          "normalized_text_overlap_by_role": text_overlap},
              "provisional_query_roles": roles, "split_rule": "SHA256(longeval-role-v1:ID), first half development",
              "judgments": judgments, "retrospective_panels": panels,
              "limits": ["No corpus membership/eligibility audit", "No vector payload audit",
                         "Official evaluation population not established", "Role split provisional until representation audit"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report
