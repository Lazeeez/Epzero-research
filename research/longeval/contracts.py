"""Fail-closed input contracts for a small, in-memory, insert-only pilot."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token(value: str) -> str:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError(f"IDs must be nonempty strings without whitespace: {value!r}")
    return value


@dataclass(frozen=True)
class Table:
    ids: tuple[str, ...]
    texts: tuple[str, ...]
    vectors: np.ndarray

    def __post_init__(self):
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("table IDs must be nonempty and unique")
        for value in self.ids:
            token(value)
        v = self.vectors
        if v.dtype != np.dtype("float32") or v.ndim != 2 or v.shape[0] != len(self.ids) or v.shape[1] == 0:
            raise ValueError("one canonical float32 vector per ID is required")
        if len(self.texts) != len(self.ids) or not all(isinstance(t, str) for t in self.texts):
            raise ValueError("one text string per ID is required")
        if not np.isfinite(v).all():
            raise ValueError("vectors must be finite")
        # Own immutable storage: callers cannot mutate the validated input via an alias.
        v = np.array(v, dtype="<f4", order="C", copy=True)
        v.setflags(write=False)
        object.__setattr__(self, "vectors", v)

    def records(self):
        return {key: (digest(text), hashlib.sha256(row.tobytes()).hexdigest())
                for key, text, row in zip(self.ids, self.texts, self.vectors)}

    def manifest(self):
        return {"count": len(self.ids), "dim": self.vectors.shape[1],
                "live_set_sha256": digest(sorted(self.ids)),
                "records_sha256": digest(self.records()), "row_order_sha256": digest(self.ids)}


def delta(previous: Table | None, current: Table) -> tuple[str, ...]:
    old = {} if previous is None else previous.records()
    new = current.records()
    if previous is not None and previous.vectors.shape[1] != current.vectors.shape[1]:
        raise ValueError("snapshot dimension changed")
    if old.keys() - new.keys():
        raise ValueError("insert-only snapshot removed IDs")
    if any(old[key] != new[key] for key in old):
        raise ValueError("shared-ID text or canonical vector changed")
    # Within-interval chronology is unavailable: lexical ordering is an explicit control.
    return tuple(sorted(new.keys() - old.keys()))


def validate_roles(roles: dict[str, Table]):
    if set(roles) != {"development", "calibration", "evaluation"}:
        raise ValueError("declare development, calibration and evaluation queries")
    seen_ids, seen_texts, seen_vectors = set(), set(), set()
    for table in roles.values():
        texts = {digest(text) for text in table.texts}
        vectors = {hashlib.sha256(v.tobytes()).hexdigest() for v in table.vectors}
        if seen_ids.intersection(table.ids) or seen_texts & texts or seen_vectors & vectors:
            raise ValueError("query-role leakage: overlapping ID, text or vector")
        seen_ids.update(table.ids)
        seen_texts.update(texts)
        seen_vectors.update(vectors)


def read_qrels(path: Path):
    qrels = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{number}: expected four TREC qrel columns")
        qid, _, doc, raw = fields
        token(qid)
        token(doc)
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError("qrel relevance must be a finite integer")
        if doc in qrels.get(qid, {}):
            raise ValueError("duplicate judgment")
        qrels.setdefault(qid, {})[doc] = int(value)
    return qrels


@dataclass(frozen=True)
class Dataset:
    spec: dict
    snapshots: tuple[Table, ...]
    queries: dict[str, Table]
    qrels: tuple[dict, ...]
    additions: tuple[tuple[str, ...], ...]
    input_hashes: dict[str, str]

    @property
    def ordered_ids(self):
        return tuple(key for batch in self.additions for key in batch)

    def manifest(self):
        return {"spec": self.spec, "files": self.input_hashes,
                "snapshots": [s.manifest() for s in self.snapshots],
                "queries": {role: t.manifest() for role, t in self.queries.items()},
                "delta_counts": [len(x) for x in self.additions],
                "label_mapping_sha256": digest(self.ordered_ids)}


def load_dataset(path: Path) -> Dataset:
    path = path.resolve()
    root = path.parent
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != 1 or not isinstance(spec.get("synthetic"), bool):
        raise ValueError("expected schema=1 and explicit synthetic flag")
    if not isinstance(spec.get("release"), str) or not spec["release"].strip():
        raise ValueError("dataset release identity required")
    representation = spec["representation"]
    required = {"model", "revision", "preprocessing", "normalization", "dtype", "dim", "metric"}
    if set(representation) != required or any(v is None or v == "" for v in representation.values()):
        raise ValueError("complete representation identity required")
    if representation["metric"] not in ("l2", "ip", "cosine") or representation["dtype"] != "float32":
        raise ValueError("unsupported representation")
    if representation["normalization"] not in ("none", "unit"):
        raise ValueError("normalization must be none or unit")
    if representation["metric"] == "cosine" and representation["normalization"] != "unit":
        raise ValueError("cosine requires canonical unit-normalized inputs")
    hashes = {path.name: file_hash(path)}

    def source(name):
        p = (root / name).resolve()
        if not p.is_relative_to(root):
            raise ValueError("dataset files must be inside the manifest directory")
        hashes[p.relative_to(root).as_posix()] = file_hash(p)
        return p

    def table(entry):
        records = [json.loads(line) for line in source(entry["records"]).read_text(encoding="utf-8").splitlines()]
        out = Table(tuple(r["id"] for r in records), tuple(r["text"] for r in records),
                    np.load(source(entry["vectors"]), allow_pickle=False))
        if out.vectors.shape[1] != representation["dim"]:
            raise ValueError("representation dimension mismatch")
        if representation["normalization"] == "unit":
            norms = np.linalg.norm(out.vectors.astype(np.float64), axis=1)
            if not np.allclose(norms, 1, rtol=0, atol=1e-6):
                raise ValueError("canonical vectors must be unit normalized; zero vectors are invalid")
        return out

    if len(spec["snapshots"]) != 3:
        raise ValueError("this pilot requires exactly three snapshots")
    names = [token(s["name"]) for s in spec["snapshots"]]
    if len(set(names)) != len(names):
        raise ValueError("snapshot names must be unique")
    snapshots = tuple(table(entry) for entry in spec["snapshots"])
    additions = tuple(delta(snapshots[i - 1] if i else None, s) for i, s in enumerate(snapshots))
    queries = {role: table(entry) for role, entry in spec["queries"].items()}
    validate_roles(queries)
    qrels = tuple(read_qrels(source(entry["qrels"])) for entry in spec["snapshots"])
    query_ids = {key for t in queries.values() for key in t.ids}
    for snapshot, judgments in zip(snapshots, qrels):
        if judgments.keys() - query_ids or any(set(j) - set(snapshot.ids) for j in judgments.values()):
            raise ValueError("qrels refer to unknown query or nonlive document IDs")
    return Dataset(spec, snapshots, queries, qrels, additions, hashes)
