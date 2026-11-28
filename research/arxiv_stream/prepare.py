"""Screen A1: the arXiv corpus in submission order and its two held-out query sets (docs/screen-a1-arxiv-stream.md).

    python -I -m research.arxiv_stream.prepare HARVEST_DIR OUT_DIR [--allow-partial]

Reads the harvester's pages, drops records without an abstract, keeps the first of duplicate IDs, orders by
the first version's date and then the arXiv ID, and holds out 10,000 abstracts fixed by seed 2027: 5,000
from the last 12 months of the harvest (new-content queries), then 5,000 uniformly from the rest of the
range (uniform queries). Writes `corpus.jsonl.gz` (in insertion order), `queries-new.jsonl.gz`,
`queries-uniform.jsonl.gz` and `manifest.json`. arXiv's metadata is CC0; the files stay outside the
repository all the same, as only aggregates are committed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 2027
PER_SET = 5_000
RECENT_DAYS = 365


def read_harvest(folder, *, allow_partial=False):
    folder = Path(folder)
    state = json.loads((folder / "state.json").read_text())
    if not state.get("done") and not allow_partial:
        raise ValueError(f"the harvest is not complete ({state.get('pages')} pages); pass --allow-partial to test")
    pages = sorted(folder.glob("page-*.jsonl.gz"))
    if len(pages) != state["pages"]:
        raise ValueError(f"{len(pages)} page files but the state records {state['pages']}")
    seen, rows, stats = set(), [], {"records": 0, "no_abstract": 0, "duplicates": 0, "bad_date": 0}
    for page in pages:
        with gzip.open(page, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                stats["records"] += 1
                if not r["abstract"]:
                    stats["no_abstract"] += 1
                    continue
                if r["id"] in seen:
                    stats["duplicates"] += 1
                    continue
                try:
                    dt.date.fromisoformat(r["created"])
                except ValueError:
                    stats["bad_date"] += 1
                    continue
                seen.add(r["id"])
                rows.append(r)
    rows.sort(key=lambda r: (r["created"], r["id"]))
    return rows, {**stats, "pages": len(pages), "complete": bool(state.get("done"))}


def split(rows, *, seed=SEED, per_set=PER_SET, recent_days=RECENT_DAYS):
    """Indices (into `rows`, ordered) of the new-content queries, the uniform queries and the corpus."""
    last = dt.date.fromisoformat(rows[-1]["created"])
    cutoff = (last - dt.timedelta(days=recent_days)).isoformat()
    recent = np.array([i for i, r in enumerate(rows) if r["created"] > cutoff])
    if len(recent) < 2 * per_set:
        raise ValueError(f"only {len(recent)} abstracts in the last {recent_days} days")
    rng = np.random.default_rng(seed)
    new = np.sort(rng.choice(recent, per_set, replace=False))
    rest = np.setdiff1d(np.arange(len(rows)), new)
    uniform = np.sort(rng.choice(rest, per_set, replace=False))
    corpus = np.setdiff1d(rest, uniform)
    return new, uniform, corpus, cutoff


def write_rows(path, rows, indices):
    digest = hashlib.sha256()
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        for i in indices:
            line = (json.dumps(rows[i], ensure_ascii=False, sort_keys=True) + "\n").encode()
            digest.update(line)
            f.write(line)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("harvest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    a = parser.parse_args()
    rows, stats = read_harvest(a.harvest, allow_partial=a.allow_partial)
    new, uniform, corpus, cutoff = split(rows)
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = {"design": "docs/screen-a1-arxiv-stream.md", "seed": SEED, "harvest": stats, "abstracts": len(rows),
                "first_created": rows[0]["created"], "last_created": rows[-1]["created"], "recent_after": cutoff,
                "order": "first version's date, then arXiv ID",
                "corpus": len(corpus), "queries_new": len(new), "queries_uniform": len(uniform),
                "content_sha256": {"corpus": write_rows(a.out / "corpus.jsonl.gz", rows, corpus),
                                   "queries-new": write_rows(a.out / "queries-new.jsonl.gz", rows, new),
                                   "queries-uniform": write_rows(a.out / "queries-uniform.jsonl.gz", rows, uniform)}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "content_sha256"}, indent=2))


if __name__ == "__main__":
    main()
