import gzip
import json

import numpy as np
import pytest

from research.arxiv_stream import prepare


def harvest(tmp_path, rows, done=True):
    with gzip.open(tmp_path / "page-00000.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (tmp_path / "state.json").write_text(json.dumps({"pages": 1, "done": done}))
    return tmp_path


def row(i, created, abstract="text"):
    return {"id": f"{i:07d}", "created": created, "categories": "cs.DB", "abstract": abstract}


def test_order_duplicates_and_empty_abstracts(tmp_path):
    rows = [row(3, "2020-01-02"), row(1, "2020-01-02"), row(2, "2019-05-05"), row(1, "2021-01-01"),
            row(4, "2020-01-01", abstract=""), row(5, "not-a-date")]
    out, stats = prepare.read_harvest(harvest(tmp_path, rows))
    assert [r["id"] for r in out] == ["0000002", "0000001", "0000003"]   # by date, then ID; first duplicate kept
    assert stats == {"records": 6, "no_abstract": 1, "duplicates": 1, "bad_date": 1, "pages": 1, "complete": True}


def test_partial_harvest_is_refused_unless_asked(tmp_path):
    folder = harvest(tmp_path, [row(1, "2020-01-01")], done=False)
    with pytest.raises(ValueError, match="not complete"):
        prepare.read_harvest(folder)
    assert len(prepare.read_harvest(folder, allow_partial=True)[0]) == 1


def test_split_is_disjoint_deterministic_and_recent_where_asked():
    rows = [row(i, f"{2000 + i // 400:04d}-06-01") for i in range(10_000)]   # years 2000-2024, 400 a year
    new, uniform, corpus, cutoff = prepare.split(rows, per_set=300, recent_days=800)
    assert len(set(new) | set(uniform) | set(corpus)) == len(rows)
    assert not (set(new) & set(uniform)) and not (set(new) & set(corpus)) and not (set(uniform) & set(corpus))
    assert all(rows[i]["created"] > cutoff for i in new)
    assert list(corpus) == sorted(corpus)
    again = prepare.split(rows, per_set=300, recent_days=800)
    assert all(np.array_equal(x, y) for x, y in zip((new, uniform, corpus), again[:3]))


def test_split_refuses_too_few_recent_abstracts():
    rows = [row(i, "2000-01-01") for i in range(100)] + [row(100, "2024-01-01")]
    with pytest.raises(ValueError, match="only 1 abstracts"):
        prepare.split(rows, per_set=10)
