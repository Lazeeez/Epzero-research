"""Harvest arXiv's metadata (CC0) through OAI-PMH: one gzip JSON-lines file per page, resumable.

    python -I -m research.arxiv_stream.harvest OUT_DIR [--delay 3] [--max-pages N]

Each record keeps the arXiv ID, the first version's date, the categories and the abstract; nothing
else. The state (the next resumption token and the page count) is written after every page, so a
rerun continues where it stopped. The server's flow control is honoured: a 503 with Retry-After waits
that long. Pages are parsed as data with the standard library's XML parser, which resolves no external
entities.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://oaipmh.arxiv.org/oai"
AGENT = "ann-research-metadata-harvest/1.0"
OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"


def fetch(params, *, attempts=8):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 503 or attempt == attempts - 1:
                raise
            time.sleep(int(e.headers.get("Retry-After", "30") or 30))
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("unreachable")


def records(page):
    """(records, resumption token or None, complete list size or None) of one ListRecords page."""
    root = ET.fromstring(page)
    error = root.find(f"{OAI}error")
    if error is not None:
        raise RuntimeError(f"OAI-PMH error {error.get('code')}: {error.text}")
    out = []
    for rec in root.iter(f"{OAI}record"):
        header = rec.find(f"{OAI}header")
        if header is not None and header.get("status") == "deleted":
            continue
        meta = rec.find(f"{OAI}metadata/{ARXIV}arXiv")
        if meta is None:
            continue
        text = lambda tag: " ".join((meta.findtext(f"{ARXIV}{tag}") or "").split())
        out.append({"id": text("id"), "created": text("created"), "categories": text("categories"),
                    "abstract": text("abstract")})
    token = root.find(f".//{OAI}resumptionToken")
    size = token.get("completeListSize") if token is not None else None
    return out, (token.text.strip() if token is not None and token.text and token.text.strip() else None), size


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    parser.add_argument("--max-pages", type=int, default=0, help="stop after this many pages this run (0: all)")
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    state_path = a.out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"pages": 0, "token": None, "done": False}
    if state["done"]:
        print("already complete:", state["pages"], "pages")
        return
    done_this_run = 0
    while True:
        params = {"verb": "ListRecords", "resumptionToken": state["token"]} if state["token"] else \
                 {"verb": "ListRecords", "metadataPrefix": "arXiv"}
        rows, token, size = records(fetch(params))
        page = state["pages"]
        part = a.out / f"page-{page:05d}.jsonl.gz"
        tmp = part.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(part)
        state = {"pages": page + 1, "token": token, "done": token is None, "list_size": size or state.get("list_size")}
        tmp_state = state_path.with_suffix(".tmp")
        tmp_state.write_text(json.dumps(state))
        tmp_state.replace(state_path)
        done_this_run += 1
        print(f"page {page}: {len(rows)} records, list size {state['list_size']}", flush=True)
        if state["done"] or (a.max_pages and done_this_run >= a.max_pages):
            break
        time.sleep(a.delay)


if __name__ == "__main__":
    main()
