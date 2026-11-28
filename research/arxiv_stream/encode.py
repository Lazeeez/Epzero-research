"""Screen A1: encode the prepared arXiv corpus and queries with BGE-base-en-v1.5 (docs/screen-a1-arxiv-stream.md).

    python -m research.arxiv_stream.encode MODEL_DIR PREPARED OUTPUT --device cuda [--batch-size 256]

The model, its pinned revision and the encoder (CLS pooling, float32 L2 normalisation, 512 tokens, right
truncation) are the LongEval BGE arm's (`research/longeval/bge_encoder.py`, imported unchanged). Abstracts are
encoded without an instruction prefix, queries included: a held-out abstract searches for abstracts like it.
PREPARED is `prepare.py`'s folder; its content checksums are checked first. OUTPUT receives `corpus.npy` (in
insertion order), `queries-new.npy`, `queries-uniform.npy`, the ID lists and `report.json`. A deterministic
sample of 1,000 abstracts is encoded again on the CPU one at a time and must agree to a cosine of 0.9999.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from research.longeval.bge_encoder import (DIM, MAX_LENGTH, MODEL, PROGRESS_SECONDS, REVISION, VERIFY_MIN_COSINE,
                                           VERIFY_SAMPLE, Encoder, min_cosine, throughput_check, verify_model)
from research.longeval.release import checksum
from research.longeval.storage import atomic_json

SETS = ("corpus", "queries-new", "queries-uniform")
CHUNK = 50_000


def read_set(prepared, name, expected):
    """IDs and abstracts of one prepared file, refusing it if its content checksum differs from the manifest's."""
    digest, ids, texts = hashlib.sha256(), [], []
    with gzip.open(Path(prepared) / f"{name}.jsonl.gz", "rb") as f:
        for line in f:
            digest.update(line)
            r = json.loads(line)
            ids.append(r["id"])
            texts.append(r["abstract"])
    if digest.hexdigest() != expected:
        raise ValueError(f"{name}: content differs from the prepared manifest")
    return ids, texts


def verification_rows(ids, count=VERIFY_SAMPLE):
    keyed = sorted(range(len(ids)), key=lambda i: hashlib.sha256(("arxiv-a1-verify-v1:" + ids[i]).encode()).digest())
    return np.sort(np.array(keyed[:count]))


def encode(model_folder, prepared, output, *, device="cuda", batch_size=256):
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((prepared / "manifest.json").read_text())
    report = {"status": "running", "design": "docs/screen-a1-arxiv-stream.md", "model": MODEL, "revision": REVISION,
              "pooling": "CLS", "normalisation": "float32 L2", "max_length": MAX_LENGTH, "truncation": "right",
              "instruction": None, "device": device, "batch_size": batch_size, "source_sha256": checksum(__file__),
              "prepared_manifest_sha256": checksum(prepared / "manifest.json")}
    atomic_json(output / "report.json", report)
    try:
        report["model_identity"] = verify_model(model_folder)
        report["packages"] = {n: importlib.metadata.version(n)
                              for n in ("torch", "transformers", "numpy", "safetensors", "tokenizers")}
        data = {name: read_set(prepared, name, manifest["content_sha256"][name]) for name in SETS}
        report["rows"] = {name: len(data[name][0]) for name in SETS}
        if report["rows"]["corpus"] != manifest["corpus"]:
            raise ValueError("corpus size differs from the manifest")
        encoder = Encoder(model_folder, device)
        report["model_device"] = str(next(encoder.model.parameters()).device)
        if device == "cuda":
            report["gpu"] = encoder.torch.cuda.get_device_name(0)
        atomic_json(output / "report.json", report)
        print(json.dumps({"stage": "model", "device": report["model_device"], "rows": report["rows"]}), flush=True)

        ids, texts = data["corpus"]
        counts = encoder.token_counts(texts, cap=MAX_LENGTH + 1)
        report["tokens"] = {"mean_capped": float(counts.mean()), "p50_capped": float(np.median(counts)),
                            "truncated": int((counts > MAX_LENGTH).sum()),
                            "truncated_share": float((counts > MAX_LENGTH).mean())}
        atomic_json(output / "report.json", report)
        started, shown = time.perf_counter(), [0.0]

        def progress(done):
            elapsed = time.perf_counter() - started
            rate = throughput_check(device, done, elapsed)
            if elapsed - shown[0] >= PROGRESS_SECONDS:
                shown[0] = elapsed
                print(json.dumps({"encoding": done, "of": len(ids), "seconds": round(elapsed, 1),
                                  "documents_per_second": round(rate, 1)}), flush=True)
        vectors = np.lib.format.open_memmap(output / "corpus.npy", mode="w+", dtype=np.float32, shape=(len(ids), DIM))
        for start in range(0, len(ids), CHUNK):
            vectors[start:start + CHUNK] = encoder.encode(texts[start:start + CHUNK], batch_size,
                                                          progress=lambda done, base=start: progress(base + done))
        vectors.flush()
        report["corpus_seconds"] = time.perf_counter() - started
        for name in SETS[1:]:
            np.save(output / f"{name}.npy", encoder.encode(data[name][1], batch_size))
        for name in SETS:
            with gzip.GzipFile(output / f"{name}-ids.json.gz", "wb", mtime=0) as f:
                f.write(json.dumps(data[name][0]).encode())

        rows = verification_rows(ids)
        checker, setting = (Encoder(model_folder, "cpu"), "cpu") if device == "cuda" else (encoder, "same device")
        again = np.stack([checker.encode([texts[i]], 1)[0] for i in rows])
        report["verification"] = {"abstracts": len(rows), "batch_size": 1, "second_setting": setting,
                                  "min_cosine": min_cosine(np.asarray(vectors[rows]), again), "required": VERIFY_MIN_COSINE}
        if report["verification"]["min_cosine"] < VERIFY_MIN_COSINE:
            raise ValueError("re-encoded abstracts differ beyond the declared tolerance")
        report["files"] = {p.name: checksum(p) for p in sorted(output.glob("*")) if p.name != "report.json"}
        report["status"] = "completed"
        atomic_json(output / "report.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error)[:500])
        atomic_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    a = parser.parse_args()
    print(json.dumps(encode(a.model, a.prepared, a.output, device=a.device, batch_size=a.batch_size)["status"]))


if __name__ == "__main__":
    main()
