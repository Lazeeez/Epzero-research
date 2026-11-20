"""Revision-pinned, offline Qwen reconstruction for document compatibility diagnostics.

These outputs are not accepted research query vectors. No remote model Python is loaded, and
no precision conversion or truncation is implicit in the run contract.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from urllib.request import urlopen

from .release import checksum
from .storage import atomic_json, fetch_archive

MODEL = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
# Immutable Git blob identities for small files; full LFS SHA-256 for the weight shards.
FILES = {
    "config.json": (727, "8b4b87fc69023e7a224eb6563753aaf3223d8b98"),
    "config_sentence_transformers.json": (215, "76aef3ade63553ebb698fe3c2a3264040ed093f8"),
    "modules.json": (349, "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
    "1_Pooling/config.json": (313, "81de5602eacbce382009c5af7a23085871801d8f"),
    "tokenizer_config.json": (7256, "df3a9d96759529ca1006eb6db024bbb099a97578"),
    "tokenizer.json": (11422947, "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d"),
    "vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569"),
    "merges.txt": (1671853, "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
    "model.safetensors.index.json": (30431, "3d736ef26714eee0abde3e05104ee1b3ec26c974"),
    "model-00001-of-00002.safetensors": (4965826464, "e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8"),
    "model-00002-of-00002.safetensors": (3077765624, "ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1"),
}


def git_blob_hash(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def verify_model(folder):
    folder = Path(folder)
    result = {}
    for name, (size, expected) in FILES.items():
        path = folder / name
        if path.stat().st_size != size:
            raise ValueError(f"model file size mismatch: {name}")
        sha256 = checksum(path)
        actual = sha256 if len(expected) == 64 else git_blob_hash(path.read_bytes())
        if actual != expected:
            raise ValueError(f"model file identity mismatch: {name}")
        result[name] = {"bytes": size, "sha256": sha256}
    return {"model": MODEL, "revision": REVISION, "files": result}


def fetch_model(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, (size, expected) in FILES.items():
        path = folder / name
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        if len(expected) == 64:
            fetch_archive(url, path, expected, size, progress=lambda value: print(json.dumps(value), flush=True))
        elif not path.exists():
            with urlopen(url, timeout=60) as response:
                raw = response.read(size + 1)
            if len(raw) != size or git_blob_hash(raw) != expected:
                raise ValueError(f"model configuration identity mismatch: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as target:
                target.write(raw)
    result = verify_model(folder)
    atomic_json(folder / "model-identity.json", result)
    return result


def read_records(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(not isinstance(x.get("id"), str) or not isinstance(x.get("text"), str) or not x["text"] for x in rows):
        raise ValueError("document diagnostic requires nonempty IDs/text")
    if len({x["id"] for x in rows}) != len(rows):
        raise ValueError("duplicate diagnostic document IDs")
    for row in rows:
        if "text_sha256" in row and hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("diagnostic document text changed")
    return rows


def encode_documents(model_folder, records_path, output, *, dtype="bfloat16", max_length=8192, threads=4, device="cpu"):
    """One document per batch, declared device, last-token pooling, float32 L2 normalization.

    Reject over-length inputs instead of silently truncating. A caller may explicitly supply
    another length in a fresh run. No query prompt or chat template is applied to documents.
    """
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    if dtype not in ("bfloat16", "float32") or not 0 < max_length <= 32768 or threads <= 0 or device not in ("cpu", "cuda"):
        raise ValueError("invalid diagnostic encoder settings")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "purpose": "document encoder compatibility diagnostic only",
              "dtype": dtype, "device": device, "threads": threads, "max_length": max_length,
              "batch_size": 1, "pooling": "last attention-masked token", "normalization": "float32 L2",
              "attention": "eager", "truncation": "reject", "query_prompt": None,
              "source_sha256": checksum(__file__), "rows": []}
    atomic_json(output / "encoding.json", report)
    try:
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; refusing device fallback")
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report["gpu"] = torch.cuda.get_device_name(0)
            report["tf32"] = False
        report["model_identity"] = verify_model(model_folder)
        report["input_sha256"] = checksum(records_path)
        report["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "numpy", "safetensors", "tokenizers")}
        rows = read_records(records_path)
        torch.set_num_threads(threads)
        # Torch allows this once per process; a second encode in the same process keeps the value.
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(11)
        tokenizer = AutoTokenizer.from_pretrained(str(model_folder), local_files_only=True, trust_remote_code=False, padding_side="left")
        batches = [tokenizer(row["text"], return_tensors="pt", truncation=False) for row in rows]
        if any(batch["input_ids"].shape[1] > max_length for batch in batches):
            raise ValueError("diagnostic input exceeds declared maximum length")
        started = time.perf_counter()
        model = AutoModel.from_pretrained(str(model_folder), local_files_only=True, trust_remote_code=False,
                                         use_safetensors=True, dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
                                         attn_implementation="eager").to(device).eval()
        if device == "cuda":
            torch.cuda.synchronize()
        report["model_load_seconds"] = time.perf_counter() - started
        vectors = []
        with torch.inference_mode():
            for row, batch in zip(rows, batches):
                batch = {key: value.to(device) for key, value in batch.items()}
                started = time.perf_counter()
                hidden = model(**batch, use_cache=False).last_hidden_state
                # A one-record unpadded batch has its final nonpadding token at the final column.
                vector = functional.normalize(hidden[:, -1, :].float(), p=2, dim=1).cpu().numpy()[0]
                if vector.shape != (2560,) or not np.isfinite(vector).all():
                    raise ValueError("invalid reconstructed embedding")
                vectors.append(vector.copy())
                report["rows"].append({"id": row["id"], "tokens": int(batch["input_ids"].shape[1]), "seconds": time.perf_counter() - started})
                atomic_json(output / "encoding.json", report)
                print(json.dumps(report["rows"][-1]), flush=True)
        np.save(output / "vectors.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False)
        atomic_json(output / "ids.json", [row["id"] for row in rows])
        report.update(status="completed", vectors_sha256=checksum(output / "vectors.npy"), ids_sha256=checksum(output / "ids.json"))
        atomic_json(output / "encoding.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        atomic_json(output / "encoding.json", report)
        raise
