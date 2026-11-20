import hashlib
import json

import pytest

from research.longeval import encoder


def test_local_model_files_are_checked_against_pins_not_receipt(tmp_path, monkeypatch):
    config = b'{"model_type":"test"}'
    weights = b"safetensor placeholder"
    pins = {"config.json": (len(config), encoder.git_blob_hash(config)),
            "tokenizer.json": (len(config), hashlib.sha256(config).hexdigest()),
            "model.safetensors": (len(weights), hashlib.sha256(weights).hexdigest())}
    monkeypatch.setattr(encoder, "FILES", pins)
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "tokenizer.json").write_bytes(config)
    (tmp_path / "model.safetensors").write_bytes(weights)
    assert encoder.verify_model(tmp_path)["revision"] == encoder.REVISION
    (tmp_path / "model.safetensors").write_bytes(b"x" * len(weights))
    with pytest.raises(ValueError, match="identity mismatch"):
        encoder.verify_model(tmp_path)


def test_diagnostic_input_text_identity_and_unique_ids(tmp_path):
    path = tmp_path / "sample.jsonl"
    row = {"id": "a", "text": "unaltered document", "text_sha256": hashlib.sha256(b"unaltered document").hexdigest()}
    path.write_text(json.dumps(row) + "\n")
    assert encoder.read_records(path) == [row]
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        encoder.read_records(path)
    row["text"] = "changed"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="text changed"):
        encoder.read_records(path)
