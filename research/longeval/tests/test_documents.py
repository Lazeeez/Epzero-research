import hashlib
import io
import json
import zipfile

import pytest

from research.longeval import documents
from research.longeval.documents import audit_document_zip, compare_texts, read_document


def test_default_text_preserves_exact_space_and_missing_abstract():
    assert read_document({"id": "001", "title": " title ", "abstract": " abstract "}) == ("001", " title   abstract ")
    assert read_document({"id": "001", "title": "title", "abstract": None}) == ("001", "title")
    with pytest.raises(ValueError, match="identifier"):
        read_document({"id": 1, "title": "title"})


def test_text_audit_counts_empty_and_conflicting_duplicates(tmp_path):
    path = tmp_path / "documents.zip"
    records = [{"id": "a", "title": "A", "abstract": "text"}, {"id": "b", "title": "B"},
               {"id": "a", "title": "A", "abstract": "text"}, {"id": "a", "title": "changed"},
               {"id": "empty", "title": "", "abstract": ""}]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("snapshot/documents/docs.jsonl", "\n".join(map(json.dumps, records)))
        archive.writestr("unrelated.jsonl", "not a document")
    result, hashes, sample = audit_document_zip(path, "snapshot/documents/", sample_size=2)
    assert result["rows"] == 5
    assert result["eligible_unique_ids"] == 2
    assert result["duplicate_eligible_rows"] == 2
    assert result["conflicting_duplicate_rows"] == 1
    assert result["empty_text_rows"] == 1
    assert hashes["a"] == hashlib.sha256(b"A text").digest()
    assert {x["id"] for x in sample} == {"a", "b"}


def test_changed_shared_text_and_removals_fail_contract():
    assert compare_texts({"a": b"a"}, {"b": b"b", "a": b"a"})["fixed_text_insert_only_valid"]
    result = compare_texts({"a": b"a", "b": b"b"}, {"a": b"changed", "c": b"c"})
    assert result["changed_shared_texts"] == 1 and result["removed_ids"] == 1
    assert not result["fixed_text_insert_only_valid"]


def test_document_sample_independent_of_file_row_order(tmp_path):
    records = [{"id": str(i), "title": str(i)} for i in range(30)]
    samples = []
    for name, rows in [("first", records), ("second", list(reversed(records)))]:
        path = tmp_path / (name + ".zip")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("documents/one.jsonl", "\n".join(map(json.dumps, rows)))
        _, _, sample = audit_document_zip(path, "documents/", sample_size=5)
        samples.append([(x["id"], x["text_sha256"]) for x in sample])
    assert samples[0] == samples[1]


def test_document_ranges_resume_and_verify_original_checksum(tmp_path, monkeypatch):
    data = b"a pinned archive"
    md5 = hashlib.md5(data).hexdigest()
    url = "https://example.org/data?public-access=value"
    path = tmp_path / "data.zip"
    path.with_name(path.name + ".part").write_bytes(data[:3])
    documents.atomic_json(path.with_suffix(".download.json"), {"identity": {
        "source_url_sha256": hashlib.sha256(url.encode()).hexdigest(), "md5": md5, "bytes": len(data)}})
    calls = []
    def respond(request, **kwargs):
        start, end = map(int, request.get_header("Range")[6:].split("-"))
        calls.append(start)
        response = io.BytesIO(data[start:end + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
        return response
    monkeypatch.setattr(documents, "urlopen", respond)
    documents.fetch_zip(url, path, md5, len(data), request_bytes=5)
    assert calls[0] == 3
    assert path.read_bytes() == data
    assert "public-access" not in path.with_suffix(".download.json").read_text()
