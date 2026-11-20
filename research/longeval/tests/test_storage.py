import hashlib
import io
import json
import re
import tarfile

import lz4.frame
import pytest

from research.longeval import storage


class Response(io.BytesIO):
    def __init__(self, data, start, end, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


def mock_ranges(monkeypatch, data, calls):
    def serve(request, **kwargs):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        calls.append((start, end))
        return Response(data[start:end + 1], start, end, len(data))
    monkeypatch.setattr(storage, "urlopen", serve)


def test_full_checksum_and_identity_bound_resume(tmp_path, monkeypatch):
    data = b"abcdefghijklmnop"
    sha = hashlib.sha256(data).hexdigest()
    path = tmp_path / "archive.lz4"
    path.with_name(path.name + ".part").write_bytes(data[:5])
    identity = {"url": "https://example.org/archive", "sha256": sha, "bytes": len(data)}
    storage.atomic_json(path.with_name(path.name + ".download.json"), {"identity": identity, "status": "downloading"})
    calls = []
    mock_ranges(monkeypatch, data, calls)
    storage.fetch_archive(identity["url"], path, sha, len(data), request_bytes=4)
    assert calls == [(5, 8), (9, 12), (13, 15)]
    assert path.read_bytes() == data
    assert not path.with_name(path.name + ".part").exists()
    path.write_bytes(b"x" * len(data))
    with pytest.raises(ValueError, match="existing archive"):
        storage.fetch_archive(identity["url"], path, sha, len(data))


def test_corrupt_partial_never_promoted(tmp_path, monkeypatch):
    data = b"complete archive"
    calls = []
    mock_ranges(monkeypatch, data, calls)
    path = tmp_path / "archive"
    with pytest.raises(ValueError, match="full archive checksum"):
        storage.fetch_archive("https://example.org/archive", path, "0" * 64, len(data))
    assert not path.exists()
    assert json.loads(path.with_name("archive.download.json").read_text())["status"] == "checksum_failed"


def test_unowned_partial_not_resumed(tmp_path):
    path = tmp_path / "archive"
    path.with_name("archive.part").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="identity record"):
        storage.fetch_archive("https://example.org/archive", path, "0" * 64, 100)


def test_wrong_range_is_not_appended(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "urlopen", lambda *a, **k: Response(b"data", 0, 3, 4, status=200))
    with pytest.raises(ValueError, match="requested archive range"):
        storage.fetch_archive("https://example.org/archive", tmp_path / "archive", "0" * 64, 4)
    assert not (tmp_path / "archive.part").exists()


def package(tmp_path, extra=None):
    members = {"pt_meta.json": b'{"type":"dense_index","format":"flex","doc_count":2,"vec_size":2}',
               "docnos.npids": b"test identifier map", "vecs.f4": bytes(16)}
    if extra:
        members.update(extra)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    path = tmp_path / "archive.lz4"
    path.write_bytes(lz4.frame.compress(raw.getvalue()))
    sidecar = {"expected_sha256": storage.checksum(path),
               "contents": [{"path": key, "size": len(value)} for key, value in members.items()]}
    return path, sidecar


def test_stream_extraction_and_reverification(tmp_path):
    archive, sidecar = package(tmp_path)
    out = tmp_path / "extracted"
    receipt = storage.extract_archive(archive, sidecar, out)
    assert receipt["status"] == "verified"
    assert (out / "vecs.f4").read_bytes() == bytes(16)
    assert storage.extract_archive(archive, sidecar, out) == receipt
    (out / "vecs.f4").write_bytes(bytes(15) + b"x")
    with pytest.raises(ValueError, match="corrupted"):
        storage.extract_archive(archive, sidecar, out)


def test_extraction_rejects_unverified_archive_and_paths(tmp_path):
    archive, sidecar = package(tmp_path, {"../outside": b"bad"})
    with pytest.raises(ValueError, match="member contract"):
        storage.extract_archive(archive, sidecar, tmp_path / "extracted")
    sidecar["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unverified"):
        storage.extract_archive(archive, sidecar, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_extraction_failure_is_retained_not_completed(tmp_path):
    archive, sidecar = package(tmp_path)
    sidecar["contents"][0]["size"] += 1
    out = tmp_path / "extracted"
    with pytest.raises(ValueError, match="size mismatch"):
        storage.extract_archive(archive, sidecar, out)
    assert not out.exists()
    assert json.loads((tmp_path / "extracted.staging" / "ingestion.json").read_text())["status"] == "failed"
