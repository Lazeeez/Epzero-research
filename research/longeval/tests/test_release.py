import hashlib
import io
import json
import tarfile
import zipfile

import lz4.frame
from npids import Lookup
import pytest

from research.longeval import dense, release


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


def test_download_checksums_cache_and_budget(tmp_path, monkeypatch):
    data = b"verified public fixture"
    expected = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(release, "urlopen", lambda *a, **k: Response(data))
    path = tmp_path / "input"
    release.download("https://example.org/file", path, expected)
    assert path.read_bytes() == data
    monkeypatch.setattr(release, "urlopen", lambda *a, **k: pytest.fail("verified cache should be offline"))
    release.download("https://example.org/file", path, expected)
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="existing file checksum"):
        release.download("https://example.org/file", path, expected)
    monkeypatch.setattr(release, "urlopen", lambda *a, **k: Response(data))
    with pytest.raises(ValueError, match="budget"):
        release.download("https://example.org/file", tmp_path / "too-large", expected, limit=2)
    with pytest.raises(ValueError, match="checksum"):
        release.download("https://example.org/file", tmp_path / "bad-hash", "0" * 64)
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("status,header", [(200, ""), (206, "bytes 1-3/100"), (206, "bytes 0-2/99")])
def test_range_never_falls_back_to_full_download(tmp_path, monkeypatch, status, header):
    monkeypatch.setattr(dense, "urlopen", lambda *a, **k: Response(b"abc", status, {"Content-Range": header}))
    with pytest.raises(ValueError, match="bounded range"):
        dense.fetch_prefix("https://example.org/large", tmp_path / "prefix", 3, 100)
    assert not list(tmp_path.iterdir())


def test_valid_range_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "urlopen", lambda *a, **k: Response(b"abc", 206, {"Content-Range": "bytes 0-2/100"}))
    path = tmp_path / "prefix.lz4"
    result = dense.fetch_prefix("https://example.org/large", path, 3, 100)
    assert not result["full_archive_checksum_verified"]
    path.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="identity"):
        dense.fetch_prefix("https://example.org/large", path, 3, 100)


@pytest.mark.parametrize("text", ["q 0 d 0.1", "q 0 d nan", "q 0 d 1\nq 0 d 0", "not enough columns"])
def test_release_qrel_contract(text):
    with pytest.raises(ValueError):
        release.parse_qrels(text)


def test_query_contract_and_unknown_qrels(tmp_path):
    path = tmp_path / "queries.tsv"
    path.write_text("001\tfirst query\n1\tsecond query\n")
    assert set(release.read_queries(path)) == {"001", "1"}
    path.write_text("q\tquery one\nq\tquery two\n")
    with pytest.raises(ValueError, match="duplicate"):
        release.read_queries(path)
    with pytest.raises(ValueError, match="unknown"):
        release.qrel_summary({"unseen": {"d": 1}}, {"q"})


def make_release_fixture(folder, monkeypatch):
    entries = {}
    archive = folder / "llm.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for model, prefix in release.MODELS.items():
            for step in (1, 2, 3):
                name = f"llm_synthetic_qrels/{prefix}-umbrella_zeroshot_no_desc_no_narrative-snapshot-{step}.qrels.txt"
                z.writestr(name, "e1 0 d1 1\ne2 0 d2 0\n")
        z.writestr("unselected-experiment.txt", "malformed content should not be selected")
    for key in release.SMALL_KEYS:
        if key == "llm-synthetic-qrels":
            path = archive
        else:
            path = folder / (key + ".txt")
            if key == release.TRAIN:
                text = "".join(f"t{i}\ttraining query {i}\n" for i in range(4))
            elif key == release.TEST:
                text = "e1\tevaluation one\ne2\tevaluation two\ne3\tevaluation three\n"
            elif "training" in key:
                text = "t0 0 d0 1\n"
            else:
                text = "e1 0 d1 1\ne2 0 d2 0\n"
            path.write_text(text)
        entries[key] = {"cache_path": path.name, "expected_md5": release.checksum(path, "md5"),
                        "url": f"https://researchdata.tuwien.ac.at/files/{path.name}?token=public-example"}
    manifest = folder / "downloads.json"
    manifest.write_text(json.dumps({"longeval-sci-2026": entries}))
    monkeypatch.setattr(release, "MANIFEST_SHA256", release.checksum(manifest))


def test_offline_release_audit_and_checksum_failure(tmp_path, monkeypatch):
    make_release_fixture(tmp_path, monkeypatch)
    report = release.audit_release(tmp_path, tmp_path / "report.json")
    assert report["queries"]["train"] == 4 and report["queries"]["test"] == 3
    assert report["retrospective_panels"]["dctr"]["query_ids"] == ["e1"]
    assert len(report["selected_archive_members"]) == 12
    assert {k: len(v) for k, v in report["provisional_query_roles"].items()} == {
        "development": 2, "calibration": 2, "evaluation": 3}
    assert "public-example" not in json.dumps(report)
    coverage = release.judgment_coverage(tmp_path, 1, {"d0", "d2"})
    assert coverage["gpt-oss-120b"]["missing_document_ids"] == ["d1"]
    assert coverage["gpt-oss-120b"]["missing_positive_pairs"] == 1
    assert coverage["longeval_sci_training_2026_qrels_dctr"]["missing_pairs"] == 0
    with pytest.raises(FileExistsError):
        release.audit_release(tmp_path, tmp_path / "report.json")
    (tmp_path / (release.TEST + ".txt")).write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        release.audit_release(tmp_path, tmp_path / "corrupt-report.json")
    assert not (tmp_path / "corrupt-report.json").exists()


def test_manifest_cannot_change_silently(tmp_path, monkeypatch):
    make_release_fixture(tmp_path, monkeypatch)
    with (tmp_path / "downloads.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="manifest identity"):
        release.manifest_entries(tmp_path)


def make_prefix(tmp_path, *, metadata=None):
    ids = tmp_path / "original.npids"
    with Lookup.build(["001", "1", "z"], ids):
        pass
    metadata = metadata or {"type": "dense_index", "format": "flex", "doc_count": 3, "vec_size": 2}
    members = {"docnos.npids": ids.read_bytes(), "pt_meta.json": json.dumps(metadata).encode(), "vecs.f4": bytes(24)}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, value in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    prefix = tmp_path / "sample.lz4"
    prefix.write_bytes(lz4.frame.compress(raw.getvalue()))
    return prefix, {"contents": [{"path": k, "size": len(v)} for k, v in members.items()]}


def test_dense_prefix_type_shape_and_membership(tmp_path):
    prefix, sidecar = make_prefix(tmp_path)
    ids = tmp_path / "extracted.npids"
    result = dense.inspect_prefix(prefix, sidecar, ids)
    assert result["metadata"]["vec_size"] == 2 and not result["vector_payload_verified"]
    membership, keys = dense.inspect_docnos(ids, 3, ["001", "gone"])
    assert keys == ["001", "1", "z"]
    assert membership["removed_ids"] == 1 and membership["added_ids"] == 2
    assert not membership["insert_only_membership_valid"]
    with pytest.raises(ValueError, match="count"):
        dense.inspect_docnos(ids, 4)
    sidecar["contents"][-1]["size"] = 25
    with pytest.raises(ValueError, match="matrix shape"):
        dense.inspect_prefix(prefix, sidecar)


def test_sparse_metadata_not_accepted_as_dense(tmp_path):
    prefix, sidecar = make_prefix(tmp_path, metadata={"type": "sparse_index", "format": "terrier"})
    with pytest.raises(ValueError, match="dense FlexIndex"):
        dense.inspect_prefix(prefix, sidecar)
