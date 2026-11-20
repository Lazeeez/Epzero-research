import hashlib
import json

import numpy as np
from npids import Lookup
import pytest

from research.longeval import mapped
from research.longeval.release import checksum
from research.longeval.storage import atomic_json


ARCHIVE_SHA = "a" * 64


def snapshot_fixture(folder, ids, values):
    folder.mkdir()
    values = np.asarray(values, dtype="<f4")
    values.tofile(folder / "vecs.f4")
    with Lookup.build(ids, folder / "docnos.npids"):
        pass
    atomic_json(folder / "pt_meta.json", {"type": "dense_index", "format": "flex",
                                         "doc_count": len(ids), "vec_size": values.shape[1]})
    atomic_json(folder / "ingestion.json", {"status": "verified", "archive_sha256": ARCHIVE_SHA,
                "members": {name: {"bytes": (folder / name).stat().st_size, "sha256": checksum(folder / name)}
                            for name in ("vecs.f4", "docnos.npids", "pt_meta.json")}})
    return folder


def test_mapped_scan_preserves_original_bytes_and_measures_norms(tmp_path):
    values = np.array([[1, 0], [3, 4], [-0.0, 0.0]], dtype=np.float32)
    folder = snapshot_fixture(tmp_path / "input", ["01", "1", "z"], values)
    with mapped.MappedSnapshot(folder, expected_archive_sha256=ARCHIVE_SHA) as snapshot:
        assert isinstance(snapshot.vectors, np.memmap)
        assert not snapshot.vectors.flags.writeable
        assert snapshot.ids == ["01", "1", "z"]
        result = mapped.scan_vectors(snapshot, tmp_path / "scan", chunk_rows=1)
        assert snapshot.vectors.tobytes() == values.tobytes()
    assert result["finite_rows"] == 3
    assert result["unit_norm_rows"] == 1
    assert result["zero_norm_rows"] == 1
    assert result["norm_mean"] == 2
    assert result["norm_std"] == pytest.approx(np.std([1, 5, 0]))
    assert not result["numeric_vectors_valid"]
    fingerprints = np.load(tmp_path / "scan/row_sha256.npy", allow_pickle=False)
    assert fingerprints[2].tobytes() == hashlib.sha256(values[2]).digest()
    assert fingerprints[2].tobytes() != hashlib.sha256(np.zeros(2, dtype=np.float32)).digest()


def test_nonfinite_rows_record_failure_without_fabricating_norms(tmp_path):
    folder = snapshot_fixture(tmp_path / "input", ["nan", "inf"], [[np.nan, 0], [np.inf, 0]])
    with mapped.MappedSnapshot(folder, expected_archive_sha256=ARCHIVE_SHA) as snapshot:
        result = mapped.scan_vectors(snapshot, tmp_path / "scan")
    assert result["nonfinite_rows"] == 2
    assert result["norm_mean"] is None
    assert result["norm_min"] is None
    assert not result["numeric_vectors_valid"]


def test_corrupt_vector_bytes_rejected_before_completed_audit(tmp_path):
    folder = snapshot_fixture(tmp_path / "input", ["a", "b"], [[1, 0], [0, 1]])
    (folder / "vecs.f4").write_bytes(np.array([[2, 0], [0, 1]], dtype="<f4").tobytes())
    with mapped.MappedSnapshot(folder, expected_archive_sha256=ARCHIVE_SHA) as snapshot:
        with pytest.raises(ValueError, match="payload checksum"):
            mapped.scan_vectors(snapshot, tmp_path / "scan")
    assert json.loads((tmp_path / "scan/scan.json").read_text())["status"] == "failed"
    assert not (tmp_path / "scan/row_sha256.npy").exists()


def test_metadata_corruption_and_shape_mismatch_rejected(tmp_path):
    folder = snapshot_fixture(tmp_path / "input", ["a", "b"], [[1, 0], [0, 1]])
    meta = folder / "pt_meta.json"
    meta.write_bytes(meta.read_bytes().replace(b'"doc_count": 2', b'"doc_count": 3'))
    with pytest.raises(ValueError, match="checksum"):
        mapped.MappedSnapshot(folder, expected_archive_sha256=ARCHIVE_SHA)
    receipt = json.loads((folder / "ingestion.json").read_text())
    receipt["members"]["pt_meta.json"]["sha256"] = checksum(meta)
    atomic_json(folder / "ingestion.json", receipt)
    with pytest.raises(ValueError, match="float32 shape"):
        mapped.MappedSnapshot(folder, expected_archive_sha256=ARCHIVE_SHA)


def test_row_reordering_is_joined_by_id_and_changes_are_not_hidden(tmp_path):
    fingerprints = lambda values: np.array([np.void(hashlib.sha256(x.encode()).digest()) for x in values])
    previous = fingerprints(["vector-a", "vector-b", "vector-c"])
    current = fingerprints(["vector-c", "vector-new", "vector-a", "vector-b"])
    same = mapped.compare_shared(["a", "b", "c"], previous, ["c", "new", "a", "b"], current)
    assert same["fixed_vector_insert_only_valid"]
    assert same["shared_ids_at_different_rows"] == 3
    assert same["added_ids"] == 1
    current[2] = np.void(bytes(32))
    changed = mapped.compare_shared(["a", "b", "c"], previous, ["c", "new", "a", "b"], current)
    assert not changed["fixed_vector_insert_only_valid"]
    assert changed["changed_id_examples"] == ["a"]
    removed = mapped.compare_shared(["a", "b", "c"], previous, ["c", "new"], current[:2])
    assert removed["removed_ids"] == 2
    assert not removed["fixed_vector_insert_only_valid"]


def test_snapshot_audit_requires_all_three_for_full_gate(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for step, ids, values in [(1, ["a", "b"], [[1, 0], [0, 1]]),
                              (2, ["b", "c", "a"], [[0, 1], [1, 1], [1, 0]]),
                              (3, ["c", "b", "a", "d"], [[1, 1], [0, 1], [1, 0], [2, 1]])]:
        snapshot_fixture(source / f"snapshot-{step}", ids, values)
    monkeypatch.setattr(mapped, "ARTIFACTS", tuple({"snapshot": s, "archive_sha256": ARCHIVE_SHA} for s in [1, 2, 3]))
    partial = mapped.audit_vectors(source, tmp_path / "partial", snapshots=[1])
    assert not partial["fixed_vector_gate_passed"]
    complete = mapped.audit_vectors(source, tmp_path / "complete", chunk_rows=1)
    assert complete["fixed_vector_gate_passed"]
    assert complete["snapshots"][1]["transition"]["bitwise_changed_shared_vectors"] == 0
