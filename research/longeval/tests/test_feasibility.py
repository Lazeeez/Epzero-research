import json

import pytest

from research.longeval.feasibility import select_ids, run_feasibility


def test_resource_sample_does_not_depend_on_rows_or_leak_probe_documents():
    ids = [str(i) for i in range(100)]
    corpus, probes = select_ids(ids, 50, 10, 11)
    assert (corpus, probes) == select_ids(list(reversed(ids)), 50, 10, 11)
    assert len(corpus) == 50 and len(probes) == 10
    assert not set(corpus) & set(probes)
    assert corpus == sorted(corpus)
    assert (corpus, probes) != select_ids(ids, 50, 10, 29)


def test_resource_probe_rejects_incomplete_audit_and_records_failure(tmp_path):
    audit = tmp_path / "scan.json"
    audit.write_text(json.dumps({"status": "scanning"}))
    with pytest.raises(ValueError, match="completed valid vector scan"):
        run_feasibility(tmp_path, audit, tmp_path / "out")
    assert json.loads((tmp_path / "out/report.json").read_text())["status"] == "failed"
