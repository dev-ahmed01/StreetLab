"""T000 recovery inventory tests: missing roots, scorecards and track samples."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase3_t000_forensic_inventory.py"
if not SCRIPT.is_file():
    pytest.skip("T000 inventory script omitted by sparse checkout", allow_module_level=True)
spec = importlib.util.spec_from_file_location("t000_inventory", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_absent_old_directory_reports_missing_not_deletion(tmp_path):
    result = mod.scan(tmp_path)
    assert result["matched_artifact_files"] == 0
    assert not result["truncated"]
    assert all(not check["exists"] for check in result["historical_expected_path_checks"])


def test_discovers_t000_manifest_and_14_column_sample(tmp_path):
    folder = tmp_path/"StreetLab"/"artifacts"/"phase3"/"geotrax_tuning"/"experiments"/"T000_BASELINE"
    folder.mkdir(parents=True)
    (folder/"summary.json").write_text('{"candidate": "T000", "status": "FINISHED"}', encoding="utf-8")
    (folder/"20250526_video.txt").write_text(
        "10750,1,1,2,3,4,5,6,7,8,0,0.9,3,4\n", encoding="utf-8")
    result = mod.scan(tmp_path)
    assert result["matched_artifact_files"] >= 2
    sample = next(x for x in result["matches"] if x["path"].endswith("20250526_video.txt"))
    assert sample["possible_14col_geotrax"]
    assert sample["sample_in_w04_window"] == 1
    assert any(x["reported_candidate"] == "T000" for x in result["matches"] if "reported_candidate" in x)
    assert not result["eligible_for_production"]


def test_enforces_bounded_scan(tmp_path):
    folder = tmp_path/"StreetLab"/"artifacts"/"phase3"/"geotrax_tuning"
    folder.mkdir(parents=True)
    (folder/"T000_reference.json").write_text('{"candidate":"T000"}', encoding="utf-8")
    result = mod.scan(tmp_path,max_entries=1)
    assert result["truncated"] is True


def test_does_not_traverse_virtualenv(tmp_path):
    folder = tmp_path/"StreetLab"/"artifacts"
    (folder/".venv-geotrax").mkdir(parents=True)
    (folder/".venv-geotrax"/"T000_fake.json").write_text("{}")
    assert mod.scan(tmp_path)["matched_artifact_files"] == 0
