"""Integration M1 acceptance: native box stream -> receipt -> Phase 2 HTTP UI."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from streetlab_integration.observation_bridge import (
    STATUS, ingest_track_file, latest_report, load_native_tracks,
    build_report, sha,
)


def track(frame, tid, class_id=0, conf=.90, x=50):
    return [frame, tid, x, 80, 24, 16, x, 80, 24, 16,
            class_id, conf, 24, 16]


def fixture_tracks(path):
    values = []
    for frame in range(100, 105):
        values += [track(frame, 21, 3, .88, 100 + frame - 100),
                   track(frame, 7, 0, .94, 200 + frame - 100)]
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(values)


def test_receipt_preserves_native_ids_and_refuses_real_site_sumo(tmp_path):
    source = tmp_path / "frozen_primary.txt"
    fixture_tracks(source)
    before = source.read_bytes()
    receipt, report = ingest_track_file(
        source, tmp_path / "product", fps=30., source_label="primary_ios030"
    )
    assert source.read_bytes() == before
    assert receipt.is_dir()
    assert report["status"] == STATUS
    assert report["source_tracking_sha256"] == sha(source)
    assert report["observed_points"] == 10
    assert report["observed_frames"] == 5
    assert report["frame_span"]["first"] == 100
    assert report["frame_span"]["last"] == 104
    assert report["tracking_identity_count"] == 2
    assert report["tracks_by_class"] == {"CAR": 1, "MOTORCYCLE": 1}
    assert report["mean_detector_confidence"] == .91
    assert report["fps"] == 30
    assert report["observed_duration_seconds"] == .133
    assert report["source_coordinate_system"] == "SOURCE_IMAGE_PIXELS_UNCALIBRATED"
    assert report["data_readiness"]["real_site_simulation_allowed"] is False
    assert report["data_readiness"]["speed_mps"] == "MISSING"
    assert report["study_gate"]["status"] == "NEEDS_DATA"
    assert "geometry" in report["study_gate"]["missing_evidence"]
    assert "demand" in report["study_gate"]["missing_evidence"]
    assert "trajectories" not in report["study_gate"]["missing_evidence"]
    assert report["study_gate"]["evidence_provenance"] == {
        "trajectories": "OBSERVED_AUTO"
    }
    assert report["data_readiness"]["persona_calibration_modified"] is False
    with (receipt / "track_points_pixels.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert set(r["source_track_id"] for r in rows) == {"7", "21"}
    assert "speed_mps" not in rows[0]
    assert latest_report(tmp_path / "product") == report


def test_existing_original_receipt_is_immutable_and_pointer_unchanged(tmp_path):
    path = tmp_path / "tracks.txt"
    fixture_tracks(path)
    ingest_track_file(path, tmp_path / "product")
    pointer = (tmp_path / "product" / "observations" / "latest.json").read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        ingest_track_file(path, tmp_path / "product")
    assert (tmp_path / "product" / "observations" / "latest.json").read_bytes() == pointer


def test_corrupted_receipt_is_not_silently_shown(tmp_path):
    path = tmp_path / "tracks.txt"
    fixture_tracks(path)
    receipt, _ = ingest_track_file(path, tmp_path / "product")
    with (receipt / "report.json").open("a") as out:
        out.write("\n")
    with pytest.raises(ValueError, match="integrity"):
        latest_report(tmp_path / "product")


@pytest.mark.parametrize("rows,phrase", [
    ([[0, 1, 2]], "14 numeric"),
    ([track(0, 1), track(0, 1)], "Duplicate real"),
    ([track(3, 1), track(2, 1)], "Unsorted"),
    ([track(0, -1)], "tracker ID"),
    ([track(0, 1, 4)], "unknown frozen"),
    ([track(0, 1, 0, 1.5)], "geometry or confidence"),
    ([track(0, 1, 0, float("nan"))], "nonfinite"),
])
def test_reject_invalid_track_schemas(tmp_path, rows, phrase):
    path = tmp_path / "bad.txt"
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError, match=phrase):
        load_native_tracks(path)


def test_class_change_is_flagged_not_counted_twice(tmp_path):
    p = tmp_path / "flip.txt"
    with p.open("w", newline="") as f:
        csv.writer(f).writerows([
            track(10, 3, 3), track(11, 3, 0), track(12, 3, 3),
        ])
    _, report = ingest_track_file(p, tmp_path / "product")
    assert report["class_flips_on_track_ids"] == 1
    assert report["tracking_identity_count"] == 1
    assert report["tracks_by_class"] == {"MOTORCYCLE": 1}


def test_latest_http_surface_separates_observed_from_demo(tmp_path):
    source = tmp_path / "source.txt"
    fixture_tracks(source)
    from streetlab_phase2.api import create_app
    client = TestClient(create_app(
        service=object(), observation_workdir=tmp_path / "product"
    ))
    missing = client.get("/api/observations/latest")
    assert missing.status_code == 200
    assert missing.json()["status"] == "NOT_IMPORTED"
    assert missing.json()["real_site_simulation_allowed"] is False
    ingest_track_file(source, tmp_path / "product")
    response = client.get("/api/observations/latest")
    assert response.status_code == 200
    item = response.json()
    assert item["status"] == STATUS
    assert item["tracking_identity_count"] == 2
    assert item["study_gate"]["status"] == "NEEDS_DATA"
    html = client.get("/")
    assert html.status_code == 200
    assert 'id="obsTracks"' in html.text
    assert "Synthetic Decision Lab demo" in html.text
    assert "/api/observations/latest" in html.text
    receipt = tmp_path / "product" / "observations" / sha(source)
    (receipt / "track_points_pixels.csv").write_text("tampered")
    bad = client.get("/api/observations/latest")
    assert bad.status_code == 409
    assert "integrity" in bad.json()["detail"]


def test_no_path_injection_through_latest_index(tmp_path):
    path = tmp_path / "tracks.txt"
    fixture_tracks(path)
    ingest_track_file(path, tmp_path / "product")
    index = tmp_path / "product" / "observations" / "latest.json"
    index.write_text(json.dumps({"schema_version": 1,
                                 "track_sha256": "../../secrets"}))
    with pytest.raises(ValueError, match="Corrupt"):
        latest_report(tmp_path / "product")


def test_cli_can_import_from_different_cwd(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "streetlab_integration_m1.py"
    if not script.is_file():
        pytest.skip("CLI excluded from sparse checkout")
    source = tmp_path / "tracks.txt"
    fixture_tracks(source)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(script), "--tracks", str(source),
         "--workdir", str(tmp_path / "product"), "--fps", "25"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    stdout = json.loads(proc.stdout)
    assert stdout["milestone"] == "INTEGRATION_M1_VIDEO_TRACK_TO_DECISION_LAB"
    assert stdout["real_site_study_status"] == "NEEDS_DATA"
    assert (tmp_path / "product" / "observations" / "latest.json").exists()
