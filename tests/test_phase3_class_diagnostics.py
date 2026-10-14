"""Regression tests for diagnosis of per-class failures without rerunning inference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth
from streetlab_phase3.video.class_diagnostics import class_diagnostics


def truth(frame, track, x, label):
    return FluidPixelTruth(frame, track, x, 10., label)


def pred(frame, track, x, label):
    return GeoTraxPixelPoint(frame, track, x, 10., x, 10., label, .85)


def test_class_aware_diagnosis_preserves_spatial_matching_and_wrong_class_errors():
    labels = [
        truth(1, "m", 10, "MOTORCYCLE"),
        truth(1, "c", 100, "CAR"),
        truth(2, "m", 20, "MOTORCYCLE"),
    ]
    detections = [
        pred(0, "A", 10, "MOTORCYCLE"),
        pred(0, "B", 100, "CAR"),
        pred(1, "A", 20, "CAR"),  # same position but wrong class
        pred(1, "C", 300, "CAR"), # truly unmatched prediction
    ]
    result = class_diagnostics(detections, labels, start_frame=0, end_frame=1)
    assert result["truth_points"] == 3
    assert result["predicted_points"] == 4
    assert result["matched_points"] == 3
    assert result["correct_class_matches"] == 2
    bike = result["by_class"]["MOTORCYCLE"]
    assert bike["truth_points"] == 2
    assert bike["spatial_matches"] == 2
    assert bike["correct_class_matches"] == 1
    assert bike["class_errors_on_spatial_matches"] == 1
    assert bike["spatial_recall"] == 1.
    assert bike["correct_class_recall"] == .5
    assert result["by_class"]["CAR"]["unmatched_predictions"] == 1
    assert result["by_class"]["CAR"]["correct_class_precision"] == pytest.approx(1/3)
    assert result["confusion_truth_to_predicted"] == [
        {"truth": "MOTORCYCLE", "predicted": "CAR", "points": 1}
    ]


def test_fixed_cohort_empty_prediction_does_not_hide_missed_motorcycles():
    labels = [truth(11, "M1", 10, "MOTORCYCLE"),
              truth(12, "M1", 20, "MOTORCYCLE")]
    result = class_diagnostics([], labels, start_frame=10, end_frame=12)
    assert result["truth_points"] == 2
    assert result["by_class"]["MOTORCYCLE"]["spatial_misses"] == 2
    assert result["by_class"]["MOTORCYCLE"]["spatial_recall"] == 0.
    assert result["by_class"]["MOTORCYCLE"]["spatial_precision"] is None


def test_diagnostics_disallows_offset_and_threshold_cherry_picking():
    with pytest.raises(ValueError, match="Frozen"):
        class_diagnostics([], [], start_frame=0, end_frame=1, frame_offset=0)
    with pytest.raises(ValueError, match="Frozen"):
        class_diagnostics([], [], start_frame=0, end_frame=1, max_pixel_distance=10)
    with pytest.raises(ValueError, match="Invalid evaluation"):
        class_diagnostics([], [], start_frame=2, end_frame=1)


def test_file_cli_reports_motorcycle_recall_and_refuses_overwrite(tmp_path):
    output = tmp_path / "class.json"
    tracks = tmp_path / "tracks.txt"
    fluid = tmp_path / "fluid.csv"
    tracks.write_text(
        "0,15,100,200,20,10,100,200,20,10,3,0.9,20,10\n",
        encoding="utf-8",
    )
    fluid.write_text("frame,id,cx,cy,type\n1,1,100,200,moped\n"
                     "2,1,101,200,moped\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_class_diagnostics.py"
    command = [sys.executable, str(script), "--tracks", str(tracks),
               "--fluid-tracks", str(fluid), "--start-frame", "0",
               "--end-frame", "1", "--output", str(output)]
    run = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(run.stdout)
    assert data["by_class"]["MOTORCYCLE"]["spatial_recall"] == .5
    assert data["by_class"]["MOTORCYCLE"]["correct_class_recall"] == .5
    assert json.loads(output.read_text()) == data
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0
    assert "FileExistsError" in again.stderr
