"""Box overlap analysis must use *exported* boxes only and never change scores."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.video.existing_box_forensics import (
    existing_box_forensics, overlap, read_exported_boxes,
)


def line(frame: int, tid: int, cx: float, cy: float,
         width: float, height: float, cls: int, score: float) -> str:
    return (f"{frame},{tid},{cx},{cy},{width},{height},"
            f"{cx},{cy},{width},{height},{cls},{score},{width},{height}\n")


def setup_trial(tmp_path: Path):
    source = tmp_path / "sliced.txt"
    source.write_text(
        line(10, 1, 100, 100, 24, 24, 0, .90)   # unmatched CAR over BUS
        + line(10, 2, 100, 100, 130, 190, 1, .80) # BUS that contains car
        + line(10, 3, 400, 400, 20, 20, 3, .31)   # matched motorcycle
        + line(10, 4, 404, 401, 20, 20, 3, .32)   # unmatched same bike
        + line(10, 5, 800, 800, 60, 45, 0, .81)   # isolated unmatched car
        + line(11, 6, 900, 900, 30, 30, 0, .95),  # unrelated frame
        encoding="utf-8",
    )
    examples = [
        {"video_frame": 10, "predicted_track_id": "1",
         "predicted_class": "CAR", "confidence": .90},
        {"video_frame": 10, "predicted_track_id": "4",
         "predicted_class": "MOTORCYCLE", "confidence": .32},
        {"video_frame": 10, "predicted_track_id": "5",
         "predicted_class": "CAR", "confidence": .81},
    ]
    comp = {
        "status": "EXPLORATORY_SAME_FRAME_COMPARISON_NOT_PRODUCTION_EVIDENCE",
        "evaluation_start_frame": 10,
        "evaluation_end_frame": 10,
        "results": {"sliced": {"tracks": str(source)}},
        "sliced_unmatched_proximity": {
            "status": "UNMATCHED_PROXIMITY_HEURISTIC_NOT_PROOF_OF_DUPLICATES",
            "video_frame_start": 10, "video_frame_end": 10,
            "examples": examples,
        },
    }
    return source, comp


def test_overlap_and_containment_uses_existing_boxes_no_detector():
    # An 24x24 car box contained in 130x190 bus yields small IoU,
    # but high coverage of the CAR.
    from streetlab_phase3.video.existing_box_forensics import ExportedBox
    car = ExportedBox(1, "1", "CAR", .8, 10, 10, 34, 34)
    bus = ExportedBox(1, "2", "BUS", .9, 0, 0, 130, 190)
    iou, share_car, share_bus = overlap(car, bus)
    assert iou < .1
    assert share_car == 1
    assert share_bus < .1
    assert overlap(car, ExportedBox(2, "3", "CAR", .8, 10, 10, 34, 34)) == (0., 0., 0.)


def test_box_audit_separates_bus_class_confusion_vs_bike_duplicate(tmp_path):
    source, comp = setup_trial(tmp_path)
    result = existing_box_forensics(source, comp)
    assert result["eligible_for_promotion"] is False
    assert result["unmatched_observations"] == 3
    assert result["high_confidence_unmatched_cars"] == {
        "observations": 2,
        "with_same_class_box_overlap": 0,
        "covered_by_exported_large_vehicle_box": 1,
    }
    assert result["unmatched_motorcycle"] == {
        "observations": 1, "with_same_class_box_overlap": 1}
    first = result["findings"][0]
    assert first["car_covered_by_bus_or_heavy_gte_0_60"] is True
    assert next(x for x in first["overlapping_track_boxes"] if x["class"] == "BUS")[
        "fraction_target_covered"] == 1
    assert result["findings"][2]["overlapping_track_boxes"] == []


def test_fail_closed_missing_predictions_wrong_manifest_and_duplicate_rows(tmp_path):
    source, comp = setup_trial(tmp_path)
    comp["sliced_unmatched_proximity"]["examples"][0]["confidence"] = .15
    with pytest.raises(ValueError, match="disagree"):
        existing_box_forensics(source, comp)
    comp["sliced_unmatched_proximity"]["examples"][0]["confidence"] = .90
    comp["results"]["sliced"]["tracks"] = str(tmp_path / "wrong.txt")
    with pytest.raises(ValueError, match="provenance"):
        existing_box_forensics(source, comp)
    comp["results"]["sliced"]["tracks"] = str(source)
    comp["sliced_unmatched_proximity"]["examples"].append(
        comp["sliced_unmatched_proximity"]["examples"][0])
    with pytest.raises(ValueError, match="Duplicate unmatched"):
        existing_box_forensics(source, comp)
    comp["sliced_unmatched_proximity"]["examples"].pop()
    with source.open("a", encoding="utf-8") as fh:
        fh.write(line(10, 1, 100, 100, 24, 24, 0, .9))
    with pytest.raises(ValueError, match="Duplicate confirmed"):
        existing_box_forensics(source, comp)


def test_cli_produces_immutable_json_from_exported_tracks(tmp_path):
    source, comp = setup_trial(tmp_path)
    comp_file, out = tmp_path / "comp.json", tmp_path / "evidence.json"
    comp_file.write_text(json.dumps(comp), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_box_forensics.py"
    argv = [sys.executable, str(script), "--comparison", str(comp_file),
            "--sliced-tracks", str(source), "--output", str(out)]
    first = subprocess.run(argv, check=True, text=True, capture_output=True)
    summary = json.loads(first.stdout)
    assert summary["high_confidence_unmatched_cars"]["observations"] == 2
    assert json.loads(out.read_text())["eligible_for_promotion"] is False
    again = subprocess.run(argv, text=True, capture_output=True)
    assert again.returncode != 0
    assert "FileExistsError" in again.stderr
