"""Tests that cached 21-frame metrics are NEVER compared to a new 5-frame cohort."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth
from streetlab_phase3.video.cached_detector_subset import (
    compare_cached_detector_subsets,
)
from streetlab_phase3.video.sahi_detection_audit import (
    Detection, score_detections,
)


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def make_audit(directory, *, frames, model, truth, candidate=False):
    directory.mkdir()
    std = {f: [Detection(f, 300., 20., "CAR", .8)] for f in frames}
    sli = {f: [Detection(f, 300., 20., "CAR", .8),
               Detection(f, 100., 20., "MOTORCYCLE", .8)]
           for f in frames}
    if candidate:
        sli[frames[-1]] = [Detection(frames[-1], 300., 20., "CAR", .8)]
    with truth.open(newline="", encoding="utf-8") as stream:
        labels = normalize_fluid_pixel_truth(csv.DictReader(stream))
    scores = {mode: score_detections(det, labels, frames)
              for mode, det in (("standard", std), ("sliced", sli))}
    for mode, detections in (("standard", std), ("sliced", sli)):
        with (directory / f"{mode}_detections.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.writer(fh)
            writer.writerow(("frame", "x_px", "y_px", "vehicle_class", "confidence"))
            for frame in frames:
                for point in detections[frame]:
                    writer.writerow((point.frame, point.x_px, point.y_px,
                                     point.vehicle_class, point.confidence))
        scores[mode]["latency_median_s"] = 4.0 if candidate else 12.0
    payload = {
        "status": "EXPERIMENTAL_DETECTION_ONLY",
        "sample_frames": frames,
        "trial": {
            "video": "recording.avi", "fluid_tracks": str(truth),
            "weights": str(model), "confidence": .15, "image_size": 640,
            "device": "cpu", "frame_offset": 1, "max_pixel_distance": 50.,
            "class_map_file": None, "start_frame": frames[0],
            "end_frame": frames[-1],
            "sample_step": frames[1] - frames[0],
            "slice_height": 1280 if candidate else 640,
            "slice_width": 1280 if candidate else 640,
            "overlap": .10 if candidate else .20,
        },
        "model_sha256": sha(model),
        "ground_truth_sha256": sha(truth),
        "class_map": {"motorcycle": 3, "car": 0},
        **scores,
    }
    (directory / "report.json").write_text(json.dumps(payload))
    return directory


def test_rescores_cached_reference_on_same_two_frames_and_reports_cost(tmp_path):
    model, truth = tmp_path / "model.pt", tmp_path / "truth.csv"
    model.write_bytes(b"weights")
    truth.write_text(
        "frame,id,cx,cy,type\n"
        "1,1,100,20,moped\n1,2,300,20,car\n"
        "2,1,100,20,moped\n2,2,300,20,car\n"
        "3,1,100,20,moped\n3,2,300,20,car\n")
    ref = make_audit(tmp_path / "ref", frames=[0, 1, 2],
                     model=model, truth=truth)
    cand = make_audit(tmp_path / "cand", frames=[0, 2],
                      model=model, truth=truth, candidate=True)
    result = compare_cached_detector_subsets(ref, cand)
    assert result["eligible_for_promotion"] is False
    assert result["sample_frames"] == [0, 2]
    assert result["same_truth_points"] == 4
    assert result["same_motorcycle_truth_points"] == 2
    assert result["scores"]["cached_sliced_640"]["per_class"]["MOTORCYCLE"]["recall"] == 1.
    assert result["scores"]["candidate_sliced"]["per_class"]["MOTORCYCLE"]["recall"] == .5
    assert result["sliced_candidate_minus_reference"]["motorcycle_recall_percentage_points"] == -50
    assert result["export_evidence"]["cached_sliced_640"]["original_evaluation_frame_count"] == 3
    assert result["export_evidence"]["candidate_sliced"]["original_evaluation_frame_count"] == 2
    assert len(result["export_evidence"]["candidate_sliced"]["sha256"]) == 64


def test_reject_different_checkpoint_or_ground_truth_corruption(tmp_path):
    model, truth = tmp_path / "model.pt", tmp_path / "truth.csv"
    model.write_bytes(b"weights")
    truth.write_text("frame,id,cx,cy,type\n1,1,100,20,moped\n"
                     "2,1,100,20,moped\n3,1,100,20,moped\n")
    ref = make_audit(tmp_path / "ref", frames=[0, 1, 2],
                     model=model, truth=truth)
    cand = make_audit(tmp_path / "cand", frames=[0, 2],
                      model=model, truth=truth, candidate=True)
    candidate_report = cand / "report.json"
    raw = json.loads(candidate_report.read_text())
    raw["trial"]["confidence"] = .2
    candidate_report.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="confidence"):
        compare_cached_detector_subsets(ref, cand)
    raw["trial"]["confidence"] = .15
    candidate_report.write_text(json.dumps(raw))
    truth.write_text(truth.read_text() + "4,1,100,20,moped\n")
    with pytest.raises(ValueError, match="Truth annotations differ"):
        compare_cached_detector_subsets(ref, cand)


def test_cli_no_overwrite_and_distinct_cohorts(tmp_path):
    model, truth = tmp_path / "model.pt", tmp_path / "truth.csv"
    model.write_bytes(b"weights")
    truth.write_text("frame,id,cx,cy,type\n"
                     "1,1,100,20,moped\n2,1,100,20,moped\n"
                     "3,1,100,20,moped\n")
    ref = make_audit(tmp_path / "ref", frames=[0, 1, 2],
                     model=model, truth=truth)
    cand = make_audit(tmp_path / "cand", frames=[0, 2],
                      model=model, truth=truth, candidate=True)
    out = tmp_path / "comparison.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_compare_cached_detector_subsets.py"
    args = [sys.executable, str(script),
            "--reference-dir", str(ref), "--candidate-dir", str(cand),
            "--output", str(out)]
    run = subprocess.run(args, check=True, capture_output=True, text=True)
    data = json.loads(run.stdout)
    assert data["same_motorcycle_truth_points"] == 2
    assert len(json.loads(out.read_text())["sample_frames"]) == 2
    again = subprocess.run(args, capture_output=True, text=True)
    assert again.returncode != 0
    assert "FileExistsError" in again.stderr
