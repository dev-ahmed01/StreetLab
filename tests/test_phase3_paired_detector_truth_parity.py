"""Exact-truth matched identity parity can expose equal-count vehicle substitutions."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth
from streetlab_phase3.video.openvino_export import export_isolated_openvino
from streetlab_phase3.video.paired_detector_truth_parity import (
    compare_detector_truth_identity,
)
from streetlab_phase3.video.sahi_detection_audit import Detection, score_detections


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_export(weights: Path, image_size: int):
    folder = weights.with_name("checkpoint_openvino_model")
    folder.mkdir()
    (folder / "model.xml").write_text("<model/>")
    (folder / "model.bin").write_bytes(b"converted")
    return folder


def make_cases(tmp_path):
    weights = tmp_path / "checkpoint.pt"
    weights.write_bytes(b"exact original source weights")
    truth = tmp_path / "truth.csv"
    truth.write_text(
        "frame,id,cx,cy,type\n"
        "1,a,100,20,moped\n1,b,300,20,moped\n"
        "2,a,100,20,moped\n2,b,300,20,moped\n"
        "3,a,100,20,moped\n3,b,300,20,moped\n",
        encoding="utf-8",
    )
    export = export_isolated_openvino(
        weights=weights, output_dir=tmp_path / "openvino",
        exporter=fake_export)
    ref, cand = tmp_path / "reference", tmp_path / "candidate"
    ref.mkdir()
    cand.mkdir()
    source = [(0, [100]), (1, [100, 300]), (2, [100])]
    converted = [(0, [300]), (2, [100])]
    with truth.open(newline="", encoding="utf-8") as stream:
        gt = normalize_fluid_pixel_truth(csv.DictReader(stream))
    for name, folder, samples, backend in (
        ("reference", ref, source, "pytorch"),
        ("candidate", cand, converted, "openvino"),
    ):
        frames = [frame for frame, _ in samples]
        sliced = {
            frame: [Detection(frame, float(x), 20., "MOTORCYCLE", .8)
                    for x in xs] for frame, xs in samples
        }
        std = {frame: [] for frame in frames}
        for mode, preds in (("standard", std), ("sliced", sliced)):
            with (folder / f"{mode}_detections.csv").open(
                "w", newline="", encoding="utf-8"
            ) as fh:
                w = csv.writer(fh)
                w.writerow(["frame", "x_px", "y_px", "vehicle_class", "confidence"])
                for frame in frames:
                    for p in preds[frame]:
                        w.writerow([p.frame, p.x_px, p.y_px,
                                    p.vehicle_class, p.confidence])
        trial = {
            "video": "same video path", "weights": str(weights),
            "fluid_tracks": str(truth), "confidence": .15, "device": "cpu",
            "class_map_file": None, "image_size": 640,
            "slice_height": 640, "slice_width": 640, "overlap": .2,
            "frame_offset": 1, "max_pixel_distance": 50.,
            "start_frame": frames[0], "end_frame": frames[-1],
            "sample_step": frames[1] - frames[0],
            "runtime_model_path": export["model_path"] if backend == "openvino" else None,
        }
        payload = {
            "status": "EXPERIMENTAL_DETECTION_ONLY",
            "sample_frames": frames,
            "trial": trial,
            "model_sha256": digest(weights),
            "ground_truth_sha256": digest(truth),
            "class_map": {"motorcycle": 3},
            "runtime_backend": backend,
            "runtime_model_sha256": (
                export["export_sha256"] if backend == "openvino"
                else digest(weights)),
            "standard": score_detections(std, gt, frames),
            "sliced": score_detections(sliced, gt, frames),
        }
        (folder / "report.json").write_text(json.dumps(payload), encoding="utf-8")
    return ref, cand


def test_equal_aggregate_match_count_can_hide_swapped_motorcycle(tmp_path):
    reference, candidate = make_cases(tmp_path)
    result = compare_detector_truth_identity(reference, candidate)
    bike = result["by_class"]["MOTORCYCLE"]
    assert result["eligible_for_promotion"] is False
    assert result["samples"] == [0, 2]
    assert bike == {
        "truth_observations": 4,
        "matched_by_both": 1,
        "pytorch_only": 1,
        "openvino_only": 1,
        "neither": 1,
        "annotated_vehicle_observation_disagreements": [
            {"video_frame": 0, "fluid_track_id": "a", "found_by": "pytorch_only"},
            {"video_frame": 0, "fluid_track_id": "b", "found_by": "openvino_only"},
        ],
    }
    assert result["total_annotation_observation_disagreements"] == 2
    assert result["source_annotated_frame_offset"] == 1


def test_same_settings_must_be_applied_to_reference_and_candidate(tmp_path):
    reference, candidate = make_cases(tmp_path)
    report_file = candidate / "report.json"
    report = json.loads(report_file.read_text())
    report["trial"]["overlap"] = .1
    report_file.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="overlap"):
        compare_detector_truth_identity(reference, candidate)


def test_sha_and_backend_provenance_is_mandatory(tmp_path):
    reference, candidate = make_cases(tmp_path)
    report_file = candidate / "report.json"
    report = json.loads(report_file.read_text())
    report["runtime_model_sha256"] = "0" * 64
    report_file.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_detector_truth_identity(reference, candidate)


def test_output_is_immutable(tmp_path):
    reference, candidate = make_cases(tmp_path)
    output = tmp_path / "parity.json"
    cli = (Path(__file__).resolve().parents[1] / "scripts"
           / "phase3_paired_detector_truth_parity.py")
    args = [sys.executable, str(cli), "--reference-dir", str(reference),
            "--candidate-dir", str(candidate), "--output", str(output)]
    run = subprocess.run(args, text=True, capture_output=True, check=True)
    assert json.loads(run.stdout)["motorcycle"]["pytorch_only"] == 1
    assert len(json.loads(output.read_text())["by_class"]) == 4
    second = subprocess.run(args, text=True, capture_output=True)
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
