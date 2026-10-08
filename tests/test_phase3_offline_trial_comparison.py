"""Offline A/B comparison with frozen FLUID frames, no inference dependencies."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.video.offline_trial_comparison import compare_existing_runs


def track(frame: int, tid: int, x: int, klass: int) -> str:
    return f"{frame},{tid},{x},20,10,10,{x},20,10,10,{klass},0.90,10,10\n"


def make_inputs(tmp_path):
    gt = tmp_path / "truth.csv"
    gt.write_text(
        "frame,id,cx,cy,type\n"
        "1,1,10,20,moped\n1,2,100,20,car\n"
        "2,1,12,20,moped\n2,2,100,20,car\n"
        "3,1,14,20,moped\n3,2,100,20,car\n",
        encoding="utf-8",
    )
    a, b = tmp_path / "standard.txt", tmp_path / "sliced.txt"
    a.write_text(
        track(0, 101, 10, 3) + track(0, 102, 100, 0)
        + track(1, 102, 100, 0)
        + track(2, 102, 100, 0), encoding="utf-8")
    b.write_text(
        track(0, 201, 10, 3) + track(0, 202, 100, 0)
        + track(1, 201, 12, 3) + track(1, 202, 100, 0)
        + track(2, 201, 14, 3) + track(2, 202, 100, 0)
        + track(2, 203, 500, 3), encoding="utf-8")
    shared = {
        "video": "test.mp4", "model_sha256": "a" * 64,
        "confidence": 0.15, "class_map": {"motorcycle": 3, "car": 0},
        "evaluation_frame_offset": 1,
        "track_activation_threshold": 0.2,
        "high_conf_det_threshold": 0.15,
        "lost_track_buffer": 45, "minimum_consecutive_frames": 2,
        "start_frame": 0, "end_frame": 3,
        "elapsed_seconds": 5.0, "processed_frames": 6,
    }
    for path, mode, input_size, warmup in (
        (a, "standard", 1920, 5),
        (b, "sliced", 640, 2),
    ):
        manifest = {**shared, "mode": mode, "image_size": input_size,
                    "warmup_frames": warmup,
                    "first_decoded_frame": 0,
                    "evaluation_frames": 4}
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
    return a, b, gt


def test_same_frames_reports_real_motorcycle_gain_without_promotion(tmp_path):
    a, b, gt = make_inputs(tmp_path)
    r = compare_existing_runs(
        standard_tracks=a, sliced_tracks=b, fluid_tracks=gt,
        start_frame=0, end_frame=2)
    assert r["evaluation_frame_count"] == 3
    assert r["truth_points"] == 6
    assert r["eligible_for_promotion"] is False
    assert r["results"]["standard"]["pixel"]["truth_points"] == 6
    assert r["results"]["standard"]["pixel"]["matched_points"] == 4
    assert r["results"]["standard"]["pixel"]["point_precision"] == 1.0
    assert r["results"]["sliced"]["pixel"]["matched_points"] == 6
    assert r["results"]["sliced"]["pixel"]["predicted_points"] == 7
    assert r["results"]["sliced"]["pixel"]["point_recall"] == 1.0
    assert r["results"]["sliced"]["class_diagnostics"]["by_class"]["MOTORCYCLE"]["spatial_recall"] == 1.0
    assert r["results"]["standard"]["class_diagnostics"]["by_class"]["MOTORCYCLE"]["spatial_misses"] == 2
    assert r["deltas_sliced_minus_standard"]["motorcycle_spatial_misses_delta"] == -2
    assert r["deltas_sliced_minus_standard"]["unmatched_predictions_delta"] == 1
    assert "different detector input resolutions" in r["confounders"]
    assert "different tracker warm-up lengths" in r["confounders"]
    assert any("fewer than 100 evaluation frames" in item for item in r["confounders"])


def test_same_model_hash_and_tracker_settings_required(tmp_path):
    a, b, gt = make_inputs(tmp_path)
    payload = json.loads(b.with_suffix(".manifest.json").read_text())
    payload["model_sha256"] = "b" * 64
    b.with_suffix(".manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model_sha256"):
        compare_existing_runs(
            standard_tracks=a, sliced_tracks=b, fluid_tracks=gt,
            start_frame=0, end_frame=2)
    payload["model_sha256"] = "a" * 64
    payload["lost_track_buffer"] = 90
    b.with_suffix(".manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="lost_track_buffer"):
        compare_existing_runs(
            standard_tracks=a, sliced_tracks=b, fluid_tracks=gt,
            start_frame=0, end_frame=2)


def test_disallows_unavailable_intervals_and_file_self_comparison(tmp_path):
    a, b, gt = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="does not cover"):
        compare_existing_runs(
            standard_tracks=a, sliced_tracks=b, fluid_tracks=gt,
            start_frame=0, end_frame=10)
    with pytest.raises(ValueError, match="itself"):
        compare_existing_runs(
            standard_tracks=a, sliced_tracks=a, fluid_tracks=gt,
            start_frame=0, end_frame=2)
    with pytest.raises(ValueError, match="Invalid inclusive"):
        compare_existing_runs(
            standard_tracks=a, sliced_tracks=b, fluid_tracks=gt,
            start_frame=4, end_frame=3)


def test_cli_exports_full_report_and_summary_and_never_overwrites(tmp_path):
    a, b, gt = make_inputs(tmp_path)
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts/phase3_compare_existing_trials.py"
    command = [
        sys.executable, str(script), "--standard-tracks", str(a),
        "--sliced-tracks", str(b), "--fluid-tracks", str(gt),
        "--start-frame", "0", "--end-frame", "2", "--output", str(output),
    ]
    run = subprocess.run(command, text=True, capture_output=True, check=True)
    summary = json.loads(run.stdout.split("Full diagnostic written to:")[0])
    assert summary["standard"]["motorcycle_spatial_recall"] == pytest.approx(1/3)
    assert summary["sliced"]["motorcycle_spatial_recall"] == 1
    assert summary["deltas_sliced_minus_standard"]["motorcycle_spatial_misses_delta"] == -2
    assert json.loads(output.read_text())["eligible_for_promotion"] is False
    again = subprocess.run(command, text=True, capture_output=True)
    assert again.returncode != 0
    assert "FileExistsError" in again.stderr
