"""Regression suite: geometry-only proposals, same-cohort scoring, no oracle or writes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.video.existing_box_forensics import ExportedBox
from streetlab_phase3.video.offline_geometry_suppression import (
    POLICIES, propose_geometry_suppression, run_offline_suppression,
)


def box(frame, tid, klass, score, cx, cy, w, h):
    return ExportedBox(frame, str(tid), klass, score,
                       cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def row(frame, tid, cx, cy, w, h, cls, conf):
    return f"{frame},{tid},{cx},{cy},{w},{h},{cx},{cy},{w},{h},{cls},{conf},{w},{h}\n"


def test_same_class_greedy_suppresses_weaker_duplicate_but_preserves_nearby_distinct_bike():
    inputs = [
        box(0, 11, "MOTORCYCLE", .70, 100, 100, 20, 20),
        box(0, 12, "MOTORCYCLE", .20, 102, 100, 20, 20),
        box(0, 13, "MOTORCYCLE", .90, 135, 100, 20, 20),
    ]
    decisions = propose_geometry_suppression(
        inputs, motorcycle_iou=.50, car_inside_large=None)
    assert [(p["track_id"], p["reference_track_id"]) for p in decisions] == [("12", "11")]
    assert all(p["reason"] == "motorcycle_same_class_overlap" for p in decisions)
    assert inputs[0].track_id == "11"
    assert propose_geometry_suppression(
        [inputs[2]], motorcycle_iou=.30, car_inside_large=None) == []


def test_car_inside_large_requires_area_and_is_not_truth_aware():
    inputs = [
        box(0, 2, "BUS", .35, 300, 300, 120, 130),
        box(0, 3, "CAR", .95, 300, 300, 25, 25),
        box(0, 4, "CAR", .6, 300, 300, 130, 130),
    ]
    result = propose_geometry_suppression(
        inputs, motorcycle_iou=None, car_inside_large=.80)
    assert [p["track_id"] for p in result] == ["3"]
    assert result[0]["reference_class"] == "BUS"
    assert result[0]["confidence"] == .95 # even high-conf REAL cars are at risk!


def make_files(tmp_path):
    tracks = tmp_path / "sliced.txt"
    truth = tmp_path / "truth.csv"
    comparison = tmp_path / "comparison.json"
    tracks.write_text(
        row(0, 11, 10, 20, 20, 20, 3, .9)
        + row(0, 12, 12, 20, 20, 20, 3, .28)
        + row(0, 13, 65, 20, 20, 20, 3, .8)
        + row(0, 20, 200, 200, 160, 160, 1, .7)
        + row(0, 21, 202, 200, 20, 20, 0, .85)
        + row(1, 11, 10, 20, 20, 20, 3, .8)
        + row(1, 13, 65, 20, 20, 20, 3, .8),
        encoding="utf-8",
    )
    truth.write_text(
        "frame,id,cx,cy,type\n"
        "1,1,10,20,moped\n1,2,65,20,moped\n"
        "1,3,200,200,bus\n1,4,202,200,car\n"
        "2,1,10,20,moped\n2,2,65,20,moped\n",
        encoding="utf-8",
    )
    from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks
    from streetlab_phase3.pixel_benchmark import (
        PixelBenchmark, normalize_fluid_pixel_truth,
    )
    import csv
    with truth.open(newline="", encoding="utf-8") as fh:
        labels = normalize_fluid_pixel_truth(csv.DictReader(fh))
    score = PixelBenchmark(50).evaluate(
        load_geotrax_pixel_tracks(tracks), labels,
        frame_offsets=(1,), evaluation_frame_range=(1, 2))
    comparison.write_text(json.dumps({
        "status": "EXPLORATORY_SAME_FRAME_COMPARISON_NOT_PRODUCTION_EVIDENCE",
        "eligible_for_promotion": False,
        "evaluation_start_frame": 0,
        "evaluation_end_frame": 1,
        "evaluation_frame_count": 2,
        "frozen_fluid_offset": 1,
        "frozen_matching_radius_px": 50,
        "results": {"sliced": {
            "tracks": str(tracks),
            "pixel": {
                "truth_points": score.truth_points,
                "predicted_points": score.predicted_points,
                "matched_points": score.matched_points,
            }
        }},
    }), encoding="utf-8")
    return tracks, truth, comparison


def test_posthoc_scoring_removes_geometry_only_and_reveals_recall_cost(tmp_path):
    tracks, truth, comparison = make_files(tmp_path)
    baseline = tracks.read_bytes()
    r = run_offline_suppression(
        sliced_tracks=tracks, fluid_tracks=truth, comparison_file=comparison)
    assert r["eligible_for_promotion"] is False
    assert r["rules_use_truth_or_unmatched_status"] is False
    assert len(r["policies"]) == len(POLICIES) == 6
    assert r["original_sahi_metrics"]["truth_points"] == 6
    assert r["original_sahi_metrics"]["motorcycle_truth_points"] == 4
    moto = next(p for p in r["policies"] if p["policy"] == "motorcycle_iou_0p50")
    assert moto["proposed_removals"] == 1
    assert moto["decisions"][0]["track_id"] == "12"
    assert moto["changes_vs_unchanged_sahi"]["motorcycle_recall_percentage_points"] == 0
    cross = next(p for p in r["policies"] if p["policy"] == "car_inside_large_0p80")
    assert cross["proposed_removals"] == 1
    assert cross["decisions"][0]["track_id"] == "21" # matched real car: no oracle!
    assert cross["metrics"]["predicted_points"] == r["original_sahi_metrics"]["predicted_points"] - 1
    assert cross["metrics"]["car_correct_class_recall"] == 0.
    assert tracks.read_bytes() == baseline
    assert comparison.is_file()


def test_rejects_mutated_source_cohort(tmp_path):
    tracks, truth, comparison = make_files(tmp_path)
    payload = json.loads(comparison.read_text())
    payload["results"]["sliced"]["pixel"]["matched_points"] += 1
    comparison.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Baseline diverged"):
        run_offline_suppression(
            sliced_tracks=tracks, fluid_tracks=truth, comparison_file=comparison)


def test_cli_json_is_immutable_and_carries_raw_provenance(tmp_path):
    tracks, truth, comparison = make_files(tmp_path)
    output = tmp_path / "audit.json"
    cli = Path(__file__).resolve().parents[1] / "scripts" / "phase3_offline_geometry_suppression.py"
    args = [sys.executable, str(cli), "--sliced-tracks", str(tracks),
            "--fluid-tracks", str(truth), "--comparison", str(comparison),
            "--output", str(output)]
    first = subprocess.run(args, check=True, text=True, capture_output=True)
    summary = json.loads(first.stdout)
    assert summary["eligible_for_promotion"] is False
    assert len(summary["policies"]) == 6
    record = json.loads(output.read_text())
    assert len(record["source_tracks_sha256"]) == 64
    assert record["rules_use_truth_or_unmatched_status"] is False
    again = subprocess.run(args, text=True, capture_output=True)
    assert again.returncode != 0
    assert "FileExistsError" in again.stderr
