from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth


def _pred(
    frame: int,
    track_id: str,
    x_px: float,
    *,
    vehicle_class: str = "CAR",
) -> GeoTraxPixelPoint:
    return GeoTraxPixelPoint(
        frame=frame,
        track_id=track_id,
        x_px=x_px,
        y_px=0.0,
        x_stabilized_px=x_px,
        y_stabilized_px=0.0,
        vehicle_class=vehicle_class,
        confidence=0.9,
    )


def _truth(
    frame: int,
    track_id: str,
    x_px: float,
    *,
    vehicle_class: str = "CAR",
) -> FluidPixelTruth:
    return FluidPixelTruth(
        frame=frame,
        track_id=track_id,
        x_px=x_px,
        y_px=0.0,
        vehicle_class=vehicle_class,
    )


def _evaluate(predicted, truth, *, frame_offset=0):
    from streetlab_phase3.recall_loss_diagnosis import RecallLossDiagnosis

    return RecallLossDiagnosis(max_distance_px=5).evaluate(
        predicted,
        truth,
        frame_offset=frame_offset,
    )


def test_benchmark_population_is_filtered_before_matching_and_unsupported_is_reported():
    predicted = [_pred(0, "P", 0)]
    truth = [
        _truth(0, "PEDESTRIAN", 0, vehicle_class="PEDESTRIAN"),
        _truth(0, "CAR", 1, vehicle_class="CAR"),
    ]

    result = _evaluate(predicted, truth)
    classes = {row.truth_class: row for row in result.per_class_recall}

    assert result.benchmark_compatible_report.truth_points == 1
    assert result.benchmark_compatible_report.matched_points == 1
    assert result.benchmark_compatible_report.point_recall == 1.0
    assert result.diagnostic_report.truth_points == 2
    assert result.diagnostic_report.matched_points == 1
    assert classes["PEDESTRIAN"].benchmark_supported is False
    assert classes["PEDESTRIAN"].matched_points == 1
    assert classes["CAR"].benchmark_supported is True
    assert classes["CAR"].matched_points == 0


def test_miss_runs_require_actual_consecutive_frame_numbers():
    predicted = [
        _pred(1, "P", 0),
        _pred(5, "P", 0),
        _pred(6, "WINDOW", 100),
    ]
    truth = [_truth(frame, "T", 0) for frame in (1, 2, 4, 5, 6)]

    result = _evaluate(predicted, truth)
    track = result.track_diagnostics[0]

    assert track.truth_frames == 5
    assert track.matched_frames == 2
    assert track.miss_event_count == 3
    assert track.maximum_miss_gap_frames == 1
    assert track.mean_miss_gap_frames == 1.0
    assert track.first_match_delay_frames == 0
    assert track.trailing_miss_frames == 1


def test_fragmentation_fraction_uses_matched_truth_tracks_only():
    predicted = [
        _pred(1, "A", 0),
        _pred(2, "B", 0),
        _pred(1, "C", 100),
        _pred(2, "C", 100),
        _pred(2, "WINDOW", 1000),
    ]
    truth = [
        _truth(1, "T1", 0),
        _truth(2, "T1", 0),
        _truth(1, "T2", 100),
        _truth(2, "T2", 100),
        _truth(1, "T3", 200),
        _truth(2, "T3", 200),
    ]

    result = _evaluate(predicted, truth)
    car = result.per_class_recall[0]

    assert car.truth_tracks == 3
    assert car.matched_truth_tracks == 2
    assert car.fraction_fragmented == 0.5
    assert car.mean_predicted_ids_per_truth_track == 1.5
    assert result.report_dict()["fragmentation"]["fraction_fragmented"] == 0.5


def test_gap_distribution_covers_all_buckets_and_whole_track_misses():
    run_lengths = (1, 2, 4, 11, 31)
    predicted = []
    truth = []
    for index, run_length in enumerate(run_lengths):
        x_px = index * 100.0
        track_id = f"T{run_length}"
        predicted.extend(
            [
                _pred(0, f"P{run_length}", x_px),
                _pred(run_length + 1, f"P{run_length}", x_px),
            ]
        )
        truth.extend(
            _truth(frame, track_id, x_px)
            for frame in range(run_length + 2)
        )
    truth.extend(_truth(frame, "WHOLE", 1000) for frame in range(3))

    result = _evaluate(predicted, truth)
    overall = {
        row.gap_bucket: row
        for row in result.miss_gap_distribution
        if row.truth_class == "ALL"
    }

    assert list(overall) == [
        "1 frame",
        "2-3 frames",
        "4-10 frames",
        "11-30 frames",
        "> 30 frames",
        "whole track missed",
    ]
    assert [row.gap_event_count for row in overall.values()] == [1, 1, 1, 1, 1, 1]
    assert [row.missed_frame_count for row in overall.values()] == [1, 2, 4, 11, 31, 3]
    assert all(row.gap_event_fraction == pytest.approx(1 / 6) for row in overall.values())
    assert sum(row.missed_frame_fraction for row in overall.values()) == pytest.approx(1.0)
    assert result.report_dict()["recall_loss_mode"]["dominant_mode"] == "long gaps"


def test_perfect_recall_reports_no_recall_loss_mode():
    predicted = [_pred(frame, "P", 0) for frame in range(1, 4)]
    truth = [_truth(frame, "T", 0) for frame in range(1, 4)]

    result = _evaluate(predicted, truth)

    assert result.report_dict()["recall_loss_mode"] == {
        "complete_miss_share": 0.0,
        "short_intermittent_gap_share": 0.0,
        "long_gap_share": 0.0,
        "dominant_mode": "no recall loss",
    }


def test_empty_evaluation_window_reports_not_evaluable():
    result = _evaluate([], [_truth(1, "T", 0)])

    assert result.diagnostic_report.truth_points == 0
    assert result.report_dict()["recall_loss_mode"]["dominant_mode"] == (
        "not evaluable"
    )


def test_runner_writes_required_csv_and_json_outputs(tmp_path):
    from streetlab_phase3.recall_loss_diagnosis import run_recall_loss_diagnosis

    geotrax = tmp_path / "geotrax.txt"
    geotrax.write_text(
        "0,10,100,200,20,10,100,200,20,10,0,0.9,20,10\n",
        encoding="utf-8",
    )
    fluid = tmp_path / "fluid.csv"
    fluid.write_text(
        "frame,id,cx,cy,type\n"
        "1,1,100,200,car\n"
        "1,2,500,500,pedestrian\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "diagnosis"

    result = run_recall_loss_diagnosis(
        geotrax_tracks=geotrax,
        fluid_tracks=fluid,
        output_dir=output_dir,
        frame_offset=1,
        max_distance_px=50,
    )

    report = json.loads(
        (output_dir / "recall_loss_report.json").read_text(encoding="utf-8")
    )
    with (output_dir / "per_class_recall.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        class_reader = csv.DictReader(fh)
        class_rows = list(class_reader)
    with (output_dir / "track_recall_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        track_reader = csv.DictReader(fh)
        track_rows = list(track_reader)
    with (output_dir / "miss_gap_distribution.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        gap_reader = csv.DictReader(fh)
        gap_rows = list(gap_reader)

    assert report == result.report_dict()
    assert report["frame_offset"] == 1
    assert report["max_distance_px"] == 50.0
    assert {row["truth_class"] for row in class_rows} == {"CAR", "PEDESTRIAN"}
    assert len(track_rows) == 2
    assert {row["gap_bucket"] for row in gap_rows} == {
        "1 frame",
        "2-3 frames",
        "4-10 frames",
        "11-30 frames",
        "> 30 frames",
        "whole track missed",
    }
    assert class_reader.fieldnames is not None
    assert set(
        (
            "truth_class",
            "truth_points",
            "matched_points",
            "missed_points",
            "point_recall",
            "truth_tracks",
            "matched_truth_tracks",
            "track_coverage",
            "mean_matched_fraction",
            "median_matched_fraction",
            "fraction_fragmented",
            "mean_predicted_ids_per_truth_track",
            "mean_longest_run_fraction",
            "median_longest_run_fraction",
        )
    ).issubset(class_reader.fieldnames)
    assert track_reader.fieldnames is not None
    assert set(
        (
            "truth_id",
            "truth_class",
            "truth_frames",
            "matched_frames",
            "matched_fraction",
            "number_of_predicted_ids",
            "fragmented",
            "longest_matched_run_frames",
            "longest_run_fraction",
            "miss_event_count",
            "maximum_miss_gap_frames",
            "mean_miss_gap_frames",
            "first_match_delay_frames",
            "trailing_miss_frames",
        )
    ).issubset(track_reader.fieldnames)


def test_cli_prints_report_and_uses_explicit_fixed_parameters(tmp_path):
    geotrax = tmp_path / "geotrax.txt"
    geotrax.write_text(
        "0,10,100,200,20,10,100,200,20,10,0,0.9,20,10\n",
        encoding="utf-8",
    )
    fluid = tmp_path / "fluid.csv"
    fluid.write_text(
        "frame,id,cx,cy,type\n1,1,100,200,car\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "diagnosis"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "phase3_recall_loss_diagnosis.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--geotrax-tracks",
            str(geotrax),
            "--fluid-tracks",
            str(fluid),
            "--output-dir",
            str(output_dir),
            "--frame-offset",
            "1",
            "--max-pixel-distance",
            "50",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == json.loads(
        (output_dir / "recall_loss_report.json").read_text(encoding="utf-8")
    )
