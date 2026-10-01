from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parse_geotrax_pixel_tracks_uses_documented_14_column_schema(tmp_path):
    from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks

    path = tmp_path / "video.txt"
    path.write_text(
        "0,10,100,200,20,10,101,201,20,10,3,0.9,21,11\n"
        "1,10,102,201,20,10,103,202,20,10,3,0.8,21,11\n",
        encoding="utf-8",
    )

    rows = load_geotrax_pixel_tracks(path)

    assert len(rows) == 2
    assert rows[0].frame == 0
    assert rows[0].track_id == "10"
    assert rows[0].x_px == 100.0
    assert rows[0].y_px == 200.0
    assert rows[0].x_stabilized_px == 101.0
    assert rows[0].vehicle_class == "MOTORCYCLE"
    assert rows[0].confidence == pytest.approx(0.9)


def test_parse_geotrax_pixel_tracks_rejects_bad_column_count(tmp_path):
    from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks

    path = tmp_path / "bad.txt"
    path.write_text("0,1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 14 columns"):
        load_geotrax_pixel_tracks(path)


def test_fluid_pixel_ground_truth_filters_unsupported_classes_and_unknowns():
    from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth

    rows = [
        {"frame": "1", "id": "1", "cx": "100", "cy": "200", "type": "moped", "confidence": "0.9"},
        {"frame": "1", "id": "2", "cx": "110", "cy": "210", "type": "car", "confidence": "0.9"},
        {"frame": "1", "id": "3", "cx": "120", "cy": "220", "type": "tricycle", "confidence": "0.9"},
        {"frame": "1", "id": "4", "cx": "", "cy": "", "type": "car", "confidence": ""},
    ]

    truth = normalize_fluid_pixel_truth(rows)

    assert len(truth) == 2
    assert {x.vehicle_class for x in truth} == {"MOTORCYCLE", "CAR"}


def test_pixel_benchmark_auto_detects_one_frame_offset_and_matches_hungarian():
    from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
    from streetlab_phase3.pixel_benchmark import FluidPixelTruth, PixelBenchmark

    predicted = [
        GeoTraxPixelPoint(0, "p1", 100, 100, 100, 100, "CAR", 0.9),
        GeoTraxPixelPoint(0, "p2", 200, 200, 200, 200, "MOTORCYCLE", 0.8),
        GeoTraxPixelPoint(1, "p1", 102, 100, 102, 100, "CAR", 0.9),
    ]
    truth = [
        FluidPixelTruth(1, "g1", 101, 100, "CAR"),
        FluidPixelTruth(1, "g2", 199, 200, "MOTORCYCLE"),
        FluidPixelTruth(2, "g1", 103, 100, "CAR"),
    ]

    report = PixelBenchmark(max_distance_px=10).evaluate(predicted, truth)

    assert report.frame_offset == 1
    assert report.matched_points == 3
    assert report.truth_points == 3
    assert report.predicted_points == 3
    assert report.point_recall == pytest.approx(1.0)
    assert report.point_precision == pytest.approx(1.0)
    assert report.class_agreement == pytest.approx(1.0)
    assert report.pixel_mae == pytest.approx(1.0)
    assert report.track_coverage == pytest.approx(1.0)


def test_pixel_benchmark_penalizes_class_mismatch_but_still_matches_position():
    from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
    from streetlab_phase3.pixel_benchmark import FluidPixelTruth, PixelBenchmark

    predicted = [
        GeoTraxPixelPoint(0, "p1", 100, 100, 100, 100, "CAR", 0.9),
    ]
    truth = [
        FluidPixelTruth(0, "g1", 100, 100, "MOTORCYCLE"),
    ]

    report = PixelBenchmark(max_distance_px=10).evaluate(
        predicted,
        truth,
        frame_offsets=(0,),
    )

    assert report.matched_points == 1
    assert report.class_agreement == 0.0
    assert report.pixel_rmse == 0.0


def test_pixel_benchmark_uses_one_to_one_assignment():
    from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
    from streetlab_phase3.pixel_benchmark import FluidPixelTruth, PixelBenchmark

    predicted = [
        GeoTraxPixelPoint(0, "p1", 0, 0, 0, 0, "CAR", 0.9),
        GeoTraxPixelPoint(0, "p2", 2, 0, 2, 0, "CAR", 0.9),
    ]
    truth = [
        FluidPixelTruth(0, "g1", 1, 0, "CAR"),
    ]

    report = PixelBenchmark(max_distance_px=5).evaluate(
        predicted,
        truth,
        frame_offsets=(0,),
    )

    assert report.matched_points == 1
    assert report.point_precision == pytest.approx(0.5)
    assert report.point_recall == pytest.approx(1.0)


def test_geotrax_compare_cli_writes_json_scorecard(tmp_path):
    from streetlab_phase3.video_benchmark_cli import run_pixel_benchmark

    tracks = tmp_path / "geotrax.txt"
    tracks.write_text(
        "0,10,100,200,20,10,100,200,20,10,0,0.9,20,10\n",
        encoding="utf-8",
    )
    fluid = tmp_path / "fluid.csv"
    fluid.write_text(
        "frame,id,cx,cy,type\n"
        "1,1,101,200,car\n",
        encoding="utf-8",
    )
    output = tmp_path / "score.json"

    report = run_pixel_benchmark(
        geotrax_tracks=tracks,
        fluid_tracks=fluid,
        output=output,
        max_distance_px=10,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert report.frame_offset == 1
    assert payload["matched_points"] == 1
    assert payload["class_agreement"] == 1.0


def test_windows_runner_script_exists_and_pins_geotrax():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "scripts" / "phase3_setup_geotrax_windows.ps1").read_text(encoding="utf-8")
    runner = (root / "scripts" / "phase3_run_geotrax_windows.ps1").read_text(encoding="utf-8")

    assert "geo-trax==1.5.1" in setup
    assert ".venv-geotrax" in setup
    assert "--no-geo" in runner
    assert "--cut-frame-right" in runner
    assert "phase3_compare_geotrax_fluid.py" in runner
    assert "Smoke" in runner and "Full" in runner


def test_compare_script_bootstraps_repo_root_for_direct_execution():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "phase3_compare_geotrax_fluid.py").read_text(encoding="utf-8")
    assert "sys.path.insert" in script
    assert "Path(__file__).resolve().parents[1]" in script


def test_runner_sets_pythonpath_and_can_reuse_existing_tracks():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts" / "phase3_run_geotrax_windows.ps1").read_text(encoding="utf-8")
    assert "$env:PYTHONPATH = $RepoRoot" in runner
    assert "[switch]$CompareOnly" in runner
    assert "CompareOnly" in runner


def test_pixel_benchmark_limits_truth_denominator_to_overlapping_frames():
    from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
    from streetlab_phase3.pixel_benchmark import FluidPixelTruth, PixelBenchmark

    predicted = [
        GeoTraxPixelPoint(0, "p1", 100, 100, 100, 100, "CAR", 0.9),
        GeoTraxPixelPoint(1, "p1", 101, 100, 101, 100, "CAR", 0.9),
    ]
    truth = [
        FluidPixelTruth(1, "g1", 100, 100, "CAR"),
        FluidPixelTruth(2, "g1", 101, 100, "CAR"),
        FluidPixelTruth(100, "g2", 400, 400, "CAR"),
    ]

    report = PixelBenchmark(max_distance_px=10).evaluate(predicted, truth)

    assert report.frame_offset == 1
    assert report.dataset_truth_points == 3
    assert report.truth_points == 2
    assert report.dataset_truth_tracks == 2
    assert report.truth_tracks == 1
    assert report.matched_points == 2
    assert report.point_recall == pytest.approx(1.0)
    assert report.track_coverage == pytest.approx(1.0)
    assert report.evaluation_frame_start == 1
    assert report.evaluation_frame_end == 2
