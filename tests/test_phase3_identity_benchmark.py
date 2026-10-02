from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth, PixelBenchmark


def _pred(
    frame: int,
    track_id: str,
    x_px: float = 0.0,
    y_px: float = 0.0,
) -> GeoTraxPixelPoint:
    return GeoTraxPixelPoint(
        frame=frame,
        track_id=track_id,
        x_px=x_px,
        y_px=y_px,
        x_stabilized_px=x_px,
        y_stabilized_px=y_px,
        vehicle_class="CAR",
        confidence=0.9,
    )


def _truth(
    frame: int,
    track_id: str,
    x_px: float = 0.0,
    y_px: float = 0.0,
) -> FluidPixelTruth:
    return FluidPixelTruth(
        frame=frame,
        track_id=track_id,
        x_px=x_px,
        y_px=y_px,
        vehicle_class="CAR",
    )


def _evaluate(predicted, truth, *, max_switch_gap_frames=5):
    from streetlab_phase3.identity_benchmark import IdentityBenchmark

    return IdentityBenchmark(
        max_distance_px=5,
        max_switch_gap_frames=max_switch_gap_frames,
    ).evaluate(predicted, truth, frame_offsets=(0,))


def test_pixel_benchmark_exposes_fixed_offset_accepted_matches_without_score_change():
    predicted = [_pred(1, "p1", 0, 0), _pred(1, "p2", 100, 100)]
    truth = [_truth(2, "t1", 3, 4), _truth(2, "t2", 200, 200)]
    benchmark = PixelBenchmark(max_distance_px=5)

    matches = benchmark.match_points(predicted, truth, frame_offset=1)
    report = benchmark.evaluate(predicted, truth, frame_offsets=(1,))

    assert matches == [(predicted[0], truth[0], 5.0)]
    assert report.frame_offset == 1
    assert report.matched_points == 1
    assert report.pixel_mae == 5.0
    assert report.pixel_rmse == 5.0


def test_identity_perfect_one_to_one_track_is_unfragmented():
    predicted = [_pred(frame, "A") for frame in range(1, 6)]
    truth = [_truth(frame, "T") for frame in range(1, 6)]

    result = _evaluate(predicted, truth)
    track = result.truth_track_details[0]

    assert track.associated_predicted_ids == ("A",)
    assert track.fragmentation_excess == 0
    assert track.raw_id_switches == 0
    assert track.contiguous_id_switches == 0
    assert track.dominant_pair_fraction == 1.0
    assert result.scorecard.total_truth_fragmentation_excess == 0
    assert result.scorecard.fraction_truth_tracks_unfragmented == 1.0
    assert result.scorecard.mean_dominant_pair_fraction == 1.0


def test_identity_one_truth_track_split_between_two_predictions():
    predicted = [
        *[_pred(frame, "A") for frame in range(1, 6)],
        *[_pred(frame, "B") for frame in range(6, 11)],
    ]
    truth = [_truth(frame, "T") for frame in range(1, 11)]

    result = _evaluate(predicted, truth)
    track = result.truth_track_details[0]

    assert track.associated_predicted_ids == ("A", "B")
    assert track.associated_predicted_id_count == 2
    assert track.fragmentation_excess == 1
    assert track.raw_id_switches == 1
    assert track.contiguous_id_switches == 1
    assert result.scorecard.truth_tracks_with_2_predicted_ids == 1
    assert result.scorecard.total_truth_fragmentation_excess == 1


def test_identity_change_after_large_temporal_gap_is_not_contiguous_switch():
    predicted = [
        *[_pred(frame, "A") for frame in range(1, 4)],
        *[_pred(frame, "B") for frame in range(20, 23)],
    ]
    truth = [
        *[_truth(frame, "T") for frame in range(1, 4)],
        *[_truth(frame, "T") for frame in range(20, 23)],
    ]

    result = _evaluate(predicted, truth, max_switch_gap_frames=5)
    track = result.truth_track_details[0]

    assert track.raw_id_switches == 1
    assert track.contiguous_id_switches == 0
    assert result.scorecard.total_raw_id_switches == 1
    assert result.scorecard.total_contiguous_id_switches == 0


def test_identity_one_prediction_can_match_two_truth_tracks_without_fragmenting_either():
    predicted = [
        *[_pred(frame, "A") for frame in range(1, 4)],
        *[_pred(frame, "A", 100, 0) for frame in range(4, 7)],
    ]
    truth = [
        *[_truth(frame, "T1") for frame in range(1, 4)],
        *[_truth(frame, "T2", 100, 0) for frame in range(4, 7)],
    ]

    result = _evaluate(predicted, truth)

    assert result.scorecard.truth_tracks == 2
    assert result.scorecard.predicted_tracks == 1
    assert result.scorecard.matched_truth_tracks == 2
    assert result.scorecard.matched_predicted_tracks == 1
    assert result.scorecard.total_truth_fragmentation_excess == 0
    assert all(
        track.associated_predicted_id_count == 1
        for track in result.truth_track_details
    )


def test_identity_unmatched_prediction_is_not_counted_as_fragmentation():
    predicted = [
        *[_pred(frame, "A") for frame in range(1, 4)],
        *[_pred(frame, "FALSE_POSITIVE", 1000, 1000) for frame in range(1, 4)],
    ]
    truth = [_truth(frame, "T") for frame in range(1, 4)]

    result = _evaluate(predicted, truth)

    assert result.scorecard.predicted_tracks == 2
    assert result.scorecard.matched_predicted_tracks == 1
    assert result.scorecard.unmatched_predicted_tracks == 1
    assert result.scorecard.unmatched_predicted_track_fraction == 0.5
    assert result.scorecard.total_truth_fragmentation_excess == 0


def test_identity_partial_detection_measures_fraction_and_longest_run():
    predicted = [
        *[_pred(frame, "A") for frame in (1, 2, 3, 7, 8)],
        _pred(10, "A", 100, 100),
    ]
    truth = [_truth(frame, "T") for frame in range(1, 11)]

    result = _evaluate(predicted, truth)
    track = result.truth_track_details[0]

    assert track.truth_frame_count == 10
    assert track.matched_frame_count == 5
    assert track.matched_fraction == pytest.approx(5 / 10)
    assert track.longest_consecutive_match_run == 3
    assert track.longest_run_fraction == pytest.approx(3 / 10)
    assert result.scorecard.mean_truth_track_matched_fraction == pytest.approx(5 / 10)
    assert result.scorecard.mean_longest_run_fraction == pytest.approx(3 / 10)


def test_identity_aggregate_denominators_use_the_evaluation_window():
    predicted = [
        *[_pred(frame, "A") for frame in range(1, 4)],
        _pred(10, "FALSE_POSITIVE", 1000, 1000),
    ]
    truth = [
        *[_truth(frame, "MATCHED") for frame in range(1, 4)],
        *[_truth(frame, "UNMATCHED", 100, 100) for frame in range(4, 7)],
        _truth(20, "OUTSIDE_WINDOW", 200, 200),
    ]

    result = _evaluate(predicted, truth)

    assert result.scorecard.truth_tracks == 2
    assert result.scorecard.matched_truth_tracks == 1
    assert result.scorecard.unmatched_truth_tracks == 1
    assert result.scorecard.mean_truth_track_matched_fraction == 0.5
    assert result.scorecard.mean_longest_run_fraction == 0.5
    assert result.scorecard.mean_predicted_ids_per_truth_track == 1.0
    assert result.scorecard.fraction_truth_tracks_unfragmented == 1.0
    assert {detail.truth_track_id for detail in result.truth_track_details} == {
        "MATCHED",
        "UNMATCHED",
    }


def test_identity_cli_runner_writes_scorecard_and_detail_csvs(tmp_path):
    from streetlab_phase3.identity_benchmark import run_identity_benchmark

    geotrax = tmp_path / "geotrax.txt"
    geotrax.write_text(
        "0,10,100,200,20,10,100,200,20,10,0,0.9,20,10\n"
        "1,10,101,200,20,10,101,200,20,10,0,0.9,20,10\n",
        encoding="utf-8",
    )
    fluid = tmp_path / "fluid.csv"
    fluid.write_text(
        "frame,id,cx,cy,type\n"
        "1,1,100,200,car\n"
        "2,1,101,200,car\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "identity"

    result = run_identity_benchmark(
        geotrax_tracks=geotrax,
        fluid_tracks=fluid,
        output_dir=output_dir,
        max_distance_px=5,
        max_switch_gap_frames=5,
    )

    payload = json.loads(
        (output_dir / "identity_scorecard.json").read_text(encoding="utf-8")
    )
    with (output_dir / "identity_truth_tracks.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        truth_reader = csv.DictReader(fh)
        truth_rows = list(truth_reader)
    with (output_dir / "identity_pairs.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        pair_rows = list(csv.DictReader(fh))

    assert payload == result.scorecard_dict()
    assert payload["frame_offset"] == 1
    assert payload["max_distance_px"] == 5.0
    assert payload["max_switch_gap_frames"] == 5
    assert set(payload) == {
        "frame_offset",
        "max_distance_px",
        "max_switch_gap_frames",
        "truth_tracks",
        "predicted_tracks",
        "matched_truth_tracks",
        "matched_predicted_tracks",
        "unmatched_truth_tracks",
        "unmatched_predicted_tracks",
        "unmatched_predicted_track_fraction",
        "mean_predicted_ids_per_truth_track",
        "median_predicted_ids_per_truth_track",
        "max_predicted_ids_per_truth_track",
        "truth_tracks_with_exactly_1_predicted_id",
        "truth_tracks_with_2_predicted_ids",
        "truth_tracks_with_3plus_predicted_ids",
        "fraction_truth_tracks_unfragmented",
        "fraction_truth_tracks_fragmented",
        "total_truth_fragmentation_excess",
        "total_raw_id_switches",
        "total_contiguous_id_switches",
        "mean_contiguous_id_switches_per_matched_truth_track",
        "truth_tracks_with_zero_contiguous_switches",
        "fraction_truth_tracks_with_zero_contiguous_switches",
        "mean_truth_track_matched_fraction",
        "median_truth_track_matched_fraction",
        "mean_longest_run_fraction",
        "median_longest_run_fraction",
        "mean_dominant_pair_fraction",
        "median_dominant_pair_fraction",
    }
    assert truth_reader.fieldnames == [
        "truth_track_id",
        "truth_class",
        "truth_frame_count",
        "matched_frame_count",
        "matched_fraction",
        "associated_predicted_ids",
        "associated_predicted_id_count",
        "fragmentation_excess",
        "raw_id_switches",
        "contiguous_id_switches",
        "dominant_predicted_id",
        "dominant_pair_matched_frames",
        "dominant_pair_fraction",
        "longest_consecutive_match_run",
        "longest_run_fraction",
    ]
    assert truth_rows[0]["associated_predicted_ids"] == "10"
    assert truth_rows[0]["dominant_predicted_id"] == "10"
    assert pair_rows == [
        {
            "truth_track_id": "1",
            "predicted_track_id": "10",
            "matched_frames": "2",
            "first_matched_frame": "1",
            "last_matched_frame": "2",
            "mean_pixel_error": "0.0",
        }
    ]


def test_identity_rejects_non_positive_switch_gap():
    from streetlab_phase3.identity_benchmark import IdentityBenchmark

    with pytest.raises(ValueError, match="max_switch_gap_frames must be positive"):
        IdentityBenchmark(max_switch_gap_frames=0)


def test_identity_script_prints_json_summary_to_stdout(tmp_path):
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
    output_dir = tmp_path / "identity"
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_identity_benchmark.py"

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
            "--max-pixel-distance",
            "5",
            "--max-switch-gap-frames",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == json.loads(
        (output_dir / "identity_scorecard.json").read_text(encoding="utf-8")
    )
