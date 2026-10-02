from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth


def _truth(frame: int, track_id: str, x_px: float = 0, y_px: float = 0):
    return FluidPixelTruth(frame, track_id, x_px, y_px, "MOTORCYCLE")


def _pred(
    frame: int,
    track_id: str,
    x_px: float = 0,
    y_px: float = 0,
    vehicle_class: str = "MOTORCYCLE",
):
    return GeoTraxPixelPoint(
        frame,
        track_id,
        x_px,
        y_px,
        x_px,
        y_px,
        vehicle_class,
        0.9,
    )


def _audit_track(
    truth_id: str,
    truth_frames: range | tuple[int, ...],
    matched_frames: tuple[int, ...],
    predicted_ids: tuple[str, ...],
):
    from streetlab_phase3.motorcycle_visual_audit import (
        AuditTrack,
        OfficialMatch,
        TrackDiagnosticRecord,
        consecutive_frame_runs,
    )

    points = tuple(_truth(frame, truth_id) for frame in truth_frames)
    matches = tuple(
        OfficialMatch(frame, predicted_ids[index % len(predicted_ids)], "MOTORCYCLE", 1.0)
        for index, frame in enumerate(matched_frames)
    ) if predicted_ids else ()
    frame_set = {point.frame for point in points}
    missed = frame_set - set(matched_frames)
    runs = consecutive_frame_runs(missed)
    lengths = [len(run) for run in runs]
    diagnostic = TrackDiagnosticRecord(
        truth_id=truth_id,
        truth_class="MOTORCYCLE",
        truth_frames=len(points),
        matched_frames=len(matched_frames),
        matched_fraction=len(matched_frames) / len(points),
        number_of_predicted_ids=len(predicted_ids),
        maximum_miss_gap_frames=max(lengths, default=0),
    )
    return AuditTrack(
        diagnostic=diagnostic,
        truth_points=points,
        official_matches=matches,
        miss_runs=runs,
    )


def test_consecutive_frame_runs_do_not_bridge_absent_truth_frames():
    from streetlab_phase3.motorcycle_visual_audit import consecutive_frame_runs

    assert consecutive_frame_runs({1, 2, 5, 6, 10}) == (
        (1, 2),
        (5, 6),
        (10,),
    )


def test_select_cases_is_deterministic_and_has_no_duplicate_truth_ids():
    from streetlab_phase3.motorcycle_visual_audit import select_audit_cases

    tracks = []
    tracks.extend(
        _audit_track(f"A{i}", range(1, 11 + i), (), ()) for i in range(5)
    )
    tracks.extend(
        _audit_track(str(track_id), range(1, 60), (1, 59), (f"P{i}",))
        for i, track_id in enumerate((3014, 2339, 3051, 3189, 2705))
    )
    tracks.extend(
        _audit_track(f"C{i}", range(1, 22 + i), (1, 21 + i), (f"Q{i}",))
        for i in range(5)
    )
    tracks.extend(
        _audit_track(f"D{i}", range(1, 11), tuple(range(1, 11)), (f"R{i}",))
        for i in range(5)
    )

    first = select_audit_cases(tracks)
    second = select_audit_cases(list(reversed(tracks)))

    assert [(case.case_id, case.truth_id) for case in first] == [
        (case.case_id, case.truth_id) for case in second
    ]
    assert len(first) == 20
    assert len({case.truth_id for case in first}) == 20
    assert [case.truth_id for case in first[5:10]] == [
        "3014",
        "2339",
        "3051",
        "3189",
        "2705",
    ]
    assert all(case.matched_fraction == 1.0 for case in first[15:20])


def test_long_and_medium_cases_use_the_proper_longest_gap():
    from streetlab_phase3.motorcycle_visual_audit import select_audit_cases

    tracks = [
        *[_audit_track(f"A{i}", range(1, 6), (), ()) for i in range(5)],
        *[
            _audit_track(
                f"B{i}",
                tuple(range(1, 5)) + tuple(range(10, 51)),
                (1, 50),
                (f"PB{i}",),
            )
            for i in range(5)
        ],
        *[
            _audit_track(f"C{i}", range(1, 22), (1, 21), (f"PC{i}",))
            for i in range(5)
        ],
        *[
            _audit_track(f"D{i}", range(1, 6), tuple(range(1, 6)), (f"PD{i}",))
            for i in range(5)
        ],
    ]

    cases = select_audit_cases(tracks, preferred_long_gap_ids=())

    assert all(case.target_gap == tuple(range(10, 50)) for case in cases[5:10])
    assert all(case.target_gap == tuple(range(2, 21)) for case in cases[10:15])


def test_representative_gap_frames_include_available_boundaries_and_quantiles():
    from streetlab_phase3.motorcycle_visual_audit import representative_gap_frames

    truth_frames = tuple(range(9, 51))
    gap = tuple(range(10, 50))

    assert representative_gap_frames(truth_frames, gap) == (
        9,
        10,
        20,
        30,
        39,
        49,
        50,
    )


def test_complete_miss_samples_only_actual_truth_frames():
    from streetlab_phase3.motorcycle_visual_audit import (
        representative_lifetime_frames,
        select_audit_cases,
    )

    truth_frames = (1, 2, 10, 11, 20, 21, 30)

    samples = representative_lifetime_frames(truth_frames)

    assert samples == truth_frames
    assert set(samples) <= set(truth_frames)

    tracks = [
        _audit_track("A0", truth_frames, (), ()),
        *[_audit_track(f"A{i}", range(1, 6), (), ()) for i in range(1, 5)],
        *[_audit_track(f"B{i}", range(1, 40), (1, 39), (f"PB{i}",)) for i in range(5)],
        *[_audit_track(f"C{i}", range(1, 20), (1, 19), (f"PC{i}",)) for i in range(5)],
        *[_audit_track(f"D{i}", range(1, 6), tuple(range(1, 6)), (f"PD{i}",)) for i in range(5)],
    ]

    case = select_audit_cases(tracks, preferred_long_gap_ids=())[0]

    assert case.truth_id == "A0"
    assert case.target_gap == (1, 2)
    assert set(case.sampled_frames) <= set(truth_frames)


def test_nearby_prediction_lookup_uses_offset_radius_and_distance_order():
    from streetlab_phase3.motorcycle_visual_audit import nearby_predictions

    truth = _truth(10, "T", 100, 100)
    predicted = [
        _pred(9, "near", 103, 104, "MOTORCYCLE"),
        _pred(9, "edge", 200, 100, "CAR"),
        _pred(9, "far", 201, 100, "CAR"),
        _pred(10, "wrong_frame", 100, 100, "MOTORCYCLE"),
    ]

    nearby = nearby_predictions(
        truth,
        predicted,
        frame_offset=1,
        radius_px=100,
    )

    assert [(row.predicted_track_id, row.pixel_distance) for row in nearby] == [
        ("near", 5.0),
        ("edge", 100.0),
    ]
    assert nearby[1].predicted_class == "CAR"


def test_truth_crops_preserve_raw_pixels_and_hollow_truth_center():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from streetlab_phase3.motorcycle_visual_audit import create_truth_crops

    image = np.zeros((200, 220, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(220, dtype=np.uint8)
    image[:, :, 1] = np.arange(200, dtype=np.uint8)[:, None]
    image[:, :, 2] = 73
    metadata = {
        "source_video_frame": 10,
        "truth_point": {
            "truth_id": "truth-at-edge",
            "raw_fluid_class": "moped",
            "canonical_class": "MOTORCYCLE",
            "benchmark_supported": True,
            "x_px": 2.0,
            "y_px": 3.0,
        },
        "matched": True,
        "official_benchmark_match": {
            "predicted_track_id": "prediction-with-a-long-id",
            "predicted_class": "MOTORCYCLE",
            "pixel_distance": 4.25,
        },
        "nearby_geotrax_observations": [
            {
                "predicted_track_id": "prediction-with-a-long-id",
                "predicted_class": "MOTORCYCLE",
                "predicted_frame": 9,
                "x_px": 2.0,
                "y_px": 3.0,
                "pixel_distance": 4.25,
            }
        ],
    }

    raw_zoom, overlay_zoom, bounds = create_truth_crops(
        image,
        metadata,
        crop_size=160,
        zoom_scale=3,
    )

    expected_raw = cv2.resize(
        image[0:160, 0:160],
        (480, 480),
        interpolation=cv2.INTER_NEAREST,
    )
    truth_zoom_xy = (6, 9)

    assert bounds == (0, 0, 160, 160)
    assert raw_zoom.shape == overlay_zoom.shape == (480, 480, 3)
    assert np.array_equal(raw_zoom, expected_raw)
    assert tuple(overlay_zoom[truth_zoom_xy[1], truth_zoom_xy[0]]) == tuple(
        raw_zoom[truth_zoom_xy[1], truth_zoom_xy[0]]
    )
    assert not np.array_equal(raw_zoom, overlay_zoom)


def test_truth_crop_bounds_shift_at_bottom_right_image_boundary():
    from streetlab_phase3.motorcycle_visual_audit import truth_crop_bounds

    assert truth_crop_bounds((200, 220, 3), 219, 199, crop_size=160) == (
        60,
        40,
        220,
        200,
    )


def test_raw_fluid_labels_are_preserved_without_changing_canonical_class(tmp_path):
    from streetlab_phase3.motorcycle_visual_audit import load_raw_fluid_labels

    source = tmp_path / "fluid.csv"
    source.write_text(
        "frame,id,cx,cy,type\n10,T,12,14,Scooter / Moped\n",
        encoding="utf-8",
    )

    assert load_raw_fluid_labels(source) == {("T", 10): "Scooter / Moped"}


def test_build_audit_tracks_rejects_duplicate_diagnostics_and_bad_fraction():
    from streetlab_phase3.motorcycle_visual_audit import (
        TrackDiagnosticRecord,
        build_audit_tracks,
    )

    predicted = [_pred(0, "P", 0, 0)]
    truth = [_truth(1, "T", 0, 0)]
    valid = TrackDiagnosticRecord("T", "MOTORCYCLE", 1, 1, 1.0, 1, 0)

    with pytest.raises(ValueError, match="Duplicate motorcycle diagnostic"):
        build_audit_tracks(predicted, truth, [valid, valid])

    inconsistent = TrackDiagnosticRecord(
        "T", "MOTORCYCLE", 1, 1, 0.25, 1, 0
    )
    with pytest.raises(ValueError, match="matched_fraction"):
        build_audit_tracks(predicted, truth, [inconsistent])


def test_manifest_paths_and_manual_review_fields_are_portable(tmp_path):
    from streetlab_phase3.motorcycle_visual_audit import manifest_row, select_audit_cases

    tracks = [
        *[_audit_track(f"A{i}", range(1, 6), (), ()) for i in range(5)],
        *[_audit_track(f"B{i}", range(1, 40), (1, 39), (f"PB{i}",)) for i in range(5)],
        *[_audit_track(f"C{i}", range(1, 20), (1, 19), (f"PC{i}",)) for i in range(5)],
        *[_audit_track(f"D{i}", range(1, 6), tuple(range(1, 6)), (f"PD{i}",)) for i in range(5)],
    ]
    case = select_audit_cases(tracks, preferred_long_gap_ids=())[0]

    row = manifest_row(case, tmp_path)

    assert row["contact_sheet_path"] == f"cases/{case.case_id}_{case.truth_id}/contact_sheet.jpg"
    assert row["raw_fluid_class"] == "MOTORCYCLE"
    assert row["canonical_truth_class"] == "MOTORCYCLE"
    assert row["benchmark_supported"] is True
    assert row["review_label"] == ""
    assert row["review_notes"] == ""


def test_cli_executes_on_synthetic_video_and_writes_twenty_cases(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (480, 400),
    )
    assert writer.isOpened()
    for frame in range(40):
        image = np.zeros((400, 480, 3), dtype=np.uint8)
        cv2.putText(
            image,
            str(frame + 1),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(image)
    writer.release()

    fluid = tmp_path / "fluid.csv"
    geotrax = tmp_path / "geotrax.txt"
    diagnostics = tmp_path / "track_diagnostics.csv"
    truth_rows = []
    prediction_rows = []
    diagnostic_rows = []
    predicted_id = 1

    groups = [
        ("A", range(1, 6), ()),
        ("B", range(1, 41), (1, 40)),
        ("C", range(1, 21), (1, 20)),
        ("D", range(1, 6), tuple(range(1, 6))),
    ]
    for group_index, (prefix, frames, matches) in enumerate(groups):
        for index in range(5):
            truth_id = f"{prefix}{index}"
            x_px = 40 + (index * 90)
            y_px = 40 + (group_index * 90)
            for frame in frames:
                truth_rows.append((frame, truth_id, x_px, y_px, "moped"))
            for frame in matches:
                prediction_rows.append(
                    (
                        frame - 1,
                        predicted_id,
                        x_px,
                        y_px,
                    )
                )
            truth_count = len(frames)
            matched_count = len(matches)
            max_gap = {"A": 5, "B": 38, "C": 18, "D": 0}[prefix]
            diagnostic_rows.append(
                {
                    "truth_id": truth_id,
                    "truth_class": "MOTORCYCLE",
                    "truth_frames": truth_count,
                    "matched_frames": matched_count,
                    "matched_fraction": matched_count / truth_count,
                    "number_of_predicted_ids": 1 if matches else 0,
                    "fragmented": False,
                    "longest_matched_run_frames": matched_count,
                    "longest_run_fraction": matched_count / truth_count,
                    "miss_event_count": 1 if max_gap else 0,
                    "maximum_miss_gap_frames": max_gap,
                    "mean_miss_gap_frames": float(max_gap),
                    "first_match_delay_frames": "" if not matches else 0,
                    "trailing_miss_frames": "" if not matches else 0,
                }
            )
            predicted_id += 1

    with fluid.open("w", encoding="utf-8", newline="") as fh:
        writer_csv = csv.writer(fh)
        writer_csv.writerow(("frame", "id", "cx", "cy", "type"))
        writer_csv.writerows(truth_rows)
    with geotrax.open("w", encoding="utf-8", newline="") as fh:
        writer_csv = csv.writer(fh)
        for frame, track_id, x_px, y_px in prediction_rows:
            writer_csv.writerow(
                (
                    frame,
                    track_id,
                    x_px,
                    y_px,
                    20,
                    10,
                    x_px,
                    y_px,
                    20,
                    10,
                    3,
                    0.9,
                    20,
                    10,
                )
            )
    with diagnostics.open("w", encoding="utf-8", newline="") as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=diagnostic_rows[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(diagnostic_rows)

    output_dir = tmp_path / "audit"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "phase3_motorcycle_visual_audit.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--video",
            str(video),
            "--fluid-tracks",
            str(fluid),
            "--geotrax-tracks",
            str(geotrax),
            "--track-diagnostics",
            str(diagnostics),
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

    summary = json.loads(completed.stdout)
    with (output_dir / "audit_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        manifest = list(csv.DictReader(fh))

    assert summary["case_count"] == 20
    assert len(manifest) == 20
    assert len({row["truth_id"] for row in manifest}) == 20
    assert all(row["review_label"] == "" for row in manifest)
    for row in manifest:
        contact_sheet = output_dir / row["contact_sheet_path"]
        metadata = contact_sheet.parent / "metadata.json"
        assert contact_sheet.is_file()
        assert metadata.is_file()
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        assert payload["sampled_frames"]
        assert payload["allowed_review_labels"][0] == "VISIBLE_CLEAR"
        assert payload["raw_fluid_classes"] == ["moped"]
        assert payload["truth_class"] == "MOTORCYCLE"
        assert payload["canonical_truth_class"] == "MOTORCYCLE"
        assert payload["benchmark_supported"] is True
        assert payload["review_label"] == ""
        assert payload["review_notes"] == ""
        assert payload["truth_coordinate_statistics"]["first_sampled_truth"]
        assert payload["truth_coordinate_statistics"]["last_sampled_truth"]
        assert "sampled_pixel_displacement" in payload["truth_coordinate_statistics"]
        assert payload["truth_coordinate_statistics"][
            "selected_interval_bounding_box"
        ]
        for frame_metadata in payload["frames"]:
            assert frame_metadata["truth_point"]["raw_fluid_class"] == "moped"
            assert frame_metadata["truth_point"]["truth_class"] == "MOTORCYCLE"
            assert frame_metadata["truth_point"]["canonical_class"] == "MOTORCYCLE"
            assert frame_metadata["truth_point"]["benchmark_supported"] is True
            assert frame_metadata["crop_bounds"] == frame_metadata[
                "raw_crop_bounds"
            ]
            assert frame_metadata["crop_bounds"] == frame_metadata[
                "overlay_crop_bounds"
            ]
            raw_crop = contact_sheet.parent / frame_metadata["raw_crop_path"]
            overlay_crop = contact_sheet.parent / frame_metadata[
                "overlay_crop_path"
            ]
            composite = contact_sheet.parent / frame_metadata["image_path"]
            assert raw_crop.is_file()
            assert overlay_crop.is_file()
            assert composite.is_file()
