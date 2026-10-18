"""Real local May-25 cross-date regression with frozen W04 model and one locked policy.

NOT a holdout: May 25 has previously been a regression reference. W04 May 26
remains tuned development. No historical T000 comparison or production claims.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from .w04_holdout_intake import digest, read_json, validate_lock
from streetlab_phase3.class_mapping import map_vehicle_class
from streetlab_phase3.models import VehicleClass

STATUS = "W04_MAY25_CROSSDATE_REGRESSION_RECORDED_NOT_BLIND_HOLDOUT"
WINDOW = (60, 179)  # 30-frame warmup from absolute source frame 30
WARMUP = 30
SUPPORTED_CLASSES = frozenset({
    VehicleClass.CAR.value, VehicleClass.BUS.value,
    VehicleClass.HEAVY_VEHICLE.value, VehicleClass.MOTORCYCLE.value,
})


class CrossDateError(ValueError):
    """Frozen inputs are incomplete, contradictory or unsuitable for comparison."""


def check_inputs(*, protocol_file: Path, lock_file: Path, video: Path,
                 truth: Path, model_dir: Path, capture: Any = None,
                 hasher: Any = None) -> dict:
    protocol = read_json(protocol_file)
    lock = read_json(lock_file)
    validate_lock(protocol, lock, digest(lock_file))
    if (not video.is_file() or video.is_symlink() or
        video.name.casefold() != "20250525_video.mp4"):
        raise CrossDateError("Only original May-25 FLUID regression reference video is allowed")
    if (not truth.is_file() or truth.is_symlink() or
        truth.name.casefold() != "20250525_video_traj.csv"):
        raise CrossDateError("Matching original May-25 FLUID Traj.csv is required")
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise CrossDateError("Frozen W04 OpenVINO model directory is required")
    if hasher is None:
        from streetlab_phase3.video.openvino_export import hash_model_tree
        hasher = hash_model_tree
    model_hash = hasher(model_dir)
    if model_hash != lock["source_model_sha256"]:
        raise CrossDateError("Actual OpenVINO model does not match preregistered W04 SHA")
    video_sha = digest(video)
    truth_sha = digest(truth)
    if (video_sha == lock["source_development_video_sha256"]
        or truth_sha == lock["source_fluid_sha256"]):
        raise CrossDateError("Cross-date source reused exact tuned W04 bytes")
    # This fixed cohort is established before detector scores are visible.
    left, right = WINDOW
    labelled = set()
    counts: dict[str, int] = {}
    source_pixel_points: list[tuple[float, float, int]] = []
    supported_rows_in_window = 0
    skipped_blank_geometry = 0
    with truth.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"frame", "id", "cx", "cy", "type"}.issubset(reader.fieldnames):
            raise CrossDateError("Original FLUID trajectory schema not available")
        for row in reader:
            frame_raw = row.get("frame")
            if frame_raw in (None, ""):
                continue
            try:
                index = int(float(frame_raw))
            except (ValueError, OverflowError) as exc:
                raise CrossDateError("Nonnumeric FLUID frame index") from exc
            if not left + 1 <= index <= right + 1:
                continue
            cls = str(row.get("type", "")).strip().casefold()
            # Source-class eligibility must match normalize_fluid_pixel_truth.
            if map_vehicle_class(cls).canonical.value not in SUPPORTED_CLASSES:
                continue
            supported_rows_in_window += 1
            # The existing FLUID PixelBenchmark normalizer omits rows whose
            # cx/cy cells are literally empty. Treat them the same way here,
            # without editing the source CSV or adding fake coordinates.
            if row.get("cx") in (None, "") or row.get("cy") in (None, ""):
                skipped_blank_geometry += 1
                continue
            try:
                x = float(row["cx"])
                y = float(row["cy"])
            except (ValueError, TypeError) as exc:
                raise CrossDateError("Invalid original FLUID geometry in fixed window") from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise CrossDateError("Invalid original FLUID nonfinite center")
            labelled.add(index)
            counts[cls] = counts.get(cls, 0) + 1
            source_pixel_points.append((x, y, index))
    if not labelled:
        raise CrossDateError(
            "No supported FLUID truth labels with valid cx/cy in fixed "
            f"May-25 120-frame cohort ({skipped_blank_geometry} skipped blank "
            "coordinate rows, source CSV unchanged)"
        )
    if capture is None:
        import cv2 as capture
    cap = capture.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise CrossDateError("Original May-25 video failed OpenCV metadata preflight")
    try:
        width = cap.get(capture.CAP_PROP_FRAME_WIDTH)
        height = cap.get(capture.CAP_PROP_FRAME_HEIGHT)
        frames = cap.get(capture.CAP_PROP_FRAME_COUNT)
        fps = cap.get(capture.CAP_PROP_FPS)
    finally:
        cap.release()
    raw_metadata = {
        "width": width, "height": height,
        "frame_count": frames, "fps": fps,
    }
    if (not all(math.isfinite(float(v)) for v in raw_metadata.values())
        or not 640 <= width <= 7680 or not 360 <= height <= 4320
        or not float(width).is_integer() or not float(height).is_integer()
        or frames < right + 1 or not 10 <= fps <= 60):
        raise CrossDateError(
            f"Unsupported original May-25 video metadata: {raw_metadata}; "
            "require native 640..7680px width, 360..4320px height, "
            "at least 180 frames and 10..60 FPS. No resizing or resampling."
        )
    # Detection and FLUID benchmark must remain in the SAME pixel geometry.
    # Never silently rescale truth coordinates to make scoring appear valid.
    outside = [(frame, x, y) for x, y, frame in source_pixel_points
               if not (0 <= x < width and 0 <= y < height)]
    if outside:
        raise CrossDateError(
            f"Original May-25 FLUID coordinate/video geometry mismatch: "
            f"{len(outside)} of {len(source_pixel_points)} usable truth centers "
            f"outside native {int(width)}x{int(height)} video; "
            f"first failing (frame,cx,cy)={outside[0]}. "
            "No automatic label scaling or repair permitted."
        )
    return {
        "schema_version": 1,
        "status": "W04_MAY25_CROSSDATE_PREFLIGHT_VERIFIED_REGRESSION_ONLY",
        "source_history": "FLUID_MAY25_REGRESSION_REFERENCE_NOT_UNSEEN",
        "policy_lock_sha256": digest(lock_file),
        "protocol_sha256": digest(protocol_file),
        "selected_policy": lock["selected_policy"],
        "selected_settings": lock["selected_settings"],
        "model_tree_sha256": model_hash,
        "source_video": str(video),
        "source_video_sha256": video_sha,
        "source_truth": str(truth),
        "source_truth_sha256": truth_sha,
        "fixed_source_evaluation_frames": [left, right],
        "fixed_fluid_evaluation_frames": [left+1, right+1],
        "tracker_warmup_frames": WARMUP,
        "first_decoded_source_frame": left-WARMUP,
        "source_frame_rate_reported": fps,
        "source_video_dimensions": [int(width), int(height)],
        "source_frame_count_reported": int(frames),
        "video_and_truth_coordinates_in_original_native_pixels": True,
        "video_resized_or_resampled": False,
        "expected_w04_video_dimensions_for_comparison": [3840, 2160],
        "crossdate_uses_original_source_fps": True,
        "supported_fluid_truth_points": sum(counts.values()),
        "supported_fluid_rows_before_blank_coordinate_filter": supported_rows_in_window,
        "skipped_fluid_rows_with_blank_coordinates": skipped_blank_geometry,
        "blank_coordinate_handling": (
            "Skip only literally missing/empty cx or cy, matching the "
            "existing PixelBenchmark FLUID normalizer; reject malformed "
            "nonempty or nonfinite coordinate values, never repair CSV."
        ),
        "supported_fluid_labeled_frames": len(labelled),
        "original_fluid_label_types": counts,
        "same_window_t000_available": False,
        "comparison_to_original_W04_is_across_dates_and_window_lengths": True,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }


def execute(*, preflight: dict, video: Path, truth: Path, model_dir: Path,
            batch_output: Path) -> dict:
    """Executes one ACTUAL 150-frame detector pass (30 warmup + 120 evaluation)."""
    if preflight.get("status") != "W04_MAY25_CROSSDATE_PREFLIGHT_VERIFIED_REGRESSION_ONLY":
        raise CrossDateError("Cannot run unverified May-25 source")
    if batch_output.exists():
        raise FileExistsError("Refusing overwrite of cross-date tracking evidence")
    from streetlab_phase3.video.continuous_box_tracking_lab import (
        cached_window_scorer, policy_matrix, sequential_video_rgb_frames,
        stream_tracking_matrix,
    )
    from streetlab_phase3.video.openvino_export import hash_model_tree
    from streetlab_phase3.video.premerge_box_capture import capture_boxes
    import cv2
    from sahi import AutoDetectionModel
    from sahi.slicing import get_slice_bboxes
    from sahi.predict import get_prediction
    from supervision import Detections
    from trackers import ByteTrackTracker

    if digest(video) != preflight["source_video_sha256"]:
        raise CrossDateError("May-25 video bytes modified after preflight")
    if digest(truth) != preflight["source_truth_sha256"]:
        raise CrossDateError("May-25 FLUID annotations modified after preflight")
    if hash_model_tree(model_dir) != preflight["model_tree_sha256"]:
        raise CrossDateError("Frozen W04 model bytes modified after preflight")
    settings = preflight["selected_settings"]
    if policy_matrix([preflight["selected_policy"]]) != [settings]:
        raise CrossDateError("Selected tracking policy drifted from frozen matrix")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(model_dir),
        confidence_threshold=.15, device="cpu", image_size=640,
    )
    left, right = WINDOW
    def raw_stream():
        for frame, rgb in sequential_video_rgb_frames(
            video, left-WARMUP, right, cv2_module=cv2,
            expected_dimensions=tuple(preflight["source_video_dimensions"]),
        ):
            boxes, times = capture_boxes(
                [(frame, rgb)], model, slicer=get_slice_bboxes,
                predictor=get_prediction, overlap=.20, slice_size=640,
            )
            yield frame, boxes, times[0]["elapsed_s"]
    provenance = {
        "status": "W04_MAY25_CROSSDATE_REGRESSION_SOURCE_NOT_BLIND",
        "eligible_for_promotion": False,
        "video_path": str(video),
        "video_sha256": preflight["source_video_sha256"],
        "fluid_path": str(truth),
        "fluid_sha256": preflight["source_truth_sha256"],
        "openvino_model_sha256": preflight["model_tree_sha256"],
        "video_frame_window": [left, right],
        "policies": [preflight["selected_policy"]],
        "source_fps_user_supplied": preflight["source_frame_rate_reported"],
        "video_dimensions_native": preflight["source_video_dimensions"],
        "video_resized_or_resampled": False,
        "reuses_historical_regression_reference": True,
        "source_is_independent_holdout": False,
    }
    scored = stream_tracking_matrix(
        raw_stream=raw_stream(), first_frame=left-WARMUP,
        start_frame=left, end_frame=right, output_dir=batch_output,
        fluid_tracks=truth, configs=[settings],
        tracker_factory=ByteTrackTracker, detection_class=Detections,
        score_fn=cached_window_scorer(truth, left, right),
        fps=preflight["source_frame_rate_reported"],
        provenance=provenance,
    )
    from streetlab_phase3.video.holdout_validation import verify_batch
    verified = verify_batch(batch_output)
    if not verified["original_source_sha256_verified"] or verified["policy_count"] != 1:
        raise CrossDateError("Saved cross-date result did not reproduce source hashes")
    policy = scored["policies"][0]
    pixel = policy["pixel"]
    classes = policy["class_diagnostics"]["by_class"]
    prec, rec = pixel.get("point_precision"), pixel.get("point_recall")
    f1 = 2*prec*rec/(prec+rec) if prec is not None and rec is not None and prec+rec else None
    return {
        "schema_version": 1, "status": STATUS,
        "selected_policy": preflight["selected_policy"],
        "cohort_role": "PREVIOUSLY_USED_MAY25_REGRESSION_REFERENCE",
        "source_video_sha256": preflight["source_video_sha256"],
        "original_fluid_sha256": preflight["source_truth_sha256"],
        "frozen_w04_model_sha256": preflight["model_tree_sha256"],
        "source_window": [left, right],
        "truth_frame_window": [left+1, right+1],
        "source_video_dimensions": preflight["source_video_dimensions"],
        "source_frame_rate_reported": preflight["source_frame_rate_reported"],
        "uses_native_fps_without_resampling": True,
        "decoded_frames": scored["processed_frames"],
        "evaluation_frames": scored["evaluated_frames"],
        "truth_label_precision": prec, "truth_label_recall": rec,
        "truth_label_f1": f1,
        "contiguous_id_switches": policy["identity"].get("total_contiguous_id_switches"),
        "per_class_fluid_label_diagnostics": classes,
        "detector_cpu_median_seconds_per_frame": scored["detector_median_seconds"],
        "tracker_cpu_median_seconds_per_frame": policy["tracking_median_seconds"],
        "postprocess_cpu_median_seconds_per_frame": policy["postprocess_median_seconds"],
        "stage_medians_not_end_to_end_p95": True,
        "original_saved_batch_sha256": verified["batch_sha256"],
        "same_window_t000_available": False,
        "not_a_blind_or_untouched_holdout": True,
        "cross_date_comparison_not_same_camera_or_same_window": True,
        "no_independent_physical_object_review": True,
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "limitations": (
            "Cross-date previously designated regression data, not unseen holdout; "
            "May 25 120-frame and W04 201-frame cohort differ. FLUID label taxonomy "
            "and camera scene may differ. No same-cohort original T000; no p95 or "
            "two-reviewer physical annotations. Never infer site transfer from this run."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="May-25 fixed cross-date regression; NEVER blind holdout")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-lock", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-inference", action="store_true",
                        help="Execute real 150-frame OpenVINO+ByteTrack experiment after preflight")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing cross-date run directory: {args.output_dir}")
    preliminary = check_inputs(
        protocol_file=args.protocol, lock_file=args.policy_lock,
        video=args.video, truth=args.truth, model_dir=args.model_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "preflight.json").open("x", encoding="utf-8") as file:
        json.dump(preliminary, file, indent=2)
        file.write("\n")
    if args.run_inference:
        result = execute(
            preflight=preliminary, video=args.video, truth=args.truth,
            model_dir=args.model_dir, batch_output=args.output_dir / "batch",
        )
        result["created_utc"] = datetime.now(timezone.utc).isoformat()
        with (args.output_dir / "cross_date_summary.json").open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
        print(json.dumps({
            "status": result["status"], "precision": result["truth_label_precision"],
            "recall": result["truth_label_recall"], "f1": result["truth_label_f1"],
            "independent_holdout_pass": False,
            "output": str(args.output_dir / "cross_date_summary.json"),
        }, indent=2))
    else:
        print(json.dumps({
            "status": preliminary["status"], "run_inference": False,
            "labelled_frames": preliminary["supported_fluid_labeled_frames"],
            "label_points": preliminary["supported_fluid_truth_points"],
            "independent_holdout_pass": False,
            "output": str(args.output_dir / "preflight.json"),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
