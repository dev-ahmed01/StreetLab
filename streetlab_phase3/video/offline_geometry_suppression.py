"""Read-only post-ByteTrack suppression hypotheses on existing Geo-trax exports.

These deliberately post-hoc, geometry-only policies were inspired by the
May-26 W04 review; no ground-truth labels or 'unmatched' statuses enter rule
selection. This CANNOT simulate suppression before ByteTrack, nor validate
production thresholds on the same tuned 3 video frames.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint, load_geotrax_pixel_tracks
from streetlab_phase3.identity_benchmark import IdentityBenchmark
from streetlab_phase3.pixel_benchmark import PixelBenchmark, normalize_fluid_pixel_truth
from streetlab_phase3.video.class_diagnostics import class_diagnostics
from streetlab_phase3.video.existing_box_forensics import (
    ExportedBox, overlap, read_exported_boxes,
)

# Frozen test matrix written before running these candidate rules on local W04:
# two isolated error mechanisms, followed by conservative/aggressive combination.
# IoU and coverage thresholds are investigative, not fitted on held-out data.
POLICIES: tuple[tuple[str, float | None, float | None], ...] = (
    ("motorcycle_iou_0p50", .50, None),
    ("motorcycle_iou_0p30", .30, None),
    ("car_inside_large_0p80", None, .80),
    ("car_inside_large_0p60", None, .60),
    ("combined_conservative", .50, .80),
    ("combined_exploratory", .30, .60),
)
CATEGORIES = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")
LARGE = frozenset(("BUS", "HEAVY_VEHICLE"))


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def propose_geometry_suppression(
    boxes: Sequence[ExportedBox], *,
    motorcycle_iou: float | None,
    car_inside_large: float | None,
) -> list[dict[str, Any]]:
    """Return decisions without consulting FLUID, matching or a track score."""
    if motorcycle_iou is not None and not 0 < motorcycle_iou <= 1:
        raise ValueError("Invalid motorcycle IoU gate")
    if car_inside_large is not None and not 0 < car_inside_large <= 1:
        raise ValueError("Invalid car containment gate")
    frames: dict[int, list[ExportedBox]] = defaultdict(list)
    keys = set()
    for box in boxes:
        if not (math.isfinite(box.confidence) and 0 <= box.confidence <= 1
                and all(math.isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2))
                and box.area > 0):
            raise ValueError("Invalid geometry input")
        key = (box.frame, box.track_id)
        if key in keys:
            raise ValueError(f"Repeated source track ID within video frame: {key}")
        keys.add(key)
        frames[box.frame].append(box)

    removed: list[dict[str, Any]] = []
    for frame, candidates in sorted(frames.items()):
        if motorcycle_iou is not None:
            motorcycle = sorted(
                (p for p in candidates if p.vehicle_class == "MOTORCYCLE"),
                key=lambda p: (-p.confidence, -p.area, p.track_id),
            )
            retained: list[ExportedBox] = []
            for bike in motorcycle:
                closest = next(
                    (kept for kept in retained
                     if overlap(bike, kept)[0] >= motorcycle_iou), None)
                if closest is None:
                    retained.append(bike)
                else:
                    removed.append({
                        "video_frame": frame, "track_id": bike.track_id,
                        "vehicle_class": bike.vehicle_class,
                        "confidence": bike.confidence,
                        "reason": "motorcycle_same_class_overlap",
                        "reference_track_id": closest.track_id,
                        "reference_class": closest.vehicle_class,
                        "iou": overlap(bike, closest)[0],
                    })

        if car_inside_large is not None:
            large = [p for p in candidates if p.vehicle_class in LARGE]
            for car in (p for p in candidates if p.vehicle_class == "CAR"):
                # Require larger reference object, never remove a CAR simply
                # because two same-sized vehicles overlap in congestion.
                containing = sorted(
                    (
                        (overlap(car, vessel)[1], vessel)
                        for vessel in large
                        if vessel.area >= 2 * car.area
                        and overlap(car, vessel)[1] >= car_inside_large
                    ),
                    key=lambda pair: (-pair[0], -pair[1].confidence,
                                      pair[1].track_id),
                )
                if containing:
                    fraction, vessel = containing[0]
                    removed.append({
                        "video_frame": frame, "track_id": car.track_id,
                        "vehicle_class": car.vehicle_class,
                        "confidence": car.confidence,
                        "reason": "car_box_contained_in_large_vehicle",
                        "reference_track_id": vessel.track_id,
                        "reference_class": vessel.vehicle_class,
                        "fraction_car_covered": fraction,
                    })
    if len({(x["video_frame"], x["track_id"]) for x in removed}) != len(removed):
        raise AssertionError("Suppression decisions contain duplicate source rows")
    return sorted(removed, key=lambda x: (x["video_frame"], x["track_id"]))


def _evaluate(points: Sequence[GeoTraxPixelPoint], truth: Sequence[Any],
              first: int, last: int) -> dict[str, Any]:
    window = (first + 1, last + 1)
    p = PixelBenchmark(50.).evaluate(
        points, truth, frame_offsets=(1,), evaluation_frame_range=window)
    detail = class_diagnostics(points, truth, start_frame=first, end_frame=last)
    identity = IdentityBenchmark(50.).evaluate(
        points, truth, frame_offsets=(1,),
        evaluation_frame_range=window).scorecard_dict()
    motorcycle = detail["by_class"]["MOTORCYCLE"]
    car = detail["by_class"]["CAR"]
    return {
        "truth_points": p.truth_points,
        "predicted_points": p.predicted_points,
        "matched_points": p.matched_points,
        "unmatched_predictions": p.predicted_points - p.matched_points,
        "point_recall": p.point_recall,
        "point_precision": p.point_precision,
        "motorcycle_truth_points": motorcycle["truth_points"],
        "motorcycle_correct_class_matches": motorcycle["correct_class_matches"],
        "motorcycle_correct_class_recall": motorcycle["correct_class_recall"],
        "motorcycle_unmatched_predictions": motorcycle["unmatched_predictions"],
        "car_correct_class_recall": car["correct_class_recall"],
        "identity": {
            k: identity[k] for k in
            ("matched_truth_tracks", "predicted_tracks",
             "fraction_truth_tracks_fragmented",
             "total_contiguous_id_switches")
        },
    }


def run_offline_suppression(
    *, sliced_tracks: Path, fluid_tracks: Path,
    comparison_file: Path,
) -> dict[str, Any]:
    original = json.loads(comparison_file.read_text(encoding="utf-8"))
    if original.get("status") != "EXPLORATORY_SAME_FRAME_COMPARISON_NOT_PRODUCTION_EVIDENCE":
        raise ValueError("Expected the fixed-cohort existing-track comparison")
    expected_tracks = Path(original["results"]["sliced"]["tracks"])
    if sliced_tracks.resolve() != expected_tracks.resolve():
        raise ValueError("SAHI source path differs from recorded comparison")
    first = original["evaluation_start_frame"]
    last = original["evaluation_end_frame"]
    if not isinstance(first, int) or not isinstance(last, int) or first > last:
        raise ValueError("Invalid recorded cohort")
    if original.get("frozen_fluid_offset") != 1 or original.get("frozen_matching_radius_px") != 50:
        raise ValueError("Frozen FLUID scoring offset/distance changed")
    if (original.get("evaluation_frame_count") != last - first + 1
        or original.get("eligible_for_promotion") is not False):
        raise ValueError("Inconsistent fixed-cohort comparison")
    with fluid_tracks.open("r", encoding="utf-8-sig", newline="") as fh:
        truth = normalize_fluid_pixel_truth(csv.DictReader(fh))
    boxes = read_exported_boxes(sliced_tracks, start_frame=first, end_frame=last)
    track_points = [
        p for p in load_geotrax_pixel_tracks(sliced_tracks)
        if first <= p.frame <= last and p.vehicle_class in CATEGORIES
    ]
    all_keys = {(p.frame, p.track_id) for p in track_points}
    box_keys = {(b.frame, b.track_id) for b in boxes}
    if all_keys != box_keys or len(track_points) != len(all_keys):
        raise ValueError("Tracked boxes and pixel rows disagree")
    baseline = _evaluate(track_points, truth, first, last)
    original_pixel = original["results"]["sliced"]["pixel"]
    for key in ("truth_points", "predicted_points", "matched_points"):
        if baseline[key] != original_pixel[key]:
            raise ValueError(f"Baseline diverged from recorded frozen score: {key}")
    if not baseline["motorcycle_truth_points"]:
        raise ValueError("No annotated motorcycles in cohort")

    evaluated = []
    for name, moto_gate, car_gate in POLICIES:
        decisions = propose_geometry_suppression(
            boxes, motorcycle_iou=moto_gate, car_inside_large=car_gate)
        drop = {(x["video_frame"], x["track_id"]) for x in decisions}
        filtered = [p for p in track_points if (p.frame, p.track_id) not in drop]
        measures = _evaluate(filtered, truth, first, last)
        if measures["truth_points"] != baseline["truth_points"]:
            raise AssertionError("Ground truth denominator changed")
        if measures["motorcycle_truth_points"] != baseline["motorcycle_truth_points"]:
            raise AssertionError("Motorcycle denominator changed")
        assert len(filtered) + len(drop) == len(track_points)
        evaluated.append({
            "policy": name,
            "geometry_only": True,
            "motorcycle_iou_threshold": moto_gate,
            "car_inside_large_coverage_threshold": car_gate,
            "proposed_removals": len(decisions),
            "removed_by_class": {
                cls: sum(d["vehicle_class"] == cls for d in decisions)
                for cls in CATEGORIES
            },
            "changes_vs_unchanged_sahi": {
                "point_recall_percentage_points": (
                    100 * (measures["point_recall"] - baseline["point_recall"])
                    if measures["point_recall"] is not None
                    and baseline["point_recall"] is not None else None),
                "point_precision_percentage_points": (
                    100 * (measures["point_precision"] - baseline["point_precision"])
                    if measures["point_precision"] is not None
                    and baseline["point_precision"] is not None else None),
                "motorcycle_recall_percentage_points": (
                    100 * (measures["motorcycle_correct_class_recall"] -
                           baseline["motorcycle_correct_class_recall"])
                    if measures["motorcycle_correct_class_recall"] is not None
                    and baseline["motorcycle_correct_class_recall"] is not None else None),
                "matched_points": measures["matched_points"] - baseline["matched_points"],
                "unmatched_predictions": (measures["unmatched_predictions"]
                                          - baseline["unmatched_predictions"]),
            },
            "metrics": measures,
            "decisions": decisions,
        })
    return {
        "status": "EXPLORATORY_POST_TRACK_GEOMETRY_SIMULATION_NOT_PRODUCTION",
        "eligible_for_promotion": False,
        "evidence_scope": "May-26 W04 tuned cohort, only 3 frames",
        "source_tracks": str(sliced_tracks),
        "source_tracks_sha256": _digest(sliced_tracks),
        "fluid_tracks": str(fluid_tracks),
        "fluid_tracks_sha256": _digest(fluid_tracks),
        "source_comparison": str(comparison_file),
        "source_comparison_sha256": _digest(comparison_file),
        "frozen_fluid_frame_offset": 1,
        "frozen_matching_radius_px": 50.,
        "video_frame_window": [first, last],
        "policy_matrix_is_posthoc_exploratory": True,
        "rules_use_truth_or_unmatched_status": False,
        "original_sahi_metrics": baseline,
        "policies": evaluated,
        "limitations": (
            "Policies use existing *confirmed* tracker boxes only; no FLUID "
            "truth, unmatched status or human labels are used to choose rows "
            "for removal. FLUID is used only to score each candidate afterward. "
            "Per-frame removals do not rerun detector NMS or ByteTrack, "
            "cannot repair tracker IDs and can create temporal gaps. Adjacent "
            "real motorcycles may overlap; BUS predictions may mislabel cars, "
            "making cross-class containment dangerous. Thresholds were "
            "hypothesized after reviewing this May-26 tuning cohort, so these "
            "numbers cannot serve as holdout validation or production gates."
        ),
    }
