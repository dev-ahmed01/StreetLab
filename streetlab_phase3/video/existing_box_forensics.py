"""Audit *existing confirmed track boxes* for overlap; no video or inference needed.

The frozen Geo-trax 14-column export already contains raw center x,y and
bounding-box width,height in columns 2-5. The legacy pixel benchmark ignores
boxes. This module inspects possible same-class duplicate tracks and car
detections placed on larger BUS/HEAVY_VEHICLE boxes. It cannot assess
pre-ByteTrack raw SAHI NMS or prove a detection is incorrect.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLASSES = {0: "CAR", 1: "BUS", 2: "HEAVY_VEHICLE", 3: "MOTORCYCLE"}
LARGE = frozenset({"BUS", "HEAVY_VEHICLE"})


@dataclass(frozen=True, slots=True)
class ExportedBox:
    frame: int
    track_id: str
    vehicle_class: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def read_exported_boxes(path: Path, *, start_frame: int,
                        end_frame: int) -> list[ExportedBox]:
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid inclusive evaluation interval")
    result: list[ExportedBox] = []
    seen: set[tuple[int, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        for lineno, row in enumerate(csv.reader(fh), 1):
            if not row:
                continue
            if len(row) < 14:
                raise ValueError(f"Track row {lineno}: expected Geo-trax 14 columns")
            numbers = [float(row[j]) for j in (0, 1, 2, 3, 4, 5, 10, 11)]
            if any(not math.isfinite(z) for z in numbers):
                raise ValueError(f"Track row {lineno}: non-finite export value")
            f, tid, x, y, w, h, klass, confidence = numbers
            if (f != int(f) or tid != int(tid) or klass != int(klass)
                    or w <= 0 or h <= 0 or not 0 <= confidence <= 1):
                raise ValueError(f"Track row {lineno}: invalid box, ID, frame or confidence")
            if not start_frame <= int(f) <= end_frame:
                continue
            if int(klass) not in CLASSES:
                continue
            key = int(f), str(int(tid))
            if key in seen:
                raise ValueError(f"Duplicate confirmed track ID within frame: {key}")
            seen.add(key)
            result.append(ExportedBox(
                int(f), str(int(tid)), CLASSES[int(klass)], confidence,
                x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return result


def overlap(a: ExportedBox, b: ExportedBox) -> tuple[float, float, float]:
    """Return box IoU, share of A covered, share of B covered."""
    if a.frame != b.frame:
        return 0., 0., 0.
    intersection = max(0., min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0., min(a.y2, b.y2) - max(a.y1, b.y1))
    union = a.area + b.area - intersection
    return intersection / union, intersection / a.area, intersection / b.area


def existing_box_forensics(
    sliced_tracks: Path, comparison: dict[str, Any],
) -> dict[str, Any]:
    if comparison.get("status") != "EXPLORATORY_SAME_FRAME_COMPARISON_NOT_PRODUCTION_EVIDENCE":
        raise ValueError("Expected fixed-cohort P3B offline comparison")
    if sliced_tracks.resolve() != Path(comparison["results"]["sliced"]["tracks"]).resolve():
        raise ValueError("Selected sliced track file does not match provenance")
    start, end = (comparison["evaluation_start_frame"],
                  comparison["evaluation_end_frame"])
    proximity = comparison["sliced_unmatched_proximity"]
    if (proximity["status"] != "UNMATCHED_PROXIMITY_HEURISTIC_NOT_PROOF_OF_DUPLICATES"
            or proximity["video_frame_start"] != start
            or proximity["video_frame_end"] != end):
        raise ValueError("Unmatched prediction cohort inconsistent with report")
    boxes = read_exported_boxes(sliced_tracks, start_frame=start,
                               end_frame=end)
    by_frame: dict[int, list[ExportedBox]] = defaultdict(list)
    by_key: dict[tuple[int, str], ExportedBox] = {}
    for box in boxes:
        by_frame[box.frame].append(box)
        by_key[(box.frame, box.track_id)] = box

    seen_targets = set()
    findings: list[dict[str, Any]] = []
    for item in proximity["examples"]:
        key = item["video_frame"], str(item["predicted_track_id"])
        if key in seen_targets:
            raise ValueError(f"Duplicate unmatched prediction in report: {key}")
        seen_targets.add(key)
        target = by_key.get(key)
        if target is None:
            raise ValueError(f"Unmatched prediction absent from raw track export: {key}")
        if (target.vehicle_class != item["predicted_class"]
            or not math.isclose(target.confidence, item["confidence"], abs_tol=1e-5)):
            raise ValueError("Track export and unmatched prediction report disagree")

        neighbors: list[dict[str, Any]] = []
        for box in by_frame[target.frame]:
            if box.track_id == target.track_id:
                continue
            iou, target_coverage, other_coverage = overlap(target, box)
            if iou == 0:
                continue
            neighbors.append({
                "track_id": box.track_id,
                "class": box.vehicle_class,
                "confidence": box.confidence,
                "iou": iou,
                "fraction_target_covered": target_coverage,
                "fraction_other_covered": other_coverage,
            })
        neighbors.sort(key=lambda x: (-x["iou"], -x["fraction_target_covered"],
                                      x["track_id"]))
        same_class_overlapping = [
            x for x in neighbors if x["class"] == target.vehicle_class
            and x["iou"] >= .30]
        overlapping_large = [
            x for x in neighbors if target.vehicle_class == "CAR"
            and x["class"] in LARGE and x["fraction_target_covered"] >= .60]
        findings.append({
            "video_frame": target.frame,
            "track_id": target.track_id,
            "class": target.vehicle_class,
            "confidence": target.confidence,
            "bbox_xyxy": [target.x1, target.y1, target.x2, target.y2],
            "same_class_iou_gte_0_30": bool(same_class_overlapping),
            "car_covered_by_bus_or_heavy_gte_0_60": bool(overlapping_large),
            "overlapping_track_boxes": neighbors,
            "caution": "Proximity/overlap of confirmed tracks is not proof of an NMS failure.",
        })

    high_car = [x for x in findings if x["class"] == "CAR" and x["confidence"] >= .5]
    bike = [x for x in findings if x["class"] == "MOTORCYCLE"]
    return {
        "status": "CONFIRMED_TRACK_BOX_FORENSICS_NOT_RAW_DETECTOR_NMS",
        "eligible_for_promotion": False,
        "video_frame_window": [start, end],
        "source": str(sliced_tracks),
        "unmatched_observations": len(findings),
        "thresholds_predeclared": {
            "same_class_iou_at_least": .30,
            "car_covered_by_large_vehicle_at_least": .60,
            "high_car_confidence_at_least": .50,
        },
        "high_confidence_unmatched_cars": {
            "observations": len(high_car),
            "with_same_class_box_overlap": sum(
                x["same_class_iou_gte_0_30"] for x in high_car),
            "covered_by_exported_large_vehicle_box": sum(
                x["car_covered_by_bus_or_heavy_gte_0_60"] for x in high_car),
        },
        "unmatched_motorcycle": {
            "observations": len(bike),
            "with_same_class_box_overlap": sum(
                x["same_class_iou_gte_0_30"] for x in bike),
        },
        "findings": findings,
        "limitations": (
            "These are *confirmed track* boxes after SAHI merging and ByteTrack, "
            "not raw slices or original SAHI NMS inputs. In-camera overlapping "
            "vehicles are real, and a CAR on a bus roof may be a wrong class, "
            "partial object, or association artifact. Visually adjudicate before "
            "suppressing anything. May-26 W04 remains tuned, only three frames, "
            "and not valid for production parameter selection."
        ),
    }
