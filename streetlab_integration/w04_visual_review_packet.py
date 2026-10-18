"""Create an offline, source-hashed, track-diverse May25 visual review packet.

This produces original-frame evidence crops and blank independent review sheets.
It does NOT run detector inference, change original labels, or adjudicate truth.
The compact images resize only for human display; evaluation pixels stay intact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .w04_holdout_intake import digest, read_json
from .w04_crossdate_forensics import AUDIT_STATUS, EVENT_FIELDS

PACKET_STATUS = "W04_MAY25_TRACK_DIVERSE_VISUAL_REVIEW_PACKET_NOT_ADJUDICATED"
CATEGORIES = (
    ("HEAVY_WRONG_CLASS", 2, 3),
    ("HEAVY_TRUTH_UNMATCHED", 2, 1),
    ("BUS_PRED_MATCHED_NONBUS_TRUTH", 1, 1),
    ("BUS_PRED_UNMATCHED", 2, 1),
    ("MOTORCYCLE_TRUTH_UNMATCHED", 3, 1),
)
REVIEW_COLUMNS = [
    "sample_id", "video_frame", "event", "truth_track_id", "predicted_track_id",
    "image_file", "original_video_object_visible", "physical_vehicle_type",
    "FLUID_class_accurate", "model_class_accurate",
    "source_annotation_ambiguous", "occlusion_or_overlap",
    "reviewer_comments",
]


class ReviewError(ValueError):
    """Fail closed if a source file, review event, or output is inconsistent."""


def _str_value(value: Any) -> str:
    return "" if value is None else str(value)


def _event_identity(event: dict) -> tuple[str, str, str]:
    return (
        event["event"], _str_value(event.get("truth_track_id")),
        _str_value(event.get("predicted_track_id")),
    )


def _validate_event(event: dict) -> None:
    if (set(event) != set(EVENT_FIELDS)
        or event.get("event") not in {p[0] for p in CATEGORIES}
        or type(event.get("video_frame")) is not int
        or not 60 <= event["video_frame"] <= 179
        or event.get("fluid_frame") != event["video_frame"] + 1
        or event.get("annotation_status") != "FLUID_LABEL_ONLY_NOT_PHYSICALLY_ADJUDICATED"):
        raise ReviewError("Unexpected May25 event schema/window; not the frozen review queue")
    for dimension in ("truth_cx", "truth_cy", "predicted_cx", "predicted_cy"):
        value = event[dimension]
        if value is not None and (type(value) not in (int, float) or
                                  not math.isfinite(value) or value < 0):
            raise ReviewError("Malformed candidate review coordinates")
    if ((event["truth_cx"] is None) != (event["truth_cy"] is None) or
        (event["predicted_cx"] is None) != (event["predicted_cy"] is None) or
        (event["truth_cx"] is None and event["predicted_cx"] is None)):
        raise ReviewError("Candidate event has incomplete/absent visual anchor")


def make_plan(events: list[dict], max_samples: int = 14) -> list[dict]:
    """Select distinct evidence pairs, not 14 copies of one persistent track."""
    if not 1 <= max_samples <= 30:
        raise ReviewError("Human review sample cap must be between 1 and 30")
    if not isinstance(events, list) or not events:
        raise ReviewError("Original May25 forensic review events absent")
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            raise ReviewError("Malformed candidate event")
        _validate_event(event)
        grouped[_event_identity(event)].append(event)
    selected: list[dict] = []
    for kind, group_limit, per_group in CATEGORIES:
        candidates = [(key, sorted(group, key=lambda x: x["video_frame"]))
                      for key, group in grouped.items() if key[0] == kind]
        candidates.sort(key=lambda entry: (
            -len(entry[1]), entry[1][0]["video_frame"], entry[0][1], entry[0][2],
        ))
        for key, group in candidates[:group_limit]:
            if len(selected) >= max_samples:
                break
            count = min(per_group, len(group), max_samples-len(selected))
            if count == 1:
                picks = [group[len(group)//2]]
            else:
                indices = [int(i*(len(group)-1)/(count-1))
                           for i in range(count)]
                picks = [group[i] for i in indices]
            for event in picks:
                selected.append({
                    "sample_id": f"R{len(selected)+1:03d}",
                    "event": event["event"],
                    "video_frame": event["video_frame"],
                    "fluid_frame": event["fluid_frame"],
                    "truth_track_id": event["truth_track_id"],
                    "predicted_track_id": event["predicted_track_id"],
                    "truth_class": event["truth_class"],
                    "predicted_class": event["predicted_class"],
                    "truth_cx": event["truth_cx"],
                    "truth_cy": event["truth_cy"],
                    "predicted_cx": event["predicted_cx"],
                    "predicted_cy": event["predicted_cy"],
                    "matching_distance_px": event["matching_distance_px"],
                    "observed_repetitions_in_18_review_frames": len(group),
                    "original_frame_not_physical_object_adjudication": True,
                    "image_file": f"crops/R{len(selected)+1:03d}.jpg",
                })
    if not selected:
        raise ReviewError("No reviewable source event was selected")
    return selected


def _csv_matches_forensics(csv_file: Path, events: list[dict]) -> None:
    if not csv_file.is_file() or csv_file.is_symlink():
        raise ReviewError("Original review_events.csv file is missing or linked")
    with csv_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(EVENT_FIELDS):
            raise ReviewError("Original event CSV header differs from saved forensics")
        rows = list(reader)
    expected = [{k: _str_value(event[k]) for k in EVENT_FIELDS} for event in events]
    if rows != expected:
        raise ReviewError("review_events.csv does not exactly match forensic JSON events")


def check_sources(*, run_dir: Path, forensics_dir: Path) -> tuple[Path, dict, list[dict]]:
    """Hash the original May25 source and prove queue/batch correspondences."""
    forensic_path = forensics_dir / "regression_forensics.json"
    events_csv = forensics_dir / "review_events.csv"
    report = read_json(forensic_path)
    summary_path = run_dir / "cross_date_summary.json"
    preflight_path = run_dir / "preflight.json"
    batch_path = run_dir / "batch" / "batch_report.json"
    origin_path = run_dir / "batch" / "source_provenance.json"
    summary = read_json(summary_path)
    preflight = read_json(preflight_path)
    batch = read_json(batch_path)
    origin = read_json(origin_path)
    if (report.get("status") != AUDIT_STATUS
        or report.get("original_video_and_FLUID_SHA_verified_from_saved_batch") is not True
        or report.get("original_saved_scores_recomputed_from_tracks_and_FLUID") is not True
        or report.get("independent_holdout_pass") is not False
        or report.get("production_release_approved") is not False
        or report.get("saved_regression_summary_sha256") != digest(summary_path)
        or report.get("preflight_sha256") != digest(preflight_path)
        or report.get("saved_regression_batch_sha256") != digest(batch_path)
        or report.get("saved_regression_batch_sha256") != summary.get("original_saved_batch_sha256")
        or report.get("selected_policy") != "hard_nms_ios_0.30"
        or preflight.get("source_video_sha256") != origin.get("video_sha256")
        or summary.get("source_video_sha256") != origin.get("video_sha256")
        or summary.get("original_fluid_sha256") != origin.get("fluid_sha256")
        or preflight.get("source_truth_sha256") != origin.get("fluid_sha256")
        or summary.get("source_window") != [60, 179]
        or summary.get("evaluation_frames") != 120
        or batch.get("eligible_for_promotion") is not False
        or origin.get("eligible_for_promotion") is not False
        or origin.get("source_is_independent_holdout") is not False):
        raise ReviewError("Saved forensics/preflight/batch is incomplete, changed or falsely promoted")
    video = Path(origin.get("video_path", ""))
    truth = Path(origin.get("fluid_path", ""))
    if (video.is_symlink() or not video.is_file()
        or truth.is_symlink() or not truth.is_file()
        or digest(video) != origin["video_sha256"]
        or digest(truth) != origin["fluid_sha256"]):
        raise ReviewError("Original May25 video/FLUID source changed or unavailable")
    events = report.get("review", {}).get("events", [])
    if not isinstance(events, list) or len(events) != report["review"].get("exported_candidate_events"):
        raise ReviewError("Forensic events changed or incomplete")
    _csv_matches_forensics(events_csv, events)
    return video, {
        "forensics_sha256": digest(forensic_path),
        "original_events_csv_sha256": digest(events_csv),
        "original_video_sha256": origin["video_sha256"],
        "original_truth_sha256": origin["fluid_sha256"],
        "original_regression_batch_sha256": digest(batch_path),
        "source_video_dimensions": summary["source_video_dimensions"],
        "source_fps": summary["source_frame_rate_reported"],
    }, events


def _tile(rgb: Any, sample: dict, frame_size: tuple[int, int]):
    """Draw source-space labels on a review-only display crop."""
    from PIL import Image, ImageDraw, ImageFont
    width, height = frame_size
    if rgb is None or tuple(rgb.shape[:2]) != (height, width):
        raise ReviewError(f"Decoded May25 source frame size differs from frozen {frame_size}")
    coords = [(sample[k], sample[k.replace("cx", "cy")])
              for k in ("truth_cx", "predicted_cx") if sample[k] is not None]
    cx = sum(p[0] for p in coords)/len(coords)
    cy = sum(p[1] for p in coords)/len(coords)
    crop_w, crop_h = min(850, width), min(575, height)
    left = min(max(int(round(cx-crop_w/2)), 0), width-crop_w)
    top = min(max(int(round(cy-crop_h/2)), 0), height-crop_h)
    img = Image.fromarray(rgb[top:top+crop_h, left:left+crop_w], mode="RGB")
    display_w, display_h = 500, 338
    img = img.resize((display_w, display_h), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (540, 454), (245, 247, 249))
    result.paste(img, (20, 98))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    header = f'{sample["sample_id"]}  source frame {sample["video_frame"]} | FLUID {sample["fluid_frame"]}'
    subtitle = sample["event"]
    third = f'FLUID: {sample["truth_class"] or "--"} #{sample["truth_track_id"] or "--"}'
    fourth = f'Model: {sample["predicted_class"] or "--"} #{sample["predicted_track_id"] or "--"}'
    draw.text((20, 10), header, fill=(20, 29, 44), font=font)
    draw.text((20, 29), subtitle, fill=(96, 40, 25), font=font)
    draw.text((20, 47), third, fill=(5, 93, 64), font=font)
    draw.text((20, 65), fourth, fill=(177, 78, 10), font=font)
    for prefix, color, radius in (
        ("truth", (10, 207, 111), 13),
        ("predicted", (255, 158, 36), 22),
    ):
        x, y = sample.get(prefix+"_cx"), sample.get(prefix+"_cy")
        if x is None:
            continue
        px = 20 + (x-left) * display_w/crop_w
        py = 98 + (y-top) * display_h/crop_h
        draw.ellipse((px-radius,py-radius,px+radius,py+radius),
                     outline=color, width=4)
    draw.text((20, 441), "Review candidate only | colors: green=FLUID, orange=model",
              fill=(60, 65, 77), font=font)
    return result


def render_packet(*, samples: list[dict], source_frames: Iterable[tuple[int, Any]],
                  frame_size: tuple[int, int], output_dir: Path) -> list[dict]:
    """Pure renderer for original decoded frames; callable with synthetic tests."""
    from PIL import Image, ImageDraw, ImageFont
    needed = defaultdict(list)
    for sample in samples:
        needed[sample["video_frame"]].append(sample)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite previous review packet: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "crops").mkdir()
    rendered = {}
    for idx, frame in source_frames:
        if idx not in needed:
            continue
        if idx in rendered:
            raise ReviewError(f"Duplicate decoded source frame {idx}")
        for sample in needed[idx]:
            tile = _tile(frame, sample, frame_size)
            tile.save(output_dir / sample["image_file"], format="JPEG", quality=92)
        rendered[idx] = True
    if set(rendered) != set(needed):
        raise ReviewError(f"Missing original video frames: {sorted(set(needed)-set(rendered))}")

    cols = 3
    rows = math.ceil(len(samples)/cols)
    sheet = Image.new("RGB", (cols*540, rows*454+40), (232, 236, 243))
    pen = ImageDraw.Draw(sheet)
    pen.text((15, 12),
             "MAY25 FLUID-LABEL CANDIDATES | NOT PHYSICALLY ADJUDICATED",
             font=ImageFont.load_default(), fill=(20, 29, 44))
    for i, sample in enumerate(samples):
        with Image.open(output_dir / sample["image_file"]) as tile:
            sheet.paste(tile, (i % cols * 540, 40 + (i//cols)*454))
    sheet.save(output_dir/"review_contact_sheet.jpg", format="JPEG", quality=89)
    for n in (1, 2):
        with (output_dir / f"reviewer_{n}_blank.csv").open(
            "x", newline="", encoding="utf-8"
        ) as out:
            writer = csv.DictWriter(out, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            for sample in samples:
                writer.writerow({
                    "sample_id": sample["sample_id"],
                    "video_frame": sample["video_frame"],
                    "event": sample["event"],
                    "truth_track_id": _str_value(sample["truth_track_id"]),
                    "predicted_track_id": _str_value(sample["predicted_track_id"]),
                    "image_file": sample["image_file"],
                })
    return [{
        **sample,
        "image_sha256": digest(output_dir/sample["image_file"]),
    } for sample in samples]


def create_packet(*, run_dir: Path, forensics_dir: Path, output_dir: Path,
                  max_samples: int = 14) -> dict:
    if output_dir.exists():
        raise FileExistsError("Refusing overwrite of previous review evidence")
    video, sources, events = check_sources(run_dir=run_dir, forensics_dir=forensics_dir)
    samples = make_plan(events, max_samples=max_samples)
    from streetlab_phase3.video.continuous_box_tracking_lab import sequential_video_rgb_frames
    frame_size = tuple(sources["source_video_dimensions"])
    images = render_packet(
        samples=samples,
        source_frames=sequential_video_rgb_frames(
            video, min(s["video_frame"] for s in samples),
            max(s["video_frame"] for s in samples),
            expected_dimensions=frame_size,
        ),
        frame_size=frame_size, output_dir=output_dir,
    )
    event_groups: dict[tuple[str, str, str], int] = {}
    for item in events:
        key = _event_identity(item)
        event_groups[key] = event_groups.get(key, 0) + 1
    results = {
        "schema_version": 1,
        "status": PACKET_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_evidence": sources,
        "selected_sample_count": len(images),
        "selected_source_frame_count": len({i["video_frame"] for i in images}),
        "original_candidate_event_count": len(events),
        "original_candidate_unique_track_event_groups": len(event_groups),
        "source_frames_are_zero_based": True,
        "FLUID_frame_offset": 1,
        "review_crops_are_resized_for_display_only": True,
        "model_inference_performed": False,
        "scoring_or_policy_modified": False,
        "independent_physical_object_adjudication_performed": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "samples": images,
        "contact_sheet_sha256": digest(output_dir / "review_contact_sheet.jpg"),
        "reviewer_1_blank_sha256": digest(output_dir / "reviewer_1_blank.csv"),
        "reviewer_2_blank_sha256": digest(output_dir / "reviewer_2_blank.csv"),
        "research_instruction": (
            "Two blinded reviewers independently classify visible physical objects; "
            "candidate predictions and FLUID labels on the evidence images are NOT "
            "ground truth. Do not mutate FLUID/track files or infer population-wide "
            "metrics from this high-priority selected sample."
        ),
    }
    with (output_dir / "review_packet_manifest.json").open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create track-diverse visual review crops from frozen May25 evidence")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--forensics-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=14)
    args = p.parse_args(argv)
    result = create_packet(
        run_dir=args.run_dir, forensics_dir=args.forensics_dir,
        output_dir=args.output_dir, max_samples=args.max_samples,
    )
    print(json.dumps({
        "status": result["status"],
        "distinct_event_track_groups": result["original_candidate_unique_track_event_groups"],
        "selected_samples": result["selected_sample_count"],
        "selected_frames": result["selected_source_frame_count"],
        "independent_holdout_pass": False,
        "contact_sheet": str(args.output_dir / "review_contact_sheet.jpg"),
        "manifest": str(args.output_dir / "review_packet_manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
