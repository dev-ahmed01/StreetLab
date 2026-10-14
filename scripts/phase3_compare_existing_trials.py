"""Report fair same-frame comparison from existing FLUID tracking outputs.

No GPU, SAHI, Ultralytics, video decoding or further inference required.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.offline_trial_comparison import compare_existing_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-tracks", type=Path, required=True)
    parser.add_argument("--sliced-tracks", type=Path, required=True)
    parser.add_argument("--fluid-tracks", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_existing_runs(
        standard_tracks=args.standard_tracks, sliced_tracks=args.sliced_tracks,
        fluid_tracks=args.fluid_tracks, start_frame=args.start_frame,
        end_frame=args.end_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
    summary = {
        "status": result["status"],
        "window": [result["evaluation_start_frame"], result["evaluation_end_frame"]],
        "truth_points": result["truth_points"],
        "confounders": result["confounders"],
        "deltas_sliced_minus_standard": result["deltas_sliced_minus_standard"],
    }
    for label in ("standard", "sliced"):
        item = result["results"][label]
        motorcycle = item["class_diagnostics"]["by_class"]["MOTORCYCLE"]
        summary[label] = {
            "point_recall": item["pixel"]["point_recall"],
            "point_precision": item["pixel"]["point_precision"],
            "matched_points": item["pixel"]["matched_points"],
            "unmatched_predictions": (
                item["pixel"]["predicted_points"] - item["pixel"]["matched_points"]),
            "motorcycle_truth": motorcycle["truth_points"],
            "motorcycle_spatial_recall": motorcycle["spatial_recall"],
            "motorcycle_correct_class_recall": motorcycle["correct_class_recall"],
            "motorcycle_spatial_misses": motorcycle["spatial_misses"],
            "fraction_truth_tracks_fragmented": item["identity"]["fraction_truth_tracks_fragmented"],
            "matched_truth_tracks": item["identity"]["matched_truth_tracks"],
        }
    audit = result["paired_error_audit"]
    motorcycles = audit["by_truth_class"]["MOTORCYCLE"]
    summary["paired_motorcycle_outcomes"] = {
        "spatial": motorcycles["spatial"],
        "correct_class": motorcycles["correct_class"],
        "spatial_rescue_examples": motorcycles["spatially_rescued_by_sliced"],
        "spatial_loss_examples": motorcycles["spatially_lost_by_sliced"],
    }
    summary["unmatched_predictions_forensics"] = {
        engine: {
            kind: audit[f"{engine}_unmatched"][kind]
            for kind in ("MOTORCYCLE", "CAR", "BUS", "HEAVY_VEHICLE")
        }
        for engine in ("standard", "sliced")
    }
    sensitivity = result["post_track_confidence_sensitivity"]
    summary["confidence_sensitivity"] = {
        "status": sensitivity["status"],
        "note": sensitivity["note"],
        "rows": [
            {
                "policy": r["policy"],
                "cutoff": r["confidence_cutoff_inclusive"],
                "kept": r["predictions_kept"],
                "unmatched": r["unmatched_predictions"],
                "point_recall": r["overall_point_recall"],
                "point_precision": r["overall_point_precision"],
                "motorcycle_correct_class_recall": r["motorcycle_correct_class_recall"],
            }
            for r in sensitivity["results"]
        ],
    }
    proximity = result["sliced_unmatched_proximity"]
    summary["unmatched_proximity"] = {
        "status": proximity["status"],
        "unmatched_total": proximity["unmatched_total"],
        "by_class": proximity["by_class"],
        "note": proximity["note"],
    }
    shadow = result["shadow_matching_audit"]
    summary["shadow_matching"] = {
        "status": shadow["status"],
        "important": shadow["important"],
        "engines": {
            mode: {
                "frozen_matches": shadow[mode]["frozen_p3b_hungarian_matches"],
                "shadow_matches": shadow[mode]["shadow_max_cardinality_matches"],
                "extra_valid_matches": shadow[mode]["additional_in_gate_matches"],
                "unmatched_after_shadow": shadow[mode]["shadow_unmatched_predictions"],
                "previously_unmatched_high_confidence_cars_newly_assigned":
                    shadow[mode]["previously_unmatched_high_confidence_car_predictions_now_assigned"],
                "new_truth_matches_by_class": shadow[mode]["newly_matched_truth_by_class"],
            }
            for mode in ("standard", "sliced")
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"Full diagnostic written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
