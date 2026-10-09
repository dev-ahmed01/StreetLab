"""Re-score a cached 21-frame SAHI audit on EXACTLY the new 5-frame probe.

Requires an existing detector-only reference directory (21 sampled frames)
and a detector-only candidate directory (5 selected frames). No inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.cached_detector_subset import compare_cached_detector_subsets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--candidate-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    result = compare_cached_detector_subsets(
        args.reference_dir, args.candidate_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    summary = {
        "status": result["status"],
        "samples": result["sample_frames"],
        "same_motorcycle_truth_points": result["same_motorcycle_truth_points"],
        "reference_runtime_backend": result["reference_runtime_backend"],
        "candidate_runtime_backend": result["candidate_runtime_backend"],
        "runtime_conversion_changes_numerics": result["runtime_conversion_changes_numerics"],
        "configurations": {
            "cached_sliced": result["reference_slice_config"],
            "candidate_sliced": result["candidate_slice_config"],
        },
        "scores": {
            key: {
                "overall_recall": data["recall"],
                "overall_precision": data["precision"],
                "motorcycle_recall": data["per_class"]["MOTORCYCLE"]["recall"],
                "motorcycle_precision": data["per_class"]["MOTORCYCLE"]["precision"],
                "unmatched_detections": (
                    data["predicted_points"] - data["matched_points"]),
            }
            for key, data in result["scores"].items()
        },
        "deltas_candidate_minus_cached_sliced":
            result["sliced_candidate_minus_reference"],
        "caution": result["limitations"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
