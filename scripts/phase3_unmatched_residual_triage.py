"""Summarize unpaired OpenVINO W04 detections into manual-review tiers.

Consumes existing SHA-linked audit, frozen 187-unmatched precision review and
21-cross-class shadow pairs. Does not decode video, execute inference,
suppress objects, or change official scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.unmatched_residual_triage import triage_unmatched_residual


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir", required=True, type=Path)
    p.add_argument("--precision-review", required=True, type=Path)
    p.add_argument("--shadow-report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    result = triage_unmatched_residual(
        audit_dir=args.audit_dir, review_file=args.precision_review,
        shadow_file=args.shadow_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({
        "status": result["status"],
        "official_unmatched": result["original_frozen_class_aware"]["unmatched"],
        "shadow_cross_class_pairs": result["shadow_cross_class_pairs"],
        "residual_unpaired_predictions": result["residual_unpaired_predictions"],
        "exclusive_review_tiers": result["exclusive_review_tiers"],
        "overlapping_proximity_flags": result["overlapping_proximity_flags"],
        "exclusive_tiers_by_prediction_class":
            result["exclusive_tiers_by_prediction_class"],
        "manual_review_examples": result["representative_review_queue"][:12],
        "report": str(args.output),
        "caution": result["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
