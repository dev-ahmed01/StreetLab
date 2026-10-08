"""Generate immutable 21-frame OpenVINO unmatched-detection manual-review evidence.

No model inference, no video decoding, no suppression. Source scoring and
frozen FLUID labels remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.sahi_unmatched_detection_review import audit_unmatched_detections


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", required=True, type=Path)
    ap.add_argument("--parity", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--review-limit", type=int, default=12)
    args = ap.parse_args(argv)
    result = audit_unmatched_detections(
        args.audit_dir, parity_file=args.parity, review_limit=args.review_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({
        "status": result["status"],
        "eligible_for_promotion": False,
        "frames": len(result["frames"]),
        "matched_predictions": result["matched_predictions"],
        "unmatched_predictions": result["unmatched_predictions"],
        "classes": result["classes"],
        "parity_disagreements": result["parity_disagreements"],
        "review_queue": result["review_queue"],
        "report": str(args.output),
        "note": result["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
