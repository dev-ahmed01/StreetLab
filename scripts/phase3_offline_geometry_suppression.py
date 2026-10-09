"""Score post-track geometry suppression policies without rerunning inference.

Uses existing exported SAHI boxes and the frozen May-26 W04 cohort.
No source files or benchmark scores are overwritten; results are exploratory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.offline_geometry_suppression import run_offline_suppression


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sliced-tracks", required=True, type=Path)
    ap.add_argument("--fluid-tracks", required=True, type=Path)
    ap.add_argument("--comparison", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    report = run_offline_suppression(
        sliced_tracks=args.sliced_tracks,
        fluid_tracks=args.fluid_tracks,
        comparison_file=args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({
        "status": report["status"],
        "eligible_for_promotion": False,
        "baseline": {
            key: value for key, value in report["original_sahi_metrics"].items()
            if key in ("matched_points", "unmatched_predictions", "point_recall",
                       "point_precision", "motorcycle_correct_class_recall")
        },
        "policies": [
            {
                "policy": p["policy"],
                "proposed_removals": p["proposed_removals"],
                "removed_by_class": p["removed_by_class"],
                "matched_points": p["metrics"]["matched_points"],
                "unmatched_predictions": p["metrics"]["unmatched_predictions"],
                "point_recall": p["metrics"]["point_recall"],
                "point_precision": p["metrics"]["point_precision"],
                "motorcycle_recall": p["metrics"]["motorcycle_correct_class_recall"],
                "delta": p["changes_vs_unchanged_sahi"],
            }
            for p in report["policies"]
        ],
        "output": str(args.output),
        "caution": report["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
