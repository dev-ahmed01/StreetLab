"""Offline local visual review of genuine CCTV, no model inference.

Requires the JSON report from phase3_compare_existing_trials.py plus unchanged
video, FLUID truth and two raw tracking outputs; renders review crops locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.visual_error_review import write_visual_review


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--fluid-tracks", required=True, type=Path)
    ap.add_argument("--standard-tracks", required=True, type=Path)
    ap.add_argument("--sliced-tracks", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--max-items", type=int, default=22)
    ap.add_argument("--crop-size", type=int, default=360)
    args = ap.parse_args(argv)
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    output = write_visual_review(
        report, video=args.video, fluid_tracks=args.fluid_tracks,
        standard_tracks=args.standard_tracks, sliced_tracks=args.sliced_tracks,
        output_dir=args.output_dir,
        max_items=args.max_items, crop_size=args.crop_size,
    )
    print(json.dumps({
        "status": output["status"],
        "image_count": output["image_count"],
        "directory": str(args.output_dir),
        "index": str(args.output_dir / "index.json"),
        "eligible_for_promotion": False,
        "caution": output["caution"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
