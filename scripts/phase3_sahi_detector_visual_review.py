"""Save side-by-side source-video visual crops for W04 OpenVINO precision review.

Read-only CCTV decoding and cached prediction overlays; no model inference.
Only CENTER markers are available; do not interpret them as object boxes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.sahi_detector_visual_review import render_detector_review


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--openvino-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-items", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=384)
    args = p.parse_args(argv)
    result = render_detector_review(
        review_json=args.review,
        reference_dir=args.reference_dir,
        openvino_dir=args.openvino_dir,
        output_dir=args.output_dir,
        max_items=args.max_items,
        crop_size=args.crop_size)
    print(json.dumps({
        "status": result["status"],
        "image_count": result["image_count"],
        "index": str(args.output_dir / "index.json"),
        "directory": str(args.output_dir),
        "eligible_for_promotion": False,
        "caution": result["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
