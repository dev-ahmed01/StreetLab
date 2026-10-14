"""Offline box overlap check on existing SAHI confirmed tracks; no inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.existing_box_forensics import existing_box_forensics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--sliced-tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    result = existing_box_forensics(args.sliced_tracks, comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({
        "status": result["status"],
        "video_frame_window": result["video_frame_window"],
        "unmatched_observations": result["unmatched_observations"],
        "high_confidence_unmatched_cars": result["high_confidence_unmatched_cars"],
        "unmatched_motorcycle": result["unmatched_motorcycle"],
        "limitations": result["limitations"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
