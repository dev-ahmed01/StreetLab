from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streetlab_phase3.motorcycle_visual_audit import run_motorcycle_visual_audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fluid-tracks", required=True)
    ap.add_argument("--geotrax-tracks", required=True)
    ap.add_argument("--track-diagnostics", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frame-offset", type=int, choices=(1,), default=1)
    ap.add_argument(
        "--max-pixel-distance",
        type=float,
        choices=(50.0,),
        default=50.0,
    )
    args = ap.parse_args()

    summary = run_motorcycle_visual_audit(
        video=args.video,
        fluid_tracks=args.fluid_tracks,
        geotrax_tracks=args.geotrax_tracks,
        track_diagnostics=args.track_diagnostics,
        output_dir=args.output_dir,
        frame_offset=args.frame_offset,
        max_distance_px=args.max_pixel_distance,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
