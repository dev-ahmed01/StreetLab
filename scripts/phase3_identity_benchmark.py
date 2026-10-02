from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streetlab_phase3.identity_benchmark import run_identity_benchmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geotrax-tracks", required=True)
    ap.add_argument("--fluid-tracks", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-pixel-distance", type=float, default=50.0)
    ap.add_argument("--max-switch-gap-frames", type=int, default=5)
    args = ap.parse_args()

    result = run_identity_benchmark(
        geotrax_tracks=args.geotrax_tracks,
        fluid_tracks=args.fluid_tracks,
        output_dir=args.output_dir,
        max_distance_px=args.max_pixel_distance,
        max_switch_gap_frames=args.max_switch_gap_frames,
    )
    print(json.dumps(result.scorecard_dict(), indent=2))


if __name__ == "__main__":
    main()
