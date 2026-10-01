from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streetlab_phase3.serialization import package_to_dict
from streetlab_phase3.video_benchmark_cli import run_pixel_benchmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geotrax-tracks", required=True)
    ap.add_argument("--fluid-tracks", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-pixel-distance", type=float, default=50.0)
    args = ap.parse_args()

    report = run_pixel_benchmark(
        geotrax_tracks=args.geotrax_tracks,
        fluid_tracks=args.fluid_tracks,
        output=args.output,
        max_distance_px=args.max_pixel_distance,
    )
    print(json.dumps(package_to_dict(report), indent=2))


if __name__ == "__main__":
    main()
