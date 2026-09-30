from __future__ import annotations

import argparse
import json

from streetlab_phase3.video.geotrax_provider import GeoTraxVideoProvider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--orthophotos")
    ap.add_argument("--segmentations")
    ap.add_argument("--master-frames")
    ap.add_argument("--executable", default="geotrax")
    args = ap.parse_args()

    plan = GeoTraxVideoProvider(executable=args.executable).plan(
        video=args.video,
        orthophotos=args.orthophotos,
        segmentations=args.segmentations,
        master_frames=args.master_frames,
    )

    print(json.dumps({
        "command": list(plan.command),
        "georeferenced": plan.georeferenced,
        "calibration_ready": plan.calibration_ready,
        "license_boundary": plan.license_boundary,
        "expected_output": plan.expected_output,
    }, indent=2))


if __name__ == "__main__":
    main()
