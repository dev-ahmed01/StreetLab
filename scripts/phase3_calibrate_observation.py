from __future__ import annotations

import argparse
import json
from pathlib import Path

from streetlab_phase3.adapters.generic import GenericTrajectoryMapping
from streetlab_phase3.ingestion import CalibrationPipeline
from streetlab_phase3.serialization import package_to_dict


def _mapping(path: str | None):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GenericTrajectoryMapping(**payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--signals")
    ap.add_argument("--routes")
    ap.add_argument("--mapping")
    ap.add_argument("--output", default="artifacts/phase3_observation_package.json")
    args = ap.parse_args()

    package = CalibrationPipeline().calibrate_files(
        track_file=args.tracks,
        signal_file=args.signals,
        route_file=args.routes,
        generic_mapping=_mapping(args.mapping),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(package_to_dict(package), indent=2),
        encoding="utf-8",
    )

    print(f"Provider: {package.source_provider}")
    print(f"Tracks: {package.summary.unique_tracks}")
    print(f"Classes: {package.summary.class_counts}")
    print(f"Movements: {package.summary.movement_counts}")
    print(f"Mean speed: {package.summary.mean_speed_mps}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
