from __future__ import annotations

import argparse
import json

from streetlab_phase3.benchmark import FluidBenchmark
from streetlab_phase3.ingestion import load_rows
from streetlab_phase3.serialization import package_to_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-id", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--signals")
    ap.add_argument("--routes")
    ap.add_argument("--telemetry")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    report = FluidBenchmark().build(
        study_id=args.study_id,
        track_rows=load_rows(args.tracks),
        signal_rows=load_rows(args.signals) if args.signals else None,
        route_rows=load_rows(args.routes) if args.routes else None,
        telemetry_rows=load_rows(args.telemetry) if args.telemetry else None,
    )

    payload = package_to_dict(report)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
