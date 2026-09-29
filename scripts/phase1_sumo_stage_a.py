from __future__ import annotations

import argparse
import json
from pathlib import Path

from streetlab_phase1.sumo_stage_a import (
    StageAConfig,
    prepare_corridor,
    run_stage_a_once,
    calibrate_coordinate_descent,
    make_overall_report,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "calibrate"], default="baseline")
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    cfg = StageAConfig()
    ctx = prepare_corridor(args.project_root, cfg)

    if args.mode == "baseline":
        result = run_stage_a_once(
            ctx, cfg, {1: 1.0, 2: 1.0, 6: 1.0}, "baseline"
        )
        out_name = "sprint1e_stage_a_baseline_report.json"
    else:
        result = calibrate_coordinate_descent(ctx, cfg)
        out_name = "sprint1e_stage_a_calibration_report.json"

    report = make_overall_report(ctx, cfg, result, args.mode)
    out = Path(args.project_root) / "data" / "processed" / out_name
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("SPRINT 1E-A — CHENNAI CORRIDOR / SPEED STAGE")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")

    if args.mode == "baseline":
        print("\nSTOP HERE FIRST. Review the infrastructure gate before running calibration.")
    else:
        print("\nDo not unlock Stage B until speed-fit results are reviewed.")


if __name__ == "__main__":
    main()
