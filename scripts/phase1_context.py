from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from streetlab_phase1.context import (
    ContextConfig,
    add_local_stream_speed,
    merge_motion_fields,
    derive_thresholds,
    apply_context_labels,
    add_calibration_baselines,
    aggregate_context_behavior,
    compare_context_behavior,
    build_context_report,
)


def load_split(prefix: str, processed: Path):
    inter = pd.read_parquet(processed / f"{prefix}_interactions.parquet")
    clean = pd.read_parquet(processed / f"{prefix}_clean.parquet")
    veh = pd.read_parquet(processed / f"{prefix}_vehicle_features.parquet")
    frames = add_local_stream_speed(inter)
    frames = merge_motion_fields(frames, clean, veh)
    return frames


def main():
    p=argparse.ArgumentParser(description="StreetLab Sprint 1B.6 context normalization")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--min-context-cell-n", type=int, default=30)
    args=p.parse_args()
    cfg=ContextConfig(min_context_cell_n=args.min_context_cell_n)

    cal=load_split("calibration", args.processed)
    val=load_split("validation", args.processed)

    thresholds=derive_thresholds(cal)
    cal=apply_context_labels(cal, thresholds, cfg)
    val=apply_context_labels(val, thresholds, cfg)
    cal,val,baseline_meta=add_calibration_baselines(cal,val,cfg)

    cal_beh=aggregate_context_behavior(cal)
    val_beh=aggregate_context_behavior(val)
    drift=compare_context_behavior(cal_beh,val_beh,cfg)
    report=build_context_report(cal,val,cal_beh,val_beh,thresholds,baseline_meta,drift,cfg)

    args.processed.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(args.processed / "calibration_context_frames.parquet", index=False)
    val.to_parquet(args.processed / "validation_context_frames.parquet", index=False)
    cal_beh.to_parquet(args.processed / "calibration_context_behavior.parquet", index=False)
    val_beh.to_parquet(args.processed / "validation_context_behavior.parquet", index=False)
    drift.to_csv(args.processed / "context_normalized_drift.csv", index=False)
    (args.processed / "context_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (args.processed / "sprint1b6_context_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSPRINT 1B.6 CONTEXT NORMALISATION")
    print(json.dumps(report, indent=2))
    print("\nSaved context-normalised frames, per-vehicle behaviour features, drift report, and thresholds.")
    print("Do not cluster yet. Review sprint1b6_context_report.json first.")

if __name__ == "__main__":
    main()
