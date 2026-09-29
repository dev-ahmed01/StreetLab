from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from streetlab_phase1.sanity import build_report, save_report


def main() -> None:
    p = argparse.ArgumentParser(description="StreetLab Sprint 1B.5 feature sanity + split-drift checks")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    d = args.processed
    required = {
        "cal_inter": d / "calibration_interactions.parquet",
        "val_inter": d / "validation_interactions.parquet",
        "cal_beh": d / "calibration_behavior_features.parquet",
        "val_beh": d / "validation_behavior_features.parquet",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Sprint 1B files:\n" + "\n".join(missing))

    cal_inter = pd.read_parquet(required["cal_inter"])
    val_inter = pd.read_parquet(required["val_inter"])
    cal_beh = pd.read_parquet(required["cal_beh"])
    val_beh = pd.read_parquet(required["val_beh"])

    report, profile, drift = build_report(cal_inter, val_inter, cal_beh, val_beh)
    save_report(report, profile, drift, d)

    print("\nSPRINT 1B.5 SANITY REPORT")
    print(json.dumps(report, indent=2))
    print("\nSaved:")
    print(d / "sprint1b_sanity.json")
    print(d / "behavior_profile_by_class.csv")
    print(d / "calibration_validation_drift.csv")
    print("\nDo not run clustering yet. Review this report first.")


if __name__ == "__main__":
    main()
