from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from streetlab_phase1.behavior_models import (
    ContestConfig,
    PRIMARY_CLASSES,
    run_class_contest,
    build_overall_report,
    save_contest_outputs,
)


def main():
    p = argparse.ArgumentParser(description="StreetLab Sprint 1C behavior model contest")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--models", type=Path, default=Path("data/models"))
    args = p.parse_args()

    cal_path = args.processed / "calibration_context_behavior.parquet"
    val_path = args.processed / "validation_context_behavior.parquet"
    if not cal_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Sprint 1B.6 outputs missing. Expected calibration_context_behavior.parquet "
            "and validation_context_behavior.parquet in data/processed."
        )

    cal = pd.read_parquet(cal_path)
    val = pd.read_parquet(val_path)
    cfg = ContestConfig()

    results = []
    for cls in PRIMARY_CLASSES:
        print(f"\nTesting vehicle class {cls}...")
        r = run_class_contest(cal, val, cls, cfg)
        results.append(r)
        d = r["decision"]
        print(json.dumps({
            "vehicle_class": cls,
            "vehicle_class_name": r["vehicle_class_name"],
            "chosen_k": d["chosen_k_by_calibration_bic"],
            "bic_improvement_vs_k1": d["bic_improvement_vs_k1"],
            "validation_loglik_gain_vs_k1": d["validation_loglik_gain_vs_k1"],
            "persona_evidence_supported": d["persona_evidence_supported"],
            "representation_decision": d["representation_decision"],
        }, indent=2))

    report = build_overall_report(results, cfg)
    save_contest_outputs(results, report, args.processed, args.models)

    print("\nSPRINT 1C BEHAVIOR MODEL CONTEST")
    print(json.dumps(report, indent=2))
    print("\nSaved:")
    print(args.processed / "behavior_model_contest_scores.csv")
    print(args.processed / "persona_profiles.csv")
    print(args.processed / "calibration_behavior_dna.parquet")
    print(args.processed / "validation_behavior_dna.parquet")
    print(args.processed / "sprint1c_model_contest.json")
    print(args.models / "behavior_gmm_class_1.joblib")
    print(args.models / "behavior_gmm_class_2.joblib")
    print(args.models / "behavior_gmm_class_6.joblib")
    print("\nNext gate: interpret persona stability before mapping any traits to SUMO parameters.")


if __name__ == "__main__":
    main()
