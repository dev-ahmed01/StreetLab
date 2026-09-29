from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_longitudinal_heterogeneity_audit import (
    run_longitudinal_heterogeneity_audit,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_longitudinal_heterogeneity_audit(args.project_root)

    compact = {
        "sprint": report["sprint"],
        "evidence": report["evidence"],
        "dna_columns": report["dna_columns"],
        "persona_analysis": report["persona_analysis"],
        "continuous_dna_analysis": report["continuous_dna_analysis"],
        "persona_signal_classes": report["persona_signal_classes"],
        "continuous_signal_classes": report["continuous_signal_classes"],
        "recommended_next_model": report["recommended_next_model"],
        "decision_rules": report["decision_rules"],
    }

    print("\nSPRINT 1E-A10 — LONGITUDINAL HETEROGENEITY AUDIT")
    print(json.dumps(compact, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a10_longitudinal_heterogeneity_audit.json")
    print("Seeds: data\\models\\a10_persona_tau_seeds.csv")
    print("\nDo not run a heterogeneous SUMO model until this evidence audit is reviewed.")


if __name__ == "__main__":
    main()
