from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_metric_parity import run_metric_parity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_metric_parity(args.project_root)

    print("\nSPRINT 1E-A9 — INTERACTION METRIC PARITY")
    compact = {
        "sprint": report["sprint"],
        "common_metric_definition": report["common_metric_definition"],
        "legacy_observed_interaction_parity": report["legacy_observed_interaction_parity"],
        "observed_common_interaction_rows": report["observed_common_interaction_rows"],
        "results": report["results"],
        "capacity_passing_cases": report["capacity_passing_cases"],
        "decision_rules": report["decision_rules"],
    }
    print(json.dumps(compact, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a9_metric_parity_report.json")
    print("\nDo not introduce heterogeneous tau/minGap until metric parity is reviewed.")


if __name__ == "__main__":
    main()
