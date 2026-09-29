from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_joint_longitudinal import run_joint_longitudinal_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_joint_longitudinal_grid(args.project_root)

    compact = {
        "sprint": report["sprint"],
        "observed_spacing_source": report["observed_spacing_source"],
        "fixed_parameters": report["fixed_parameters"],
        "grid": report["grid"],
        "capacity_passing_cases": report["capacity_passing_cases"],
        "pareto_front": report["pareto_front"],
        "best_joint_candidate_if_any": report["best_joint_candidate_if_any"],
        "decision_rules": report["decision_rules"],
    }

    print("\nSPRINT 1E-A8 — JOINT TAU / minGap GRID")
    print(json.dumps(compact, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a8_joint_longitudinal_grid.json")
    print("\nDo not touch validation or freeze a pair until the Pareto/spacing results are reviewed.")


if __name__ == "__main__":
    main()
