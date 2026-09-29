from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_coupling import run_coupling_diagnostic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_coupling_diagnostic(args.project_root)
    print("\nSPRINT 1E-A3 — SPEED/CAPACITY COUPLING DIAGNOSTIC")
    # Keep console concise; the full report is written to disk.
    compact = {
        "sprint": report["sprint"],
        "fixed_longitudinal_baseline": report["fixed_longitudinal_baseline"],
        "candidate_speed_multipliers": report["candidate_speed_multipliers"],
        "strict_gate": report["strict_gate"],
        "results": report["results"],
        "best_strictly_passing_case": report["best_strictly_passing_case"],
        "next_step_if_coupling_confirmed": report["next_step_if_coupling_confirmed"],
    }
    print(json.dumps(compact, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a3_coupling_report.json")
    print("\nDo not freeze Stage-A speed parameters yet. Review this diagnostic first.")


if __name__ == "__main__":
    main()
