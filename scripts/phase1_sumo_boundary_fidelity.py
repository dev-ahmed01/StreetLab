from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_boundary_fidelity import run_boundary_fidelity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_boundary_fidelity(args.project_root)

    print("\nSPRINT 1E-A6 — BOUNDARY FIDELITY")
    compact = {
        "sprint": report["sprint"],
        "observed_boundary": report["observed_boundary"],
        "fixed_behavior": report["fixed_behavior"],
        "gates": report["gates"],
        "results": [
            (
                {
                    "case": r["case"],
                    "status": "CASE_ERROR",
                    "error_type": r.get("error_type"),
                    "error": r.get("error"),
                    "legacy_gate_5s": False,
                    "boundary_gate_1s": False,
                }
                if r.get("status") == "CASE_ERROR"
                else {
                    "case": r["case"],
                    "boundary_speed_handling": r.get("boundary_speed_handling"),
                    "discretization": r["discretization"],
                    "objective": r["objective"],
                    "legacy_gate_5s": r["legacy_gate_5s"],
                    "boundary_gate_1s": r["boundary_gate_1s"],
                    "simulation": r["simulation"],
                    "speed_fit": r["speed_fit"],
                }
            )
            for r in report["results"]
        ],
        "any_boundary_gate_pass": report["any_boundary_gate_pass"],
        "any_legacy_gate_pass": report["any_legacy_gate_pass"],
    }
    print(json.dumps(compact, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a6_boundary_fidelity_report.json")
    print("\nDo not tune behavior parameters until this diagnostic is reviewed.")


if __name__ == "__main__":
    main()
