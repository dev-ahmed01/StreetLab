from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_interaction_forensics import run_forensics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_forensics(args.project_root)

    compact = {
        "sprint": report["sprint"],
        "legacy": report["legacy"],
        "forensic_signal": report["forensic_signal"],
        "ranked_candidates": report["ranked_candidates"],
        "decision_rules": report["decision_rules"],
    }

    print("\nSPRINT 1E-A9.2 — LEADER/GAP FORENSICS")
    print(json.dumps(compact, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a9_2_leader_gap_forensics.json")
    print("\nDo not rerun SUMO or change longitudinal parameters until this forensic report is reviewed.")


if __name__ == "__main__":
    main()
