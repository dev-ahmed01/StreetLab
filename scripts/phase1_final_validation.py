from __future__ import annotations

import argparse
import json

from streetlab_phase1.phase1_final_validation import run_final_validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    print(
        "\nWARNING: this opens the untouched held-out validation period. "
        "After a completed report is written, Phase 1 is closed and this "
        "script intentionally refuses to rerun.\n"
    )

    report = run_final_validation(args.project_root)

    compact = {
        "sprint": report["sprint"],
        "frozen_model": report["frozen_model"],
        "validation_dataset": report["validation_dataset"],
        "simulation": report["simulation"],
        "capacity_gate_pass": report["capacity_gate_pass"],
        "speed_objective": report["speed_objective"],
        "spacing_objective": report["spacing_objective"],
        "precommitted_generalization_guards": report[
            "precommitted_generalization_guards"
        ],
        "holdout_generalization_pass": report[
            "holdout_generalization_pass"
        ],
        "phase1_status": report["phase1_status"],
        "known_limitations": report["known_limitations"],
        "retuning_allowed": report["retuning_allowed"],
        "phase1_closed": report["phase1_closed"],
        "next_project_phase": report["next_project_phase"],
    }

    print("\nPHASE 1 — FINAL HELD-OUT VALIDATION")
    print(json.dumps(compact, indent=2))
    print(
        "\nSaved: data\\processed\\phase1_final_heldout_validation.json"
    )
    print(
        "\nPHASE 1 IS NOW CLOSED. Do not tune against this validation result."
    )


if __name__ == "__main__":
    main()
