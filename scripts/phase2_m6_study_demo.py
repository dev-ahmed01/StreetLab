from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from streetlab_phase2.api import create_app
from streetlab_phase2.web_service import DecisionLabService


METRICS = [
    "max_approach_queue_vehicles",
    "mean_network_speed_mps",
    "rerouted_vehicles",
]

INCOMPLETE_EVIDENCE = [
    {"name": "geometry", "provenance": "ASSUMED", "note": "synthetic demo network"},
    {"name": "demand", "provenance": "ASSUMED", "note": "synthetic demo demand"},
    {"name": "turn_movements", "provenance": "ASSUMED", "note": "synthetic demo split"},
    {"name": "speeds", "provenance": "CALIBRATED", "note": "Phase-1 speed evidence"},
]

SUPPORTED_EVIDENCE = INCOMPLETE_EVIDENCE + [
    {
        "name": "alternate_route",
        "provenance": "ASSUMED",
        "note": "synthetic WJ -> JE -> EN -> N2 -> NS route",
    }
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--workdir", default=".streetlab-m6")
    ap.add_argument("--output", default="artifacts/phase2_m6_study.json")
    ap.add_argument("--decision-at-s", type=float, default=60.0)
    ap.add_argument("--members", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    service = DecisionLabService(project_root=root, workdir=args.workdir)
    client = TestClient(create_app(service=service))

    ui = client.get("/")
    assert ui.status_code == 200
    assert "APPLY_DETOUR" in ui.text
    assert "Evidence sufficiency" in ui.text

    incomplete_study = client.post(
        "/api/study/check",
        json={
            "decision_type": "APPLY_DETOUR",
            "area": "demo_junction",
            "baseline": "current_state",
            "scenario_family": "temporary_route_management",
            "requested_metrics": METRICS,
            "evidence": INCOMPLETE_EVIDENCE,
        },
    )
    assert incomplete_study.status_code == 200
    incomplete = incomplete_study.json()
    assert incomplete["status"] == "NEEDS_DATA"
    assert incomplete["can_simulate"] is False
    assert incomplete["missing_evidence"] == ["alternate_route"]
    assert "confidence" not in incomplete

    started = client.post("/api/simulation/start")
    assert started.status_code == 200
    assert started.json()["status"] == "RUNNING"

    advanced = client.post(
        "/api/simulation/advance",
        json={"target_time_s": args.decision_at_s},
    )
    assert advanced.status_code == 200
    state_at_decision = advanced.json()
    assert abs(state_at_decision["simulation_time_s"] - args.decision_at_s) <= 0.21

    refused = client.post(
        "/api/decision/evaluate",
        json={
            "decision_type": "APPLY_DETOUR",
            "from_edge": "WJ",
            "blocked_edge": "JN",
            "duration_s": 120.0,
            "guided_share": 0.50,
            "members": args.members,
            "horizon_s": 120.0,
            "requested_metrics": METRICS,
            "evidence": INCOMPLETE_EVIDENCE,
        },
    )
    assert refused.status_code == 200
    refused_body = refused.json()
    assert refused_body["study"]["status"] == "NEEDS_DATA"
    assert refused_body["decision_evaluated"] is False
    assert refused_body["snapshot"] is None
    assert refused_body["ensemble"] is None
    assert "confidence" not in refused_body

    still_live = client.get("/api/simulation/state")
    assert still_live.status_code == 200
    assert still_live.json()["status"] == "RUNNING"
    assert still_live.json()["simulation_time_s"] == state_at_decision["simulation_time_s"]

    supported_study = client.post(
        "/api/study/check",
        json={
            "decision_type": "APPLY_DETOUR",
            "area": "demo_junction",
            "baseline": "current_state",
            "scenario_family": "temporary_route_management",
            "requested_metrics": METRICS,
            "evidence": SUPPORTED_EVIDENCE,
        },
    )
    assert supported_study.status_code == 200
    supported = supported_study.json()
    assert supported["status"] == "SUPPORTED"
    assert supported["can_simulate"] is True
    assert supported["evidence_provenance"]["alternate_route"] == "ASSUMED"
    assert supported["evidence_provenance"]["speeds"] == "CALIBRATED"

    evaluated = client.post(
        "/api/decision/evaluate",
        json={
            "decision_type": "APPLY_DETOUR",
            "from_edge": "WJ",
            "blocked_edge": "JN",
            "duration_s": 120.0,
            "guided_share": 0.50,
            "members": args.members,
            "horizon_s": 120.0,
            "requested_metrics": METRICS,
            "evidence": SUPPORTED_EVIDENCE,
        },
    )
    assert evaluated.status_code == 200, evaluated.text
    result = evaluated.json()

    assert result["study"]["status"] == "SUPPORTED"
    assert result["decision_evaluated"] is True
    assert result["decision"]["type"] == "APPLY_DETOUR"

    meta = result["ensemble"]["ensemble"]
    assert meta["members"] == args.members
    assert meta["provenance"] == "ASSUMED"
    assert meta["same_snapshot_for_all_members"] is True

    summaries = result["ensemble"]["branch_summaries"]
    assert set(summaries) == {
        "BASELINE",
        "DETOUR_GUIDED",
        "DETOUR_MIXED",
    }
    mixed = summaries["DETOUR_MIXED"]["metrics"]
    assert mixed["heterogeneous_guided_responders"]["min"] > 0
    assert mixed["heterogeneous_local_responders"]["min"] > 0
    assert mixed["rerouted_vehicles"]["min"] > 0

    paused = client.get("/api/simulation/state")
    assert paused.status_code == 200
    assert paused.json()["status"] == "PAUSED"
    assert paused.json()["simulation_time_s"] == state_at_decision["simulation_time_s"]

    g = result["guardrails"]
    assert g["confidence_fabricated"] is False
    assert g["authority_remains_final_decision_maker"] is True
    assert g["live_runtime_mutated_by_evaluated_branch"] is False

    client.post("/api/simulation/close")

    report = {
        "milestone": "PHASE2_M6_STUDY_EVIDENCE_DECISIONS",
        "objective": (
            "Gate counterfactual simulation on explicit evidence sufficiency, "
            "refuse unsupported studies with NEEDS_DATA, and add APPLY_DETOUR "
            "without changing StreetLab's decision-assurance role."
        ),
        "incomplete_study": incomplete,
        "refused_evaluation": {
            "study": refused_body["study"],
            "decision_evaluated": refused_body["decision_evaluated"],
            "snapshot": refused_body["snapshot"],
            "ensemble": refused_body["ensemble"],
        },
        "supported_study": supported,
        "decision": result["decision"],
        "snapshot": result["snapshot"],
        "ensemble": {
            "metadata": meta,
            "branch_summaries": summaries,
        },
        "guardrails": g,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSTREETLAB PHASE 2 M6 — STUDY EVIDENCE + APPLY_DETOUR")
    print("Incomplete study:", incomplete["status"], incomplete["missing_evidence"])
    print("Supported study:", supported["status"])
    print(f"Decision: {result['decision']['type']} at {result['decision']['at_s']:.1f}s")
    for name, branch in summaries.items():
        queue = branch["metrics"]["max_approach_queue_vehicles"]
        speed = branch["metrics"]["mean_network_speed_mps"]
        print(
            f"{name:<18} queue median={queue['median']:.2f} "
            f"speed median={speed['median']:.3f}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
