from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from streetlab_phase2.api import create_app
from streetlab_phase2.web_service import DecisionLabService


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--workdir", default=".streetlab-m5")
    ap.add_argument("--output", default="artifacts/phase2_m5_web_smoke.json")
    ap.add_argument("--decision-at-s", type=float, default=60.0)
    ap.add_argument("--members", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    service = DecisionLabService(
        project_root=root,
        workdir=args.workdir,
    )
    client = TestClient(create_app(service=service))

    ui = client.get("/")
    assert ui.status_code == 200
    assert "StreetLab Decision Lab" in ui.text
    assert "Scenario sensitivity" in ui.text

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["product_role"] == "decision_assurance"

    started = client.post("/api/simulation/start")
    assert started.status_code == 200
    assert started.json()["status"] == "RUNNING"

    advanced = client.post(
        "/api/simulation/advance",
        json={"target_time_s": args.decision_at_s},
    )
    assert advanced.status_code == 200
    state = advanced.json()
    assert abs(state["simulation_time_s"] - args.decision_at_s) <= 0.21

    evaluated = client.post(
        "/api/decision/evaluate",
        json={
            "decision_type": "BLOCK_TURN",
            "from_edge": "WJ",
            "blocked_edge": "JN",
            "duration_s": 120.0,
            "guided_share": 0.50,
            "members": args.members,
            "horizon_s": 120.0,
        },
    )
    assert evaluated.status_code == 200, evaluated.text
    result = evaluated.json()

    assert result["decision"]["type"] == "BLOCK_TURN"
    assert result["snapshot"]["simulation_time_s"] == state["simulation_time_s"]

    guardrails = result["guardrails"]
    assert guardrails["phase1_retuned"] is False
    assert guardrails["microscopic_parameters_varied"] is False
    assert guardrails["routing_behavior_empirically_learned"] is False
    assert guardrails["uncertainty_is_probability_forecast"] is False
    assert guardrails["authority_remains_final_decision_maker"] is True
    assert guardrails["live_runtime_mutated_by_evaluated_branch"] is False

    ensemble = result["ensemble"]
    meta = ensemble["ensemble"]
    assert meta["members"] == args.members
    assert meta["provenance"] == "ASSUMED"
    assert meta["same_snapshot_for_all_members"] is True

    summaries = ensemble["branch_summaries"]
    assert set(summaries) == {
        "BASELINE",
        "NATURAL_RESPONSE",
        "GUIDED_DIVERSION",
        "MIXED_RESPONSE",
    }
    assert summaries["MIXED_RESPONSE"]["metrics"][
        "heterogeneous_guided_responders"
    ]["min"] > 0
    assert summaries["MIXED_RESPONSE"]["metrics"][
        "heterogeneous_local_responders"
    ]["min"] > 0

    after = client.get("/api/simulation/state")
    assert after.status_code == 200
    assert after.json()["status"] == "PAUSED"
    assert after.json()["simulation_time_s"] == state["simulation_time_s"]

    closed = client.post("/api/simulation/close")
    assert closed.status_code == 200
    assert closed.json()["status"] == "CLOSED"

    report = {
        "milestone": "PHASE2_M5_DECISION_LAB_WEB",
        "objective": (
            "Expose the existing runtime, decision branching, and uncertainty "
            "engine through a human-facing Decision Lab API/UI without adding "
            "policy recommendation logic."
        ),
        "health": health.json(),
        "state_at_decision": state,
        "decision": result["decision"],
        "snapshot": result["snapshot"],
        "ensemble": {
            "metadata": meta,
            "branch_summaries": summaries,
        },
        "guardrails": guardrails,
        "ui_checks": {
            "title_present": "StreetLab Decision Lab" in ui.text,
            "scenario_sensitivity_present": "Scenario sensitivity" in ui.text,
            "network_svg_present": 'id="networkMap"' in ui.text,
            "decision_control_present": 'id="evaluateDecision"' in ui.text,
        },
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSTREETLAB PHASE 2 M5 — DECISION LAB WEB")
    print(f"Decision time: {state['simulation_time_s']:.1f}s")
    print(f"Snapshot: {result['snapshot']['sha256']}")
    print(f"Members: {meta['members']}")
    for name, branch in summaries.items():
        queue = branch["metrics"]["max_approach_queue_vehicles"]
        speed = branch["metrics"]["mean_network_speed_mps"]
        print(
            f"{name:<20} queue median={queue['median']:.2f} "
            f"p10-p90={queue['p10']:.2f}-{queue['p90']:.2f} "
            f"speed median={speed['median']:.3f}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
