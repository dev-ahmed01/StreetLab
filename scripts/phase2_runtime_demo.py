from __future__ import annotations

import argparse
import json
from pathlib import Path

from streetlab_phase2.decision_lab import _write_routes
from streetlab_phase2.demo_network import write_demo_network, write_vtypes
from streetlab_phase2.models import BranchSpec, DecisionEvent, DecisionType, ResponsePolicy
from streetlab_phase2.personas import load_phase1_personas, profiles_by_class
from streetlab_phase2.runtime import RuntimeSimulation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--workdir", default=".streetlab-m2")
    ap.add_argument("--output", default="artifacts/phase2_m2_runtime.json")
    ap.add_argument("--decision-at-s", type=float, default=60.0)
    ap.add_argument("--horizon-s", type=float, default=120.0)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = root / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    profiles = load_phase1_personas(root / "data" / "models" / "sumo_persona_priors.csv")
    grouped = profiles_by_class(profiles)
    net = write_demo_network(workdir)
    types = write_vtypes(workdir, profiles)
    routes = _write_routes(workdir, grouped)

    runtime = RuntimeSimulation(
        net=net,
        routes=routes,
        types=types,
        workdir=workdir / "snapshots",
        simulation_id="m2-demo",
    )

    try:
        started = runtime.start()
        runtime.step_until(args.decision_at_s)
        paused = runtime.pause()

        injected = DecisionEvent(
            DecisionType.BLOCK_TURN,
            at_s=args.decision_at_s,
            duration_s=120.0,
            from_edge="WJ",
            blocked_edge="JN",
            response_policy=ResponsePolicy.GUIDED_DETOUR,
            metadata={
                "provenance": "ASSUMED",
                "note": "Runtime route-response behavior is a scenario assumption.",
            },
        )
        receipt = runtime.inject_decision(injected)
        snapshot = runtime.capture_snapshot("decision_point")
        runtime.resume()
        runtime.step(1)
        live_after_injection = runtime.state()

        branch_decision_natural = DecisionEvent(
            DecisionType.BLOCK_TURN,
            at_s=snapshot.simulation_time_s,
            duration_s=120.0,
            from_edge="WJ",
            blocked_edge="JN",
            response_policy=ResponsePolicy.NATURAL_REROUTE,
        )
        branch_decision_guided = DecisionEvent(
            DecisionType.BLOCK_TURN,
            at_s=snapshot.simulation_time_s,
            duration_s=120.0,
            from_edge="WJ",
            blocked_edge="JN",
            response_policy=ResponsePolicy.GUIDED_DETOUR,
        )

        comparison = runtime.run_branches(
            snapshot_id=snapshot.snapshot_id,
            horizon_s=args.horizon_s,
            branches=[
                BranchSpec("BASELINE", None, "No intervention."),
                BranchSpec(
                    "NATURAL_RESPONSE",
                    branch_decision_natural,
                    "Block turn with local/natural rerouting assumption.",
                ),
                BranchSpec(
                    "GUIDED_DIVERSION",
                    branch_decision_guided,
                    "Block turn with guided upstream diversion assumption.",
                ),
            ],
        )

        report = {
            "milestone": "PHASE2_M2_RUNTIME_STATE_MANAGER",
            "objective": (
                "Run a reusable live simulation lifecycle, inject a structured "
                "decision, snapshot the exact decision state, and fork comparable "
                "counterfactual branches with deterministic provenance."
            ),
            "lifecycle": {
                "started": started,
                "paused_at_decision": paused,
                "decision_receipt": receipt,
                "live_after_injection": live_after_injection,
            },
            "snapshot": {
                "snapshot_id": snapshot.snapshot_id,
                "simulation_time_s": snapshot.simulation_time_s,
                "path": snapshot.path,
                "sha256": snapshot.sha256,
                "injected_decision_count": snapshot.injected_decision_count,
            },
            "comparison": comparison,
            "scientific_guardrail": (
                "StreetLab evaluates plausible counterfactual outcomes under "
                "explicit assumptions; it does not claim exact future prediction."
            ),
        }
    finally:
        runtime.close()

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSTREETLAB PHASE 2 M2 — RUNTIME STATE MANAGER")
    print(f"Decision snapshot: {snapshot.simulation_time_s:.1f}s")
    print(f"Snapshot SHA-256: {snapshot.sha256}")
    print(f"Shared snapshot: {comparison['same_snapshot_for_all_branches']}")
    for branch in comparison["branches"]:
        m = branch["metrics"]
        print(
            f"{branch['name']:<20} arrivals={m['arrivals_after_decision']:>3} "
            f"rerouted={m['rerouted_vehicles']:>3} "
            f"maxQ={m['max_approach_queue_vehicles']:>3} "
            f"speed={m['mean_network_speed_mps']:.3f}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
