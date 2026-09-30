from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from dataclasses import asdict
from statistics import mean

from .demo_network import write_demo_network, write_vtypes
from .models import BranchSpec, DecisionEvent, DecisionType, ResponsePolicy
from .personas import deterministic_persona, load_phase1_personas, profiles_by_class
from .response import (
    ResponseAssumptions,
    ResponseMode,
    apply_heterogeneous_response,
)


def _ensure_sumo_tools() -> None:
    try:
        import traci  # noqa: F401
        import sumolib  # noqa: F401
        return
    except ImportError:
        pass

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        tools_path = str(Path(sumo_home) / "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)

    import traci  # noqa: F401
    import sumolib  # noqa: F401


def _sumo_binary() -> str:
    exe = shutil.which("sumo")
    if exe:
        return exe
    raise FileNotFoundError("sumo binary not found on PATH")


def default_branches(decision_at_s: float = 60.0) -> list[BranchSpec]:
    return [
        BranchSpec("BASELINE", None, "No turn restriction."),
        BranchSpec(
            "BLOCK_NATURAL_120",
            DecisionEvent(
                DecisionType.BLOCK_TURN,
                decision_at_s,
                120.0,
                "WJ",
                "JN",
                ResponsePolicy.NATURAL_REROUTE,
            ),
            "Block direct north for 120 s and let SUMO reroute dynamically.",
        ),
        BranchSpec(
            "BLOCK_GUIDED_120",
            DecisionEvent(
                DecisionType.BLOCK_TURN,
                decision_at_s,
                120.0,
                "WJ",
                "JN",
                ResponsePolicy.GUIDED_DETOUR,
            ),
            "Block direct north for 120 s and guide upstream traffic to the detour.",
        ),
        BranchSpec(
            "BLOCK_GUIDED_60",
            DecisionEvent(
                DecisionType.BLOCK_TURN,
                decision_at_s,
                60.0,
                "WJ",
                "JN",
                ResponsePolicy.GUIDED_DETOUR,
            ),
            "Shorter 60 s closure with guided diversion.",
        ),
    ]


def _write_routes(workdir: Path, grouped, duration_s: float = 300.0, headway_s: float = 1.5) -> Path:
    path = workdir / "decision_lab.rou.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        '  <route id="north_direct" edges="WJ JN NS"/>',
        '  <route id="north_detour" edges="WJ JE EN N2 NS"/>',
        '  <route id="east" edges="WJ JE EE"/>',
    ]

    classes = [1, 2, 6]
    i = 0
    depart = 0.0
    while depart <= duration_s:
        cls = classes[i % len(classes)]
        is_north = (i % 10) < 7
        prefix = "n" if is_north else "e"
        vid = f"{prefix}_{i:04d}"
        persona = deterministic_persona(vid, cls, grouped)
        route = "north_direct" if is_north else "east"
        lines.append(
            f'  <vehicle id="{vid}" type="{persona.type_id}" route="{route}" '
            f'depart="{depart:.2f}" departLane="best" departSpeed="max"/>'
        )
        depart += headway_s
        i += 1

    lines.append("</routes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _start_sumo(net: Path, routes: Path, types: Path, label: str):
    _ensure_sumo_tools()
    import traci

    cmd = [
        _sumo_binary(),
        "-n", str(net),
        "-r", str(routes),
        "-a", str(types),
        "--step-length", "0.2",
        "--lateral-resolution", "0.2",
        "--time-to-teleport", "-1",
        "--collision.action", "warn",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--seed", "42",
    ]
    traci.start(cmd, label=label)
    return traci.getConnection(label)


def _set_turn_block(conn, blocked: bool) -> None:
    """Logical closure marker.

    We deliberately do not change lane permissions at runtime. SUMO validates
    routes for loaded/departing vehicles against lane permissions, so making JN
    disallowed can invalidate still-pending northbound routes before TraCI has a
    chance to divert them. M1 enforces the closed movement by route replacement
    while vehicles are still on the upstream WJ edge.
    """
    return None


def _guide_northbound(conn) -> int:
    changed = 0
    for vid in list(conn.vehicle.getIDList()):
        if not str(vid).startswith("n_") or conn.vehicle.getRoadID(vid) != "WJ":
            continue
        try:
            current = tuple(conn.vehicle.getRoute(vid))
            target = ["WJ", "JE", "EN", "N2", "NS"]
            if list(current) != target:
                conn.vehicle.setRoute(vid, target)
                changed += 1
        except Exception:
            continue
    return changed


def _natural_reroute(conn) -> int:
    """Local/late response: divert only when a driver reaches the closure area."""
    changed = 0
    for vid in list(conn.vehicle.getIDList()):
        if not str(vid).startswith("n_") or conn.vehicle.getRoadID(vid) != "WJ":
            continue
        try:
            # WJ is about 220 m long. Natural response represents a driver
            # discovering the closure locally, ~70 m before the junction.
            if float(conn.vehicle.getLanePosition(vid)) < 150.0:
                continue
            current = tuple(conn.vehicle.getRoute(vid))
            target = ["WJ", "JE", "EN", "N2", "NS"]
            if list(current) != target:
                conn.vehicle.setRoute(vid, target)
                changed += 1
        except Exception:
            continue
    return changed


def _run_to_snapshot(net: Path, routes: Path, types: Path, workdir: Path, decision_at_s: float) -> Path:
    label = "streetlab_predecision"
    conn = _start_sumo(net, routes, types, label)
    snapshot = workdir / "decision_point_state.xml"
    try:
        while conn.simulation.getTime() < decision_at_s:
            conn.simulationStep()
        conn.simulation.saveState(str(snapshot))
    finally:
        conn.close()
    return snapshot


def _branch_run(net: Path, routes: Path, types: Path, snapshot: Path, spec: BranchSpec, horizon_s: float) -> dict:
    label = f"branch_{spec.name.lower()}"
    conn = _start_sumo(net, routes, types, label)
    conn.simulation.loadState(str(snapshot))

    decision_time = spec.decision.at_s if spec.decision is not None else float(conn.simulation.getTime())
    end_time = decision_time + horizon_s

    blocked = False
    rerouted_ids: set[str] = set()
    affected_ids: set[str] = set()
    arrived_ids: set[str] = set()
    speed_samples: list[float] = []
    occupancy_samples: list[int] = []
    queue_samples: list[int] = []
    waiting_samples: list[float] = []
    response_assignments: dict[str, ResponseMode] = {}
    heterogeneous_rerouted: dict[ResponseMode, set[str]] = {
        ResponseMode.GUIDED: set(),
        ResponseMode.LOCAL: set(),
    }
    response_assumptions: ResponseAssumptions | None = None

    try:
        while conn.simulation.getTime() < end_time:
            now = float(conn.simulation.getTime())
            decision = spec.decision
            should_block = bool(decision is not None and decision.active_at(now))
            if should_block != blocked:
                _set_turn_block(conn, should_block)
                blocked = should_block

            if blocked and decision is not None:
                for vid in list(conn.vehicle.getIDList()):
                    if str(vid).startswith("n_") and conn.vehicle.getRoadID(vid) == "WJ":
                        affected_ids.add(str(vid))

                before = {
                    str(v): tuple(conn.vehicle.getRoute(v))
                    for v in conn.vehicle.getIDList()
                    if str(v).startswith("n_")
                }
                if decision.response_policy == ResponsePolicy.GUIDED_DETOUR:
                    _guide_northbound(conn)
                elif decision.response_policy == ResponsePolicy.NATURAL_REROUTE:
                    _natural_reroute(conn)
                elif decision.response_policy == ResponsePolicy.HETEROGENEOUS_RESPONSE:
                    if response_assumptions is None:
                        response_assumptions = ResponseAssumptions.from_metadata(
                            decision.metadata
                        )
                    changed = apply_heterogeneous_response(
                        conn, response_assumptions, response_assignments
                    )
                    for vid in changed:
                        heterogeneous_rerouted[response_assignments[vid]].add(vid)

                active_after = set(str(v) for v in conn.vehicle.getIDList())
                for vid, old in before.items():
                    if vid in active_after and tuple(conn.vehicle.getRoute(vid)) != old:
                        rerouted_ids.add(vid)

            conn.simulationStep()
            arrived_ids.update(str(x) for x in conn.simulation.getArrivedIDList())

            active = list(conn.vehicle.getIDList())
            occupancy_samples.append(len(active))
            queue_samples.append(
                sum(
                    1
                    for vid in active
                    if conn.vehicle.getRoadID(vid) == "WJ" and conn.vehicle.getSpeed(vid) < 0.5
                )
            )
            speed_samples.extend(float(conn.vehicle.getSpeed(v)) for v in active)
            waiting_samples.extend(float(conn.vehicle.getWaitingTime(v)) for v in active)
    finally:
        conn.close()

    return {
        "name": spec.name,
        "description": spec.description,
        "decision": asdict(spec.decision) if spec.decision else None,
        "metrics": {
            "arrivals_after_decision": len(arrived_ids),
            "affected_upstream_northbound": len(affected_ids),
            "rerouted_vehicles": len(rerouted_ids),
            "mean_active_vehicles": mean(occupancy_samples) if occupancy_samples else 0.0,
            "max_approach_queue_vehicles": max(queue_samples, default=0),
            "mean_network_speed_mps": mean(speed_samples) if speed_samples else 0.0,
            "mean_instant_waiting_time_s": mean(waiting_samples) if waiting_samples else 0.0,
            "heterogeneous_guided_responders": sum(
                1 for mode in response_assignments.values()
                if mode == ResponseMode.GUIDED
            ),
            "heterogeneous_local_responders": sum(
                1 for mode in response_assignments.values()
                if mode == ResponseMode.LOCAL
            ),
            "heterogeneous_guided_rerouted": len(
                heterogeneous_rerouted[ResponseMode.GUIDED]
            ),
            "heterogeneous_local_rerouted": len(
                heterogeneous_rerouted[ResponseMode.LOCAL]
            ),
        },
        "response_assumptions": (
            response_assumptions.as_provenance()
            if response_assumptions is not None
            else None
        ),
    }


def _add_deltas(branches: list[dict]) -> None:
    baseline = next(x for x in branches if x["name"] == "BASELINE")["metrics"]
    for row in branches:
        m = row["metrics"]
        row["vs_baseline"] = {
            "arrivals_delta": m["arrivals_after_decision"] - baseline["arrivals_after_decision"],
            "max_queue_delta": m["max_approach_queue_vehicles"] - baseline["max_approach_queue_vehicles"],
            "mean_speed_delta_mps": m["mean_network_speed_mps"] - baseline["mean_network_speed_mps"],
            "mean_waiting_delta_s": m["mean_instant_waiting_time_s"] - baseline["mean_instant_waiting_time_s"],
        }


def run_decision_lab(
    project_root: str | Path = ".",
    workdir: str | Path = ".streetlab-m1",
    output: str | Path = "artifacts/phase2_m1_decision_lab.json",
    decision_at_s: float = 60.0,
    horizon_s: float = 180.0,
) -> dict:
    project_root = Path(project_root)
    workdir = Path(workdir)
    if not workdir.is_absolute():
        workdir = project_root / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    profiles = load_phase1_personas(project_root / "data" / "models" / "sumo_persona_priors.csv")
    grouped = profiles_by_class(profiles)

    net = write_demo_network(workdir)
    types = write_vtypes(workdir, profiles)
    routes = _write_routes(workdir, grouped)
    snapshot = _run_to_snapshot(net, routes, types, workdir, decision_at_s)

    branches = [
        _branch_run(net, routes, types, snapshot, spec, horizon_s)
        for spec in default_branches(decision_at_s)
    ]
    _add_deltas(branches)

    report = {
        "milestone": "PHASE2_M1_DECISION_IN_THE_LOOP",
        "objective": "Fork one running SUMO state and compare multiple responses to a BLOCK_TURN decision.",
        "phase1_reuse": {
            "persona_priors": "data/models/sumo_persona_priors.csv",
            "longitudinal_tau": 0.50,
            "vehicle_classes": ["MOTORCYCLE", "CAR", "AUTO_RICKSHAW"],
            "note": (
                "Persona speedFactor/minGapLat seeds provide prototype heterogeneity. "
                "Routing-response behavior is an explicit scenario assumption, not a Phase-1 empirical claim."
            ),
        },
        "branching": {
            "decision_time_s": decision_at_s,
            "horizon_s": horizon_s,
            "snapshot": str(snapshot),
            "same_predecision_state_for_all_branches": True,
        },
        "branches": branches,
    }

    output = Path(output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
