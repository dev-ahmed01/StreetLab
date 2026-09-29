from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import shutil
import subprocess
import numpy as np
import pandas as pd


@dataclass
class CapacityCase:
    name: str
    feeder_length_m: float
    tau_s: float
    min_gap_m: float
    min_gap_lat_m: float
    lc_pushy: float
    depart_lane: str
    depart_pos: str
    depart_speed: str
    extrapolate_departpos: bool


CASES = [
    CapacityCase(
        name="C0_CURRENT_STYLE",
        feeder_length_m=80.0,
        tau_s=1.20,
        min_gap_m=2.50,
        min_gap_lat_m=0.60,
        lc_pushy=0.00,
        depart_lane="best_prob",
        depart_pos="last",
        depart_speed="desired",
        extrapolate_departpos=False,
    ),
    CapacityCase(
        name="C1_INSERTION_OPTIMIZED_NEUTRAL",
        feeder_length_m=400.0,
        tau_s=1.20,
        min_gap_m=2.50,
        min_gap_lat_m=0.60,
        lc_pushy=0.00,
        depart_lane="best_prob",
        depart_pos="last",
        depart_speed="avg",
        extrapolate_departpos=True,
    ),
    CapacityCase(
        name="C2_TAU_0P90",
        feeder_length_m=400.0,
        tau_s=0.90,
        min_gap_m=2.50,
        min_gap_lat_m=0.60,
        lc_pushy=0.00,
        depart_lane="best_prob",
        depart_pos="last",
        depart_speed="avg",
        extrapolate_departpos=True,
    ),
    CapacityCase(
        name="C3_TAU_0P72",
        feeder_length_m=400.0,
        tau_s=0.72,
        min_gap_m=2.50,
        min_gap_lat_m=0.60,
        lc_pushy=0.00,
        depart_lane="best_prob",
        depart_pos="last",
        depart_speed="avg",
        extrapolate_departpos=True,
    ),
    CapacityCase(
        name="C4_TAU_0P72_LATERAL_DIAGNOSTIC",
        feeder_length_m=400.0,
        tau_s=0.72,
        min_gap_m=2.50,
        min_gap_lat_m=0.40,
        lc_pushy=0.20,
        depart_lane="best_prob",
        depart_pos="last",
        depart_speed="avg",
        extrapolate_departpos=True,
    ),
]


def _type_id(cls: int, persona: str) -> str:
    return f"sl_c{int(cls)}_{str(persona).lower()}"


def _vclass_for(cls: int) -> str:
    return {1: "motorcycle", 2: "passenger", 6: "passenger"}.get(int(cls), "passenger")


def _build_network(workdir: Path, case: CapacityCase, stage_cfg, netconvert_binary: str) -> Path:
    nodes = workdir / "capacity.nod.xml"
    edges = workdir / "capacity.edg.xml"
    net = workdir / "capacity.net.xml"

    nodes.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<nodes>\n'
        f'  <node id="npre" x="{-case.feeder_length_m:.3f}" y="0" type="priority"/>\n'
        '  <node id="n0" x="0" y="0" type="priority"/>\n'
        f'  <node id="n1" x="{stage_cfg.corridor_length_m:.3f}" y="0" type="priority"/>\n'
        '</nodes>\n',
        encoding="utf-8",
    )
    edges.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<edges>\n'
        f'  <edge id="E_pre" from="npre" to="n0" numLanes="{stage_cfg.lanes}" '
        f'speed="{stage_cfg.edge_speed_mps:.4f}" width="{stage_cfg.lane_width_m:.3f}"/>\n'
        f'  <edge id="E0" from="n0" to="n1" numLanes="{stage_cfg.lanes}" '
        f'speed="{stage_cfg.edge_speed_mps:.4f}" width="{stage_cfg.lane_width_m:.3f}"/>\n'
        '</edges>\n',
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            netconvert_binary,
            "--node-files", str(nodes),
            "--edge-files", str(edges),
            "--output-file", str(net),
            "--no-turnarounds", "true",
        ],
        capture_output=True, text=True,
    )
    if p.returncode:
        raise RuntimeError(f"netconvert failed for {case.name}\n{p.stdout}\n{p.stderr}")
    return net


def _write_types(workdir: Path, case: CapacityCase, ctx) -> Path:
    stats = (
        ctx["clean"]
        .groupby("vehicle_class", as_index=False)
        .agg(length_m=("length_m", "median"), width_m=("width_m", "median"))
        .set_index("vehicle_class")
    )
    pairs = sorted(set((int(r.vehicle_class), str(r.persona)) for r in ctx["demand"].itertuples()))
    path = workdir / "capacity_types.add.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional>',
        f'  <!-- Diagnostic case {case.name}; not final calibrated behavior. -->',
    ]
    for cls, persona in pairs:
        if cls in stats.index:
            length = float(stats.loc[cls, "length_m"])
            width = float(stats.loc[cls, "width_m"])
        else:
            length, width = 4.5, 1.8
        lines.append(
            f'  <vType id="{_type_id(cls, persona)}" vClass="{_vclass_for(cls)}" '
            f'length="{length:.4f}" width="{width:.4f}" '
            f'carFollowModel="Krauss" laneChangeModel="SL2015" '
            f'tau="{case.tau_s:.4f}" minGap="{case.min_gap_m:.4f}" '
            f'minGapLat="{case.min_gap_lat_m:.4f}" lcPushy="{case.lc_pushy:.4f}" '
            f'lcSublane="1.0" speedFactor="1.0" speedDev="0.0"/>'
        )
    lines.append("</additional>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_routes(workdir: Path, case: CapacityCase, ctx, speed_factors: dict[str, float]) -> Path:
    path = workdir / "capacity.rou.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '  <route id="corridor" edges="E_pre E0"/>',
    ]
    for r in ctx["demand"].itertuples():
        vid = str(r.vehicle_id)
        sf = float(speed_factors.get(vid, 1.0))
        lines.append(
            f'  <vehicle id="{vid}" type="{_type_id(r.vehicle_class, r.persona)}" '
            f'route="corridor" depart="{float(r.depart_s):.3f}" '
            f'departLane="{case.depart_lane}" departPos="{case.depart_pos}" '
            f'departSpeed="{case.depart_speed}" speedFactor="{sf:.6f}"/>'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_case(project_root: Path, ctx, stage_cfg, case: CapacityCase) -> dict:
    import sumolib
    import traci
    from streetlab_phase1.sumo_stage_a import make_vehicle_speed_factors

    simroot = project_root / "data" / "simulation" / "capacity_diagnostic"
    rundir = simroot / case.name
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    sumo_binary = sumolib.checkBinary("sumo")
    netconvert_binary = sumolib.checkBinary("netconvert")
    net = _build_network(rundir, case, stage_cfg, netconvert_binary)
    types = _write_types(rundir, case, ctx)
    speed_factors = make_vehicle_speed_factors(
        ctx["demand"], ctx["clean"], ctx["priors"], ctx["dna"],
        {1: 1.0, 2: 1.0, 6: 1.0}, stage_cfg
    )
    routes = _write_routes(rundir, case, ctx, speed_factors)

    last_depart = float(ctx["demand"]["depart_s"].max())
    first_depart = float(ctx["demand"]["depart_s"].min())
    # Give a longer tail so we can distinguish discard-at-120s from a true
    # network throughput problem. max-depart-delay remains 120s.
    end_s = last_depart + 180.0

    cmd = [
        sumo_binary,
        "-n", str(net),
        "-r", str(routes),
        "-a", str(types),
        "--step-length", str(stage_cfg.step_length_s),
        "--lateral-resolution", str(stage_cfg.lateral_resolution_m),
        "--time-to-teleport", "-1",
        "--max-depart-delay", "120",
        "--collision.action", "warn",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--seed", "42",
    ]
    if case.extrapolate_departpos:
        cmd += ["--extrapolate-departpos", "true"]

    scheduled = dict(zip(ctx["demand"]["vehicle_id"].astype(str), ctx["demand"]["depart_s"].astype(float)))
    departed_at = {}
    e0_entered = set()
    e0_entry_times = []
    teleports = 0
    collisions = 0

    traci.start(cmd)
    try:
        while traci.simulation.getTime() <= end_s:
            traci.simulationStep()
            now = float(traci.simulation.getTime())
            for vid in traci.simulation.getDepartedIDList():
                departed_at[str(vid)] = now
            for vid in traci.vehicle.getIDList():
                svid = str(vid)
                if svid not in e0_entered and traci.vehicle.getRoadID(vid) == "E0":
                    e0_entered.add(svid)
                    e0_entry_times.append(now)
            try:
                teleports += len(traci.simulation.getStartingTeleportIDList())
            except Exception:
                pass
            try:
                collisions += len(traci.simulation.getCollidingVehiclesIDList())
            except Exception:
                pass
    finally:
        traci.close(False)

    delays = [
        float(actual - scheduled[vid])
        for vid, actual in departed_at.items()
        if vid in scheduled
    ]
    obs_duration = max(last_depart - first_depart, 1.0)
    scheduled_rate = len(scheduled) / obs_duration * 3600.0
    departed_rate = len(departed_at) / obs_duration * 3600.0
    e0_rate = len(e0_entered) / obs_duration * 3600.0

    result = {
        "case": case.name,
        "case_config": asdict(case),
        "scheduled_vehicles": int(len(scheduled)),
        "departed_vehicles": int(len(departed_at)),
        "departed_fraction": float(len(departed_at) / len(scheduled)),
        "e0_entered_vehicles": int(len(e0_entered)),
        "e0_entered_fraction": float(len(e0_entered) / len(scheduled)),
        "mean_depart_delay_s": float(np.mean(delays)) if delays else None,
        "p95_depart_delay_s": float(np.quantile(delays, 0.95)) if delays else None,
        "max_depart_delay_observed_s": float(np.max(delays)) if delays else None,
        "teleport_events": int(teleports),
        "collision_events": int(collisions),
        "scheduled_rate_veh_per_h": float(scheduled_rate),
        "departed_rate_veh_per_h": float(departed_rate),
        "e0_entry_rate_veh_per_h": float(e0_rate),
    }
    (rundir / "capacity_case_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_capacity_diagnostic(project_root: str | Path = ".") -> dict:
    from streetlab_phase1.sumo_stage_a import StageAConfig, prepare_corridor

    project_root = Path(project_root)
    stage_cfg = StageAConfig()
    ctx = prepare_corridor(project_root, stage_cfg)

    results = []
    for case in CASES:
        print(f"Running {case.name}...")
        r = _run_case(project_root, ctx, stage_cfg, case)
        results.append(r)
        print(
            f"  departed={r['departed_fraction']:.3f}, "
            f"p95_delay={r['p95_depart_delay_s']}, "
            f"E0={r['e0_entered_fraction']:.3f}"
        )

    report = {
        "sprint": "1E_A0_CAPACITY_DIAGNOSTIC",
        "purpose": (
            "Determine whether Stage-A failure is caused mainly by insertion mechanics "
            "or by insufficient mixed-traffic carrying capacity under neutral SUMO physics."
        ),
        "calibration_status": "NO_PARAMETER_SET_IN_THIS REPORT IS FINAL",
        "observed_demand_note": (
            "Demand schedule is the calibration-period first-appearance schedule after "
            "removing initial stock. Rates below are diagnostic, not a claim of official roadway capacity."
        ),
        "cases": results,
        "interpretation_rule": [
            "If C1 fixes delay, insertion mechanics were the main problem.",
            "If C1 fails but C2/C3 fix delay, longitudinal following capacity is the main problem.",
            "If only C4 fixes delay, lateral mixed-traffic capacity is essential before speed calibration.",
            "If C4 also fails, stop and revisit geometry/demand reconstruction rather than tuning speed.",
        ],
        "important": (
            "Do not run Stage-A speed calibration until a capacity case carries the demand "
            "without large departure delay and without teleports/collisions."
        ),
    }
    out = project_root / "data" / "processed" / "sprint1e_a0_capacity_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
