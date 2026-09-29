from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

from streetlab_phase1.sumo_stage_a import (
    StageAConfig,
    load_clean,
    load_persona_priors,
    load_behavior_dna,
    assign_personas,
    make_vehicle_speed_factors,
    derive_class_stats,
    evaluate_speed_fit,
)


ENTRY_LINE_M = 55.0
OBS_END_M = 245.0
INTERIOR_BUFFER_M = 10.0
PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}

CASES = [
    {
        "name": "O0_REALISTIC_SPEEDS",
        "multipliers": {1: 0.90, 2: 0.90, 6: 0.90},
    },
    {
        "name": "O1_PREVIOUS_FEASIBLE_REFERENCE",
        "multipliers": {1: 0.90, 2: 1.00, 6: 0.96},
    },
]


def _crossing_state(g: pd.DataFrame, line_m: float) -> tuple[float, float] | None:
    g = g.sort_values("time_s")
    x = g["x_m"].to_numpy(float)
    t = g["time_s"].to_numpy(float)
    v = g["speed_mps"].to_numpy(float)

    # All vehicles should begin upstream of 55 m in this calibration set, but
    # retain a safe exact-point path.
    exact = np.where(np.isclose(x, line_m, atol=1e-9))[0]
    if exact.size:
        i = int(exact[0])
        return float(t[i]), float(v[i])

    idx = np.where((x[:-1] < line_m) & (x[1:] > line_m))[0]
    if idx.size == 0:
        return None

    i = int(idx[0])
    dx = x[i + 1] - x[i]
    if abs(dx) < 1e-9:
        return None
    frac = (line_m - x[i]) / dx
    ct = t[i] + frac * (t[i + 1] - t[i])
    cv = v[i] + frac * (v[i + 1] - v[i])
    return float(ct), float(cv)


def build_observed_clock_demand(clean: pd.DataFrame, priors: pd.DataFrame, dna: pd.DataFrame | None, warmup_s: float) -> pd.DataFrame:
    rows = []
    for vid, g in clean.groupby("vehicle_id", sort=False):
        state = _crossing_state(g, ENTRY_LINE_M)
        if state is None:
            continue
        ct, cv = state
        rows.append({
            "vehicle_id": vid,
            "vehicle_class": int(g["vehicle_class"].iloc[0]),
            "crossing_time_s": ct,
            "crossing_speed_mps": cv,
        })

    d = pd.DataFrame(rows).sort_values(["crossing_time_s", "vehicle_id"]).reset_index(drop=True)
    if d.empty:
        raise RuntimeError("No vehicles crossed the observed entry line.")

    t0 = float(d["crossing_time_s"].min())
    d["depart_s"] = d["crossing_time_s"] - t0 + warmup_s
    d = assign_personas(d, priors, dna)
    return d


def observed_speed_targets_after_entry(clean: pd.DataFrame) -> pd.DataFrame:
    lo = ENTRY_LINE_M + INTERIOR_BUFFER_M
    hi = OBS_END_M - INTERIOR_BUFFER_M
    x = clean[
        (clean["x_m"] >= lo)
        & (clean["x_m"] <= hi)
        & (clean["vehicle_class"].isin(PRIMARY_CLASSES))
    ].copy()
    return (
        x.groupby(["vehicle_id", "vehicle_class"], as_index=False)
        .agg(mean_speed_mps=("speed_mps", "mean"))
    )


def _write_network(workdir: Path, cfg: StageAConfig, netconvert_binary: str) -> Path:
    length = OBS_END_M - ENTRY_LINE_M
    nodes = workdir / "observed_clock.nod.xml"
    edges = workdir / "observed_clock.edg.xml"
    net = workdir / "observed_clock.net.xml"

    nodes.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<nodes>\n'
        '  <node id="n0" x="0" y="0" type="priority"/>\n'
        f'  <node id="n1" x="{length:.3f}" y="0" type="priority"/>\n'
        '</nodes>\n',
        encoding="utf-8",
    )
    edges.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<edges>\n'
        f'  <edge id="E0" from="n0" to="n1" numLanes="{cfg.lanes}" '
        f'speed="{cfg.edge_speed_mps:.4f}" width="{cfg.lane_width_m:.3f}"/>\n'
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
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise RuntimeError(f"netconvert failed:\n{p.stdout}\n{p.stderr}")
    return net


def _vclass_for(cls: int) -> str:
    return {1: "motorcycle", 2: "passenger", 6: "passenger"}.get(int(cls), "passenger")


def _write_types(workdir: Path, cfg: StageAConfig, demand: pd.DataFrame, clean: pd.DataFrame) -> Path:
    stats = derive_class_stats(clean).set_index("vehicle_class")
    pairs = sorted(set((int(r.vehicle_class), str(r.persona)) for r in demand.itertuples()))
    path = workdir / "observed_clock_types.add.xml"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional>',
        '  <!-- A5: observed-crossing demand clock. tau remains provisional at 0.72 s. -->',
    ]
    for cls, persona in pairs:
        if cls in stats.index:
            length = float(stats.loc[cls, "length_m"])
            width = float(stats.loc[cls, "width_m"])
        else:
            length, width = 4.5, 1.8
        lines.append(
            f'  <vType id="sl_c{cls}_{persona.lower()}" vClass="{_vclass_for(cls)}" '
            f'length="{length:.4f}" width="{width:.4f}" '
            f'carFollowModel="Krauss" laneChangeModel="SL2015" '
            f'tau="{cfg.default_tau_s:.4f}" minGap="{cfg.default_min_gap_m:.4f}" '
            f'minGapLat="{cfg.default_min_gap_lat_m:.4f}" '
            f'lcPushy="{cfg.default_lc_pushy:.4f}" lcSublane="{cfg.default_lc_sublane:.4f}" '
            f'speedFactor="1.0" speedDev="0.0"/>'
        )
    lines.append("</additional>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_routes(workdir: Path, demand: pd.DataFrame, speed_factors: dict[str, float]) -> Path:
    path = workdir / "observed_clock.rou.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '  <route id="corridor" edges="E0"/>',
    ]
    for r in demand.itertuples():
        vid = str(r.vehicle_id)
        cls = int(r.vehicle_class)
        persona = str(r.persona)
        sf = float(speed_factors.get(vid, 1.0))
        lines.append(
            f'  <vehicle id="{vid}" type="sl_c{cls}_{persona.lower()}" route="corridor" '
            f'depart="{float(r.depart_s):.6f}" departLane="best_prob" '
            f'departPos="base" departSpeed="avg" speedFactor="{sf:.6f}"/>'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_case(project_root: Path, ctx: dict, cfg: StageAConfig, case: dict) -> dict:
    import sumolib
    import traci

    outroot = project_root / "data" / "simulation" / "observed_clock"
    rundir = outroot / case["name"]
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    sumo_binary = sumolib.checkBinary("sumo")
    netconvert_binary = sumolib.checkBinary("netconvert")
    net = _write_network(rundir, cfg, netconvert_binary)
    types = _write_types(rundir, cfg, ctx["demand"], ctx["clean"])

    speed_factors = make_vehicle_speed_factors(
        ctx["demand"], ctx["clean"], ctx["priors"], ctx["dna"],
        {int(k): float(v) for k, v in case["multipliers"].items()},
        cfg,
    )
    routes = _write_routes(rundir, ctx["demand"], speed_factors)

    last_depart = float(ctx["demand"]["depart_s"].max())
    end_s = last_depart + 90.0

    cmd = [
        sumo_binary,
        "-n", str(net),
        "-r", str(routes),
        "-a", str(types),
        "--step-length", str(cfg.step_length_s),
        "--lateral-resolution", str(cfg.lateral_resolution_m),
        "--time-to-teleport", "-1",
        "--max-depart-delay", "120",
        "--collision.action", "warn",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--extrapolate-departpos", "true",
        "--seed", "42",
    ]

    scheduled = dict(zip(ctx["demand"]["vehicle_id"].astype(str), ctx["demand"]["depart_s"].astype(float)))
    class_map = dict(zip(ctx["demand"]["vehicle_id"].astype(str), ctx["demand"]["vehicle_class"].astype(int)))
    departed_at = {}
    rows = []
    teleports = 0
    collisions = 0

    traci.start(cmd)
    try:
        while traci.simulation.getTime() <= end_s:
            traci.simulationStep()
            now = float(traci.simulation.getTime())

            for vid in traci.simulation.getDepartedIDList():
                departed_at[str(vid)] = now

            try:
                teleports += len(traci.simulation.getStartingTeleportIDList())
            except Exception:
                pass
            try:
                collisions += len(traci.simulation.getCollidingVehiclesIDList())
            except Exception:
                pass

            for vid in traci.vehicle.getIDList():
                if traci.vehicle.getRoadID(vid) != "E0":
                    continue
                pos = float(traci.vehicle.getLanePosition(vid))
                # Network coordinate 0 corresponds to observed x=55 m.
                if pos < INTERIOR_BUFFER_M:
                    continue
                if pos > (OBS_END_M - ENTRY_LINE_M - INTERIOR_BUFFER_M):
                    continue
                rows.append({
                    "time_s": now,
                    "vehicle_id": str(vid),
                    "vehicle_class": class_map.get(str(vid)),
                    "speed_mps": float(traci.vehicle.getSpeed(vid)),
                    "lane_position_m": pos,
                })
    finally:
        traci.close(False)

    sim = pd.DataFrame(rows)
    fit, objective = evaluate_speed_fit(ctx["observed"], sim)

    delays = [
        float(actual - scheduled[vid])
        for vid, actual in departed_at.items()
        if vid in scheduled
    ]
    simulation = {
        "scheduled_vehicles": int(len(scheduled)),
        "departed_vehicles": int(len(departed_at)),
        "departed_fraction": float(len(departed_at) / len(scheduled)) if scheduled else 0.0,
        "mean_depart_delay_s": float(np.mean(delays)) if delays else None,
        "p95_depart_delay_s": float(np.quantile(delays, 0.95)) if delays else None,
        "teleport_events": int(teleports),
        "collision_events": int(collisions),
        "trajectory_rows": int(len(sim)),
    }
    strict_gate = bool(
        simulation["departed_fraction"] >= 0.99
        and simulation["p95_depart_delay_s"] is not None
        and simulation["p95_depart_delay_s"] <= 5.0
        and simulation["teleport_events"] == 0
        and simulation["collision_events"] == 0
    )

    sim.to_parquet(rundir / "simulation_frames.parquet", index=False)
    fit.to_csv(rundir / "speed_fit.csv", index=False)

    return {
        "case": case["name"],
        "class_multipliers": case["multipliers"],
        "objective": float(objective),
        "strict_gate_pass": strict_gate,
        "simulation": simulation,
        "speed_fit": fit.to_dict("records"),
    }


def run_observed_clock_experiment(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    models = project_root / "data" / "models"

    clean, original_cols = load_clean(processed / "calibration_clean.parquet")
    priors = load_persona_priors(models / "sumo_persona_priors.csv")
    dna = load_behavior_dna(processed / "calibration_behavior_dna.parquet")

    cfg = StageAConfig()
    cfg.default_tau_s = 0.72
    cfg.feeder_length_m = 0.0
    cfg.default_min_gap_lat_m = 0.60
    cfg.default_lc_pushy = 0.0

    demand = build_observed_clock_demand(clean, priors, dna, cfg.warmup_s)
    observed = observed_speed_targets_after_entry(clean)

    ctx = {
        "clean": clean,
        "priors": priors,
        "dna": dna,
        "demand": demand,
        "observed": observed,
    }

    results = []
    for case in CASES:
        print(f"Running {case['name']} with observed 55 m crossing clock...")
        r = _run_case(project_root, ctx, cfg, case)
        results.append(r)
        print(
            f"  objective={r['objective']:.4f}, "
            f"departed={r['simulation']['departed_fraction']:.3f}, "
            f"p95_delay={r['simulation']['p95_depart_delay_s']}, "
            f"collisions={r['simulation']['collision_events']}, "
            f"strict_gate={r['strict_gate_pass']}"
        )

    demand.to_csv(
        project_root / "data" / "simulation" / "observed_clock" / "observed_55m_demand.csv",
        index=False,
    )

    report = {
        "sprint": "1E_A5_OBSERVED_CROSSING_CLOCK",
        "entry_line_m": ENTRY_LINE_M,
        "why_55m": (
            "A4 found max first observed x = 54.0098 m and all vehicles move downstream, "
            "so x=55 m gives an observed interpolated crossing time for every calibration vehicle "
            "without backward extrapolation or dropping left-censored vehicles."
        ),
        "demand_clock": {
            "vehicles": int(len(demand)),
            "first_depart_s": float(demand["depart_s"].min()),
            "last_depart_s": float(demand["depart_s"].max()),
            "all_crossings_observed": bool(len(demand) == clean["vehicle_id"].nunique()),
        },
        "measurement_window_observed_x_m": [
            ENTRY_LINE_M + INTERIOR_BUFFER_M,
            OBS_END_M - INTERIOR_BUFFER_M,
        ],
        "fixed_physics": {
            "tau_s": 0.72,
            "min_gap_m": cfg.default_min_gap_m,
            "min_gap_lat_m": cfg.default_min_gap_lat_m,
            "lc_pushy": cfg.default_lc_pushy,
            "meaning": "Still diagnostic; not final calibrated spacing/lateral behavior.",
        },
        "strict_gate": {
            "departed_fraction_min": 0.99,
            "p95_depart_delay_s_max": 5.0,
            "teleports_required": 0,
            "collisions_required": 0,
        },
        "results": results,
        "decision": (
            "If O0 passes, the previous speed/capacity conflict was largely a demand-clock artifact. "
            "If O0 fails while O1 passes, realistic speeds still expose a longitudinal-capacity mismatch. "
            "If both fail, revisit minGap/longitudinal physics before any more speed tuning."
        ),
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a5_observed_clock_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
