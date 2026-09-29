from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
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


@dataclass
class Case:
    name: str
    step_length_s: float
    eager_insert: bool
    lane_mode: str          # "best_prob" or "observed_band"
    speed_mode: str         # "avg" or "observed"


CASES = [
    Case("R0_REPRO_0P5_BEST_AVG", 0.50, False, "best_prob", "avg"),
    Case("R1_0P1_BEST_AVG",       0.10, False, "best_prob", "avg"),
    Case("R2_0P1_EAGER_BEST_AVG", 0.10, True,  "best_prob", "avg"),
    Case("R3_0P1_EAGER_OBS_LANE_AVG", 0.10, True, "observed_band", "avg"),
    Case("R4_0P1_EAGER_OBS_LANE_OBS_SPEED", 0.10, True, "observed_band", "observed"),
    Case("R5_0P05_EAGER_OBS_LANE_OBS_SPEED", 0.05, True, "observed_band", "observed"),
]


def _interp_crossing_state(g: pd.DataFrame, line_m: float) -> dict | None:
    g = g.sort_values("time_s")
    x = g["x_m"].to_numpy(float)
    t = g["time_s"].to_numpy(float)
    v = g["speed_mps"].to_numpy(float)

    # Recover lateral position directly from the canonical clean parquet.
    # load_clean() does not include lat_pos, so use the original group if present.
    ycol = "lat_pos" if "lat_pos" in g.columns else None
    y = g[ycol].to_numpy(float) if ycol else None

    exact = np.where(np.isclose(x, line_m, atol=1e-9))[0]
    if exact.size:
        i = int(exact[0])
        return {
            "crossing_time_s": float(t[i]),
            "crossing_speed_mps": float(v[i]),
            "crossing_lat_m": float(y[i]) if y is not None else np.nan,
        }

    idx = np.where((x[:-1] < line_m) & (x[1:] > line_m))[0]
    if idx.size == 0:
        return None

    i = int(idx[0])
    dx = x[i + 1] - x[i]
    if abs(dx) < 1e-12:
        return None
    frac = (line_m - x[i]) / dx

    return {
        "crossing_time_s": float(t[i] + frac * (t[i + 1] - t[i])),
        "crossing_speed_mps": float(v[i] + frac * (v[i + 1] - v[i])),
        "crossing_lat_m": (
            float(y[i] + frac * (y[i + 1] - y[i])) if y is not None else np.nan
        ),
    }


def _canonical_clean_with_lat(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path).copy()
    required = [
        "timestamp", "vehicle_id", "vehicle_class", "length", "width",
        "long_pos", "long_speed", "lat_pos"
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing canonical columns {missing}; available={list(raw.columns)}")

    d = pd.DataFrame({
        "vehicle_id": raw["vehicle_id"],
        "vehicle_class": pd.to_numeric(raw["vehicle_class"], errors="coerce"),
        "time_s": pd.to_numeric(raw["timestamp"], errors="coerce"),
        "x_m": pd.to_numeric(raw["long_pos"], errors="coerce"),
        "speed_mps": pd.to_numeric(raw["long_speed"], errors="coerce"),
        "lat_pos": pd.to_numeric(raw["lat_pos"], errors="coerce"),
        "length_m": pd.to_numeric(raw["length"], errors="coerce"),
        "width_m": pd.to_numeric(raw["width"], errors="coerce"),
    }).dropna(subset=["vehicle_id", "vehicle_class", "time_s", "x_m", "speed_mps", "lat_pos"])
    d["vehicle_class"] = d["vehicle_class"].astype(int)
    return d.sort_values(["vehicle_id", "time_s"]).reset_index(drop=True)


def build_boundary_demand(
    clean: pd.DataFrame,
    priors: pd.DataFrame,
    dna: pd.DataFrame | None,
    cfg: StageAConfig,
) -> pd.DataFrame:
    rows = []
    for vid, g in clean.groupby("vehicle_id", sort=False):
        state = _interp_crossing_state(g, ENTRY_LINE_M)
        if state is None:
            continue
        rows.append({
            "vehicle_id": vid,
            "vehicle_class": int(g["vehicle_class"].iloc[0]),
            **state,
        })

    d = pd.DataFrame(rows).sort_values(["crossing_time_s", "vehicle_id"]).reset_index(drop=True)
    if d.empty:
        raise RuntimeError("No observed x=55 m crossings were reconstructed.")

    t0 = float(d["crossing_time_s"].min())
    d["depart_s"] = d["crossing_time_s"] - t0 + 60.0

    # Diagnostic three-band mapping. Reversing band order would not change
    # capacity on this straight symmetric road, so unknown real-world left/right
    # orientation is not material for this test.
    y = d["crossing_lat_m"].astype(float)
    lane = np.floor(y / cfg.lane_width_m).astype(int)
    d["observed_lane_band"] = np.clip(lane, 0, cfg.lanes - 1)

    d = assign_personas(d, priors, dna)
    return d


def observed_speed_targets(clean: pd.DataFrame) -> pd.DataFrame:
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
    nodes = workdir / "boundary.nod.xml"
    edges = workdir / "boundary.edg.xml"
    net = workdir / "boundary.net.xml"

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
    path = workdir / "boundary_types.add.xml"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional>',
        '  <!-- A6 diagnostic: behavior parameters remain provisional/fixed. -->',
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


def _write_routes(
    workdir: Path,
    case: Case,
    demand: pd.DataFrame,
    speed_factors: dict[str, float],
    cfg: StageAConfig,
) -> tuple[Path, dict]:
    path = workdir / "boundary.rou.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '  <route id="corridor" edges="E0"/>',
    ]

    lifted = 0
    max_lift = 0.0
    max_observed_speed = 0.0
    over_edge_speed = 0

    for r in demand.itertuples():
        vid = str(r.vehicle_id)
        cls = int(r.vehicle_class)
        persona = str(r.persona)
        desired_sf = float(speed_factors.get(vid, 1.0))

        lane = (
            str(int(r.observed_lane_band))
            if case.lane_mode == "observed_band"
            else "best_prob"
        )

        xml_sf = desired_sf
        if case.speed_mode == "observed":
            observed_speed = max(float(r.crossing_speed_mps), 0.01)
            max_observed_speed = max(max_observed_speed, observed_speed)
            if observed_speed > cfg.edge_speed_mps:
                over_edge_speed += 1

            # Numeric departSpeed is checked against the vehicle's allowed
            # speed on the departure edge, which is affected by speedFactor.
            # Boundary speed is an observed state; downstream speedFactor is
            # the behavior parameter under test. Keep them separate.
            required_sf = observed_speed / max(cfg.edge_speed_mps, 1e-9)
            xml_sf = max(desired_sf, required_sf * 1.01)

            if xml_sf > desired_sf + 1e-12:
                lifted += 1
                max_lift = max(max_lift, xml_sf - desired_sf)

            depart_speed = f"{observed_speed:.6f}"
        else:
            depart_speed = "avg"

        lines.append(
            f'  <vehicle id="{vid}" type="sl_c{cls}_{persona.lower()}" route="corridor" '
            f'depart="{float(r.depart_s):.6f}" departLane="{lane}" '
            f'departPos="base" departSpeed="{depart_speed}" speedFactor="{xml_sf:.6f}"/>'
        )

    lines.append("</routes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return path, {
        "speed_mode": case.speed_mode,
        "vehicles_with_temporary_speedfactor_lift": int(lifted),
        "max_temporary_speedfactor_lift": float(max_lift),
        "max_observed_crossing_speed_mps": float(max_observed_speed),
        "observed_crossing_speeds_above_edge_speed": int(over_edge_speed),
        "edge_speed_mps": float(cfg.edge_speed_mps),
        "method": (
            "Observed-speed cases temporarily lift route-XML speedFactor only "
            "enough to permit the measured departSpeed. The intended downstream "
            "speedFactor is restored immediately after insertion through TraCI."
        ),
    }

def _discretization_report(demand: pd.DataFrame, step: float) -> dict:
    t = demand["depart_s"].to_numpy(float)
    # insertion attempts happen on simulation ticks
    bucket = np.ceil((t - 1e-12) / step).astype(int)
    _, counts = np.unique(bucket, return_counts=True)
    return {
        "step_length_s": float(step),
        "scheduled_vehicles": int(len(t)),
        "nonempty_departure_ticks": int(len(counts)),
        "max_vehicles_due_same_tick": int(np.max(counts)),
        "p95_vehicles_due_same_nonempty_tick": float(np.quantile(counts, 0.95)),
        "fraction_nonempty_ticks_with_ge_2": float((counts >= 2).mean()),
        "fraction_nonempty_ticks_with_ge_3": float((counts >= 3).mean()),
        "fraction_nonempty_ticks_with_ge_4": float((counts >= 4).mean()),
    }


def _run_case(
    project_root: Path,
    clean: pd.DataFrame,
    priors: pd.DataFrame,
    dna: pd.DataFrame | None,
    demand: pd.DataFrame,
    observed: pd.DataFrame,
    case: Case,
) -> dict:
    import sumolib
    import traci

    cfg = StageAConfig()
    cfg.default_tau_s = 0.72
    cfg.default_min_gap_m = 2.5
    cfg.default_min_gap_lat_m = 0.6
    cfg.default_lc_pushy = 0.0
    cfg.step_length_s = case.step_length_s

    outroot = project_root / "data" / "simulation" / "boundary_fidelity"
    rundir = outroot / case.name
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    sumo_binary = sumolib.checkBinary("sumo")
    netconvert_binary = sumolib.checkBinary("netconvert")
    net = _write_network(rundir, cfg, netconvert_binary)
    types = _write_types(rundir, cfg, demand, clean)

    multipliers = {1: 0.90, 2: 0.90, 6: 0.90}
    speed_factors = make_vehicle_speed_factors(
        demand, clean, priors, dna, multipliers, cfg
    )
    routes, boundary_speed_report = _write_routes(
        rundir, case, demand, speed_factors, cfg
    )

    last_depart = float(demand["depart_s"].max())
    end_s = last_depart + 90.0

    cmd = [
        sumo_binary,
        "-n", str(net),
        "-r", str(routes),
        "-a", str(types),
        "--step-length", str(case.step_length_s),
        "--lateral-resolution", str(cfg.lateral_resolution_m),
        "--time-to-teleport", "-1",
        "--max-depart-delay", "120",
        "--collision.action", "warn",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--extrapolate-departpos", "true",
        "--seed", "42",
    ]
    if case.eager_insert:
        cmd += ["--eager-insert", "true"]

    scheduled = dict(zip(demand["vehicle_id"].astype(str), demand["depart_s"].astype(float)))
    class_map = dict(zip(demand["vehicle_id"].astype(str), demand["vehicle_class"].astype(int)))
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
                svid = str(vid)
                departed_at[svid] = now
                if case.speed_mode == "observed" and svid in speed_factors:
                    traci.vehicle.setSpeedFactor(svid, float(speed_factors[svid]))

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
        try:
            traci.close(False)
        except Exception:
            pass

    sim = pd.DataFrame(rows)
    fit, objective = evaluate_speed_fit(observed, sim)

    delays = np.asarray([
        float(actual - scheduled[vid])
        for vid, actual in departed_at.items()
        if vid in scheduled
    ], dtype=float)

    def frac_le(x: float) -> float | None:
        return float((delays <= x + 1e-9).mean()) if delays.size else None

    simulation = {
        "scheduled_vehicles": int(len(scheduled)),
        "departed_vehicles": int(len(departed_at)),
        "departed_fraction": float(len(departed_at) / len(scheduled)) if scheduled else 0.0,
        "mean_depart_delay_s": float(np.mean(delays)) if delays.size else None,
        "median_depart_delay_s": float(np.median(delays)) if delays.size else None,
        "p95_depart_delay_s": float(np.quantile(delays, 0.95)) if delays.size else None,
        "max_depart_delay_s": float(np.max(delays)) if delays.size else None,
        "fraction_delay_within_one_step": frac_le(case.step_length_s + 1e-9),
        "fraction_delay_le_1s": frac_le(1.0),
        "fraction_delay_le_5s": frac_le(5.0),
        "teleport_events": int(teleports),
        "collision_events": int(collisions),
        "trajectory_rows": int(len(sim)),
    }

    legacy_gate_5s = bool(
        simulation["departed_fraction"] >= 0.99
        and simulation["p95_depart_delay_s"] is not None
        and simulation["p95_depart_delay_s"] <= 5.0
        and simulation["teleport_events"] == 0
        and simulation["collision_events"] == 0
    )
    boundary_gate_1s = bool(
        simulation["departed_fraction"] >= 0.99
        and simulation["p95_depart_delay_s"] is not None
        and simulation["p95_depart_delay_s"] <= 1.0
        and simulation["teleport_events"] == 0
        and simulation["collision_events"] == 0
    )

    sim.to_parquet(rundir / "simulation_frames.parquet", index=False)
    fit.to_csv(rundir / "speed_fit.csv", index=False)

    result = {
        "case": asdict(case),
        "boundary_speed_handling": boundary_speed_report,
        "discretization": _discretization_report(demand, case.step_length_s),
        "objective": float(objective),
        "legacy_gate_5s": legacy_gate_5s,
        "boundary_gate_1s": boundary_gate_1s,
        "simulation": simulation,
        "speed_fit": fit.to_dict("records"),
    }
    (rundir / "run_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_boundary_fidelity(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    models = project_root / "data" / "models"

    clean = _canonical_clean_with_lat(processed / "calibration_clean.parquet")
    priors = load_persona_priors(models / "sumo_persona_priors.csv")
    dna = load_behavior_dna(processed / "calibration_behavior_dna.parquet")

    base_cfg = StageAConfig()
    demand = build_boundary_demand(clean, priors, dna, base_cfg)
    observed = observed_speed_targets(clean)

    outroot = project_root / "data" / "simulation" / "boundary_fidelity"
    outroot.mkdir(parents=True, exist_ok=True)
    demand.to_csv(outroot / "observed_55m_boundary_state.csv", index=False)

    lane_counts = {
        str(int(k)): int(v)
        for k, v in demand["observed_lane_band"].value_counts().sort_index().items()
    }

    results = []
    for case in CASES:
        print(
            f"Running {case.name}: step={case.step_length_s}, "
            f"eager={case.eager_insert}, lane={case.lane_mode}, speed={case.speed_mode}"
        )
        try:
            r = _run_case(project_root, clean, priors, dna, demand, observed, case)
            results.append(r)
            print(
                f"  p95_delay={r['simulation']['p95_depart_delay_s']}, "
                f"<=1s={r['simulation']['fraction_delay_le_1s']:.3f}, "
                f"objective={r['objective']:.4f}, "
                f"collisions={r['simulation']['collision_events']}, "
                f"gate1s={r['boundary_gate_1s']}"
            )
        except Exception as exc:
            failed = {
                "case": asdict(case),
                "status": "CASE_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "legacy_gate_5s": False,
                "boundary_gate_1s": False,
            }
            results.append(failed)
            print(f"  CASE ERROR: {type(exc).__name__}: {exc}")
            print("  Continuing with remaining diagnostic cases...")

    passing_1s = [
        r for r in results
        if r.get("status") != "CASE_ERROR" and r.get("boundary_gate_1s", False)
    ]
    passing_5s = [
        r for r in results
        if r.get("status") != "CASE_ERROR" and r.get("legacy_gate_5s", False)
    ]

    report = {
        "sprint": "1E_A6_BOUNDARY_FIDELITY",
        "purpose": (
            "Separate simulation-time discretization / insertion-policy artifacts "
            "from true longitudinal-capacity mismatch while preserving the observed "
            "x=55 m boundary clock."
        ),
        "observed_boundary": {
            "entry_line_m": ENTRY_LINE_M,
            "vehicles": int(len(demand)),
            "lane_band_counts": lane_counts,
            "all_have_observed_time_speed_lat": bool(
                demand[["crossing_time_s", "crossing_speed_mps", "crossing_lat_m"]]
                .notna().all().all()
            ),
        },
        "fixed_behavior": {
            "speed_multipliers": {"1": 0.90, "2": 0.90, "6": 0.90},
            "tau_s": 0.72,
            "min_gap_m": 2.5,
            "min_gap_lat_m": 0.6,
            "lc_pushy": 0.0,
            "note": "No behavior parameter is tuned in A6.",
        },
        "gates": {
            "boundary_fidelity_gate": "p95 depart delay <= 1.0 s, >=99% departed, zero teleports/collisions",
            "legacy_gate": "p95 depart delay <= 5.0 s, >=99% departed, zero teleports/collisions",
        },
        "results": results,
        "any_boundary_gate_pass": bool(passing_1s),
        "any_legacy_gate_pass": bool(passing_5s),
        "interpretation": [
            "If R1/R2 sharply reduce delay, 0.5 s simulation resolution / insertion retry policy was a major artifact.",
            "If R3 improves further, preserving observed lane-band demand matters.",
            "If R4/R5 pass, the correct Stage-A boundary condition must include observed crossing time, lane band and speed.",
            "If R5 still fails badly, do not lower speedFactor or relax lateral rules further; investigate longitudinal safety spacing (especially minGap and tau) against observed same-lane gaps.",
        ],
        "hotfix": "A6.1_OBSERVED_DEPART_SPEED_DECOUPLING",
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a6_boundary_fidelity_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
