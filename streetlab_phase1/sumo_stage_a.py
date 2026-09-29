from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}


def _pick_col(df: pd.DataFrame, names: Iterable[str]) -> str:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(f"Could not find any of columns {list(names)}. Available: {list(df.columns)}")


def _safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _stable_u01(value: object) -> float:
    h = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12 - 1)


@dataclass
class StageAConfig:
    # Geometry is PROVISIONAL for this speed-calibration stage.
    corridor_length_m: float = 245.0
    feeder_length_m: float = 400.0
    lanes: int = 3
    lane_width_m: float = 3.5
    edge_speed_mps: float = 16.7
    lateral_resolution_m: float = 0.2

    step_length_s: float = 0.5
    warmup_s: float = 60.0
    cooldown_s: float = 90.0
    interior_buffer_m: float = 10.0

    default_tau_s: float = 0.72
    default_min_gap_m: float = 2.5
    default_min_gap_lat_m: float = 0.6
    default_lc_pushy: float = 0.0
    default_lc_sublane: float = 1.0

    max_depart_delay_s: float = 120.0
    time_to_teleport_s: float = -1.0

    # Infrastructure gate only. Speed mismatch is expected before calibration.
    min_departed_fraction_gate: float = 0.95
    max_p95_depart_delay_gate_s: float = 5.0
    max_teleports_gate: int = 0

    # Stage A2 coordinate-descent search if --mode calibrate is used.
    coarse_multipliers: tuple = (0.80, 0.90, 1.00, 1.10, 1.20)
    fine_offsets: tuple = (-0.08, -0.04, 0.0, 0.04, 0.08)


def load_clean(path: str | Path) -> tuple[pd.DataFrame, dict]:
    d = pd.read_parquet(path).copy()
    cols = {
        "vehicle_id": _pick_col(d, ["vehicle_id", "vehicle number", "vehicle_number"]),
        "vehicle_class": _pick_col(d, ["vehicle_class", "vehicle type", "vehicle_type"]),
        "time": _pick_col(d, ["timestamp", "time", "time_s", "time_sec", "time (sec)"]),
        "x": _pick_col(d, ["long_pos", "long_position", "long_distance", "long_distance_m", "long distance (m)", "x_m"]),
        "speed": _pick_col(d, ["long_speed", "long_speed_mps", "long speed (m/sec)", "speed_mps"]),
        "length": _pick_col(d, ["length", "length_m", "length (m)"]),
        "width": _pick_col(d, ["width", "width_m", "width (m)"]),
    }
    out = pd.DataFrame({
        "vehicle_id": d[cols["vehicle_id"]],
        "vehicle_class": pd.to_numeric(d[cols["vehicle_class"]], errors="coerce"),
        "time_s": pd.to_numeric(d[cols["time"]], errors="coerce"),
        "x_m": pd.to_numeric(d[cols["x"]], errors="coerce"),
        "speed_mps": pd.to_numeric(d[cols["speed"]], errors="coerce"),
        "length_m": pd.to_numeric(d[cols["length"]], errors="coerce"),
        "width_m": pd.to_numeric(d[cols["width"]], errors="coerce"),
    }).dropna(subset=["vehicle_id", "vehicle_class", "time_s", "x_m", "speed_mps"])
    out["vehicle_class"] = out["vehicle_class"].astype(int)
    return out, cols


def load_persona_priors(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_behavior_dna(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    d = pd.read_parquet(p).copy()
    try:
        vid = _pick_col(d, ["vehicle_id", "vehicle number", "vehicle_number"])
        cls = _pick_col(d, ["vehicle_class", "vehicle type", "vehicle_type"])
    except Exception:
        return None

    persona = None
    for c in ["persona", "persona_label", "cluster_persona"]:
        if c in d.columns:
            persona = c
            break
    speed_pref = None
    for c in ["speed_preference", "speed_pref_raw", "speed_preference_raw"]:
        if c in d.columns:
            speed_pref = c
            break

    out = pd.DataFrame({
        "vehicle_id": d[vid],
        "vehicle_class": pd.to_numeric(d[cls], errors="coerce"),
    })
    if persona:
        out["persona"] = d[persona].astype(str)
    if speed_pref:
        out["speed_preference"] = pd.to_numeric(d[speed_pref], errors="coerce")
    out = out.dropna(subset=["vehicle_id", "vehicle_class"]).drop_duplicates("vehicle_id")
    out["vehicle_class"] = out["vehicle_class"].astype(int)
    return out


def derive_class_stats(clean: pd.DataFrame) -> pd.DataFrame:
    per_vehicle = (
        clean.groupby("vehicle_id", as_index=False)
        .agg(
            vehicle_class=("vehicle_class", "first"),
            length_m=("length_m", "median"),
            width_m=("width_m", "median"),
            mean_speed_mps=("speed_mps", "mean"),
            p95_speed_mps=("speed_mps", lambda s: float(s.quantile(0.95))),
        )
    )
    return (
        per_vehicle.groupby("vehicle_class", as_index=False)
        .agg(
            vehicles=("vehicle_id", "size"),
            length_m=("length_m", "median"),
            width_m=("width_m", "median"),
            median_vehicle_mean_speed_mps=("mean_speed_mps", "median"),
            median_vehicle_p95_speed_mps=("p95_speed_mps", "median"),
        )
    )


def derive_demand(clean: pd.DataFrame, cfg: StageAConfig) -> pd.DataFrame:
    t0 = float(clean["time_s"].min())
    first = (
        clean.sort_values(["vehicle_id", "time_s"])
        .groupby("vehicle_id", as_index=False)
        .first()[["vehicle_id", "vehicle_class", "time_s", "x_m"]]
        .rename(columns={"time_s": "first_time_s", "x_m": "first_x_m"})
    )
    first["depart_s"] = first["first_time_s"] - t0

    # Remove only vehicles already inside the observed segment at the start.
    # Keep genuine arrivals during the first 60 s so they populate SUMO's
    # warm-up period instead of leaving the corridor artificially empty.
    initial_stock_window_s = max(cfg.step_length_s * 2.0, 1.0)
    first = first[first["depart_s"] > initial_stock_window_s].copy()
    first = first.sort_values(["depart_s", "vehicle_id"]).reset_index(drop=True)
    return first


def assign_personas(
    demand: pd.DataFrame,
    priors: pd.DataFrame,
    dna: pd.DataFrame | None,
) -> pd.DataFrame:
    out = demand.copy()
    dna_map = {}
    if dna is not None and "persona" in dna.columns:
        dna_map = dict(zip(dna["vehicle_id"].astype(str), dna["persona"].astype(str)))

    class_personas: Dict[int, list[tuple[str, float]]] = {}
    for cls, g in priors.groupby("vehicle_class"):
        g = g.sort_values("persona")
        shares = g["calibration_share"].astype(float).to_numpy()
        shares = shares / shares.sum()
        class_personas[int(cls)] = list(zip(g["persona"].astype(str), shares))

    personas = []
    for _, r in out.iterrows():
        vid = str(r["vehicle_id"])
        cls = int(r["vehicle_class"])
        if vid in dna_map and cls in PRIMARY_CLASSES:
            personas.append(dna_map[vid])
            continue
        options = class_personas.get(cls)
        if not options:
            personas.append("BG")
            continue
        u = _stable_u01(vid)
        cum = 0.0
        chosen = options[-1][0]
        for label, share in options:
            cum += float(share)
            if u <= cum:
                chosen = label
                break
        personas.append(chosen)
    out["persona"] = personas
    return out


def make_vehicle_speed_factors(
    demand: pd.DataFrame,
    clean: pd.DataFrame,
    priors: pd.DataFrame,
    dna: pd.DataFrame | None,
    class_multipliers: dict[int, float],
    cfg: StageAConfig,
) -> dict[str, float]:
    stats = derive_class_stats(clean).set_index("vehicle_class")
    prior_lookup = {
        (int(r.vehicle_class), str(r.persona)): float(r.speed_relative_to_class)
        for r in priors.itertuples()
    }

    dna_speed = {}
    class_dna_median = {}
    if dna is not None and "speed_preference" in dna.columns:
        valid = dna.dropna(subset=["speed_preference"])
        dna_speed = dict(zip(valid["vehicle_id"].astype(str), valid["speed_preference"].astype(float)))
        class_dna_median = valid.groupby("vehicle_class")["speed_preference"].median().to_dict()

    factors = {}
    for _, r in demand.iterrows():
        vid = str(r["vehicle_id"])
        cls = int(r["vehicle_class"])
        persona = str(r["persona"])

        if cls in stats.index:
            target_ref = float(stats.loc[cls, "median_vehicle_p95_speed_mps"])
        else:
            target_ref = cfg.edge_speed_mps * 0.5
        class_base = target_ref / cfg.edge_speed_mps

        if vid in dna_speed and cls in class_dna_median and class_dna_median[cls] > 1e-6:
            relative = float(dna_speed[vid] / class_dna_median[cls])
        else:
            relative = prior_lookup.get((cls, persona), 1.0)

        multiplier = float(class_multipliers.get(cls, 1.0))
        factors[vid] = float(np.clip(class_base * relative * multiplier, 0.15, 1.45))
    return factors


def write_network(workdir: Path, cfg: StageAConfig, netconvert_binary: str) -> Path:
    nodes = workdir / "corridor.nod.xml"
    edges = workdir / "corridor.edg.xml"
    net = workdir / "corridor.net.xml"

    # E_pre is an insertion reservoir. It prevents high-flow insertion mechanics
    # from contaminating the 245 m measurement edge E0.
    nodes.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<nodes>\n'
        f'  <node id="npre" x="{-cfg.feeder_length_m:.3f}" y="0.0" type="priority"/>\n'
        '  <node id="n0" x="0.0" y="0.0" type="priority"/>\n'
        f'  <node id="n1" x="{cfg.corridor_length_m:.3f}" y="0.0" type="priority"/>\n'
        '</nodes>\n',
        encoding="utf-8",
    )
    edges.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<edges>\n'
        f'  <edge id="E_pre" from="npre" to="n0" numLanes="{cfg.lanes}" '
        f'speed="{cfg.edge_speed_mps:.4f}" width="{cfg.lane_width_m:.3f}"/>\n'
        f'  <edge id="E0" from="n0" to="n1" numLanes="{cfg.lanes}" '
        f'speed="{cfg.edge_speed_mps:.4f}" width="{cfg.lane_width_m:.3f}"/>\n'
        '</edges>\n',
        encoding="utf-8",
    )
    cmd = [
        netconvert_binary,
        "--node-files", str(nodes),
        "--edge-files", str(edges),
        "--output-file", str(net),
        "--no-turnarounds", "true",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"netconvert failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return net

def _vclass_for(cls: int) -> str:
    return {1: "motorcycle", 2: "passenger", 6: "passenger"}.get(cls, "passenger")


def write_types(
    workdir: Path,
    demand: pd.DataFrame,
    clean: pd.DataFrame,
    priors: pd.DataFrame,
    cfg: StageAConfig,
) -> Path:
    stats = derive_class_stats(clean).set_index("vehicle_class")
    types_path = workdir / "stage_a_types.add.xml"
    type_rows = sorted(set((int(r.vehicle_class), str(r.persona)) for r in demand.itertuples()))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional>',
        '  <!-- Stage A isolates desired-speed calibration. Other behavior parameters stay neutral/fixed. -->',
    ]
    for cls, persona in type_rows:
        if cls in stats.index:
            length = float(stats.loc[cls, "length_m"])
            width = float(stats.loc[cls, "width_m"])
        else:
            length, width = 4.5, 1.8

        type_id = f"sl_c{cls}_{persona.lower()}"
        lines.append(
            f'  <vType id="{type_id}" vClass="{_vclass_for(cls)}" '
            f'length="{length:.4f}" width="{width:.4f}" '
            f'carFollowModel="Krauss" laneChangeModel="SL2015" '
            f'tau="{cfg.default_tau_s:.4f}" minGap="{cfg.default_min_gap_m:.4f}" '
            f'minGapLat="{cfg.default_min_gap_lat_m:.4f}" '
            f'lcPushy="{cfg.default_lc_pushy:.4f}" lcSublane="{cfg.default_lc_sublane:.4f}" '
            f'speedFactor="1.0" speedDev="0.0"/>'
        )
    lines.append("</additional>")
    types_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return types_path


def write_routes(
    workdir: Path,
    demand: pd.DataFrame,
    speed_factors: dict[str, float],
) -> Path:
    path = workdir / "stage_a.rou.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '  <route id="corridor" edges="E_pre E0"/>',
    ]
    for r in demand.itertuples():
        vid = str(r.vehicle_id)
        cls = int(r.vehicle_class)
        persona = str(r.persona)
        type_id = f"sl_c{cls}_{persona.lower()}"
        sf = float(speed_factors.get(vid, 1.0))
        lines.append(
            f'  <vehicle id="{vid}" type="{type_id}" route="corridor" '
            f'depart="{float(r.depart_s):.3f}" departLane="best_prob" '
            f'departPos="last" departSpeed="avg" speedFactor="{sf:.6f}"/>'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def observed_speed_targets(clean: pd.DataFrame, cfg: StageAConfig) -> pd.DataFrame:
    t0 = float(clean["time_s"].min())
    x = clean[
        (clean["time_s"] - t0 >= cfg.warmup_s)
        & (clean["x_m"] >= cfg.interior_buffer_m)
        & (clean["x_m"] <= cfg.corridor_length_m - cfg.interior_buffer_m)
        & (clean["vehicle_class"].isin(PRIMARY_CLASSES))
    ].copy()
    return (
        x.groupby(["vehicle_id", "vehicle_class"], as_index=False)
        .agg(mean_speed_mps=("speed_mps", "mean"))
    )


def run_simulation(
    workdir: Path,
    net: Path,
    routes: Path,
    types_path: Path,
    demand: pd.DataFrame,
    speed_factors: dict[str, float],
    cfg: StageAConfig,
    sumo_binary: str,
) -> tuple[pd.DataFrame, dict]:
    import traci

    last_depart = float(demand["depart_s"].max()) if len(demand) else cfg.warmup_s
    end_s = last_depart + cfg.cooldown_s

    tripinfo = workdir / "tripinfo.xml"
    cmd = [
        sumo_binary,
        "-n", str(net),
        "-r", str(routes),
        "-a", str(types_path),
        "--step-length", str(cfg.step_length_s),
        "--lateral-resolution", str(cfg.lateral_resolution_m),
        "--time-to-teleport", str(cfg.time_to_teleport_s),
        "--max-depart-delay", str(cfg.max_depart_delay_s),
        "--collision.action", "warn",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--tripinfo-output", str(tripinfo),
        "--seed", "42",
        "--extrapolate-departpos", "true",
    ]

    scheduled = dict(zip(demand["vehicle_id"].astype(str), demand["depart_s"].astype(float)))
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

            try:
                teleports += len(traci.simulation.getStartingTeleportIDList())
            except Exception:
                pass
            try:
                collisions += len(traci.simulation.getCollidingVehiclesIDList())
            except Exception:
                pass

            if now < cfg.warmup_s:
                continue

            for vid in traci.vehicle.getIDList():
                # Feeder-edge trajectories are never part of the observed
                # 245 m comparison window.
                if traci.vehicle.getRoadID(vid) != "E0":
                    continue
                pos = float(traci.vehicle.getLanePosition(vid))
                if pos < cfg.interior_buffer_m or pos > cfg.corridor_length_m - cfg.interior_buffer_m:
                    continue
                rows.append({
                    "time_s": now,
                    "vehicle_id": str(vid),
                    "speed_mps": float(traci.vehicle.getSpeed(vid)),
                    "lane_position_m": pos,
                })
    finally:
        traci.close(False)

    sim = pd.DataFrame(rows)
    class_map = dict(zip(demand["vehicle_id"].astype(str), demand["vehicle_class"].astype(int)))
    if not sim.empty:
        sim["vehicle_class"] = sim["vehicle_id"].map(class_map)

    delays = []
    for vid, actual in departed_at.items():
        if vid in scheduled:
            delays.append(float(actual - scheduled[vid]))

    summary = {
        "scheduled_vehicles": int(len(demand)),
        "departed_vehicles": int(len(departed_at)),
        "departed_fraction": float(len(departed_at) / len(demand)) if len(demand) else 0.0,
        "mean_depart_delay_s": float(np.mean(delays)) if delays else None,
        "p95_depart_delay_s": float(np.quantile(delays, 0.95)) if delays else None,
        "teleport_events": int(teleports),
        "collision_events": int(collisions),
        "simulation_end_s": float(end_s),
        "trajectory_rows": int(len(sim)),
    }
    return sim, summary


def evaluate_speed_fit(
    observed_per_vehicle: pd.DataFrame,
    sim_frames: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    if sim_frames.empty:
        return pd.DataFrame(), float("inf")

    sim_per_vehicle = (
        sim_frames.groupby(["vehicle_id", "vehicle_class"], as_index=False)
        .agg(mean_speed_mps=("speed_mps", "mean"))
    )

    rows = []
    objectives = []
    for cls, name in PRIMARY_CLASSES.items():
        o = observed_per_vehicle.loc[
            observed_per_vehicle["vehicle_class"] == cls, "mean_speed_mps"
        ].dropna().to_numpy(float)
        s = sim_per_vehicle.loc[
            sim_per_vehicle["vehicle_class"] == cls, "mean_speed_mps"
        ].dropna().to_numpy(float)
        if len(o) < 20 or len(s) < 20:
            rows.append({
                "vehicle_class": cls,
                "vehicle_class_name": name,
                "observed_n": len(o),
                "simulated_n": len(s),
                "status": "INSUFFICIENT",
            })
            objectives.append(10.0)
            continue

        iqr = float(np.quantile(o, 0.75) - np.quantile(o, 0.25))
        scale = max(iqr, 0.5)
        wd = float(wasserstein_distance(o, s))
        med_delta = float(np.median(s) - np.median(o))
        ks = float(ks_2samp(o, s).statistic)
        norm_wd = wd / scale
        norm_med = abs(med_delta) / scale
        obj = norm_wd + 0.5 * norm_med + 0.5 * ks
        objectives.append(obj)

        rows.append({
            "vehicle_class": cls,
            "vehicle_class_name": name,
            "observed_n": int(len(o)),
            "simulated_n": int(len(s)),
            "observed_median_speed_mps": float(np.median(o)),
            "simulated_median_speed_mps": float(np.median(s)),
            "median_delta_mps": med_delta,
            "ks_statistic": ks,
            "wasserstein_mps": wd,
            "normalized_wasserstein": norm_wd,
            "objective": obj,
            "status": "OK",
        })
    return pd.DataFrame(rows), float(np.mean(objectives))


def prepare_corridor(
    project_root: str | Path,
    cfg: StageAConfig,
) -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    models = project_root / "data" / "models"
    simroot = project_root / "data" / "simulation" / "stage_a"
    simroot.mkdir(parents=True, exist_ok=True)

    clean, original_cols = load_clean(processed / "calibration_clean.parquet")
    priors = load_persona_priors(models / "sumo_persona_priors.csv")
    dna = load_behavior_dna(processed / "calibration_behavior_dna.parquet")
    demand = derive_demand(clean, cfg)
    demand = assign_personas(demand, priors, dna)

    demand.to_csv(simroot / "calibration_demand.csv", index=False)
    observed = observed_speed_targets(clean, cfg)
    observed.to_parquet(simroot / "observed_speed_targets.parquet", index=False)

    return {
        "project_root": project_root,
        "processed": processed,
        "models": models,
        "simroot": simroot,
        "clean": clean,
        "priors": priors,
        "dna": dna,
        "demand": demand,
        "observed": observed,
        "original_columns": original_cols,
    }


def run_stage_a_once(
    ctx: dict,
    cfg: StageAConfig,
    class_multipliers: dict[int, float],
    run_name: str,
) -> dict:
    import sumolib

    simroot = ctx["simroot"]
    rundir = simroot / "runs" / run_name
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    sumo_binary = sumolib.checkBinary("sumo")
    netconvert_binary = sumolib.checkBinary("netconvert")

    net = write_network(rundir, cfg, netconvert_binary)
    types_path = write_types(rundir, ctx["demand"], ctx["clean"], ctx["priors"], cfg)

    speed_factors = make_vehicle_speed_factors(
        ctx["demand"], ctx["clean"], ctx["priors"], ctx["dna"],
        class_multipliers, cfg
    )
    routes = write_routes(rundir, ctx["demand"], speed_factors)
    sim, sim_summary = run_simulation(
        rundir, net, routes, types_path, ctx["demand"], speed_factors, cfg, sumo_binary
    )
    fit, objective = evaluate_speed_fit(ctx["observed"], sim)

    sim.to_parquet(rundir / "simulation_frames.parquet", index=False)
    fit.to_csv(rundir / "speed_fit.csv", index=False)
    (rundir / "class_multipliers.json").write_text(
        json.dumps({str(k): float(v) for k, v in class_multipliers.items()}, indent=2),
        encoding="utf-8",
    )

    p95_delay = sim_summary.get("p95_depart_delay_s")
    infrastructure_gate = (
        sim_summary["departed_fraction"] >= cfg.min_departed_fraction_gate
        and sim_summary["teleport_events"] <= cfg.max_teleports_gate
        and p95_delay is not None
        and p95_delay <= cfg.max_p95_depart_delay_gate_s
    )
    result = {
        "run_name": run_name,
        "class_multipliers": {str(k): float(v) for k, v in class_multipliers.items()},
        "objective": objective,
        "infrastructure_gate_pass": bool(infrastructure_gate),
        "simulation": sim_summary,
        "speed_fit": fit.to_dict("records"),
    }
    (rundir / "run_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def calibrate_coordinate_descent(ctx: dict, cfg: StageAConfig) -> dict:
    current = {1: 1.0, 2: 1.0, 6: 1.0}
    history = []

    baseline = run_stage_a_once(ctx, cfg, current, "baseline")
    history.append(baseline)
    if not baseline["infrastructure_gate_pass"]:
        return {
            "status": "STOPPED_INFRASTRUCTURE_GATE_FAILED",
            "best": baseline,
            "history": history,
        }

    best = baseline
    for cls in [1, 2, 6]:
        cls_best = best
        for mult in cfg.coarse_multipliers:
            candidate = dict(current)
            candidate[cls] = float(mult)
            name = f"coarse_c{cls}_{mult:.3f}".replace(".", "p")
            r = run_stage_a_once(ctx, cfg, candidate, name)
            history.append(r)
            if r["infrastructure_gate_pass"] and r["objective"] < cls_best["objective"]:
                cls_best = r
        current = {int(k): float(v) for k, v in cls_best["class_multipliers"].items()}
        best = cls_best

    for cls in [1, 2, 6]:
        center = float(current[cls])
        cls_best = best
        for off in cfg.fine_offsets:
            mult = max(0.5, min(1.5, center + float(off)))
            candidate = dict(current)
            candidate[cls] = mult
            name = f"fine_c{cls}_{mult:.3f}".replace(".", "p")
            r = run_stage_a_once(ctx, cfg, candidate, name)
            history.append(r)
            if r["infrastructure_gate_pass"] and r["objective"] < cls_best["objective"]:
                cls_best = r
        current = {int(k): float(v) for k, v in cls_best["class_multipliers"].items()}
        best = cls_best

    return {
        "status": "CALIBRATION_COMPLETE",
        "best": best,
        "history": history,
    }


def make_overall_report(ctx: dict, cfg: StageAConfig, result: dict, mode: str) -> dict:
    class_counts = ctx["demand"]["vehicle_class"].value_counts().sort_index().to_dict()
    return {
        "sprint": "1E_A_SPEED_CORRIDOR",
        "mode": mode,
        "geometry_status": "PROVISIONAL_STAGE_A_NOT_FINAL_CHENNAI_GEOMETRY",
        "purpose": "Validate SUMO execution and calibrate desired-speed behavior only, using a provisional capacity-enabling tau baseline.",
        "capacity_baseline": {
            "source_case": "C3_TAU_0P72",
            "provisional_tau_s": cfg.default_tau_s,
            "final_spacing_calibration": False,
        },
        "explicit_non_goals": [
            "Do not claim final Chennai geometry from this stage.",
            "Do not interpret tau=0.72 as final spacing calibration; it is a provisional capacity-enabling baseline for Stage A.",
            "Do not calibrate lateral behavior in Stage A.",
            "Do not use the untouched validation period for tuning.",
        ],
        "corridor": {
            "measurement_length_m": cfg.corridor_length_m,
            "feeder_length_m": cfg.feeder_length_m,
            "lanes": cfg.lanes,
            "lane_width_m": cfg.lane_width_m,
            "edge_speed_mps": cfg.edge_speed_mps,
            "lateral_resolution_m": cfg.lateral_resolution_m,
            "step_length_s": cfg.step_length_s,
        },
        "demand": {
            "calibration_only": True,
            "initial_stock_rule": "remove only vehicles first seen within ~1 s of recording start",
            "scheduled_vehicles": int(len(ctx["demand"])),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        },
        "result": result,
        "config": asdict(cfg),
        "next_gate": (
            "If baseline infrastructure gate passes, run --mode calibrate. "
            "If calibration completes, inspect speed fit before unlocking Stage B."
        ),
    }
