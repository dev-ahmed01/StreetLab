from __future__ import annotations

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from streetlab_phase1.sumo_stage_a import (
    StageAConfig,
    load_persona_priors,
    load_behavior_dna,
    make_vehicle_speed_factors,
    evaluate_speed_fit,
)
from streetlab_phase1.sumo_boundary_fidelity import (
    Case,
    _canonical_clean_with_lat,
    build_boundary_demand,
    observed_speed_targets,
    _write_network,
    _write_types,
    _write_routes,
)

PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}
PER_VEHICLE_LATERAL_SAFETY_MARGIN_M = 0.20
PAIRWISE_LATERAL_CLEARANCE_THRESHOLD_M = 0.40
LEADER_LOOKAHEAD_FRONT_FRONT_M = 50.0
OBSERVED_X_MIN_M = 65.0
OBSERVED_X_MAX_M = 235.0

CASES = [
    {"name": "P1_TAU_0P50_GAP_0P50", "tau_s": 0.50, "min_gap_m": 0.50},
    {"name": "P2_TAU_0P40_GAP_1P00", "tau_s": 0.40, "min_gap_m": 1.00},
    {"name": "P3_TAU_0P40_GAP_0P50", "tau_s": 0.40, "min_gap_m": 0.50},
]


def _q(a) -> dict:
    x = np.asarray(pd.Series(a).dropna(), dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def _dist_metrics(obs: np.ndarray, sim: np.ndarray, floor_scale: float) -> dict:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    obs = obs[np.isfinite(obs)]
    sim = sim[np.isfinite(sim)]
    if len(obs) < 20 or len(sim) < 20:
        return {"status": "INSUFFICIENT", "observed_n": len(obs), "simulated_n": len(sim)}

    iqr = float(np.quantile(obs, 0.75) - np.quantile(obs, 0.25))
    scale = max(iqr, floor_scale)
    wd = float(wasserstein_distance(obs, sim))
    ks = float(ks_2samp(obs, sim).statistic)
    med_delta = float(np.median(sim) - np.median(obs))
    p10_delta = float(np.quantile(sim, 0.10) - np.quantile(obs, 0.10))
    score = (
        wd / scale
        + 0.35 * ks
        + 0.25 * abs(med_delta) / scale
        + 0.25 * abs(p10_delta) / scale
    )
    return {
        "status": "OK",
        "observed_n": int(len(obs)),
        "simulated_n": int(len(sim)),
        "observed_p10": float(np.quantile(obs, 0.10)),
        "simulated_p10": float(np.quantile(sim, 0.10)),
        "p10_delta": p10_delta,
        "observed_median": float(np.median(obs)),
        "simulated_median": float(np.median(sim)),
        "median_delta": med_delta,
        "ks_statistic": ks,
        "wasserstein": wd,
        "normalized_wasserstein": float(wd / scale),
        "score": float(score),
    }


def reconstruct_interactions(
    frames: pd.DataFrame,
    *,
    time_col: str,
    vehicle_col: str,
    class_col: str,
    x_col: str,
    y_col: str,
    speed_col: str,
    length_col: str,
    width_col: str,
    pairwise_lateral_clearance_m: float = PAIRWISE_LATERAL_CLEARANCE_THRESHOLD_M,
    lookahead_front_front_m: float = LEADER_LOOKAHEAD_FRONT_FRONT_M,
) -> pd.DataFrame:
    """
    One common offline leader reconstruction for BOTH observed and simulated data.

    Canonical Sprint-1B interaction rule recovered by A9.2:

    - retain all vehicle classes as potential leaders;
    - leader must be ahead by front-bumper coordinate;
    - front-to-front lookahead is capped at 50 m;
    - each vehicle body is treated with a 0.20 m lateral safety expansion,
      equivalent to allowing up to 0.40 m pairwise body-edge clearance;
    - among eligible candidates, choose the nearest forward vehicle.

    `x_col` is treated as front-bumper longitudinal position.
    """
    d = frames[
        [time_col, vehicle_col, class_col, x_col, y_col, speed_col, length_col, width_col]
    ].copy()
    d.columns = [
        "time_s", "vehicle_id", "vehicle_class", "x_front_m", "y_center_m",
        "speed_mps", "length_m", "width_m"
    ]
    for c in [
        "time_s", "vehicle_class", "x_front_m", "y_center_m",
        "speed_mps", "length_m", "width_m"
    ]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().copy()
    d["vehicle_class"] = d["vehicle_class"].astype(int)

    # IMPORTANT: keep ALL observed/simulated vehicle classes as possible
    # leaders. Metrics are scored only for primary follower classes later.
    # Removing buses/trucks/LCVs here biases leader choice toward farther
    # vehicles and inflates front-gap/headway distributions.
    rows = []

    for t, g in d.groupby("time_s", sort=True):
        if len(g) < 2:
            continue

        vids = g["vehicle_id"].astype(str).to_numpy()
        cls = g["vehicle_class"].to_numpy(int)
        x = g["x_front_m"].to_numpy(float)
        y = g["y_center_m"].to_numpy(float)
        v = g["speed_mps"].to_numpy(float)
        length = g["length_m"].to_numpy(float)
        width = g["width_m"].to_numpy(float)

        # Pairwise follower i -> possible leader j.
        dx_front = x[None, :] - x[:, None]
        dy = np.abs(y[None, :] - y[:, None])
        body_clearance = dy - (width[:, None] + width[None, :]) / 2.0

        candidate = (
            (dx_front > 0.0)
            & (dx_front <= lookahead_front_front_m)
            & (body_clearance <= pairwise_lateral_clearance_m)
        )
        np.fill_diagonal(candidate, False)

        for i in range(len(g)):
            js = np.flatnonzero(candidate[i])
            if js.size == 0:
                continue

            # Nearest forward front position.
            j = int(js[np.argmin(dx_front[i, js])])
            front_gap = float(x[j] - length[j] - x[i])
            time_headway = (
                float(front_gap / v[i])
                if v[i] > 0.05 and front_gap >= 0.0
                else np.nan
            )
            side_clearance = float(body_clearance[i, j])

            rows.append({
                "time_s": float(t),
                "vehicle_id": vids[i],
                "vehicle_class": int(cls[i]),
                "leader_id": vids[j],
                "leader_class": int(cls[j]),
                "follower_x_front_m": float(x[i]),
                "leader_x_front_m": float(x[j]),
                "front_gap_m": front_gap,
                "time_headway_s": time_headway,
                "side_clearance_m": side_clearance,
                "follower_speed_mps": float(v[i]),
            })

    return pd.DataFrame(rows)


def _legacy_interaction_file(processed: Path) -> Path | None:
    preferred = [
        processed / "calibration_interactions.parquet",
        processed / "calibration_interaction.parquet",
        processed / "interactions_calibration.parquet",
    ]
    for p in preferred:
        if p.exists():
            return p
    cands = sorted(processed.glob("*calib*interaction*.parquet"))
    return cands[0] if cands else None


def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def _legacy_parity(processed: Path, common_obs: pd.DataFrame) -> dict:
    p = _legacy_interaction_file(processed)
    if p is None:
        return {"status": "LEGACY_INTERACTION_FILE_NOT_FOUND"}

    old = pd.read_parquet(p).copy()
    gap_col = _pick_col(old, ["front_gap_m", "front_gap", "gap_m"])
    th_col = _pick_col(old, ["time_headway_s", "time_headway", "headway_s"])
    cls_col = _pick_col(old, ["vehicle_class", "vehicle_type", "class_id"])
    pos_col = _pick_col(old, ["long_pos", "long_position", "x_m", "long_distance"])

    if gap_col is None or th_col is None or cls_col is None:
        return {
            "status": "LEGACY_COLUMNS_INCOMPLETE",
            "file": str(p),
            "available_columns": [str(c) for c in old.columns],
        }

    x = pd.DataFrame({
        "vehicle_class": pd.to_numeric(old[cls_col], errors="coerce"),
        "front_gap_m": pd.to_numeric(old[gap_col], errors="coerce"),
        "time_headway_s": pd.to_numeric(old[th_col], errors="coerce"),
    })
    if pos_col:
        x["long_pos_m"] = pd.to_numeric(old[pos_col], errors="coerce")
        m = (x["long_pos_m"] >= OBSERVED_X_MIN_M) & (x["long_pos_m"] <= OBSERVED_X_MAX_M)
        if int(m.sum()) >= 5000:
            x = x[m]

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x[
        x["vehicle_class"].isin(PRIMARY_CLASSES)
        & x["front_gap_m"].notna()
        & (x["front_gap_m"] >= 0)
        & x["time_headway_s"].notna()
        & (x["time_headway_s"] >= 0)
    ].copy()
    x["vehicle_class"] = x["vehicle_class"].astype(int)

    c = common_obs[
        (common_obs["front_gap_m"] >= 0)
        & common_obs["time_headway_s"].notna()
        & (common_obs["time_headway_s"] >= 0)
    ].copy()

    by_class = []
    max_rel_median_error = 0.0
    for cls, name in PRIMARY_CLASSES.items():
        a = x[x["vehicle_class"] == cls]
        b = c[c["vehicle_class"] == cls]
        old_gap_med = float(a["front_gap_m"].median()) if len(a) else None
        new_gap_med = float(b["front_gap_m"].median()) if len(b) else None
        old_th_med = float(a["time_headway_s"].median()) if len(a) else None
        new_th_med = float(b["time_headway_s"].median()) if len(b) else None

        errors = []
        if old_gap_med and new_gap_med is not None:
            errors.append(abs(new_gap_med - old_gap_med) / old_gap_med)
        if old_th_med and new_th_med is not None:
            errors.append(abs(new_th_med - old_th_med) / old_th_med)
        if errors:
            max_rel_median_error = max(max_rel_median_error, max(errors))

        by_class.append({
            "vehicle_class": cls,
            "vehicle_class_name": name,
            "legacy_rows": int(len(a)),
            "common_rows": int(len(b)),
            "legacy_front_gap": _q(a["front_gap_m"]),
            "common_front_gap": _q(b["front_gap_m"]),
            "legacy_time_headway": _q(a["time_headway_s"]),
            "common_time_headway": _q(b["time_headway_s"]),
        })

    return {
        "status": "OK",
        "file": str(p),
        "max_relative_median_error": float(max_rel_median_error),
        "parity_close_within_10pct_medians": bool(max_rel_median_error <= 0.10),
        "by_class": by_class,
    }


def _spacing_fit(obs: pd.DataFrame, sim: pd.DataFrame) -> tuple[list[dict], float]:
    rows = []
    scores = []
    for cls, name in PRIMARY_CLASSES.items():
        o = obs[
            (obs["vehicle_class"] == cls)
            & (obs["front_gap_m"] >= 0)
            & obs["time_headway_s"].notna()
            & (obs["time_headway_s"] >= 0)
        ]
        s = sim[
            (sim["vehicle_class"] == cls)
            & (sim["front_gap_m"] >= 0)
            & sim["time_headway_s"].notna()
            & (sim["time_headway_s"] >= 0)
        ]

        gap = _dist_metrics(
            o["front_gap_m"].to_numpy(float),
            s["front_gap_m"].to_numpy(float),
            1.0,
        )
        th = _dist_metrics(
            o["time_headway_s"].to_numpy(float),
            s["time_headway_s"].to_numpy(float),
            0.25,
        )
        valid = [m["score"] for m in (gap, th) if m.get("status") == "OK"]
        class_score = float(np.mean(valid)) if valid else float("inf")
        if np.isfinite(class_score):
            scores.append(class_score)

        rows.append({
            "vehicle_class": cls,
            "vehicle_class_name": name,
            "front_gap": gap,
            "time_headway": th,
            "class_spacing_score": class_score,
        })

    return rows, float(np.mean(scores)) if scores else float("inf")


def _run_case(
    project_root: Path,
    clean: pd.DataFrame,
    priors: pd.DataFrame,
    dna: pd.DataFrame | None,
    demand: pd.DataFrame,
    observed_speed: pd.DataFrame,
    observed_interactions: pd.DataFrame,
    case_def: dict,
) -> dict:
    import sumolib
    import traci

    tau_s = float(case_def["tau_s"])
    min_gap_m = float(case_def["min_gap_m"])

    case = Case(
        name=case_def["name"],
        step_length_s=0.10,
        eager_insert=True,
        lane_mode="best_prob",
        speed_mode="avg",
    )

    cfg = StageAConfig()
    cfg.default_tau_s = tau_s
    cfg.default_min_gap_m = min_gap_m
    cfg.default_min_gap_lat_m = 0.60
    cfg.default_lc_pushy = 0.0
    cfg.step_length_s = 0.10

    outroot = project_root / "data" / "simulation" / "metric_parity"
    rundir = outroot / case.name
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    sumo_binary = sumolib.checkBinary("sumo")
    netconvert_binary = sumolib.checkBinary("netconvert")
    net = _write_network(rundir, cfg, netconvert_binary)
    types = _write_types(rundir, cfg, demand, clean)

    speed_multipliers = {1: 0.90, 2: 0.90, 6: 0.90}
    speed_factors = make_vehicle_speed_factors(
        demand, clean, priors, dna, speed_multipliers, cfg
    )
    routes, _ = _write_routes(rundir, case, demand, speed_factors, cfg)

    last_depart = float(demand["depart_s"].max())
    end_s = last_depart + 90.0

    cmd = [
        sumo_binary,
        "-n", str(net),
        "-r", str(routes),
        "-a", str(types),
        "--step-length", "0.1",
        "--lateral-resolution", str(cfg.lateral_resolution_m),
        "--time-to-teleport", "-1",
        "--max-depart-delay", "120",
        "--collision.action", "warn",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--extrapolate-departpos", "true",
        "--eager-insert", "true",
        "--seed", "42",
    ]

    scheduled = dict(zip(demand["vehicle_id"].astype(str), demand["depart_s"].astype(float)))
    class_map = dict(zip(demand["vehicle_id"].astype(str), demand["vehicle_class"].astype(int)))
    departed_at = {}
    rows = []
    teleports = 0
    collisions = 0
    step_idx = 0

    traci.start(cmd)
    try:
        while traci.simulation.getTime() <= end_s:
            traci.simulationStep()
            step_idx += 1
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

            # Match source trajectory cadence: 0.5 s.
            if step_idx % 5 != 0:
                continue

            for vid in traci.vehicle.getIDList():
                svid = str(vid)
                cls = class_map.get(svid)
                if cls is None:
                    continue
                if traci.vehicle.getRoadID(vid) != "E0":
                    continue

                lane_pos = float(traci.vehicle.getLanePosition(vid))

                # Keep ALL classes and the whole measurement edge as interaction
                # context. The scored follower window is applied only after
                # leader reconstruction. This prevents edge/window censoring.
                xy = traci.vehicle.getPosition(vid)
                rows.append({
                    "time_s": now,
                    "vehicle_id": svid,
                    "vehicle_class": int(cls),
                    "x_front_m": lane_pos,
                    "y_center_m": float(xy[1]),
                    "speed_mps": float(traci.vehicle.getSpeed(vid)),
                    "length_m": float(traci.vehicle.getLength(vid)),
                    "width_m": float(traci.vehicle.getWidth(vid)),
                })
    finally:
        try:
            traci.close(False)
        except Exception:
            pass

    frames = pd.DataFrame(rows)
    sim_interactions_all = reconstruct_interactions(
        frames,
        time_col="time_s",
        vehicle_col="vehicle_id",
        class_col="vehicle_class",
        x_col="x_front_m",
        y_col="y_center_m",
        speed_col="speed_mps",
        length_col="length_m",
        width_col="width_m",
    )

    # Sim x=0 corresponds to observed x=55. Therefore the observed scored
    # follower window 65..235 maps to sim x=10..180.
    sim_interactions = sim_interactions_all[
        sim_interactions_all["vehicle_class"].isin(PRIMARY_CLASSES)
        & (sim_interactions_all["follower_x_front_m"] >= 10.0)
        & (sim_interactions_all["follower_x_front_m"] <= 180.0)
    ].copy()

    speed_frames = frames[
        frames["vehicle_class"].isin(PRIMARY_CLASSES)
        & (frames["x_front_m"] >= 10.0)
        & (frames["x_front_m"] <= 180.0)
    ][["time_s", "vehicle_id", "vehicle_class", "speed_mps"]].copy()

    speed_fit, speed_objective = evaluate_speed_fit(observed_speed, speed_frames)
    spacing_fit, spacing_objective = _spacing_fit(
        observed_interactions, sim_interactions
    )

    delays = np.asarray([
        float(actual - scheduled[vid])
        for vid, actual in departed_at.items()
        if vid in scheduled
    ], dtype=float)

    simulation = {
        "scheduled_vehicles": int(len(scheduled)),
        "departed_vehicles": int(len(departed_at)),
        "departed_fraction": float(len(departed_at) / len(scheduled)),
        "mean_depart_delay_s": float(np.mean(delays)) if delays.size else None,
        "median_depart_delay_s": float(np.median(delays)) if delays.size else None,
        "p95_depart_delay_s": float(np.quantile(delays, 0.95)) if delays.size else None,
        "max_depart_delay_s": float(np.max(delays)) if delays.size else None,
        "teleport_events": int(teleports),
        "collision_events": int(collisions),
        "trajectory_rows": int(len(frames)),
        "interaction_rows_all_context": int(len(sim_interactions_all)),
        "interaction_rows_scored": int(len(sim_interactions)),
    }

    capacity_gate = bool(
        simulation["departed_fraction"] >= 0.99
        and simulation["p95_depart_delay_s"] is not None
        and simulation["p95_depart_delay_s"] <= 5.0
        and simulation["teleport_events"] == 0
        and simulation["collision_events"] == 0
    )

    frames.to_parquet(rundir / "simulation_frames_common_metric.parquet", index=False)
    sim_interactions_all.to_parquet(
        rundir / "simulation_interactions_all_context.parquet", index=False
    )
    sim_interactions.to_parquet(
        rundir / "simulation_interactions_common_metric.parquet", index=False
    )
    speed_fit.to_csv(rundir / "speed_fit.csv", index=False)

    result = {
        "case": case_def,
        "capacity_gate_pass": capacity_gate,
        "simulation": simulation,
        "speed_objective": float(speed_objective),
        "speed_fit": speed_fit.to_dict("records"),
        "common_metric_spacing_objective": float(spacing_objective),
        "common_metric_spacing_fit": spacing_fit,
    }
    (rundir / "run_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_metric_parity(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    models = project_root / "data" / "models"

    clean = _canonical_clean_with_lat(processed / "calibration_clean.parquet")
    priors = load_persona_priors(models / "sumo_persona_priors.csv")
    dna = load_behavior_dna(processed / "calibration_behavior_dna.parquet")

    cfg = StageAConfig()
    demand = build_boundary_demand(clean, priors, dna, cfg)
    observed_speed = observed_speed_targets(clean)

    # Reconstruct interactions from the FULL observed frame so a primary
    # follower can legitimately choose a bus/truck/LCV leader and so leaders
    # just outside the scored follower window are not accidentally discarded.
    observed_frames = clean.copy()

    observed_interactions_all = reconstruct_interactions(
        observed_frames,
        time_col="time_s",
        vehicle_col="vehicle_id",
        class_col="vehicle_class",
        x_col="x_m",
        y_col="lat_pos",
        speed_col="speed_mps",
        length_col="length_m",
        width_col="width_m",
    )

    observed_interactions = observed_interactions_all[
        observed_interactions_all["vehicle_class"].isin(PRIMARY_CLASSES)
        & (observed_interactions_all["follower_x_front_m"] >= OBSERVED_X_MIN_M)
        & (observed_interactions_all["follower_x_front_m"] <= OBSERVED_X_MAX_M)
    ].copy()

    outroot = project_root / "data" / "simulation" / "metric_parity"
    outroot.mkdir(parents=True, exist_ok=True)
    observed_interactions_all.to_parquet(
        outroot / "observed_interactions_all_context.parquet", index=False
    )
    observed_interactions.to_parquet(
        outroot / "observed_interactions_common_metric.parquet", index=False
    )

    legacy_check = _legacy_parity(processed, observed_interactions)

    results = []
    for i, case_def in enumerate(CASES, start=1):
        print(
            f"[{i}/{len(CASES)}] {case_def['name']} "
            f"tau={case_def['tau_s']:.2f}, minGap={case_def['min_gap_m']:.2f}"
        )
        try:
            r = _run_case(
                project_root,
                clean,
                priors,
                dna,
                demand,
                observed_speed,
                observed_interactions,
                case_def,
            )
            results.append(r)
            print(
                f"  p95_delay={r['simulation']['p95_depart_delay_s']}, "
                f"spacing_obj={r['common_metric_spacing_objective']:.3f}, "
                f"capacity={r['capacity_gate_pass']}"
            )
        except Exception as exc:
            results.append({
                "case": case_def,
                "status": "CASE_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "capacity_gate_pass": False,
            })
            print(f"  CASE ERROR: {type(exc).__name__}: {exc}")

    valid = [r for r in results if r.get("status") != "CASE_ERROR"]
    passing = [r for r in valid if r.get("capacity_gate_pass")]

    report = {
        "sprint": "1E_A9_3_CANONICAL_INTERACTION_METRIC",
        "purpose": (
            "Re-evaluate A8's capacity-passing longitudinal candidates using one "
            "identical offline leader/front-gap reconstruction for observed and "
            "simulated trajectories."
        ),
        "why_needed": (
            "A9.2 recovered the canonical Sprint-1B interaction rule from the "
            "calibration trajectories: 50 m front-to-front leader lookahead and "
            "0.20 m lateral safety expansion per vehicle (0.40 m pairwise body-edge "
            "clearance threshold). A9.3 applies that same rule to observed and SUMO "
            "trajectories before deciding whether a global tau/minGap model is adequate."
        ),
        "hotfix": "A9.3_CANONICAL_50M_LOOKAHEAD_AND_0P20M_PER_VEHICLE_LATERAL_MARGIN",
        "common_metric_definition": {
            "per_vehicle_lateral_safety_expansion_m": PER_VEHICLE_LATERAL_SAFETY_MARGIN_M,
            "pairwise_body_clearance_threshold_m": PAIRWISE_LATERAL_CLEARANCE_THRESHOLD_M,
            "leader_lookahead_front_front_m": LEADER_LOOKAHEAD_FRONT_FRONT_M,
            "leader_candidate": (
                "forward vehicle within 50 m front-to-front whose body-edge "
                "clearance is <=0.40 m; this equals 0.20 m lateral safety expansion "
                "applied to each vehicle"
            ),
            "leader_population": "all vehicle classes; only follower scoring is restricted to classes 1/2/6",
            "window_rule": "leader reconstruction uses full corridor/edge context; scoring window is applied to followers afterward",
            "leader_choice": "nearest forward vehicle by front position",
            "front_gap": "leader_front - leader_length - follower_front",
            "time_headway": "nonnegative front_gap / follower_speed when speed > 0.05 m/s",
            "observed_window_m": [OBSERVED_X_MIN_M, OBSERVED_X_MAX_M],
            "sampling_s": 0.5,
        },
        "legacy_observed_interaction_parity": legacy_check,
        "observed_all_context_interaction_rows": int(len(observed_interactions_all)),
        "observed_common_interaction_rows": int(len(observed_interactions)),
        "cases": CASES,
        "results": results,
        "capacity_passing_cases": int(len(passing)),
        "decision_rules": [
            "Observed parity should now be <=10% because A9.2 recovered the canonical interaction definition; if not, stop and inspect the remaining unmatched legacy rows.",
            "If parity passes and every capacity-passing global tau/minGap case still produces materially shorter gaps/headways than observed, reject the single-global longitudinal model and move to Behaviour-DNA/persona heterogeneity.",
            "If parity passes and one global case reproduces spacing acceptably while retaining capacity, prefer the simpler global model.",
            "Do not select a candidate from capacity alone; spacing and speed fit must both be reviewed.",
            "Validation period remains untouched.",
        ],
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a9_metric_parity_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
