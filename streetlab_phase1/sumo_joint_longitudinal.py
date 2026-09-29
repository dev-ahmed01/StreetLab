from __future__ import annotations

from pathlib import Path
import json
import math
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

TAU_VALUES_S = [0.72, 0.60, 0.50, 0.40]
MIN_GAP_VALUES_M = [2.00, 1.50, 1.00, 0.50]


def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def _find_interaction_file(processed: Path) -> Path:
    preferred = [
        processed / "calibration_interactions.parquet",
        processed / "calibration_interaction.parquet",
        processed / "interactions_calibration.parquet",
    ]
    for p in preferred:
        if p.exists():
            return p
    candidates = sorted(processed.glob("*calib*interaction*.parquet"))
    if not candidates:
        candidates = sorted(processed.glob("*interaction*.parquet"))
    if not candidates:
        raise FileNotFoundError("Could not locate calibration interactions parquet.")
    return candidates[0]


def _load_observed_spacing(processed: Path) -> tuple[pd.DataFrame, dict]:
    p = _find_interaction_file(processed)
    d = pd.read_parquet(p).copy()

    gap_col = _pick_col(d, ["front_gap_m", "front_gap", "gap_m"])
    th_col = _pick_col(d, ["time_headway_s", "time_headway", "headway_s"])
    cls_col = _pick_col(d, ["vehicle_class", "vehicle_type", "class_id"])
    pos_col = _pick_col(d, ["long_pos", "long_position", "x_m", "long_distance"])
    speed_col = _pick_col(
        d,
        ["long_speed", "long_speed_mps", "speed_mps", "follower_speed_mps", "speed"],
    )

    if gap_col is None or cls_col is None:
        raise ValueError(
            f"Interactions parquet must contain front gap and vehicle class. "
            f"Available columns: {list(d.columns)}"
        )

    out = pd.DataFrame({
        "vehicle_class": pd.to_numeric(d[cls_col], errors="coerce"),
        "front_gap_m": pd.to_numeric(d[gap_col], errors="coerce"),
    })
    if th_col:
        out["time_headway_s"] = pd.to_numeric(d[th_col], errors="coerce")
    elif speed_col:
        speed = pd.to_numeric(d[speed_col], errors="coerce")
        out["time_headway_s"] = out["front_gap_m"] / speed.where(speed > 0.05)

    if pos_col:
        out["long_pos_m"] = pd.to_numeric(d[pos_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[
        out["vehicle_class"].isin(PRIMARY_CLASSES)
        & out["front_gap_m"].notna()
        & (out["front_gap_m"] >= 0)
        & out["time_headway_s"].notna()
        & (out["time_headway_s"] >= 0)
    ].copy()
    out["vehicle_class"] = out["vehicle_class"].astype(int)

    # Match the simulated observation window whenever the interaction parquet
    # contains a longitudinal coordinate.
    spatial_filter_applied = False
    if "long_pos_m" in out.columns:
        mask = (out["long_pos_m"] >= 65.0) & (out["long_pos_m"] <= 235.0)
        if int(mask.sum()) >= 5000:
            out = out[mask].copy()
            spatial_filter_applied = True

    meta = {
        "file": str(p),
        "rows": int(len(out)),
        "spatial_filter_applied": spatial_filter_applied,
        "spatial_window_m": [65.0, 235.0] if spatial_filter_applied else None,
    }
    return out, meta


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
    score = wd / scale + 0.35 * ks + 0.25 * abs(med_delta) / scale + 0.25 * abs(p10_delta) / scale

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


def _spacing_fit(observed: pd.DataFrame, simulated: pd.DataFrame) -> tuple[list[dict], float]:
    rows = []
    scores = []
    for cls, name in PRIMARY_CLASSES.items():
        o = observed[observed["vehicle_class"] == cls]
        s = simulated[simulated["vehicle_class"] == cls]

        gap = _dist_metrics(
            o["front_gap_m"].to_numpy(float),
            s["front_gap_m"].to_numpy(float),
            floor_scale=1.0,
        )
        th = _dist_metrics(
            o["time_headway_s"].to_numpy(float),
            s["time_headway_s"].to_numpy(float),
            floor_scale=0.25,
        )

        valid_scores = [
            x["score"] for x in [gap, th]
            if x.get("status") == "OK"
        ]
        class_score = float(np.mean(valid_scores)) if valid_scores else float("inf")
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


def _empirical_envelope(observed: pd.DataFrame, tau_s: float, min_gap_m: float) -> dict:
    # This is an equilibrium-style compatibility diagnostic, not a claim that
    # every instantaneous observation must satisfy a steady-state formula.
    # For Krauss, the constant-speed net headway approximation is:
    #   gap ~= minGap + tau * speed
    # The interaction parquet does not always carry speed after filtering, so
    # derive the equivalent headway threshold from observed gap/headway:
    # speed ~= gap / headway.
    x = observed[(observed["time_headway_s"] > 0.05) & (observed["front_gap_m"] >= 0)].copy()
    speed = x["front_gap_m"] / x["time_headway_s"]
    predicted_min_gap = min_gap_m + tau_s * speed
    violating = x["front_gap_m"] < predicted_min_gap

    med_speed = float(np.median(speed[np.isfinite(speed)])) if len(x) else None
    implied_net_time = (
        float(tau_s + min_gap_m / med_speed)
        if med_speed is not None and med_speed > 0
        else None
    )

    return {
        "rows": int(len(x)),
        "fraction_observed_below_equilibrium_envelope": float(violating.mean()) if len(x) else None,
        "median_inferred_speed_mps": med_speed,
        "implied_net_time_headway_at_median_speed_s": implied_net_time,
        "note": (
            "Diagnostic only: instantaneous car-following is dynamic; this uses "
            "the Krauss constant-speed relation to compare candidate restrictiveness."
        ),
    }


def _run_case(
    project_root: Path,
    clean: pd.DataFrame,
    priors: pd.DataFrame,
    dna: pd.DataFrame | None,
    demand: pd.DataFrame,
    observed_speed: pd.DataFrame,
    observed_spacing: pd.DataFrame,
    tau_s: float,
    min_gap_m: float,
) -> dict:
    import sumolib
    import traci

    case_name = f"TAU_{str(tau_s).replace('.', 'P')}_GAP_{str(min_gap_m).replace('.', 'P')}"
    case = Case(
        name=case_name,
        step_length_s=0.10,
        eager_insert=True,
        lane_mode="best_prob",
        speed_mode="avg",
    )

    cfg = StageAConfig()
    cfg.default_tau_s = float(tau_s)
    cfg.default_min_gap_m = float(min_gap_m)
    cfg.default_min_gap_lat_m = 0.60
    cfg.default_lc_pushy = 0.0
    cfg.step_length_s = 0.10

    outroot = project_root / "data" / "simulation" / "joint_longitudinal_grid"
    rundir = outroot / case_name
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
    speed_rows = []
    spacing_rows = []
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

            # Match the original 0.5 s trajectory sampling cadence even though
            # SUMO runs at 0.1 s.
            if step_idx % 5 != 0:
                continue

            for vid in traci.vehicle.getIDList():
                svid = str(vid)
                cls = class_map.get(svid)
                if cls not in PRIMARY_CLASSES:
                    continue
                if traci.vehicle.getRoadID(vid) != "E0":
                    continue
                pos = float(traci.vehicle.getLanePosition(vid))
                if pos < 10.0 or pos > 180.0:
                    continue

                speed = float(traci.vehicle.getSpeed(vid))
                speed_rows.append({
                    "vehicle_id": svid,
                    "vehicle_class": cls,
                    "speed_mps": speed,
                })

                try:
                    leader = traci.vehicle.getLeader(vid, 60.0)
                except Exception:
                    leader = None
                if not leader:
                    continue

                gap = float(leader[1])
                if not np.isfinite(gap) or gap < 0:
                    continue
                if speed <= 0.05:
                    continue

                spacing_rows.append({
                    "vehicle_class": cls,
                    "front_gap_m": gap,
                    "time_headway_s": gap / speed,
                })
    finally:
        try:
            traci.close(False)
        except Exception:
            pass

    speed_frames = pd.DataFrame(speed_rows)
    if speed_frames.empty:
        speed_fit = pd.DataFrame()
        speed_objective = float("inf")
    else:
        # evaluate_speed_fit expects per-frame speed data and will aggregate
        # vehicle means internally.
        speed_fit, speed_objective = evaluate_speed_fit(observed_speed, speed_frames)

    sim_spacing = pd.DataFrame(spacing_rows)
    spacing_fit, spacing_objective = _spacing_fit(observed_spacing, sim_spacing)

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
        "fraction_delay_le_1s": float((delays <= 1.0).mean()) if delays.size else None,
        "fraction_delay_le_5s": float((delays <= 5.0).mean()) if delays.size else None,
        "teleport_events": int(teleports),
        "collision_events": int(collisions),
        "speed_rows": int(len(speed_frames)),
        "spacing_rows": int(len(sim_spacing)),
    }

    capacity_gate = bool(
        simulation["departed_fraction"] >= 0.99
        and simulation["p95_depart_delay_s"] is not None
        and simulation["p95_depart_delay_s"] <= 5.0
        and simulation["teleport_events"] == 0
        and simulation["collision_events"] == 0
    )

    # Do not let a great fit compensate for a failed capacity gate.
    joint_objective = (
        float(speed_objective + spacing_objective)
        if capacity_gate and np.isfinite(speed_objective) and np.isfinite(spacing_objective)
        else None
    )

    result = {
        "tau_s": float(tau_s),
        "min_gap_m": float(min_gap_m),
        "speed_multipliers": {"1": 0.90, "2": 0.90, "6": 0.90},
        "capacity_gate_pass": capacity_gate,
        "simulation": simulation,
        "speed_objective": float(speed_objective),
        "speed_fit": speed_fit.to_dict("records") if not speed_fit.empty else [],
        "spacing_objective": float(spacing_objective),
        "spacing_fit": spacing_fit,
        "joint_objective_if_capacity_pass": joint_objective,
        "empirical_equilibrium_envelope": _empirical_envelope(
            observed_spacing, tau_s, min_gap_m
        ),
    }

    if not sim_spacing.empty:
        sim_spacing.to_parquet(rundir / "simulated_spacing_samples.parquet", index=False)
    if not speed_fit.empty:
        speed_fit.to_csv(rundir / "speed_fit.csv", index=False)
    (rundir / "run_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _pareto_front(results: list[dict]) -> list[dict]:
    valid = [
        r for r in results
        if r.get("status") != "CASE_ERROR"
        and r.get("capacity_gate_pass")
        and r.get("joint_objective_if_capacity_pass") is not None
    ]
    if not valid:
        return []

    # Keep candidates that are not dominated simultaneously on speed and spacing.
    front = []
    for a in valid:
        dominated = False
        for b in valid:
            if a is b:
                continue
            if (
                b["speed_objective"] <= a["speed_objective"]
                and b["spacing_objective"] <= a["spacing_objective"]
                and (
                    b["speed_objective"] < a["speed_objective"]
                    or b["spacing_objective"] < a["spacing_objective"]
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(a)

    return sorted(
        front,
        key=lambda r: (
            r["joint_objective_if_capacity_pass"],
            r["tau_s"],
            r["min_gap_m"],
        ),
    )


def run_joint_longitudinal_grid(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    models = project_root / "data" / "models"

    clean = _canonical_clean_with_lat(processed / "calibration_clean.parquet")
    priors = load_persona_priors(models / "sumo_persona_priors.csv")
    dna = load_behavior_dna(processed / "calibration_behavior_dna.parquet")

    cfg = StageAConfig()
    demand = build_boundary_demand(clean, priors, dna, cfg)
    observed_speed = observed_speed_targets(clean)
    observed_spacing, spacing_meta = _load_observed_spacing(processed)

    results = []
    total = len(TAU_VALUES_S) * len(MIN_GAP_VALUES_M)
    idx = 0
    for tau in TAU_VALUES_S:
        for gap in MIN_GAP_VALUES_M:
            idx += 1
            print(f"[{idx}/{total}] Running tau={tau:.2f}s, minGap={gap:.2f}m...")
            try:
                r = _run_case(
                    project_root,
                    clean,
                    priors,
                    dna,
                    demand,
                    observed_speed,
                    observed_spacing,
                    tau,
                    gap,
                )
                results.append(r)
                print(
                    f"  departed={r['simulation']['departed_fraction']:.3f}, "
                    f"p95_delay={r['simulation']['p95_depart_delay_s']}, "
                    f"collisions={r['simulation']['collision_events']}, "
                    f"speed_obj={r['speed_objective']:.3f}, "
                    f"spacing_obj={r['spacing_objective']:.3f}, "
                    f"gate={r['capacity_gate_pass']}"
                )
            except Exception as exc:
                results.append({
                    "tau_s": float(tau),
                    "min_gap_m": float(gap),
                    "status": "CASE_ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "capacity_gate_pass": False,
                })
                print(f"  CASE ERROR: {type(exc).__name__}: {exc}")

    passing = [
        r for r in results
        if r.get("status") != "CASE_ERROR" and r.get("capacity_gate_pass")
    ]
    pareto = _pareto_front(results)

    best_joint = None
    if passing:
        best_joint = min(
            passing,
            key=lambda r: r["joint_objective_if_capacity_pass"]
            if r["joint_objective_if_capacity_pass"] is not None
            else float("inf"),
        )

    report = {
        "sprint": "1E_A8_JOINT_LONGITUDINAL_GRID",
        "purpose": (
            "Test tau and minGap jointly because Krauss longitudinal capacity "
            "depends on both parameters; do not use minGap alone to force capacity."
        ),
        "observed_spacing_source": spacing_meta,
        "fixed_parameters": {
            "speed_multipliers": {"1": 0.90, "2": 0.90, "6": 0.90},
            "min_gap_lat_m": 0.60,
            "lc_pushy": 0.0,
            "step_length_s": 0.10,
            "eager_insert": True,
            "lane_mode": "best_prob",
            "depart_speed": "avg",
            "sampling_for_spacing_s": 0.50,
        },
        "grid": {
            "tau_s": TAU_VALUES_S,
            "min_gap_m": MIN_GAP_VALUES_M,
            "cases": total,
        },
        "capacity_gate": {
            "departed_fraction_min": 0.99,
            "p95_depart_delay_s_max": 5.0,
            "teleports_required": 0,
            "collisions_required": 0,
        },
        "results": results,
        "capacity_passing_cases": int(len(passing)),
        "pareto_front": pareto,
        "best_joint_candidate_if_any": best_joint,
        "decision_rules": [
            "Do not accept a candidate that fails the capacity gate regardless of fit score.",
            "Among capacity-passing candidates, inspect both speed and spacing distributions; the report exposes a Pareto front rather than hiding tradeoffs in one number.",
            "A low tau/minGap pair is not final simply because it passes. Check observed p10/median front-gap and time-headway fit.",
            "If no globally uniform pair passes with reasonable spacing fit, the next model should use heterogeneous tau/minGap by persona/continuous Behaviour DNA rather than pushing one global value lower.",
            "Validation period remains untouched.",
        ],
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a8_joint_longitudinal_grid.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
