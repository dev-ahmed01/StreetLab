from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

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

ENTRY_LINE_M = 55.0
MIN_GAP_CASES_M = [2.50, 2.00, 1.50, 1.00, 0.50]


def _q(a) -> dict:
    x = np.asarray(pd.Series(a).dropna(), dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def _boundary_headway_audit(demand: pd.DataFrame, tau_s: float) -> dict:
    by_band = {}
    all_gaps = []
    for band, g in demand.groupby("observed_lane_band"):
        t = np.sort(g["crossing_time_s"].to_numpy(float))
        gaps = np.diff(t)
        all_gaps.extend(gaps.tolist())
        by_band[str(int(band))] = {
            "vehicles": int(len(g)),
            "interarrival_s": _q(gaps),
            "fraction_below_tau": float((gaps < tau_s).mean()) if len(gaps) else None,
            "fraction_below_0p5s": float((gaps < 0.5).mean()) if len(gaps) else None,
            "fraction_below_1s": float((gaps < 1.0).mean()) if len(gaps) else None,
        }
    return {
        "tau_s_reference": float(tau_s),
        "all_observed_lane_bands_interarrival_s": _q(all_gaps),
        "by_lane_band": by_band,
    }


def _find_interaction_file(processed: Path) -> Path | None:
    preferred = [
        processed / "calibration_interactions.parquet",
        processed / "calibration_interaction.parquet",
        processed / "interactions_calibration.parquet",
    ]
    for p in preferred:
        if p.exists():
            return p
    candidates = sorted(processed.glob("*calib*interaction*.parquet"))
    if candidates:
        return candidates[0]
    candidates = sorted(processed.glob("*interaction*.parquet"))
    return candidates[0] if candidates else None


def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def _interaction_gap_audit(processed: Path, tau_s: float) -> dict:
    p = _find_interaction_file(processed)
    if p is None:
        return {
            "status": "INTERACTION_FILE_NOT_FOUND",
            "note": "Boundary-headway audit still ran; no empirical front-gap parquet was located.",
        }

    d = pd.read_parquet(p)
    gap_col = _pick_col(d, ["front_gap_m", "front_gap", "gap_m"])
    th_col = _pick_col(d, ["time_headway_s", "time_headway", "headway_s"])
    cls_col = _pick_col(d, ["vehicle_class", "vehicle_type", "class_id"])
    speed_col = _pick_col(
        d,
        ["long_speed", "long_speed_mps", "speed_mps", "follower_speed_mps", "speed"],
    )

    if gap_col is None:
        return {
            "status": "FRONT_GAP_COLUMN_NOT_FOUND",
            "file": str(p),
            "available_columns": [str(c) for c in d.columns],
        }

    x = pd.DataFrame({
        "front_gap_m": pd.to_numeric(d[gap_col], errors="coerce"),
    })
    if th_col:
        x["time_headway_s"] = pd.to_numeric(d[th_col], errors="coerce")
    if cls_col:
        x["vehicle_class"] = pd.to_numeric(d[cls_col], errors="coerce")
    if speed_col:
        x["speed_mps"] = pd.to_numeric(d[speed_col], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x[x["front_gap_m"].notna() & (x["front_gap_m"] >= 0)].copy()

    report = {
        "status": "OK",
        "file": str(p),
        "rows_with_nonnegative_front_gap": int(len(x)),
        "front_gap_m": _q(x["front_gap_m"]),
        "fraction_front_gap_below_2p5m": float((x["front_gap_m"] < 2.5).mean()) if len(x) else None,
        "fraction_front_gap_below_2p0m": float((x["front_gap_m"] < 2.0).mean()) if len(x) else None,
        "fraction_front_gap_below_1p5m": float((x["front_gap_m"] < 1.5).mean()) if len(x) else None,
        "fraction_front_gap_below_1p0m": float((x["front_gap_m"] < 1.0).mean()) if len(x) else None,
    }

    if "time_headway_s" in x.columns:
        valid = x["time_headway_s"].dropna()
        report["time_headway_s"] = _q(valid)
        report["fraction_time_headway_below_tau"] = (
            float((valid < tau_s).mean()) if len(valid) else None
        )

    if "speed_mps" in x.columns:
        valid = x.dropna(subset=["speed_mps"]).copy()
        valid["residual_gap_after_tau_m"] = (
            valid["front_gap_m"] - tau_s * valid["speed_mps"]
        )
        report["speed_mps"] = _q(valid["speed_mps"])
        report["front_gap_minus_tau_v_m"] = _q(valid["residual_gap_after_tau_m"])
        report["fraction_gap_below_tau_v_plus_2p5"] = float(
            (
                valid["front_gap_m"]
                < tau_s * valid["speed_mps"] + 2.5
            ).mean()
        ) if len(valid) else None

        # Low/moderate-speed evidence: useful for judging whether 2.5 m is
        # obviously too restrictive even when traffic is not stopped.
        speed_bins = {}
        for label, lo, hi in [
            ("0_to_1", 0.0, 1.0),
            ("1_to_3", 1.0, 3.0),
            ("3_to_6", 3.0, 6.0),
            ("6_plus", 6.0, np.inf),
        ]:
            g = valid[(valid["speed_mps"] >= lo) & (valid["speed_mps"] < hi)]
            speed_bins[label] = {
                "rows": int(len(g)),
                "front_gap_m": _q(g["front_gap_m"]),
            }
        report["front_gap_by_speed_bin"] = speed_bins

    if "vehicle_class" in x.columns:
        by_class = {}
        for cls, g in x.dropna(subset=["vehicle_class"]).groupby("vehicle_class"):
            by_class[str(int(cls))] = {
                "rows": int(len(g)),
                "front_gap_m": _q(g["front_gap_m"]),
                "fraction_below_2p5m": float((g["front_gap_m"] < 2.5).mean()),
            }
        report["by_vehicle_class"] = by_class

    return report


def _run_min_gap_case(
    project_root: Path,
    clean: pd.DataFrame,
    priors: pd.DataFrame,
    dna: pd.DataFrame | None,
    demand: pd.DataFrame,
    observed: pd.DataFrame,
    min_gap_m: float,
) -> dict:
    import sumolib
    import traci

    case = Case(
        name=f"GAP_{str(min_gap_m).replace('.', 'P')}M",
        step_length_s=0.10,
        eager_insert=True,
        lane_mode="best_prob",
        speed_mode="avg",
    )

    cfg = StageAConfig()
    cfg.default_tau_s = 0.72
    cfg.default_min_gap_m = float(min_gap_m)
    cfg.default_min_gap_lat_m = 0.60
    cfg.default_lc_pushy = 0.0
    cfg.step_length_s = 0.10

    outroot = project_root / "data" / "simulation" / "longitudinal_audit"
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
                if pos < 10.0 or pos > 180.0:
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
        "trajectory_rows": int(len(sim)),
    }

    strict_gate = bool(
        simulation["departed_fraction"] >= 0.99
        and simulation["p95_depart_delay_s"] is not None
        and simulation["p95_depart_delay_s"] <= 5.0
        and simulation["teleport_events"] == 0
        and simulation["collision_events"] == 0
    )

    result = {
        "min_gap_m": float(min_gap_m),
        "tau_s": 0.72,
        "speed_multipliers": {"1": 0.90, "2": 0.90, "6": 0.90},
        "boundary_mechanics": {
            "step_length_s": 0.10,
            "eager_insert": True,
            "lane_mode": "best_prob",
            "speed_mode": "avg",
        },
        "objective": float(objective),
        "strict_gate_pass": strict_gate,
        "simulation": simulation,
        "speed_fit": fit.to_dict("records"),
    }

    sim.to_parquet(rundir / "simulation_frames.parquet", index=False)
    fit.to_csv(rundir / "speed_fit.csv", index=False)
    (rundir / "run_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_longitudinal_audit(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    models = project_root / "data" / "models"

    clean = _canonical_clean_with_lat(processed / "calibration_clean.parquet")
    priors = load_persona_priors(models / "sumo_persona_priors.csv")
    dna = load_behavior_dna(processed / "calibration_behavior_dna.parquet")

    cfg = StageAConfig()
    demand = build_boundary_demand(clean, priors, dna, cfg)
    observed = observed_speed_targets(clean)

    empirical = {
        "boundary_lane_headways": _boundary_headway_audit(demand, tau_s=0.72),
        "interaction_front_gaps": _interaction_gap_audit(processed, tau_s=0.72),
    }

    results = []
    for gap in MIN_GAP_CASES_M:
        print(f"Running minGap={gap:.2f} m sensitivity...")
        try:
            r = _run_min_gap_case(
                project_root, clean, priors, dna, demand, observed, gap
            )
            results.append(r)
            print(
                f"  departed={r['simulation']['departed_fraction']:.3f}, "
                f"p95_delay={r['simulation']['p95_depart_delay_s']}, "
                f"collisions={r['simulation']['collision_events']}, "
                f"objective={r['objective']:.4f}, "
                f"gate={r['strict_gate_pass']}"
            )
        except Exception as exc:
            results.append({
                "min_gap_m": float(gap),
                "status": "CASE_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "strict_gate_pass": False,
            })
            print(f"  CASE ERROR: {type(exc).__name__}: {exc}")

    valid = [r for r in results if r.get("status") != "CASE_ERROR"]
    passing = [r for r in valid if r.get("strict_gate_pass", False)]

    report = {
        "sprint": "1E_A7_LONGITUDINAL_SPACING_AUDIT",
        "purpose": (
            "Test whether the remaining boundary-capacity mismatch is driven by "
            "SUMO's 2.5 m longitudinal minGap while holding the corrected observed "
            "55 m clock, speed targets, tau and boundary mechanics fixed."
        ),
        "why_now": (
            "A6 showed that smaller time steps, eager insertion, observed lane band "
            "and observed entry speed do not remove the long-delay tail. Boundary "
            "mechanics are no longer the leading explanation."
        ),
        "empirical_spacing_evidence": empirical,
        "fixed_parameters": {
            "speed_multipliers": {"1": 0.90, "2": 0.90, "6": 0.90},
            "tau_s": 0.72,
            "min_gap_lat_m": 0.60,
            "lc_pushy": 0.0,
            "step_length_s": 0.10,
            "eager_insert": True,
            "lane_mode": "best_prob",
            "speed_mode": "avg",
        },
        "min_gap_sensitivity_m": MIN_GAP_CASES_M,
        "results": results,
        "any_strict_gate_pass": bool(passing),
        "interpretation": [
            "If reducing minGap sharply removes the delay tail without collisions, the default 2.5 m longitudinal gap is a major structural mismatch for this mixed-traffic stream.",
            "If only extremely small minGap values pass, do not freeze them automatically; compare against empirical front-gap distributions and then jointly calibrate tau/minGap.",
            "If minGap reduction barely helps, tau or another longitudinal car-following assumption is the next target.",
            "Validation data remains untouched in this sprint.",
        ],
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a7_longitudinal_spacing_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
