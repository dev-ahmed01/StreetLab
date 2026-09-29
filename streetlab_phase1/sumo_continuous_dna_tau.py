from __future__ import annotations

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
    derive_class_stats,
)
from streetlab_phase1.sumo_boundary_fidelity import (
    _canonical_clean_with_lat,
    build_boundary_demand,
    observed_speed_targets,
    _write_network,
)
from streetlab_phase1.sumo_metric_parity import (
    reconstruct_interactions,
    _spacing_fit,
)

PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}

# Hard-stop calibration experiment:
# H0 is the A9.3 best global compromise.
# D1-D3 are bounded monotonic mappings from within-class DNA spacing percentile
# to SUMO tau. No free optimizer is used.
CASES = [
    {
        "name": "H0_GLOBAL_TAU_0P50",
        "mode": "global",
        "tau_s": 0.50,
    },
    {
        "name": "D1_DNA_NARROW_0P35_0P75",
        "mode": "dna_percentile",
        "tau_min_s": 0.35,
        "tau_max_s": 0.75,
    },
    {
        "name": "D2_DNA_MEDIUM_0P35_0P95",
        "mode": "dna_percentile",
        "tau_min_s": 0.35,
        "tau_max_s": 0.95,
    },
    {
        "name": "D3_DNA_WIDE_0P30_1P10",
        "mode": "dna_percentile",
        "tau_min_s": 0.30,
        "tau_max_s": 1.10,
    },
]

PROVISIONAL_MIN_GAP_M = 0.50
TAU_BIN_S = 0.05
STEP_LENGTH_S = 0.10
OBSERVED_X_MIN_M = 65.0
OBSERVED_X_MAX_M = 235.0


def _tau_type_token(tau_s: float) -> str:
    return f"{int(round(tau_s * 100)):03d}"


def _vclass_for(cls: int) -> str:
    return {1: "motorcycle", 2: "passenger", 6: "passenger"}.get(
        int(cls), "passenger"
    )


def _load_spacing_dna(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"DNA artifact not found: {path}")

    d = pd.read_parquet(path).copy()
    required = {"vehicle_id", "vehicle_class", "dna_spacing_preference"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            "A11 requires the continuous DNA recovered in A10.1. "
            f"Missing columns={sorted(missing)}, available={list(d.columns)}"
        )

    out = pd.DataFrame({
        "vehicle_id": d["vehicle_id"].astype(str),
        "vehicle_class": pd.to_numeric(d["vehicle_class"], errors="coerce"),
        "dna_spacing_preference": pd.to_numeric(
            d["dna_spacing_preference"], errors="coerce"
        ),
    }).dropna()

    out["vehicle_class"] = out["vehicle_class"].astype(int)
    out = out.drop_duplicates("vehicle_id").copy()

    # Rank within each vehicle class so the mapping is scale-free and robust
    # to the units of the DNA feature.
    out["dna_spacing_percentile"] = out.groupby("vehicle_class")[
        "dna_spacing_preference"
    ].rank(method="average", pct=True)

    return out


def _assign_tau(
    demand: pd.DataFrame,
    spacing_dna: pd.DataFrame,
    case: dict,
) -> pd.DataFrame:
    d = demand.copy()
    d["vehicle_id"] = d["vehicle_id"].astype(str)

    x = d.merge(
        spacing_dna,
        on=["vehicle_id", "vehicle_class"],
        how="left",
    )

    if case["mode"] == "global":
        x["tau_raw_s"] = float(case["tau_s"])
        x["tau_source"] = "GLOBAL"
    else:
        lo = float(case["tau_min_s"])
        hi = float(case["tau_max_s"])
        pct = x["dna_spacing_percentile"]

        # Higher observed DNA spacing preference -> larger tau.
        x["tau_raw_s"] = lo + (hi - lo) * pct
        x["tau_raw_s"] = x["tau_raw_s"].fillna(0.50)
        x["tau_source"] = np.where(
            pct.notna(),
            "DNA_PERCENTILE",
            "FALLBACK_GLOBAL_0P50",
        )

    # Quantize to keep the number of SUMO vTypes bounded while preserving the
    # monotonic DNA ordering.
    x["tau_s"] = (
        np.round(x["tau_raw_s"] / TAU_BIN_S) * TAU_BIN_S
    ).clip(lower=0.25, upper=1.20)

    return x


def _write_types(
    workdir: Path,
    cfg: StageAConfig,
    demand_tau: pd.DataFrame,
    clean: pd.DataFrame,
) -> Path:
    stats = derive_class_stats(clean).set_index("vehicle_class")
    path = workdir / "dna_tau_types.add.xml"

    combos = (
        demand_tau[
            ["vehicle_class", "persona", "tau_s"]
        ]
        .drop_duplicates()
        .sort_values(["vehicle_class", "persona", "tau_s"])
    )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional>',
        '  <!-- A11: tau varies only through the bounded DNA mapping. -->',
    ]

    for r in combos.itertuples(index=False):
        cls = int(r.vehicle_class)
        persona = str(r.persona)
        tau = float(r.tau_s)

        if cls in stats.index:
            length = float(stats.loc[cls, "length_m"])
            width = float(stats.loc[cls, "width_m"])
        else:
            length, width = 4.5, 1.8

        type_id = f"sl_c{cls}_{persona.lower()}_t{_tau_type_token(tau)}"

        lines.append(
            f'  <vType id="{type_id}" vClass="{_vclass_for(cls)}" '
            f'length="{length:.4f}" width="{width:.4f}" '
            f'carFollowModel="Krauss" laneChangeModel="SL2015" '
            f'tau="{tau:.4f}" minGap="{PROVISIONAL_MIN_GAP_M:.4f}" '
            f'minGapLat="{cfg.default_min_gap_lat_m:.4f}" '
            f'lcPushy="{cfg.default_lc_pushy:.4f}" '
            f'lcSublane="{cfg.default_lc_sublane:.4f}" '
            f'speedFactor="1.0" speedDev="0.0"/>'
        )

    lines.append("</additional>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_routes(
    workdir: Path,
    demand_tau: pd.DataFrame,
    speed_factors: dict[str, float],
) -> Path:
    path = workdir / "dna_tau.rou.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '  <route id="corridor" edges="E0"/>',
    ]

    for r in demand_tau.itertuples():
        vid = str(r.vehicle_id)
        cls = int(r.vehicle_class)
        persona = str(r.persona)
        tau = float(r.tau_s)
        sf = float(speed_factors.get(vid, 1.0))
        type_id = f"sl_c{cls}_{persona.lower()}_t{_tau_type_token(tau)}"

        lines.append(
            f'  <vehicle id="{vid}" type="{type_id}" route="corridor" '
            f'depart="{float(r.depart_s):.6f}" '
            f'departLane="best_prob" departPos="base" departSpeed="avg" '
            f'speedFactor="{sf:.6f}"/>'
        )

    lines.append("</routes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _tau_summary(demand_tau: pd.DataFrame) -> dict:
    rows = []
    for cls, name in PRIMARY_CLASSES.items():
        g = demand_tau[
            (demand_tau["vehicle_class"] == cls)
            & demand_tau["tau_s"].notna()
        ]
        if len(g):
            rows.append({
                "vehicle_class": cls,
                "vehicle_class_name": name,
                "vehicles": int(len(g)),
                "tau_min_s": float(g["tau_s"].min()),
                "tau_p10_s": float(g["tau_s"].quantile(0.10)),
                "tau_median_s": float(g["tau_s"].median()),
                "tau_p90_s": float(g["tau_s"].quantile(0.90)),
                "tau_max_s": float(g["tau_s"].max()),
                "dna_mapped_fraction": float(
                    (g["tau_source"] == "DNA_PERCENTILE").mean()
                ),
            })
    return {
        "by_class": rows,
        "overall_tau_min_s": float(demand_tau["tau_s"].min()),
        "overall_tau_median_s": float(demand_tau["tau_s"].median()),
        "overall_tau_max_s": float(demand_tau["tau_s"].max()),
        "unique_tau_bins": sorted(
            float(x) for x in demand_tau["tau_s"].dropna().unique()
        ),
    }


def _run_case(
    project_root: Path,
    clean: pd.DataFrame,
    priors: pd.DataFrame,
    dna_for_persona: pd.DataFrame | None,
    spacing_dna: pd.DataFrame,
    demand: pd.DataFrame,
    observed_speed: pd.DataFrame,
    observed_interactions: pd.DataFrame,
    case: dict,
) -> dict:
    import sumolib
    import traci

    cfg = StageAConfig()
    cfg.default_tau_s = 0.50  # only fallback / metadata; vTypes carry tau.
    cfg.default_min_gap_m = PROVISIONAL_MIN_GAP_M
    cfg.default_min_gap_lat_m = 0.60
    cfg.default_lc_pushy = 0.0
    cfg.step_length_s = STEP_LENGTH_S

    demand_tau = _assign_tau(demand, spacing_dna, case)

    outroot = project_root / "data" / "simulation" / "continuous_dna_tau"
    rundir = outroot / case["name"]
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    sumo_binary = sumolib.checkBinary("sumo")
    netconvert_binary = sumolib.checkBinary("netconvert")

    net = _write_network(rundir, cfg, netconvert_binary)
    types = _write_types(rundir, cfg, demand_tau, clean)

    speed_multipliers = {1: 0.90, 2: 0.90, 6: 0.90}
    speed_factors = make_vehicle_speed_factors(
        demand_tau,
        clean,
        priors,
        dna_for_persona,
        speed_multipliers,
        cfg,
    )
    routes = _write_routes(rundir, demand_tau, speed_factors)

    last_depart = float(demand_tau["depart_s"].max())
    end_s = last_depart + 90.0

    cmd = [
        sumo_binary,
        "-n", str(net),
        "-r", str(routes),
        "-a", str(types),
        "--step-length", str(STEP_LENGTH_S),
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

    scheduled = dict(
        zip(
            demand_tau["vehicle_id"].astype(str),
            demand_tau["depart_s"].astype(float),
        )
    )
    class_map = dict(
        zip(
            demand_tau["vehicle_id"].astype(str),
            demand_tau["vehicle_class"].astype(int),
        )
    )

    departed_at = {}
    frame_rows = []
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
                teleports += len(
                    traci.simulation.getStartingTeleportIDList()
                )
            except Exception:
                pass
            try:
                collisions += len(
                    traci.simulation.getCollidingVehiclesIDList()
                )
            except Exception:
                pass

            # Source trajectory cadence is 0.5 s.
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
                xy = traci.vehicle.getPosition(vid)

                frame_rows.append({
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

    frames = pd.DataFrame(frame_rows)

    interactions_all = reconstruct_interactions(
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

    # Sim x=0 corresponds to observed x=55.
    scored_interactions = interactions_all[
        interactions_all["vehicle_class"].isin(PRIMARY_CLASSES)
        & (interactions_all["follower_x_front_m"] >= 10.0)
        & (interactions_all["follower_x_front_m"] <= 180.0)
    ].copy()

    speed_frames = frames[
        frames["vehicle_class"].isin(PRIMARY_CLASSES)
        & (frames["x_front_m"] >= 10.0)
        & (frames["x_front_m"] <= 180.0)
    ][["time_s", "vehicle_id", "vehicle_class", "speed_mps"]].copy()

    speed_fit, speed_objective = evaluate_speed_fit(
        observed_speed,
        speed_frames,
    )
    spacing_fit, spacing_objective = _spacing_fit(
        observed_interactions,
        scored_interactions,
    )

    delays = np.asarray([
        float(actual - scheduled[vid])
        for vid, actual in departed_at.items()
        if vid in scheduled
    ])

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
        "interaction_rows_scored": int(len(scored_interactions)),
    }

    capacity_gate = bool(
        simulation["departed_fraction"] >= 0.99
        and simulation["p95_depart_delay_s"] is not None
        and simulation["p95_depart_delay_s"] <= 5.0
        and simulation["teleport_events"] == 0
        and simulation["collision_events"] == 0
    )

    demand_tau.to_csv(rundir / "vehicle_tau_assignments.csv", index=False)
    frames.to_parquet(rundir / "simulation_frames.parquet", index=False)
    scored_interactions.to_parquet(
        rundir / "simulation_interactions_canonical.parquet",
        index=False,
    )
    speed_fit.to_csv(rundir / "speed_fit.csv", index=False)

    result = {
        "case": case,
        "tau_distribution": _tau_summary(demand_tau),
        "capacity_gate_pass": capacity_gate,
        "simulation": simulation,
        "speed_objective": float(speed_objective),
        "speed_fit": speed_fit.to_dict("records"),
        "spacing_objective": float(spacing_objective),
        "spacing_fit": spacing_fit,
    }

    (rundir / "run_report.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _observed_interactions(clean: pd.DataFrame) -> pd.DataFrame:
    x = reconstruct_interactions(
        clean,
        time_col="time_s",
        vehicle_col="vehicle_id",
        class_col="vehicle_class",
        x_col="x_m",
        y_col="lat_pos",
        speed_col="speed_mps",
        length_col="length_m",
        width_col="width_m",
    )
    return x[
        x["vehicle_class"].isin(PRIMARY_CLASSES)
        & (x["follower_x_front_m"] >= OBSERVED_X_MIN_M)
        & (x["follower_x_front_m"] <= OBSERVED_X_MAX_M)
    ].copy()


def _select_model(results: list[dict]) -> dict:
    ok = [
        r for r in results
        if r.get("status") != "CASE_ERROR"
    ]
    baseline = next(
        (r for r in ok if r["case"]["name"] == "H0_GLOBAL_TAU_0P50"),
        None,
    )
    if baseline is None:
        return {
            "selected_case": None,
            "status": "NO_BASELINE_RESULT",
            "reason": "Global H0 baseline did not complete.",
        }

    if not baseline["capacity_gate_pass"]:
        return {
            "selected_case": None,
            "status": "BASELINE_CAPACITY_FAILURE",
            "reason": (
                "The previously passing H0 baseline no longer passes. "
                "Treat as implementation regression; do not calibrate further."
            ),
        }

    base_spacing = baseline["spacing_objective"]
    base_speed = baseline["speed_objective"]

    qualified = []
    for r in ok:
        if r["case"]["name"] == "H0_GLOBAL_TAU_0P50":
            continue
        if not r["capacity_gate_pass"]:
            continue

        spacing_improvement = (
            (base_spacing - r["spacing_objective"]) / base_spacing
            if base_spacing > 0 else 0.0
        )
        speed_change = (
            (r["speed_objective"] - base_speed) / base_speed
            if base_speed > 0 else 0.0
        )

        r["selection_diagnostics"] = {
            "spacing_improvement_fraction_vs_H0": float(spacing_improvement),
            "speed_objective_change_fraction_vs_H0": float(speed_change),
            "material_spacing_improvement": bool(spacing_improvement >= 0.10),
            "speed_guard_pass": bool(speed_change <= 0.05),
        }

        if spacing_improvement >= 0.10 and speed_change <= 0.05:
            qualified.append(r)

    if qualified:
        best = min(
            qualified,
            key=lambda r: (
                r["spacing_objective"],
                r["speed_objective"],
                r["simulation"]["p95_depart_delay_s"],
            ),
        )
        return {
            "selected_case": best["case"]["name"],
            "status": "DNA_MODEL_SELECTED",
            "reason": (
                "Capacity passed, canonical spacing objective improved by at least "
                "10% versus H0, and speed objective worsened by no more than 5%."
            ),
            "selection_diagnostics": best["selection_diagnostics"],
        }

    return {
        "selected_case": "H0_GLOBAL_TAU_0P50",
        "status": "FREEZE_BASELINE_AND_VALIDATE",
        "reason": (
            "No bounded DNA mapping achieved a material spacing improvement "
            "without violating the speed/capacity guards. Per the Phase-1 hard "
            "stop, do not launch another calibration search; freeze H0 and run "
            "held-out validation with the spacing limitation documented."
        ),
    }


def run_continuous_dna_tau(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    models = project_root / "data" / "models"

    clean = _canonical_clean_with_lat(
        processed / "calibration_clean.parquet"
    )
    priors = load_persona_priors(
        models / "sumo_persona_priors.csv"
    )
    dna_for_persona = load_behavior_dna(
        processed / "calibration_behavior_dna.parquet"
    )
    spacing_dna = _load_spacing_dna(
        processed / "calibration_behavior_dna.parquet"
    )

    cfg = StageAConfig()
    demand = build_boundary_demand(
        clean,
        priors,
        dna_for_persona,
        cfg,
    )

    observed_speed = observed_speed_targets(clean)
    observed_interactions = _observed_interactions(clean)

    results = []

    for idx, case in enumerate(CASES, start=1):
        print(f"[{idx}/{len(CASES)}] Running {case['name']}...")
        try:
            r = _run_case(
                project_root,
                clean,
                priors,
                dna_for_persona,
                spacing_dna,
                demand,
                observed_speed,
                observed_interactions,
                case,
            )
            results.append(r)
            print(
                f"  capacity={r['capacity_gate_pass']}, "
                f"p95_delay={r['simulation']['p95_depart_delay_s']:.3f}, "
                f"speed_obj={r['speed_objective']:.4f}, "
                f"spacing_obj={r['spacing_objective']:.4f}"
            )
        except Exception as exc:
            results.append({
                "case": case,
                "status": "CASE_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "capacity_gate_pass": False,
            })
            print(f"  CASE ERROR: {type(exc).__name__}: {exc}")

    selection = _select_model(results)

    report = {
        "sprint": "1E_A11_CONTINUOUS_DNA_TAU",
        "purpose": (
            "Final Phase-1 calibration experiment: test whether the empirically "
            "supported continuous DNA spacing preference can improve canonical "
            "spacing while retaining capacity and speed fit."
        ),
        "hard_stop": True,
        "dna_mapping": {
            "source": "data/processed/calibration_behavior_dna.parquet",
            "feature": "dna_spacing_preference",
            "mapping": (
                "within-class percentile rank -> bounded monotonic tau; "
                "higher spacing preference yields higher tau"
            ),
            "tau_quantization_s": TAU_BIN_S,
            "candidate_ranges": [
                {"name": c["name"], "tau_min_s": c.get("tau_min_s"), "tau_max_s": c.get("tau_max_s")}
                for c in CASES if c["mode"] == "dna_percentile"
            ],
        },
        "fixed_parameters": {
            "min_gap_m": PROVISIONAL_MIN_GAP_M,
            "speed_multipliers": {"1": 0.90, "2": 0.90, "6": 0.90},
            "min_gap_lat_m": 0.60,
            "lc_pushy": 0.0,
            "step_length_s": STEP_LENGTH_S,
            "eager_insert": True,
            "depart_lane": "best_prob",
            "depart_speed": "avg",
            "canonical_interaction_metric": {
                "lookahead_front_front_m": 50.0,
                "per_vehicle_lateral_safety_expansion_m": 0.20,
                "pairwise_body_clearance_threshold_m": 0.40,
            },
        },
        "results": results,
        "selection": selection,
        "phase1_next_step": (
            "Run untouched held-out validation once using selection.selected_case. "
            "No more calibration searches after this sprint."
        ),
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a11_continuous_dna_tau.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
