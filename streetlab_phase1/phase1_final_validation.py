from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import numpy as np
import pandas as pd

from streetlab_phase1.sumo_stage_a import (
    StageAConfig,
    load_persona_priors,
    make_vehicle_speed_factors,
    evaluate_speed_fit,
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
from streetlab_phase1.sumo_continuous_dna_tau import (
    _write_types,
    _write_routes,
    _assign_tau,
)

PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}

# ---------------------------------------------------------------------------
# FROZEN MODEL — selected by A11. Do not alter after validation is opened.
# ---------------------------------------------------------------------------
FROZEN_CASE = {
    "name": "H0_GLOBAL_TAU_0P50",
    "mode": "global",
    "tau_s": 0.50,
}
FROZEN_MIN_GAP_M = 0.50
FROZEN_SPEED_MULTIPLIERS = {1: 0.90, 2: 0.90, 6: 0.90}
FROZEN_MIN_GAP_LAT_M = 0.60
FROZEN_LC_PUSHY = 0.0
FROZEN_STEP_LENGTH_S = 0.10

OBSERVED_X_MIN_M = 65.0
OBSERVED_X_MAX_M = 235.0

# Calibration references from the already-frozen A11 H0 run.
CALIBRATION_SPEED_OBJECTIVE = 0.960820937308807
CALIBRATION_SPACING_OBJECTIVE = 0.8169023738890587

# Precommitted generalization guards. These are pragmatic holdout guards,
# not claims of universal transportation-model validity.
MAX_OBJECTIVE_DEGRADATION_FRACTION = 0.20

REPORT_PATH = Path("data/processed/phase1_final_heldout_validation.json")


def _observed_interactions(clean: pd.DataFrame) -> pd.DataFrame:
    all_interactions = reconstruct_interactions(
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
    return all_interactions[
        all_interactions["vehicle_class"].isin(PRIMARY_CLASSES)
        & (all_interactions["follower_x_front_m"] >= OBSERVED_X_MIN_M)
        & (all_interactions["follower_x_front_m"] <= OBSERVED_X_MAX_M)
    ].copy()


def _class_counts(clean: pd.DataFrame) -> dict:
    one = clean[["vehicle_id", "vehicle_class"]].drop_duplicates("vehicle_id")
    return {
        str(int(cls)): int(n)
        for cls, n in one["vehicle_class"].value_counts().sort_index().items()
    }


def _run_frozen_validation(
    project_root: Path,
    calibration_clean: pd.DataFrame,
    validation_clean: pd.DataFrame,
    priors: pd.DataFrame,
) -> dict:
    import sumolib
    import traci

    cfg = StageAConfig()
    cfg.default_tau_s = 0.50
    cfg.default_min_gap_m = FROZEN_MIN_GAP_M
    cfg.default_min_gap_lat_m = FROZEN_MIN_GAP_LAT_M
    cfg.default_lc_pushy = FROZEN_LC_PUSHY
    cfg.step_length_s = FROZEN_STEP_LENGTH_S

    # IMPORTANT:
    # - validation observations supply only the observed x=55 arrival clock,
    #   vehicle class, and post-run evaluation targets;
    # - persona assignment is drawn deterministically from CALIBRATION priors;
    # - speed-factor class references come from CALIBRATION clean data;
    # - no validation Behaviour-DNA file is loaded.
    demand = build_boundary_demand(
        validation_clean,
        priors,
        dna=None,
        cfg=cfg,
    )
    demand_tau = _assign_tau(
        demand,
        spacing_dna=pd.DataFrame(
            columns=[
                "vehicle_id",
                "vehicle_class",
                "dna_spacing_preference",
                "dna_spacing_percentile",
            ]
        ),
        case=FROZEN_CASE,
    )

    observed_speed = observed_speed_targets(validation_clean)
    observed_interactions = _observed_interactions(validation_clean)

    workdir = project_root / "data" / "simulation" / "phase1_final_validation"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    sumo_binary = sumolib.checkBinary("sumo")
    netconvert_binary = sumolib.checkBinary("netconvert")

    net = _write_network(workdir, cfg, netconvert_binary)

    # Freeze physical dimensions/types from CALIBRATION rather than recomputing
    # them from validation observations.
    types = _write_types(
        workdir,
        cfg,
        demand_tau,
        calibration_clean,
    )

    # Freeze desired-speed references from CALIBRATION.
    speed_factors = make_vehicle_speed_factors(
        demand_tau,
        calibration_clean,
        priors,
        dna=None,
        class_multipliers=FROZEN_SPEED_MULTIPLIERS,
        cfg=cfg,
    )
    routes = _write_routes(
        workdir,
        demand_tau,
        speed_factors,
    )

    last_depart = float(demand_tau["depart_s"].max())
    end_s = last_depart + 90.0

    cmd = [
        sumo_binary,
        "-n", str(net),
        "-r", str(routes),
        "-a", str(types),
        "--step-length", str(FROZEN_STEP_LENGTH_S),
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

                pos = float(traci.vehicle.getLanePosition(vid))
                xy = traci.vehicle.getPosition(vid)

                rows.append({
                    "time_s": now,
                    "vehicle_id": svid,
                    "vehicle_class": int(cls),
                    "x_front_m": pos,
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

    # Sim x=0 corresponds to observed x=55; observed score window 65..235
    # therefore maps to sim 10..180.
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

    total_validation_vehicles = int(
        validation_clean["vehicle_id"].astype(str).nunique()
    )
    boundary_coverage = (
        len(demand_tau) / total_validation_vehicles
        if total_validation_vehicles
        else 0.0
    )

    simulation = {
        "validation_total_vehicles": total_validation_vehicles,
        "boundary_crossing_vehicles": int(len(demand_tau)),
        "boundary_crossing_coverage": float(boundary_coverage),
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
        boundary_coverage >= 0.99
        and simulation["departed_fraction"] >= 0.99
        and simulation["p95_depart_delay_s"] is not None
        and simulation["p95_depart_delay_s"] <= 5.0
        and simulation["teleport_events"] == 0
        and simulation["collision_events"] == 0
    )

    speed_limit = (
        CALIBRATION_SPEED_OBJECTIVE
        * (1.0 + MAX_OBJECTIVE_DEGRADATION_FRACTION)
    )
    spacing_limit = (
        CALIBRATION_SPACING_OBJECTIVE
        * (1.0 + MAX_OBJECTIVE_DEGRADATION_FRACTION)
    )

    speed_generalization_guard = bool(
        float(speed_objective) <= speed_limit
    )
    spacing_generalization_guard = bool(
        float(spacing_objective) <= spacing_limit
    )

    holdout_generalization_pass = bool(
        capacity_gate
        and speed_generalization_guard
        and spacing_generalization_guard
    )

    # This distinction is intentional:
    # generalization can pass even though the frozen model has a known absolute
    # spacing bias. We must not rewrite that limitation into a "good fit" claim.
    phase1_status = (
        "CLOSED_HELDOUT_GENERALIZATION_PASS_WITH_KNOWN_SPACING_LIMITATION"
        if holdout_generalization_pass
        else "CLOSED_HELDOUT_GENERALIZATION_FAIL_NO_RETUNING"
    )

    frames.to_parquet(
        workdir / "validation_simulation_frames.parquet",
        index=False,
    )
    scored_interactions.to_parquet(
        workdir / "validation_simulation_interactions.parquet",
        index=False,
    )
    speed_fit.to_csv(
        workdir / "validation_speed_fit.csv",
        index=False,
    )

    return {
        "frozen_model": {
            "selected_from_a11": "H0_GLOBAL_TAU_0P50",
            "tau_s": 0.50,
            "min_gap_m": FROZEN_MIN_GAP_M,
            "speed_multipliers": {
                str(k): v for k, v in FROZEN_SPEED_MULTIPLIERS.items()
            },
            "min_gap_lat_m": FROZEN_MIN_GAP_LAT_M,
            "lc_pushy": FROZEN_LC_PUSHY,
            "step_length_s": FROZEN_STEP_LENGTH_S,
            "eager_insert": True,
            "depart_lane": "best_prob",
            "depart_speed": "avg",
            "persona_source": (
                "deterministic assignment from calibration persona shares; "
                "no validation Behaviour-DNA used"
            ),
            "speed_reference_source": "calibration_clean.parquet",
            "physical_profile_source": "calibration_clean.parquet",
        },
        "validation_dataset": {
            "file": "data/processed/validation_clean.parquet",
            "vehicle_counts_by_class": _class_counts(validation_clean),
            "observed_speed_target_rows": int(len(observed_speed)),
            "observed_interaction_rows": int(len(observed_interactions)),
        },
        "simulation": simulation,
        "capacity_gate_pass": capacity_gate,
        "speed_objective": float(speed_objective),
        "speed_fit": speed_fit.to_dict("records"),
        "spacing_objective": float(spacing_objective),
        "spacing_fit": spacing_fit,
        "precommitted_generalization_guards": {
            "calibration_speed_objective": CALIBRATION_SPEED_OBJECTIVE,
            "validation_speed_objective_max": float(speed_limit),
            "speed_guard_pass": speed_generalization_guard,
            "calibration_spacing_objective": CALIBRATION_SPACING_OBJECTIVE,
            "validation_spacing_objective_max": float(spacing_limit),
            "spacing_guard_pass": spacing_generalization_guard,
            "max_relative_degradation": MAX_OBJECTIVE_DEGRADATION_FRACTION,
            "note": (
                "These guards test holdout degradation relative to the frozen "
                "calibration model. They do not erase the known absolute spacing bias."
            ),
        },
        "holdout_generalization_pass": holdout_generalization_pass,
        "phase1_status": phase1_status,
        "known_limitations": [
            (
                "Calibration H0 materially under-reproduced observed median "
                "front gaps and time headways even though capacity passed."
            ),
            (
                "This held-out test evaluates whether that frozen behavior "
                "generalizes without substantial further degradation; it does "
                "not claim perfect microscopic spacing fidelity."
            ),
            (
                "Only one Chennai mid-block site and two short time windows "
                "have been evaluated; no citywide or universal-India claim is supported."
            ),
        ],
        "retuning_allowed": False,
    }


def run_final_validation(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    report_path = project_root / REPORT_PATH

    # One-shot protection: after a successful completed validation report exists,
    # this script refuses to rerun. A code crash before report creation remains
    # recoverable.
    if report_path.exists():
        raise RuntimeError(
            "Held-out validation has already been opened and completed. "
            "Per the Phase-1 hard stop, do not rerun or retune. "
            f"Existing report: {report_path}"
        )

    calibration_path = (
        project_root / "data" / "processed" / "calibration_clean.parquet"
    )
    validation_path = (
        project_root / "data" / "processed" / "validation_clean.parquet"
    )
    priors_path = (
        project_root / "data" / "models" / "sumo_persona_priors.csv"
    )

    for p in (calibration_path, validation_path, priors_path):
        if not p.exists():
            raise FileNotFoundError(f"Required frozen artifact missing: {p}")

    calibration_clean = _canonical_clean_with_lat(calibration_path)
    validation_clean = _canonical_clean_with_lat(validation_path)
    priors = load_persona_priors(priors_path)

    result = _run_frozen_validation(
        project_root,
        calibration_clean,
        validation_clean,
        priors,
    )

    report = {
        "sprint": "PHASE1_FINAL_HELDOUT_VALIDATION",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_locked": True,
        "validation_was_untouched_before_this_run": True,
        "validation_is_now_unblinded": True,
        "selection_source": {
            "a11_selected_case": "H0_GLOBAL_TAU_0P50",
            "a11_status": "FREEZE_BASELINE_AND_VALIDATE",
        },
        **result,
        "phase1_closed": True,
        "next_project_phase": (
            "Phase 2 — StreetLab product workflow: Study Contract, "
            "ObservationPackage, data-sufficiency gate, Scenario Lab, "
            "simulation comparison, verification/evidence record."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
