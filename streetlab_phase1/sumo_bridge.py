from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd


PRIMARY_CLASSES = {
    1: "MOTORCYCLE",
    2: "CAR",
    6: "AUTO_RICKSHAW",
}

SUMO_VCLASS = {
    1: "motorcycle",
    2: "passenger",
    6: "passenger",
}


@dataclass
class MappingConfig:
    # These are SEEDS for the optimizer, not final calibrated values.
    base_tau_s: float = 1.20
    base_min_gap_lat_m: float = 0.60
    base_speed_factor: float = 1.00

    tau_min_s: float = 0.45
    tau_max_s: float = 2.50
    min_gap_lat_min_m: float = 0.15
    min_gap_lat_max_m: float = 1.20
    speed_factor_min: float = 0.70
    speed_factor_max: float = 1.35

    low_speed_threshold_mps: float = 1.00
    low_speed_min_n: int = 30
    low_speed_gap_floor_m: float = 0.20
    low_speed_gap_cap_m: float = 3.00

    # Lateral assertiveness is not directly mapped yet because its scale is
    # weak / small in Sprint 1C. Stage-C SUMO calibration will fit lcPushy.
    lc_pushy_seed: float = 0.0
    lc_sublane_seed: float = 1.0


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def load_profiles(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    required = {
        "vehicle_class", "vehicle_class_name", "persona", "calibration_share",
        "speed_preference__calibration_median",
        "spacing_preference__calibration_median",
        "side_clearance_preference__calibration_median",
        "lateral_assertiveness_proxy__calibration_median",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"persona_profiles.csv missing columns: {missing}")
    return d[d["vehicle_class"].isin(PRIMARY_CLASSES)].copy()


def physical_profiles(clean_path: str | Path) -> pd.DataFrame:
    d = pd.read_parquet(clean_path)
    required = {"vehicle_id", "vehicle_class", "length", "width", "long_speed"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"calibration_clean.parquet missing columns: {missing}")

    per_vehicle = (
        d.groupby("vehicle_id", as_index=False)
        .agg(
            vehicle_class=("vehicle_class", "first"),
            length_m=("length", "median"),
            width_m=("width", "median"),
            mean_speed_mps=("long_speed", "mean"),
            p95_speed_mps=("long_speed", lambda s: float(s.quantile(0.95))),
        )
    )

    out = (
        per_vehicle[per_vehicle["vehicle_class"].isin(PRIMARY_CLASSES)]
        .groupby("vehicle_class", as_index=False)
        .agg(
            vehicles=("vehicle_id", "size"),
            length_m=("length_m", "median"),
            width_m=("width_m", "median"),
            median_vehicle_mean_speed_mps=("mean_speed_mps", "median"),
            median_vehicle_p95_speed_mps=("p95_speed_mps", "median"),
        )
    )
    out["vehicle_class_name"] = out["vehicle_class"].map(PRIMARY_CLASSES)
    return out


def low_speed_gap_priors(interactions_path: str | Path, cfg: MappingConfig) -> pd.DataFrame:
    d = pd.read_parquet(interactions_path)
    required = {"vehicle_class", "long_speed", "front_gap_m"}
    missing = sorted(required - set(d.columns))
    if missing:
        # tolerate old naming if present
        if "front_gap" in d.columns:
            d = d.rename(columns={"front_gap": "front_gap_m"})
        missing = sorted(required - set(d.columns))
    if missing:
        return pd.DataFrame(columns=[
            "vehicle_class", "vehicle_class_name", "n",
            "p10_low_speed_front_gap_m", "median_low_speed_front_gap_m",
            "minGap_seed_m", "status"
        ])

    rows = []
    for cls, name in PRIMARY_CLASSES.items():
        x = d[
            (d["vehicle_class"] == cls)
            & d["long_speed"].notna()
            & d["front_gap_m"].notna()
            & (d["long_speed"] <= cfg.low_speed_threshold_mps)
            & (d["front_gap_m"] > 0)
        ]["front_gap_m"]
        n = int(x.size)
        if n >= cfg.low_speed_min_n:
            p10 = float(x.quantile(0.10))
            med = float(x.median())
            seed = _clip(p10, cfg.low_speed_gap_floor_m, cfg.low_speed_gap_cap_m)
            status = "DATA_DERIVED_SEED"
        else:
            p10 = float("nan")
            med = float("nan")
            seed = float("nan")
            status = "INSUFFICIENT_LOW_SPEED_DATA_DO_NOT_MAP"
        rows.append({
            "vehicle_class": cls,
            "vehicle_class_name": name,
            "n": n,
            "p10_low_speed_front_gap_m": p10,
            "median_low_speed_front_gap_m": med,
            "minGap_seed_m": seed,
            "status": status,
        })
    return pd.DataFrame(rows)


def persona_relative_vectors(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls, g in profiles.groupby("vehicle_class"):
        w = g["calibration_share"].to_numpy(float)
        w = w / w.sum()

        speed = g["speed_preference__calibration_median"].to_numpy(float)
        spacing = g["spacing_preference__calibration_median"].to_numpy(float)
        side = g["side_clearance_preference__calibration_median"].to_numpy(float)
        lateral = g["lateral_assertiveness_proxy__calibration_median"].to_numpy(float)

        speed_center = float(np.average(speed, weights=w))
        spacing_center = float(np.average(spacing, weights=w))
        side_center = float(np.average(side, weights=w))
        lateral_center = float(np.average(lateral, weights=w))

        lat_scale = float(np.quantile(np.abs(lateral - lateral_center), 0.75))
        if not np.isfinite(lat_scale) or lat_scale < 1e-6:
            lat_scale = 1.0

        for (_, r), sp, sg, sd, la in zip(g.iterrows(), speed, spacing, side, lateral):
            rows.append({
                "vehicle_class": int(cls),
                "vehicle_class_name": r["vehicle_class_name"],
                "persona": r["persona"],
                "calibration_share": float(r["calibration_share"]),
                "speed_pref_raw": float(sp),
                "spacing_pref_raw": float(sg),
                "side_clearance_pref_raw": float(sd),
                "lateral_proxy_raw": float(la),
                "speed_relative_to_class": float(sp / speed_center) if speed_center else 1.0,
                "spacing_relative_to_class": float(sg / spacing_center) if spacing_center else 1.0,
                "side_clearance_relative_to_class": float(sd / side_center) if side_center else 1.0,
                "lateral_score_centered": float((la - lateral_center) / lat_scale),
            })
    return pd.DataFrame(rows)


def seed_sumo_parameters(relative: pd.DataFrame, cfg: MappingConfig) -> pd.DataFrame:
    d = relative.copy()

    # IMPORTANT:
    # These convert only RELATIVE ordering into initial SUMO seeds.
    # They are not claims that observed ratios equal SUMO parameters.
    d["speedFactor_seed"] = [
        _clip(cfg.base_speed_factor * x, cfg.speed_factor_min, cfg.speed_factor_max)
        for x in d["speed_relative_to_class"]
    ]
    d["tau_seed_s"] = [
        _clip(cfg.base_tau_s * x, cfg.tau_min_s, cfg.tau_max_s)
        for x in d["spacing_relative_to_class"]
    ]
    d["minGapLat_seed_m"] = [
        _clip(cfg.base_min_gap_lat_m * x, cfg.min_gap_lat_min_m, cfg.min_gap_lat_max_m)
        for x in d["side_clearance_relative_to_class"]
    ]

    # Do not claim a direct lateral-proxy -> lcPushy mapping yet.
    d["lcPushy_seed"] = cfg.lc_pushy_seed
    d["lcSublane_seed"] = cfg.lc_sublane_seed

    # Search envelopes for staged calibration.
    d["speedFactor_search_lo"] = np.maximum(cfg.speed_factor_min, d["speedFactor_seed"] - 0.12)
    d["speedFactor_search_hi"] = np.minimum(cfg.speed_factor_max, d["speedFactor_seed"] + 0.12)

    d["tau_search_lo_s"] = np.maximum(cfg.tau_min_s, d["tau_seed_s"] * 0.70)
    d["tau_search_hi_s"] = np.minimum(cfg.tau_max_s, d["tau_seed_s"] * 1.30)

    d["minGapLat_search_lo_m"] = np.maximum(cfg.min_gap_lat_min_m, d["minGapLat_seed_m"] * 0.60)
    d["minGapLat_search_hi_m"] = np.minimum(cfg.min_gap_lat_max_m, d["minGapLat_seed_m"] * 1.40)

    # lcPushy is intentionally broad because Sprint 1C did not give us a
    # trustworthy absolute mapping.
    d["lcPushy_search_lo"] = 0.0
    d["lcPushy_search_hi"] = 0.80

    return d


def make_vtype_seed_xml(
    priors: pd.DataFrame,
    physical: pd.DataFrame,
    low_speed_gaps: pd.DataFrame,
) -> str:
    phys = physical.set_index("vehicle_class").to_dict("index")
    gaps = (
        low_speed_gaps.set_index("vehicle_class").to_dict("index")
        if not low_speed_gaps.empty else {}
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional>',
        '    <!-- StreetLab Phase 1D seed parameters. NOT final calibrated values. -->',
        '    <!-- Sublane behavior requires SUMO --lateral-resolution > 0. -->',
    ]
    for _, r in priors.iterrows():
        cls = int(r["vehicle_class"])
        p = phys.get(cls, {})
        g = gaps.get(cls, {})
        min_gap = g.get("minGap_seed_m", float("nan"))
        min_gap_attr = ""
        if isinstance(min_gap, (int, float)) and math.isfinite(min_gap):
            min_gap_attr = f' minGap="{min_gap:.4f}"'

        type_id = f'sl_{PRIMARY_CLASSES[cls].lower()}_{str(r["persona"]).lower()}'
        lines.append(
            '    <vType'
            f' id="{type_id}"'
            f' vClass="{SUMO_VCLASS[cls]}"'
            f' probability="{float(r["calibration_share"]):.6f}"'
            f' length="{float(p.get("length_m", 5.0)):.4f}"'
            f' width="{float(p.get("width_m", 2.0)):.4f}"'
            f' carFollowModel="Krauss"'
            f' laneChangeModel="SL2015"'
            f' speedFactor="{float(r["speedFactor_seed"]):.4f}"'
            f' tau="{float(r["tau_seed_s"]):.4f}"'
            f' minGapLat="{float(r["minGapLat_seed_m"]):.4f}"'
            f' lcPushy="{float(r["lcPushy_seed"]):.4f}"'
            f' lcSublane="{float(r["lcSublane_seed"]):.4f}"'
            f'{min_gap_attr}'
            '/>'
        )
    lines.append('</additional>')
    return "\n".join(lines) + "\n"


def detect_sumo() -> dict:
    out = {
        "eclipse_sumo_python_package": False,
        "sumolib_import": False,
        "traci_import": False,
        "sumo_home": None,
        "error": None,
    }
    try:
        import sumo  # type: ignore
        out["eclipse_sumo_python_package"] = True
        out["sumo_home"] = getattr(sumo, "SUMO_HOME", None)
    except Exception as e:
        out["error"] = f"import sumo failed: {e}"

    try:
        import sumolib  # noqa
        out["sumolib_import"] = True
    except Exception as e:
        if out["error"] is None:
            out["error"] = f"import sumolib failed: {e}"

    try:
        import traci  # noqa
        out["traci_import"] = True
    except Exception as e:
        if out["error"] is None:
            out["error"] = f"import traci failed: {e}"

    return out


def build_report(
    profiles: pd.DataFrame,
    relative: pd.DataFrame,
    priors: pd.DataFrame,
    physical: pd.DataFrame,
    gaps: pd.DataFrame,
    cfg: MappingConfig,
) -> dict:
    return {
        "sprint": "1D_SUMO_PARAMETER_BRIDGE",
        "status": "PARAMETER_PRIORS_ONLY_NOT_CALIBRATED",
        "mapping_principles": {
            "speed_preference": "relative ordering -> speedFactor seed; final value must be simulation-calibrated",
            "spacing_preference": "relative ordering -> tau seed; moving headway data supports tau better than minGap",
            "side_clearance_preference": "relative ordering -> minGapLat seed",
            "lateral_assertiveness_proxy": "NOT directly mapped yet; lcPushy kept broad for Stage-C calibration",
            "accel_decel": "excluded because Sprint 1B.6 showed split drift",
            "ttc": "excluded from personality; retained for safety validation",
            "minGap": "only data-derived if enough low-speed leader-gap observations exist",
        },
        "sumo_detection": detect_sumo(),
        "classes": sorted(int(x) for x in profiles["vehicle_class"].unique()),
        "persona_rows": int(len(priors)),
        "physical_profiles": physical.to_dict("records"),
        "low_speed_gap_priors": gaps.to_dict("records"),
        "config": asdict(cfg),
        "next_calibration_order": [
            "A: speedFactor -> match speed distributions",
            "B: tau (+ data-supported minGap if available) -> match headway/front-gap distributions",
            "C: minGapLat and lcPushy -> match side-clearance and lateral-motion distributions",
            "D: joint holdout validation on untouched 3:00-3:15 period",
        ],
    }
