from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class ContextConfig:
    local_speed_radius_m: float = 20.0
    min_stream_speed_mps: float = 0.5
    leader_relative_speed_threshold_mps: float = 0.5
    min_context_cell_n: int = 30
    severe_side_overlap_m: float = -0.25
    min_class_n_for_drift: int = 100
    ks_substantial: float = 0.20
    robust_delta_substantial: float = 0.50


CONTEXT_METRICS = [
    "median_speed_delta_to_stream_mps",
    "median_speed_ratio_to_stream",
    "median_headway_s",
    "p10_headway_s",
    "median_headway_context_ratio",
    "median_front_gap_context_ratio",
    "median_side_clearance_width_ratio",
    "median_side_clearance_context_ratio",
    "median_accel_excess_mps2",
    "median_braking_excess_mps2",
    "median_lateral_activity_excess_mps",
]


def _quantile_thresholds(s: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    return float(x.quantile(1 / 3)), float(x.quantile(2 / 3))


def _regime_from_thresholds(s: pd.Series, low: float, high: float, labels=("LOW", "MEDIUM", "HIGH")) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    valid = np.isfinite(x)
    out.loc[valid & (x <= low)] = labels[0]
    out.loc[valid & (x > low) & (x <= high)] = labels[1]
    out.loc[valid & (x > high)] = labels[2]
    return out


def add_local_stream_speed(interactions: pd.DataFrame, radius_m: float = 20.0) -> pd.DataFrame:
    """Add a local median stream-speed estimate using neighbours within +/- radius.

    This uses only observed trajectories at the same timestamp. The subject is excluded.
    """
    out = interactions.copy()
    stream = np.full(len(out), np.nan, dtype=float)
    counts = np.zeros(len(out), dtype=int)

    # preserve original row locations while operating per timestamp
    for _, idx in out.groupby("timestamp", sort=False).groups.items():
        locs = np.asarray(list(idx), dtype=int)
        g = out.loc[locs]
        x = g["long_pos"].to_numpy(float)
        v = g["long_speed"].to_numpy(float)
        n = len(g)
        if n <= 1:
            continue
        dx = np.abs(x[None, :] - x[:, None])
        mask = (dx <= radius_m) & (~np.eye(n, dtype=bool)) & np.isfinite(v)[None, :]
        for i in range(n):
            vals = v[mask[i]]
            if len(vals):
                stream[locs[i]] = float(np.median(vals))
                counts[locs[i]] = int(len(vals))

    out["local_stream_speed_mps"] = stream
    out["local_stream_vehicle_count"] = counts
    return out


def merge_motion_fields(interactions: pd.DataFrame, clean: pd.DataFrame, vehicle_features: pd.DataFrame) -> pd.DataFrame:
    needed = ["vehicle_id", "timestamp", "long_accel", "lat_accel"]
    motion = clean[needed].copy()
    out = interactions.merge(motion, on=["vehicle_id", "timestamp"], how="left", validate="one_to_one")

    phys_cols = [c for c in ["vehicle_id", "width_m", "length_m"] if c in vehicle_features.columns]
    if "vehicle_id" in phys_cols and len(phys_cols) > 1:
        out = out.merge(vehicle_features[phys_cols], on="vehicle_id", how="left", validate="many_to_one")
    return out


def derive_thresholds(calibration_frames: pd.DataFrame) -> dict:
    interior = calibration_frames[calibration_frames["analysis_interior"]].copy()
    density = _quantile_thresholds(interior["local_neighbor_count_20m"])
    stream = _quantile_thresholds(interior["local_stream_speed_mps"])
    gap = _quantile_thresholds(interior["front_gap_m"])
    return {
        "density_neighbor_count_20m": {"q33": density[0], "q67": density[1]},
        "local_stream_speed_mps": {"q33": stream[0], "q67": stream[1]},
        "front_gap_m": {"q33": gap[0], "q67": gap[1]},
    }


def apply_context_labels(frames: pd.DataFrame, thresholds: dict, cfg: ContextConfig) -> pd.DataFrame:
    out = frames.copy()
    d = thresholds["density_neighbor_count_20m"]
    s = thresholds["local_stream_speed_mps"]
    g = thresholds["front_gap_m"]
    out["density_regime"] = _regime_from_thresholds(out["local_neighbor_count_20m"], d["q33"], d["q67"])
    out["stream_speed_regime"] = _regime_from_thresholds(out["local_stream_speed_mps"], s["q33"], s["q67"])

    gap_regime = _regime_from_thresholds(out["front_gap_m"], g["q33"], g["q67"], labels=("NEAR", "MID", "FAR"))
    gap_regime = gap_regime.fillna("NO_LEADER")
    out["front_gap_regime"] = gap_regime

    rel = pd.to_numeric(out["relative_speed_mps"], errors="coerce")
    leader_state = pd.Series("NO_LEADER", index=out.index, dtype="string")
    finite = np.isfinite(rel)
    leader_state.loc[finite & (rel > cfg.leader_relative_speed_threshold_mps)] = "CLOSING"
    leader_state.loc[finite & (rel < -cfg.leader_relative_speed_threshold_mps)] = "OPENING"
    leader_state.loc[finite & (rel >= -cfg.leader_relative_speed_threshold_mps) & (rel <= cfg.leader_relative_speed_threshold_mps)] = "STEADY"
    out["leader_state"] = leader_state

    stream_v = pd.to_numeric(out["local_stream_speed_mps"], errors="coerce")
    own_v = pd.to_numeric(out["long_speed"], errors="coerce")
    out["speed_delta_to_stream_mps"] = own_v - stream_v
    ratio = own_v / stream_v.where(stream_v >= cfg.min_stream_speed_mps)
    out["speed_ratio_to_stream"] = ratio.replace([np.inf, -np.inf], np.nan)

    width = pd.to_numeric(out.get("width_m"), errors="coerce")
    side = pd.to_numeric(out["min_side_clearance_m"], errors="coerce")
    valid_side = np.isfinite(side) & np.isfinite(width) & (width > 0) & (side >= 0)
    out["side_clearance_width_ratio"] = np.where(valid_side, side / width, np.nan)
    out["side_overlap_excluded"] = np.isfinite(side) & (side < 0)

    return out


def _make_baseline_tables(cal: pd.DataFrame, value_col: str, positive_only: bool = False, braking_only: bool = False, min_n: int = 30):
    x = cal.copy()
    vals = pd.to_numeric(x[value_col], errors="coerce")
    if positive_only:
        x = x[np.isfinite(vals) & (vals > 0)].copy()
        x["__value"] = pd.to_numeric(x[value_col], errors="coerce")
    elif braking_only:
        x = x[np.isfinite(vals) & (vals < 0)].copy()
        x["__value"] = -pd.to_numeric(x[value_col], errors="coerce")
    else:
        x = x[np.isfinite(vals)].copy()
        x["__value"] = pd.to_numeric(x[value_col], errors="coerce")

    levels = [
        ["vehicle_class", "density_regime", "stream_speed_regime", "leader_state", "front_gap_regime"],
        ["vehicle_class", "density_regime", "stream_speed_regime", "leader_state"],
        ["vehicle_class", "density_regime", "stream_speed_regime"],
        ["vehicle_class", "density_regime"],
        ["vehicle_class"],
    ]
    tables = []
    for keys in levels:
        t = x.groupby(keys, dropna=False)["__value"].agg(["median", "size"]).reset_index()
        t = t[t["size"] >= min_n]
        mapping = {tuple(row[k] for k in keys): float(row["median"]) for _, row in t.iterrows()}
        tables.append((keys, mapping))
    return tables


def _lookup_baseline(row: pd.Series, tables) -> float:
    for keys, mapping in tables:
        key = tuple(row[k] for k in keys)
        if key in mapping:
            return mapping[key]
    return np.nan


def add_calibration_baselines(cal: pd.DataFrame, val: pd.DataFrame, cfg: ContextConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create calibration-only context baselines and apply them to both splits."""
    cal_out = cal.copy()
    val_out = val.copy()

    specs = {
        "expected_positive_accel_mps2": ("long_accel", True, False),
        "expected_braking_mps2": ("long_accel", False, True),
        "expected_abs_lat_speed_mps": ("lat_speed", False, False),
        "expected_headway_s": ("time_headway_s", False, False),
        "expected_front_gap_m": ("front_gap_m", False, False),
        "expected_side_ratio": ("side_clearance_width_ratio", False, False),
    }

    baseline_meta = {}
    for output_col, (value_col, positive_only, braking_only) in specs.items():
        source = cal_out.copy()
        if output_col == "expected_abs_lat_speed_mps":
            source[value_col] = pd.to_numeric(source[value_col], errors="coerce").abs()
        tables = _make_baseline_tables(
            source, value_col,
            positive_only=positive_only,
            braking_only=braking_only,
            min_n=cfg.min_context_cell_n,
        )
        for target in (cal_out, val_out):
            target[output_col] = target.apply(lambda r: _lookup_baseline(r, tables), axis=1)
        baseline_meta[output_col] = {
            "levels": [keys for keys, _ in tables],
            "cells_per_level": [len(mapping) for _, mapping in tables],
        }

    accel = pd.to_numeric(cal_out["long_accel"], errors="coerce")
    cal_out["accel_excess_mps2"] = np.where(accel > 0, accel - cal_out["expected_positive_accel_mps2"], np.nan)
    cal_out["braking_excess_mps2"] = np.where(accel < 0, (-accel) - cal_out["expected_braking_mps2"], np.nan)
    val_accel = pd.to_numeric(val_out["long_accel"], errors="coerce")
    val_out["accel_excess_mps2"] = np.where(val_accel > 0, val_accel - val_out["expected_positive_accel_mps2"], np.nan)
    val_out["braking_excess_mps2"] = np.where(val_accel < 0, (-val_accel) - val_out["expected_braking_mps2"], np.nan)

    for target in (cal_out, val_out):
        abs_lat = pd.to_numeric(target["lat_speed"], errors="coerce").abs()
        target["lateral_activity_excess_mps"] = abs_lat - target["expected_abs_lat_speed_mps"]
        target["headway_context_ratio"] = pd.to_numeric(target["time_headway_s"], errors="coerce") / target["expected_headway_s"].replace(0, np.nan)
        target["front_gap_context_ratio"] = pd.to_numeric(target["front_gap_m"], errors="coerce") / target["expected_front_gap_m"].replace(0, np.nan)
        target["side_clearance_context_ratio"] = target["side_clearance_width_ratio"] / target["expected_side_ratio"].replace(0, np.nan)
        target.replace([np.inf, -np.inf], np.nan, inplace=True)

    return cal_out, val_out, baseline_meta


def aggregate_context_behavior(frames: pd.DataFrame) -> pd.DataFrame:
    x = frames[frames["analysis_interior"]].copy()
    rows = []
    for vid, g in x.groupby("vehicle_id", sort=False):
        def med(col):
            z = pd.to_numeric(g[col], errors="coerce")
            z = z[np.isfinite(z)]
            return float(z.median()) if len(z) else np.nan
        def q10(col):
            z = pd.to_numeric(g[col], errors="coerce")
            z = z[np.isfinite(z)]
            return float(z.quantile(.10)) if len(z) else np.nan
        rows.append({
            "vehicle_id": vid,
            "vehicle_class": int(g["vehicle_class"].mode().iloc[0]),
            "context_samples": int(len(g)),
            "median_speed_delta_to_stream_mps": med("speed_delta_to_stream_mps"),
            "median_speed_ratio_to_stream": med("speed_ratio_to_stream"),
            "median_headway_s": med("time_headway_s"),
            "p10_headway_s": q10("time_headway_s"),
            "median_headway_context_ratio": med("headway_context_ratio"),
            "median_front_gap_context_ratio": med("front_gap_context_ratio"),
            "median_side_clearance_width_ratio": med("side_clearance_width_ratio"),
            "median_side_clearance_context_ratio": med("side_clearance_context_ratio"),
            "median_accel_excess_mps2": med("accel_excess_mps2"),
            "median_braking_excess_mps2": med("braking_excess_mps2"),
            "median_lateral_activity_excess_mps": med("lateral_activity_excess_mps"),
        })
    return pd.DataFrame(rows)


def _finite(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x[np.isfinite(x)]


def _robust_delta(a: pd.Series, b: pd.Series) -> float:
    med_a, med_b = float(a.median()), float(b.median())
    scales = [float(a.quantile(.75)-a.quantile(.25)), float(b.quantile(.75)-b.quantile(.25))]
    scale = float(np.nanmedian(scales))
    if not np.isfinite(scale) or scale <= 1e-12:
        pooled = pd.concat([a,b], ignore_index=True)
        scale = float(pooled.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 0.0 if np.isclose(med_a, med_b) else float("inf")
    return abs(med_a-med_b)/scale


def compare_context_behavior(cal: pd.DataFrame, val: pd.DataFrame, cfg: ContextConfig) -> pd.DataFrame:
    classes = sorted(set(cal.vehicle_class.astype(int)) & set(val.vehicle_class.astype(int)))
    rows=[]
    for cls in classes:
        ca=cal[cal.vehicle_class.astype(int)==cls]
        va=val[val.vehicle_class.astype(int)==cls]
        for metric in CONTEXT_METRICS:
            if metric not in ca.columns or metric not in va.columns:
                continue
            a,b=_finite(ca[metric]),_finite(va[metric])
            enough=min(len(a),len(b))>=cfg.min_class_n_for_drift
            if len(a) and len(b):
                ks=float(ks_2samp(a,b).statistic)
                rd=float(_robust_delta(a,b))
                ma,mb=float(a.median()),float(b.median())
            else:
                ks=rd=ma=mb=float("nan")
            rows.append({
                "vehicle_class":cls,"metric":metric,
                "calibration_n":len(a),"validation_n":len(b),
                "calibration_median":ma,"validation_median":mb,
                "ks_statistic":ks,"robust_median_delta_iqr":rd,
                "enough_samples":enough,
                "substantial_drift":bool(enough and np.isfinite(ks) and np.isfinite(rd) and ks>=cfg.ks_substantial and rd>=cfg.robust_delta_substantial),
            })
    return pd.DataFrame(rows)


def build_context_report(cal_frames: pd.DataFrame, val_frames: pd.DataFrame,
                         cal_beh: pd.DataFrame, val_beh: pd.DataFrame,
                         thresholds: dict, baselines: dict, drift: pd.DataFrame,
                         cfg: ContextConfig) -> dict:
    supported = drift[drift["enough_samples"]].copy()
    substantial = supported[supported["substantial_drift"]]
    primary = supported[supported["vehicle_class"].isin([1,2,6])]
    primary_sub = primary[primary["substantial_drift"]]
    return {
        "thresholds_derived_from": "CALIBRATION_ONLY",
        "context_thresholds": thresholds,
        "baseline_tables": baselines,
        "frame_coverage": {
            "calibration_rows": int(len(cal_frames)),
            "validation_rows": int(len(val_frames)),
            "calibration_local_stream_available_fraction": float(cal_frames["local_stream_speed_mps"].notna().mean()),
            "validation_local_stream_available_fraction": float(val_frames["local_stream_speed_mps"].notna().mean()),
            "calibration_side_overlap_excluded_fraction": float(cal_frames["side_overlap_excluded"].mean()),
            "validation_side_overlap_excluded_fraction": float(val_frames["side_overlap_excluded"].mean()),
        },
        "behavior_vehicles": {"calibration": int(len(cal_beh)), "validation": int(len(val_beh))},
        "drift_tests_with_enough_samples": int(len(supported)),
        "substantial_drift_tests": int(len(substantial)),
        "substantial_drift_fraction": float(len(substantial)/len(supported)) if len(supported) else None,
        "primary_classes_1_2_6_tests": int(len(primary)),
        "primary_classes_1_2_6_substantial_drift": int(len(primary_sub)),
        "primary_classes_1_2_6_drift_fraction": float(len(primary_sub)/len(primary)) if len(primary) else None,
        "top_substantial_drift": substantial.sort_values(["ks_statistic","robust_median_delta_iqr"], ascending=False).head(15).to_dict("records"),
        "config": asdict(cfg),
    }
