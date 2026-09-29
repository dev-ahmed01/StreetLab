from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


CORE_BEHAVIOR_METRICS = [
    "mean_speed_mps",
    "speed_std_mps",
    "mean_positive_accel_mps2",
    "mean_braking_mps2",
    "lateral_span_m",
    "mean_abs_lat_speed_mps",
    "median_front_gap_m",
    "p10_front_gap_m",
    "median_time_headway_s",
    "p10_time_headway_s",
    "median_min_side_clearance_m",
    "p10_min_side_clearance_m",
    "median_local_neighbor_count_20m",
]


@dataclass(frozen=True)
class SanityConfig:
    severe_side_overlap_m: float = -0.25
    implausibly_small_headway_s: float = 0.10
    very_large_headway_s: float = 20.0
    min_class_n_for_drift: int = 20
    ks_substantial: float = 0.20
    robust_delta_substantial: float = 0.50
    leader_fraction_split_tolerance: float = 0.05


def _finite(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x[np.isfinite(x)]


def _safe_fraction(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(mask.mean())


def summarize_interactions(interactions: pd.DataFrame, role: str, cfg: SanityConfig) -> dict:
    interior = interactions[interactions["analysis_interior"]].copy()
    leaders = interior[interior["leader_id"].notna()].copy()
    side = _finite(interior["min_side_clearance_m"])
    headway = _finite(leaders["time_headway_s"])
    ttc = _finite(leaders["ttc_s"])

    summary = {
        "role": role,
        "rows": int(len(interactions)),
        "interior_rows": int(len(interior)),
        "vehicles": int(interactions["vehicle_id"].nunique()),
        "timestamps": int(interactions["timestamp"].nunique()),
        "leader_fraction_interior": _safe_fraction(interior["leader_id"].notna()),
        "headway_available_fraction_of_leaders": float(len(headway) / len(leaders)) if len(leaders) else None,
        "ttc_available_fraction_of_leaders": float(len(ttc) / len(leaders)) if len(leaders) else None,
        "headway_le_0_fraction": _safe_fraction(headway <= 0) if len(headway) else None,
        "headway_lt_0_10s_fraction": _safe_fraction(headway < cfg.implausibly_small_headway_s) if len(headway) else None,
        "headway_gt_20s_fraction": _safe_fraction(headway > cfg.very_large_headway_s) if len(headway) else None,
        "ttc_le_0_fraction": _safe_fraction(ttc <= 0) if len(ttc) else None,
        "side_clearance_available_fraction": float(len(side) / len(interior)) if len(interior) else None,
        "negative_side_clearance_fraction": _safe_fraction(side < 0) if len(side) else None,
        "severe_side_overlap_fraction": _safe_fraction(side < cfg.severe_side_overlap_m) if len(side) else None,
        "median_headway_s": float(headway.median()) if len(headway) else None,
        "p10_headway_s": float(headway.quantile(0.10)) if len(headway) else None,
        "median_ttc_s": float(ttc.median()) if len(ttc) else None,
        "p10_ttc_s": float(ttc.quantile(0.10)) if len(ttc) else None,
        "median_side_clearance_m": float(side.median()) if len(side) else None,
        "p10_side_clearance_m": float(side.quantile(0.10)) if len(side) else None,
    }
    return summary


def behavior_profile(behavior: pd.DataFrame, role: str) -> pd.DataFrame:
    metrics = [m for m in CORE_BEHAVIOR_METRICS if m in behavior.columns]
    rows: list[dict] = []
    for cls, g in behavior.groupby("vehicle_class", sort=True):
        base = {
            "role": role,
            "vehicle_class": int(cls),
            "vehicles": int(g["vehicle_id"].nunique()),
        }
        for metric in metrics:
            x = _finite(g[metric])
            base[f"{metric}__available"] = int(len(x))
            base[f"{metric}__missing_fraction"] = float(1.0 - len(x) / len(g)) if len(g) else float("nan")
            base[f"{metric}__p10"] = float(x.quantile(0.10)) if len(x) else float("nan")
            base[f"{metric}__median"] = float(x.median()) if len(x) else float("nan")
            base[f"{metric}__p90"] = float(x.quantile(0.90)) if len(x) else float("nan")
        rows.append(base)
    return pd.DataFrame(rows)


def _robust_delta(a: pd.Series, b: pd.Series) -> float:
    med_a = float(a.median())
    med_b = float(b.median())
    iqr_a = float(a.quantile(0.75) - a.quantile(0.25))
    iqr_b = float(b.quantile(0.75) - b.quantile(0.25))
    scale = np.nanmedian([iqr_a, iqr_b])
    if not np.isfinite(scale) or scale <= 1e-12:
        pooled = pd.concat([a, b], ignore_index=True)
        scale = float(pooled.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 0.0 if np.isclose(med_a, med_b) else float("inf")
    return abs(med_a - med_b) / scale


def compare_splits(cal: pd.DataFrame, val: pd.DataFrame, cfg: SanityConfig) -> pd.DataFrame:
    metrics = [m for m in CORE_BEHAVIOR_METRICS if m in cal.columns and m in val.columns]
    classes = sorted(set(cal["vehicle_class"].dropna().astype(int)) & set(val["vehicle_class"].dropna().astype(int)))
    rows: list[dict] = []

    for cls in classes:
        c = cal[cal["vehicle_class"].astype(int) == cls]
        v = val[val["vehicle_class"].astype(int) == cls]
        for metric in metrics:
            a = _finite(c[metric])
            b = _finite(v[metric])
            enough = len(a) >= cfg.min_class_n_for_drift and len(b) >= cfg.min_class_n_for_drift
            if len(a) and len(b):
                ks = float(ks_2samp(a, b, alternative="two-sided", method="auto").statistic)
                rdelta = float(_robust_delta(a, b))
                med_a = float(a.median())
                med_b = float(b.median())
            else:
                ks = float("nan")
                rdelta = float("nan")
                med_a = float("nan")
                med_b = float("nan")
            substantial = bool(
                enough
                and np.isfinite(ks)
                and np.isfinite(rdelta)
                and ks >= cfg.ks_substantial
                and rdelta >= cfg.robust_delta_substantial
            )
            rows.append({
                "vehicle_class": cls,
                "metric": metric,
                "calibration_n": int(len(a)),
                "validation_n": int(len(b)),
                "calibration_median": med_a,
                "validation_median": med_b,
                "ks_statistic": ks,
                "robust_median_delta_iqr": rdelta,
                "enough_samples": enough,
                "substantial_drift": substantial,
            })
    return pd.DataFrame(rows)


def build_report(cal_inter: pd.DataFrame, val_inter: pd.DataFrame,
                 cal_beh: pd.DataFrame, val_beh: pd.DataFrame,
                 cfg: SanityConfig | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg = cfg or SanityConfig()
    cal_summary = summarize_interactions(cal_inter, "CALIBRATION", cfg)
    val_summary = summarize_interactions(val_inter, "VALIDATION", cfg)
    profile = pd.concat([
        behavior_profile(cal_beh, "CALIBRATION"),
        behavior_profile(val_beh, "VALIDATION"),
    ], ignore_index=True)
    drift = compare_splits(cal_beh, val_beh, cfg)

    warnings: list[str] = []
    lf_a = cal_summary["leader_fraction_interior"]
    lf_b = val_summary["leader_fraction_interior"]
    if lf_a is not None and lf_b is not None and abs(lf_a - lf_b) > cfg.leader_fraction_split_tolerance:
        warnings.append("Leader availability differs materially between calibration and validation splits.")

    for s in (cal_summary, val_summary):
        if s["headway_le_0_fraction"] not in (None, 0.0) and s["headway_le_0_fraction"] > 0.001:
            warnings.append(f"{s['role']}: non-positive time-headway observations detected.")
        if s["severe_side_overlap_fraction"] is not None and s["severe_side_overlap_fraction"] > 0.02:
            warnings.append(f"{s['role']}: more than 2% of side-clearance observations overlap by >0.25 m; inspect geometry/side-neighbour logic.")

    supported_classes = []
    for cls in sorted(set(cal_beh["vehicle_class"].astype(int)) & set(val_beh["vehicle_class"].astype(int))):
        ncal = int((cal_beh["vehicle_class"].astype(int) == cls).sum())
        nval = int((val_beh["vehicle_class"].astype(int) == cls).sum())
        if min(ncal, nval) >= cfg.min_class_n_for_drift:
            supported_classes.append(cls)

    drift_supported = drift[drift["enough_samples"]].copy()
    substantial_rows = drift_supported[drift_supported["substantial_drift"]]

    report = {
        "interaction_summary": {
            "calibration": cal_summary,
            "validation": val_summary,
        },
        "supported_vehicle_classes_for_distribution_tests": supported_classes,
        "drift_tests_with_enough_samples": int(len(drift_supported)),
        "substantial_drift_tests": int(len(substantial_rows)),
        "substantial_drift_fraction": float(len(substantial_rows) / len(drift_supported)) if len(drift_supported) else None,
        "top_substantial_drift": substantial_rows.sort_values(
            ["ks_statistic", "robust_median_delta_iqr"], ascending=False
        ).head(15).to_dict(orient="records"),
        "warnings": warnings,
        "config": cfg.__dict__,
    }
    return report, profile, drift


def save_report(report: dict, profile: pd.DataFrame, drift: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_dir / "behavior_profile_by_class.csv", index=False)
    drift.to_csv(output_dir / "calibration_validation_drift.csv", index=False)
    (output_dir / "sprint1b_sanity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
