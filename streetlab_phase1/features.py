from __future__ import annotations

import numpy as np
import pandas as pd


def _positive_mean(s: pd.Series) -> float:
    x = s[s > 0]
    return float(x.mean()) if len(x) else 0.0


def _negative_mean_abs(s: pd.Series) -> float:
    x = s[s < 0]
    return float((-x).mean()) if len(x) else 0.0


def extract_vehicle_features(df: pd.DataFrame, minimum_track_seconds: float = 3.0) -> pd.DataFrame:
    rows = []
    for vehicle_id, g in df.groupby("vehicle_id", sort=False):
        g = g.sort_values("timestamp")
        duration = float(g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]) if len(g) > 1 else 0.0
        if duration < minimum_track_seconds:
            continue

        speed = g["long_speed"].astype(float)
        accel = g["long_accel"].astype(float)
        lat_speed = g["lat_speed"].astype(float)
        lat_accel = g["lat_accel"].astype(float)
        lat_pos = g["lat_pos"].astype(float)

        rows.append({
            "vehicle_id": vehicle_id,
            "vehicle_class": int(g["vehicle_class"].mode().iloc[0]) if len(g["vehicle_class"].mode()) else g["vehicle_class"].iloc[0],
            "length_m": float(g["length"].median()),
            "width_m": float(g["width"].median()),
            "duration_s": duration,
            "samples": int(len(g)),
            "mean_speed_mps": float(speed.mean()),
            "median_speed_mps": float(speed.median()),
            "speed_std_mps": float(speed.std(ddof=0)),
            "speed_p05_mps": float(speed.quantile(0.05)),
            "speed_p95_mps": float(speed.quantile(0.95)),
            "mean_positive_accel_mps2": _positive_mean(accel),
            "mean_braking_mps2": _negative_mean_abs(accel),
            "accel_std_mps2": float(accel.std(ddof=0)),
            "lateral_span_m": float(lat_pos.max() - lat_pos.min()),
            "mean_abs_lat_speed_mps": float(lat_speed.abs().mean()),
            "p95_abs_lat_speed_mps": float(lat_speed.abs().quantile(0.95)),
            "mean_abs_lat_accel_mps2": float(lat_accel.abs().mean()),
        })
    return pd.DataFrame(rows)


def class_profile(features: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "duration_s", "mean_speed_mps", "speed_std_mps",
        "mean_positive_accel_mps2", "mean_braking_mps2",
        "lateral_span_m", "mean_abs_lat_speed_mps"
    ]
    grouped = features.groupby("vehicle_class")[metrics].agg(["count", "mean", "median", "std"])
    grouped.columns = [f"{a}__{b}" for a, b in grouped.columns]
    return grouped.reset_index()
