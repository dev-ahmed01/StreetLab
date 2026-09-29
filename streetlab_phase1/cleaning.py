from __future__ import annotations

import numpy as np
import pandas as pd


def clean_trajectories(df: pd.DataFrame, *, drop_flagged_rows: bool = True) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(df)
    flagged_removed = 0
    out = df.copy()

    if drop_flagged_rows and "flag" in out.columns:
        mask = out["flag"].fillna(0).astype(float) != 0
        flagged_removed = int(mask.sum())
        out = out.loc[~mask].copy()

    out = out.dropna(subset=["timestamp", "vehicle_id", "vehicle_class", "long_pos", "lat_pos"])
    duplicate_count = int(out.duplicated(subset=["vehicle_id", "timestamp"]).sum())
    out = out.drop_duplicates(subset=["vehicle_id", "timestamp"], keep="first")
    out = out.sort_values(["vehicle_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    dts = out.groupby("vehicle_id", sort=False)["timestamp"].diff()
    positive_dts = dts[(dts > 0) & np.isfinite(dts)]
    median_dt = float(positive_dts.median()) if len(positive_dts) else None

    report = {
        "rows_raw": raw_rows,
        "rows_clean": len(out),
        "flagged_rows_removed": flagged_removed,
        "duplicate_vehicle_time_rows": duplicate_count,
        "unique_vehicles": int(out["vehicle_id"].nunique()),
        "median_sample_seconds": median_dt,
    }
    return out, report
