from __future__ import annotations

import pandas as pd


def build_warnings(df: pd.DataFrame, median_sample_seconds: float | None, expected_sample_seconds: float = 0.5) -> list[str]:
    warnings: list[str] = []
    if median_sample_seconds is None:
        warnings.append("Could not estimate sampling interval.")
    elif abs(median_sample_seconds - expected_sample_seconds) > 0.1:
        warnings.append(
            f"Median sample interval is {median_sample_seconds:.3f}s; expected about {expected_sample_seconds:.3f}s. Verify timestamp units."
        )
    if (df["long_speed"] < 0).mean() > 0.01:
        warnings.append("More than 1% of longitudinal speed values are negative. Verify direction/sign convention.")
    if (df["length"] <= 0).any() or (df["width"] <= 0).any():
        warnings.append("Non-positive vehicle dimensions detected.")
    return warnings
