from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


CANDIDATE_ENTRY_LINES_M = [5.0, 10.0, 15.0, 20.0, 30.0]
OCCUPANCY_SLICES_M = [20.0, 50.0, 100.0, 150.0, 200.0, 225.0]
PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}


def _load_clean(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path).copy()
    required = {
        "timestamp", "vehicle_id", "vehicle_class",
        "length", "width", "long_pos", "long_speed", "lat_pos"
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(d.columns)}")

    out = d[list(required)].copy()
    for c in ["timestamp", "vehicle_class", "length", "width", "long_pos", "long_speed", "lat_pos"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["timestamp", "vehicle_id", "vehicle_class", "length", "width", "long_pos"])
    out["vehicle_class"] = out["vehicle_class"].astype(int)
    return out.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)


def _quantiles(s: pd.Series | np.ndarray) -> dict:
    a = np.asarray(pd.Series(s).dropna(), dtype=float)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "p01": float(np.quantile(a, 0.01)),
        "p05": float(np.quantile(a, 0.05)),
        "p10": float(np.quantile(a, 0.10)),
        "p25": float(np.quantile(a, 0.25)),
        "median": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def _direction_audit(d: pd.DataFrame) -> dict:
    rows = []
    for vid, g in d.groupby("vehicle_id", sort=False):
        g = g.sort_values("timestamp")
        dx = np.diff(g["long_pos"].to_numpy(float))
        rows.append({
            "vehicle_id": vid,
            "vehicle_class": int(g["vehicle_class"].iloc[0]),
            "net_dx_m": float(g["long_pos"].iloc[-1] - g["long_pos"].iloc[0]),
            "median_step_dx_m": float(np.median(dx)) if dx.size else np.nan,
        })
    v = pd.DataFrame(rows)
    return {
        "vehicles": int(len(v)),
        "positive_net_dx_fraction": float((v["net_dx_m"] > 0).mean()),
        "negative_net_dx_fraction": float((v["net_dx_m"] < 0).mean()),
        "near_zero_net_dx_fraction": float((v["net_dx_m"].abs() < 1.0).mean()),
        "net_dx_m": _quantiles(v["net_dx_m"]),
        "median_step_dx_m": _quantiles(v["median_step_dx_m"]),
    }


def _first_last_audit(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    first = d.groupby("vehicle_id", as_index=False).first()
    last = d.groupby("vehicle_id", as_index=False).last()
    v = first[
        ["vehicle_id", "vehicle_class", "timestamp", "long_pos", "lat_pos", "length", "width"]
    ].rename(columns={
        "timestamp": "first_time_s",
        "long_pos": "first_x_m",
        "lat_pos": "first_y_m",
    })
    v = v.merge(
        last[["vehicle_id", "timestamp", "long_pos"]].rename(columns={
            "timestamp": "last_time_s",
            "long_pos": "last_x_m",
        }),
        on="vehicle_id",
        how="left",
    )
    t0 = float(d["timestamp"].min())
    v["first_time_offset_s"] = v["first_time_s"] - t0

    by_class = {}
    for cls, g in v.groupby("vehicle_class"):
        by_class[str(int(cls))] = {
            "name": PRIMARY_CLASSES.get(int(cls), f"CLASS_{int(cls)}"),
            "vehicles": int(len(g)),
            "first_x_m": _quantiles(g["first_x_m"]),
            "first_time_offset_s": _quantiles(g["first_time_offset_s"]),
            "last_x_m": _quantiles(g["last_x_m"]),
        }

    report = {
        "all_vehicles": int(len(v)),
        "first_x_m": _quantiles(v["first_x_m"]),
        "first_time_offset_s": _quantiles(v["first_time_offset_s"]),
        "last_x_m": _quantiles(v["last_x_m"]),
        "by_class": by_class,
    }
    return v, report


def _crossing_time(g: pd.DataFrame, line_m: float) -> float | None:
    g = g.sort_values("timestamp")
    x = g["long_pos"].to_numpy(float)
    t = g["timestamp"].to_numpy(float)

    if len(x) < 2:
        return None

    # Exact sample first.
    exact = np.where(np.isclose(x, line_m, atol=1e-9))[0]
    if exact.size:
        return float(t[exact[0]])

    # Increasing-direction crossing.
    idx = np.where((x[:-1] < line_m) & (x[1:] > line_m))[0]
    if idx.size == 0:
        return None

    i = int(idx[0])
    dx = x[i + 1] - x[i]
    if abs(dx) < 1e-9:
        return None
    frac = (line_m - x[i]) / dx
    return float(t[i] + frac * (t[i + 1] - t[i]))


def _burst_metrics(times: list[float], bin_s: float) -> dict:
    if len(times) < 2:
        return {"n": int(len(times))}
    a = np.sort(np.asarray(times, dtype=float))
    # Use fixed bins anchored to integer multiples of bin_s.
    keys = np.floor(a / bin_s).astype(int)
    _, counts = np.unique(keys, return_counts=True)
    duration = max(float(a[-1] - a[0]), bin_s)
    return {
        "n": int(len(a)),
        "duration_s": duration,
        "rate_veh_per_h": float(len(a) / duration * 3600.0),
        "max_arrivals_per_bin": int(np.max(counts)),
        "p95_arrivals_per_bin": float(np.quantile(counts, 0.95)),
        "mean_arrivals_per_nonempty_bin": float(np.mean(counts)),
    }


def _crossing_audit(d: pd.DataFrame, vehicle_table: pd.DataFrame, line_m: float) -> tuple[pd.DataFrame, dict]:
    crossings = []
    left_censored = 0
    never_reached = 0

    for vid, g in d.groupby("vehicle_id", sort=False):
        first_x = float(g["long_pos"].iloc[0])
        last_x = float(g["long_pos"].iloc[-1])

        ct = _crossing_time(g, line_m)
        if ct is not None:
            crossings.append({
                "vehicle_id": vid,
                "vehicle_class": int(g["vehicle_class"].iloc[0]),
                "crossing_time_s": ct,
            })
        elif first_x > line_m:
            left_censored += 1
        elif last_x < line_m:
            never_reached += 1

    c = pd.DataFrame(crossings)
    times = c["crossing_time_s"].tolist() if not c.empty else []
    inter = np.diff(np.sort(np.asarray(times, dtype=float))) if len(times) >= 2 else np.array([])

    report = {
        "entry_line_m": float(line_m),
        "vehicles_total": int(vehicle_table.shape[0]),
        "crossing_vehicles": int(len(c)),
        "crossing_fraction_all_vehicles": float(len(c) / len(vehicle_table)) if len(vehicle_table) else 0.0,
        "left_censored_first_seen_downstream": int(left_censored),
        "never_reached_line": int(never_reached),
        "crossing_time_s": _quantiles(c["crossing_time_s"]) if not c.empty else {},
        "interarrival_s": _quantiles(inter),
        "bursts_0p5s": _burst_metrics(times, 0.5),
        "bursts_1s": _burst_metrics(times, 1.0),
        "bursts_5s": _burst_metrics(times, 5.0),
        "class_counts": (
            {str(int(k)): int(v) for k, v in c["vehicle_class"].value_counts().sort_index().items()}
            if not c.empty else {}
        ),
    }
    return c, report


def _first_appearance_schedule_audit(v: pd.DataFrame) -> dict:
    # Mirror the Stage-A initial-stock rule: drop vehicles appearing within ~1 s.
    x = v[v["first_time_offset_s"] > 1.0].copy()
    times = x["first_time_s"].astype(float).tolist()
    inter = np.diff(np.sort(np.asarray(times, dtype=float))) if len(times) >= 2 else np.array([])
    return {
        "scheduled_vehicles": int(len(x)),
        "interarrival_s": _quantiles(inter),
        "bursts_0p5s": _burst_metrics(times, 0.5),
        "bursts_1s": _burst_metrics(times, 1.0),
        "bursts_5s": _burst_metrics(times, 5.0),
    }


def _occupied_lateral_envelope(d: pd.DataFrame) -> dict:
    left = d["lat_pos"].astype(float) - d["width"].astype(float) / 2.0
    right = d["lat_pos"].astype(float) + d["width"].astype(float) / 2.0
    # This is traffic occupancy, not a surveyed curb-to-curb road width.
    q_left = float(np.quantile(left, 0.01))
    q_right = float(np.quantile(right, 0.99))
    return {
        "lat_center_m": _quantiles(d["lat_pos"]),
        "vehicle_left_edge_m": _quantiles(left),
        "vehicle_right_edge_m": _quantiles(right),
        "robust_occupied_envelope_q01_to_q99_m": {
            "left_q01_m": q_left,
            "right_q99_m": q_right,
            "width_m": float(q_right - q_left),
        },
        "warning": "Occupied envelope is not an exact physical road-width measurement.",
    }


def _cross_section_occupancy(d: pd.DataFrame, slices_m: list[float]) -> tuple[pd.DataFrame, dict]:
    # Dataset longitudinal position is treated as vehicle front position.
    x = d.copy()
    x["rear_x_m"] = x["long_pos"] - x["length"]

    rows = []
    for slice_m in slices_m:
        hit = x[(x["rear_x_m"] <= slice_m) & (x["long_pos"] >= slice_m)]
        if hit.empty:
            continue
        counts = hit.groupby("timestamp").size()
        for t, count in counts.items():
            rows.append({"slice_m": float(slice_m), "timestamp": float(t), "vehicles_abreast": int(count)})

    occ = pd.DataFrame(rows)
    report = {}
    for slice_m in slices_m:
        s = occ.loc[occ["slice_m"] == slice_m, "vehicles_abreast"]
        if s.empty:
            report[str(slice_m)] = {"n_timestamps_with_vehicle": 0}
            continue
        report[str(slice_m)] = {
            "n_timestamps_with_vehicle": int(len(s)),
            "vehicles_abreast": _quantiles(s),
            "fraction_gt_3": float((s > 3).mean()),
            "fraction_ge_4": float((s >= 4).mean()),
            "fraction_ge_5": float((s >= 5).mean()),
        }

    if not occ.empty:
        all_s = occ["vehicles_abreast"]
        report["all_slices_combined"] = {
            "n_slice_timestamp_samples": int(len(all_s)),
            "vehicles_abreast": _quantiles(all_s),
            "fraction_gt_3": float((all_s > 3).mean()),
            "fraction_ge_4": float((all_s >= 4).mean()),
            "fraction_ge_5": float((all_s >= 5).mean()),
        }

    return occ, report


def run_audit(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    out_dir = project_root / "data" / "simulation" / "demand_geometry_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = _load_clean(processed / "calibration_clean.parquet")
    vehicles, first_last_report = _first_last_audit(d)

    crossing_reports = []
    crossing_tables = {}
    for line in CANDIDATE_ENTRY_LINES_M:
        table, rep = _crossing_audit(d, vehicles, line)
        crossing_reports.append(rep)
        crossing_tables[line] = table
        table.to_csv(out_dir / f"crossings_{int(line)}m.csv", index=False)

    occ, occ_report = _cross_section_occupancy(d, OCCUPANCY_SLICES_M)
    occ.to_csv(out_dir / "cross_section_occupancy.csv", index=False)

    # Choose a diagnostic line only by observed crossing coverage, not by SUMO fit.
    eligible = [r for r in crossing_reports if r["entry_line_m"] >= 10.0]
    recommended = max(
        eligible,
        key=lambda r: (
            r["crossing_fraction_all_vehicles"],
            -r["left_censored_first_seen_downstream"],
        ),
    ) if eligible else None

    report = {
        "sprint": "1E_A4_DEMAND_GEOMETRY_AUDIT",
        "purpose": (
            "Audit the observed demand clock and cross-section occupancy before any more SUMO tuning."
        ),
        "direction": _direction_audit(d),
        "trajectory_start_end": first_last_report,
        "stage_a_first_appearance_schedule": _first_appearance_schedule_audit(vehicles),
        "candidate_entry_lines": crossing_reports,
        "recommended_observed_entry_line": recommended,
        "lateral_occupied_envelope": _occupied_lateral_envelope(d),
        "cross_section_occupancy": occ_report,
        "decision_rules": [
            "If many vehicles are first seen well downstream of the entry line, stop using first appearance as departure time.",
            "If a clean 10-30 m crossing line retains high coverage, use interpolated crossing times as the demand clock.",
            "If >=4 vehicles are frequently abreast at a cross-section, a three-lane centerline abstraction is not enough by itself; SUMO must reproduce sublane packing before speed/headway can be frozen.",
            "If first appearances are already near the upstream boundary and >3-abreast occupancy is rare, investigate longitudinal parameters such as minGap and the exact site geometry instead.",
        ],
        "important": (
            "This audit uses calibration trajectories only and does not touch the validation period."
        ),
    }

    (processed / "sprint1e_a4_demand_geometry_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    vehicles.to_csv(out_dir / "vehicle_first_last.csv", index=False)
    return report
