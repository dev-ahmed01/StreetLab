from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InteractionConfig:
    lateral_safety_margin_m: float = 0.20
    max_leader_distance_m: float = 50.0
    side_longitudinal_margin_m: float = 0.50
    local_density_radius_m: float = 20.0
    min_speed_for_headway_mps: float = 0.50
    min_closing_speed_for_ttc_mps: float = 0.10
    max_ttc_s: float = 60.0
    edge_buffer_m: float = 10.0


def _nanmin_pair(a: float, b: float) -> float:
    vals = [v for v in (a, b) if np.isfinite(v)]
    return min(vals) if vals else np.nan


def reconstruct_interactions(
    df: pd.DataFrame,
    config: InteractionConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Reconstruct local leader/follower and side-neighbour interactions.

    Coordinate convention for the Chennai data:
      * long_pos is the FRONT-centre longitudinal coordinate of a vehicle.
      * lat_pos is the lateral CENTRE coordinate.

    A candidate leader is ahead of the subject, within max_leader_distance_m,
    and its laterally expanded footprint overlaps the subject's footprint.
    The closest valid candidate is selected.
    """
    cfg = config or InteractionConfig()
    if df.empty:
        return pd.DataFrame(), {"rows": 0, "timestamps": 0, "warnings": ["empty input"]}

    needed = {
        "timestamp", "vehicle_id", "vehicle_class", "length", "width",
        "long_pos", "lat_pos", "long_speed", "lat_speed",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for interaction reconstruction: {', '.join(missing)}")

    x_min = float(df["long_pos"].min())
    x_max = float(df["long_pos"].max())
    interior_min = x_min + cfg.edge_buffer_m
    interior_max = x_max - cfg.edge_buffer_m

    out_rows: list[dict] = []
    negative_gap_candidates = 0
    candidate_pairs = 0
    impossible_side_overlaps = 0

    for ts, g in df.groupby("timestamp", sort=True):
        g = g.sort_values("vehicle_id", kind="mergesort").reset_index(drop=True)
        ids = g["vehicle_id"].to_numpy()
        classes = g["vehicle_class"].to_numpy()
        x = g["long_pos"].to_numpy(float)
        y = g["lat_pos"].to_numpy(float)
        length = g["length"].to_numpy(float)
        width = g["width"].to_numpy(float)
        vx = g["long_speed"].to_numpy(float)
        vy = g["lat_speed"].to_numpy(float)
        n = len(g)

        if n == 0:
            continue

        # Pair matrices are indexed [subject i, other j].
        dx = x[None, :] - x[:, None]  # positive => j is ahead of i
        dy = y[None, :] - y[:, None]  # positive => j is to the right of i (dataset axis)

        # Chennai longitudinal coordinate is the FRONT of the vehicle.
        # Gap from follower front to leader rear = leader_front - leader_length - follower_front.
        front_gap = dx - length[None, :]

        # Lateral footprints use centre coordinate +/- half-width, expanded by a small
        # safety margin used only for influence/leader identification.
        subj_left = (y - width / 2.0 - cfg.lateral_safety_margin_m)[:, None]
        subj_right = (y + width / 2.0 + cfg.lateral_safety_margin_m)[:, None]
        other_left = (y - width / 2.0 - cfg.lateral_safety_margin_m)[None, :]
        other_right = (y + width / 2.0 + cfg.lateral_safety_margin_m)[None, :]
        lateral_overlap = np.minimum(subj_right, other_right) - np.maximum(subj_left, other_left)

        eye = np.eye(n, dtype=bool)
        candidate = (
            (~eye)
            & (dx > 0.0)
            & (dx <= cfg.max_leader_distance_m)
            & (lateral_overlap > 0.0)
        )
        candidate_pairs += int(candidate.sum())
        negative_gap_candidates += int((candidate & (front_gap < 0)).sum())

        # For a physical following gap, require leader rear to be ahead of follower front.
        valid_leader = candidate & (front_gap >= 0.0)
        leader_gap_matrix = np.where(valid_leader, front_gap, np.inf)
        leader_idx = np.argmin(leader_gap_matrix, axis=1)
        leader_gap = leader_gap_matrix[np.arange(n), leader_idx]
        has_leader = np.isfinite(leader_gap)

        # Side neighbours: longitudinal vehicle footprints overlap (or are very close),
        # then pick the nearest lateral vehicle on each side.
        rear = x - length
        subj_rear = rear[:, None]
        subj_front = x[:, None]
        other_rear = rear[None, :]
        other_front = x[None, :]
        longitudinal_overlap = (
            np.minimum(subj_front, other_front)
            - np.maximum(subj_rear, other_rear)
        )
        side_candidate = (~eye) & (longitudinal_overlap >= -cfg.side_longitudinal_margin_m)

        # Edge-to-edge lateral clearance (negative means footprint overlap in the
        # axis-aligned approximation; retained for QA rather than silently clipped).
        lateral_clearance = np.abs(dy) - (width[:, None] + width[None, :]) / 2.0
        impossible_side_overlaps += int((side_candidate & (lateral_clearance < -0.25)).sum())

        left_mask = side_candidate & (dy < 0)
        right_mask = side_candidate & (dy > 0)
        left_dist = np.where(left_mask, lateral_clearance, np.inf)
        right_dist = np.where(right_mask, lateral_clearance, np.inf)
        left_idx = np.argmin(left_dist, axis=1)
        right_idx = np.argmin(right_dist, axis=1)
        left_gap = left_dist[np.arange(n), left_idx]
        right_gap = right_dist[np.arange(n), right_idx]
        has_left = np.isfinite(left_gap)
        has_right = np.isfinite(right_gap)

        # Local longitudinal density proxy: neighbours whose front positions are
        # within +/- radius. This is not a calibrated veh/km density measure.
        local = (~eye) & (np.abs(dx) <= cfg.local_density_radius_m)
        local_count = local.sum(axis=1)

        for i in range(n):
            lid = ids[leader_idx[i]] if has_leader[i] else None
            lclass = classes[leader_idx[i]] if has_leader[i] else None
            l_speed = vx[leader_idx[i]] if has_leader[i] else np.nan
            rel_speed = vx[i] - l_speed if has_leader[i] else np.nan

            if has_leader[i] and vx[i] > cfg.min_speed_for_headway_mps:
                time_headway = leader_gap[i] / vx[i]
            else:
                time_headway = np.nan

            if (
                has_leader[i]
                and leader_gap[i] > 0
                and rel_speed > cfg.min_closing_speed_for_ttc_mps
            ):
                ttc = leader_gap[i] / rel_speed
                if ttc > cfg.max_ttc_s:
                    ttc = np.nan
            else:
                ttc = np.nan

            lgap = float(left_gap[i]) if has_left[i] else np.nan
            rgap = float(right_gap[i]) if has_right[i] else np.nan
            min_side = _nanmin_pair(lgap, rgap)

            out_rows.append({
                "timestamp": float(ts),
                "vehicle_id": ids[i],
                "vehicle_class": classes[i],
                "long_pos": float(x[i]),
                "lat_pos": float(y[i]),
                "long_speed": float(vx[i]),
                "lat_speed": float(vy[i]),
                "analysis_interior": bool(interior_min <= x[i] <= interior_max),
                "leader_id": lid,
                "leader_class": lclass,
                "front_gap_m": float(leader_gap[i]) if has_leader[i] else np.nan,
                "relative_speed_mps": float(rel_speed) if has_leader[i] else np.nan,
                "time_headway_s": float(time_headway) if np.isfinite(time_headway) else np.nan,
                "ttc_s": float(ttc) if np.isfinite(ttc) else np.nan,
                "left_neighbor_id": ids[left_idx[i]] if has_left[i] else None,
                "left_clearance_m": lgap,
                "right_neighbor_id": ids[right_idx[i]] if has_right[i] else None,
                "right_clearance_m": rgap,
                "min_side_clearance_m": min_side,
                "local_neighbor_count_20m": int(local_count[i]),
            })

    out = pd.DataFrame(out_rows)
    interior = out[out["analysis_interior"]] if not out.empty else out

    report = {
        "rows": int(len(out)),
        "timestamps": int(out["timestamp"].nunique()) if len(out) else 0,
        "vehicles": int(out["vehicle_id"].nunique()) if len(out) else 0,
        "road_longitudinal_min_m": x_min,
        "road_longitudinal_max_m": x_max,
        "edge_buffer_m": cfg.edge_buffer_m,
        "interior_rows": int(len(interior)),
        "leader_observations_interior": int(interior["leader_id"].notna().sum()) if len(interior) else 0,
        "leader_fraction_interior": float(interior["leader_id"].notna().mean()) if len(interior) else None,
        "finite_headway_observations_interior": int(interior["time_headway_s"].notna().sum()) if len(interior) else 0,
        "finite_ttc_observations_interior": int(interior["ttc_s"].notna().sum()) if len(interior) else 0,
        "candidate_pairs": candidate_pairs,
        "negative_gap_candidate_pairs": negative_gap_candidates,
        "negative_gap_candidate_fraction": (negative_gap_candidates / candidate_pairs) if candidate_pairs else 0.0,
        "large_negative_side_overlap_pairs": impossible_side_overlaps,
        "config": cfg.__dict__,
        "warnings": [],
    }
    if report["leader_fraction_interior"] is not None and report["leader_fraction_interior"] < 0.10:
        report["warnings"].append("Very few leader observations; inspect lateral-overlap/influence settings.")
    if report["negative_gap_candidate_fraction"] > 0.05:
        report["warnings"].append("Many candidate leaders have negative geometric gaps; verify coordinate semantics.")
    return out, report


def aggregate_behavior_features(
    interactions: pd.DataFrame,
    vehicle_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate per-frame interactions into one row per vehicle.

    Only interior observations are used to reduce camera-boundary bias.
    """
    if interactions.empty:
        return pd.DataFrame()
    x = interactions[interactions["analysis_interior"]].copy()
    rows = []

    for vehicle_id, g in x.groupby("vehicle_id", sort=False):
        leader = g[g["leader_id"].notna()]
        headway = leader["time_headway_s"].dropna()
        front_gap = leader["front_gap_m"].dropna()
        ttc = leader["ttc_s"].dropna()
        side = g["min_side_clearance_m"].dropna()

        rows.append({
            "vehicle_id": vehicle_id,
            "vehicle_class": int(g["vehicle_class"].mode().iloc[0]),
            "interaction_samples": int(len(g)),
            "leader_samples": int(len(leader)),
            "leader_fraction": float(len(leader) / len(g)) if len(g) else np.nan,
            "median_front_gap_m": float(front_gap.median()) if len(front_gap) else np.nan,
            "p10_front_gap_m": float(front_gap.quantile(0.10)) if len(front_gap) else np.nan,
            "median_time_headway_s": float(headway.median()) if len(headway) else np.nan,
            "p10_time_headway_s": float(headway.quantile(0.10)) if len(headway) else np.nan,
            "min_ttc_s": float(ttc.min()) if len(ttc) else np.nan,
            "p10_ttc_s": float(ttc.quantile(0.10)) if len(ttc) else np.nan,
            "median_min_side_clearance_m": float(side.median()) if len(side) else np.nan,
            "p10_min_side_clearance_m": float(side.quantile(0.10)) if len(side) else np.nan,
            "median_local_neighbor_count_20m": float(g["local_neighbor_count_20m"].median()),
            "p90_local_neighbor_count_20m": float(g["local_neighbor_count_20m"].quantile(0.90)),
        })

    behavior = pd.DataFrame(rows)
    if vehicle_features is not None and not vehicle_features.empty:
        feature_cols = [c for c in vehicle_features.columns if c not in {"vehicle_class"}]
        behavior = behavior.merge(vehicle_features[feature_cols], on="vehicle_id", how="left", validate="one_to_one")
    return behavior
