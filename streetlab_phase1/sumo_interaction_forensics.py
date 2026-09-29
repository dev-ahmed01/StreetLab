from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}
SCORE_X_MIN_M = 65.0
SCORE_X_MAX_M = 235.0

CANDIDATES = [
    # The current A9.1 rule, retained as a reference.
    {
        "name": "F0_NO_LOOKAHEAD_CAP_MARGIN_0P20",
        "lateral_margin_m": 0.20,
        "lookahead_mode": "none",
        "lookahead_m": None,
    },
    # Main forensic hypothesis: Sprint-1B likely used ~50 m candidate lookahead.
    {
        "name": "F1_FRONT_FRONT_CAP_50M_MARGIN_0P20",
        "lateral_margin_m": 0.20,
        "lookahead_mode": "front_front",
        "lookahead_m": 50.0,
    },
    {
        "name": "F2_NET_GAP_CAP_50M_MARGIN_0P20",
        "lateral_margin_m": 0.20,
        "lookahead_mode": "net_gap",
        "lookahead_m": 50.0,
    },
    # Bracket the likely cutoff.
    {
        "name": "F3_FRONT_FRONT_CAP_40M_MARGIN_0P20",
        "lateral_margin_m": 0.20,
        "lookahead_mode": "front_front",
        "lookahead_m": 40.0,
    },
    {
        "name": "F4_FRONT_FRONT_CAP_60M_MARGIN_0P20",
        "lateral_margin_m": 0.20,
        "lookahead_mode": "front_front",
        "lookahead_m": 60.0,
    },
    # Check whether the 0.20 m lateral margin itself was implemented differently.
    {
        "name": "F5_FRONT_FRONT_CAP_50M_MARGIN_0P00",
        "lateral_margin_m": 0.00,
        "lookahead_mode": "front_front",
        "lookahead_m": 50.0,
    },
    {
        "name": "F6_FRONT_FRONT_CAP_50M_MARGIN_0P40",
        "lateral_margin_m": 0.40,
        "lookahead_mode": "front_front",
        "lookahead_m": 50.0,
    },
]


def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lut = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lut:
            return lut[n.lower()]
    return None


def _q(a) -> dict:
    x = np.asarray(pd.Series(a).dropna(), dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def _load_clean(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path).copy()
    required = [
        "timestamp", "vehicle_id", "vehicle_class",
        "length", "width", "long_pos", "long_speed", "lat_pos",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(
            f"Clean parquet missing {missing}. Available={list(d.columns)}"
        )

    out = pd.DataFrame({
        "time_s": pd.to_numeric(d["timestamp"], errors="coerce"),
        "vehicle_id": d["vehicle_id"].astype(str),
        "vehicle_class": pd.to_numeric(d["vehicle_class"], errors="coerce"),
        "length_m": pd.to_numeric(d["length"], errors="coerce"),
        "width_m": pd.to_numeric(d["width"], errors="coerce"),
        "x_front_m": pd.to_numeric(d["long_pos"], errors="coerce"),
        "speed_mps": pd.to_numeric(d["long_speed"], errors="coerce"),
        "y_center_m": pd.to_numeric(d["lat_pos"], errors="coerce"),
    }).dropna()

    out["vehicle_class"] = out["vehicle_class"].astype(int)
    return out.sort_values(["time_s", "vehicle_id"]).reset_index(drop=True)


def _load_legacy(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(path).copy()

    cols = {
        "time": _pick_col(raw, ["timestamp", "time_s", "time", "t"]),
        "vehicle_id": _pick_col(raw, ["vehicle_id", "vehicle_number", "follower_id"]),
        "class": _pick_col(raw, ["vehicle_class", "vehicle_type", "class_id"]),
        "leader_id": _pick_col(raw, ["leader_id", "front_vehicle_id"]),
        "front_gap": _pick_col(raw, ["front_gap_m", "front_gap", "gap_m"]),
        "headway": _pick_col(raw, ["time_headway_s", "time_headway", "headway_s"]),
        "long_pos": _pick_col(raw, ["long_pos", "long_position", "x_m", "long_distance"]),
        "side_clearance": _pick_col(raw, ["side_clearance_m", "side_clearance"]),
    }

    if cols["front_gap"] is None or cols["headway"] is None or cols["class"] is None:
        raise ValueError(
            "Legacy interaction parquet does not expose required gap/headway/class columns. "
            f"Detected={cols}; available={list(raw.columns)}"
        )

    out = pd.DataFrame({
        "vehicle_class": pd.to_numeric(raw[cols["class"]], errors="coerce"),
        "front_gap_m": pd.to_numeric(raw[cols["front_gap"]], errors="coerce"),
        "time_headway_s": pd.to_numeric(raw[cols["headway"]], errors="coerce"),
    })

    if cols["time"] is not None:
        out["time_s"] = pd.to_numeric(raw[cols["time"]], errors="coerce")
    if cols["vehicle_id"] is not None:
        out["vehicle_id"] = raw[cols["vehicle_id"]].astype(str)
    if cols["leader_id"] is not None:
        out["leader_id"] = raw[cols["leader_id"]].astype(str)
    if cols["long_pos"] is not None:
        out["follower_x_front_m"] = pd.to_numeric(raw[cols["long_pos"]], errors="coerce")
    if cols["side_clearance"] is not None:
        out["side_clearance_m"] = pd.to_numeric(raw[cols["side_clearance"]], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[
        out["vehicle_class"].isin(PRIMARY_CLASSES)
        & out["front_gap_m"].notna()
        & (out["front_gap_m"] >= 0)
        & out["time_headway_s"].notna()
        & (out["time_headway_s"] >= 0)
    ].copy()
    out["vehicle_class"] = out["vehicle_class"].astype(int)

    spatial_filter_applied = False
    if "follower_x_front_m" in out.columns:
        m = (
            (out["follower_x_front_m"] >= SCORE_X_MIN_M)
            & (out["follower_x_front_m"] <= SCORE_X_MAX_M)
        )
        if int(m.sum()) >= 5000:
            out = out[m].copy()
            spatial_filter_applied = True

    meta = {
        "file": str(path),
        "detected_columns": {k: (str(v) if v is not None else None) for k, v in cols.items()},
        "rows_after_filters": int(len(out)),
        "spatial_filter_applied": spatial_filter_applied,
        "legacy_front_gap": _q(out["front_gap_m"]),
        "legacy_time_headway": _q(out["time_headway_s"]),
    }
    return out, meta


def _reconstruct(clean: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    margin = float(cfg["lateral_margin_m"])
    mode = cfg["lookahead_mode"]
    cap = cfg["lookahead_m"]

    rows = []

    for t, g in clean.groupby("time_s", sort=True):
        if len(g) < 2:
            continue

        vids = g["vehicle_id"].astype(str).to_numpy()
        cls = g["vehicle_class"].to_numpy(int)
        x = g["x_front_m"].to_numpy(float)
        y = g["y_center_m"].to_numpy(float)
        speed = g["speed_mps"].to_numpy(float)
        length = g["length_m"].to_numpy(float)
        width = g["width_m"].to_numpy(float)

        dx_front = x[None, :] - x[:, None]
        dy = np.abs(y[None, :] - y[:, None])
        side_clearance = dy - (width[:, None] + width[None, :]) / 2.0

        # Candidate leaders must be ahead in front-bumper coordinate and
        # laterally overlapping/near-overlapping.
        candidate = (dx_front > 0.0) & (side_clearance <= margin)
        np.fill_diagonal(candidate, False)

        # Net front gap follower i -> leader j.
        net_gap = x[None, :] - length[None, :] - x[:, None]

        if mode == "front_front":
            candidate &= dx_front <= float(cap)
        elif mode == "net_gap":
            candidate &= net_gap <= float(cap)
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown lookahead mode: {mode}")

        for i in range(len(g)):
            js = np.flatnonzero(candidate[i])
            if js.size == 0:
                continue

            j = int(js[np.argmin(dx_front[i, js])])
            fg = float(net_gap[i, j])
            th = float(fg / speed[i]) if speed[i] > 0.05 and fg >= 0 else np.nan

            rows.append({
                "time_s": float(t),
                "vehicle_id": vids[i],
                "vehicle_class": int(cls[i]),
                "leader_id": vids[j],
                "leader_class": int(cls[j]),
                "follower_x_front_m": float(x[i]),
                "leader_x_front_m": float(x[j]),
                "front_front_m": float(dx_front[i, j]),
                "front_gap_m": fg,
                "time_headway_s": th,
                "side_clearance_m": float(side_clearance[i, j]),
            })

    d = pd.DataFrame(rows)
    if d.empty:
        return d

    # Score only primary followers in the same spatial window used by A8/A9.
    d = d[
        d["vehicle_class"].isin(PRIMARY_CLASSES)
        & (d["follower_x_front_m"] >= SCORE_X_MIN_M)
        & (d["follower_x_front_m"] <= SCORE_X_MAX_M)
        & (d["front_gap_m"] >= 0)
        & d["time_headway_s"].notna()
        & (d["time_headway_s"] >= 0)
    ].copy()
    return d


def _class_distribution_summary(d: pd.DataFrame) -> list[dict]:
    rows = []
    for cls, name in PRIMARY_CLASSES.items():
        g = d[d["vehicle_class"] == cls]
        rows.append({
            "vehicle_class": cls,
            "vehicle_class_name": name,
            "rows": int(len(g)),
            "front_gap_m": _q(g["front_gap_m"]),
            "time_headway_s": _q(g["time_headway_s"]),
        })
    return rows


def _distribution_error(legacy: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    rows = []
    rel_errors = []

    for cls, name in PRIMARY_CLASSES.items():
        a = legacy[legacy["vehicle_class"] == cls]
        b = candidate[candidate["vehicle_class"] == cls]

        gm_a = float(a["front_gap_m"].median()) if len(a) else np.nan
        gm_b = float(b["front_gap_m"].median()) if len(b) else np.nan
        hm_a = float(a["time_headway_s"].median()) if len(a) else np.nan
        hm_b = float(b["time_headway_s"].median()) if len(b) else np.nan

        gap_rel = abs(gm_b - gm_a) / gm_a if np.isfinite(gm_a) and gm_a > 0 else np.nan
        th_rel = abs(hm_b - hm_a) / hm_a if np.isfinite(hm_a) and hm_a > 0 else np.nan

        for x in (gap_rel, th_rel):
            if np.isfinite(x):
                rel_errors.append(float(x))

        rows.append({
            "vehicle_class": cls,
            "vehicle_class_name": name,
            "legacy_rows": int(len(a)),
            "candidate_rows": int(len(b)),
            "legacy_gap_median": gm_a,
            "candidate_gap_median": gm_b,
            "gap_relative_median_error": float(gap_rel) if np.isfinite(gap_rel) else None,
            "legacy_headway_median": hm_a,
            "candidate_headway_median": hm_b,
            "headway_relative_median_error": float(th_rel) if np.isfinite(th_rel) else None,
        })

    return {
        "max_relative_median_error": max(rel_errors) if rel_errors else None,
        "mean_relative_median_error": float(np.mean(rel_errors)) if rel_errors else None,
        "within_10pct_all_medians": bool(rel_errors) and max(rel_errors) <= 0.10,
        "by_class": rows,
    }


def _row_level_parity(legacy: pd.DataFrame, cand: pd.DataFrame) -> dict:
    needed = {"time_s", "vehicle_id"}
    if not needed.issubset(legacy.columns):
        return {
            "status": "KEY_COLUMNS_NOT_AVAILABLE_IN_LEGACY",
            "available_legacy_columns": list(legacy.columns),
        }

    a = legacy.copy()
    b = cand.copy()

    # Source timestamps are 0.5 s; rounding protects against float serialization.
    a["time_key"] = a["time_s"].round(3)
    b["time_key"] = b["time_s"].round(3)

    a["vehicle_id"] = a["vehicle_id"].astype(str)
    b["vehicle_id"] = b["vehicle_id"].astype(str)

    keep_a = [
        "time_key", "vehicle_id", "front_gap_m", "time_headway_s"
    ] + (["leader_id"] if "leader_id" in a.columns else [])
    keep_b = [
        "time_key", "vehicle_id", "front_gap_m", "time_headway_s", "leader_id",
        "front_front_m", "side_clearance_m",
    ]

    a = a[keep_a].drop_duplicates(["time_key", "vehicle_id"])
    b = b[keep_b].drop_duplicates(["time_key", "vehicle_id"])

    m = a.merge(
        b,
        on=["time_key", "vehicle_id"],
        how="outer",
        suffixes=("_legacy", "_candidate"),
        indicator=True,
    )

    both = m[m["_merge"] == "both"].copy()
    report = {
        "status": "OK",
        "legacy_key_rows": int(len(a)),
        "candidate_key_rows": int(len(b)),
        "matched_key_rows": int(len(both)),
        "legacy_only_rows": int((m["_merge"] == "left_only").sum()),
        "candidate_only_rows": int((m["_merge"] == "right_only").sum()),
        "matched_fraction_of_legacy": float(len(both) / len(a)) if len(a) else None,
    }

    if "leader_id_legacy" in both.columns:
        li = both["leader_id_legacy"].astype(str)
        ci = both["leader_id_candidate"].astype(str)
        report["leader_id_exact_match_fraction"] = float((li == ci).mean()) if len(both) else None

    if len(both):
        report["front_gap_abs_error_m"] = _q(
            np.abs(both["front_gap_m_candidate"] - both["front_gap_m_legacy"])
        )
        report["headway_abs_error_s"] = _q(
            np.abs(
                both["time_headway_s_candidate"]
                - both["time_headway_s_legacy"]
            )
        )

    return report


def run_forensics(project_root: str | Path = ".") -> dict:
    root = Path(project_root)
    processed = root / "data" / "processed"

    clean = _load_clean(processed / "calibration_clean.parquet")
    legacy, legacy_meta = _load_legacy(
        processed / "calibration_interactions.parquet"
    )

    candidate_results = []

    for cfg in CANDIDATES:
        print(f"Testing {cfg['name']}...")
        d = _reconstruct(clean, cfg)
        dist = _distribution_error(legacy, d)
        row = _row_level_parity(legacy, d)

        result = {
            "candidate": cfg,
            "rows": int(len(d)),
            "distribution_error": dist,
            "row_level_parity": row,
            "summary_by_class": _class_distribution_summary(d),
        }
        candidate_results.append(result)

        print(
            f"  rows={len(d)}, "
            f"max_median_err={dist['max_relative_median_error']:.4f}, "
            f"within10={dist['within_10pct_all_medians']}"
        )

    ranked = sorted(
        candidate_results,
        key=lambda r: (
            float("inf")
            if r["distribution_error"]["max_relative_median_error"] is None
            else r["distribution_error"]["max_relative_median_error"],
            float("inf")
            if r["distribution_error"]["mean_relative_median_error"] is None
            else r["distribution_error"]["mean_relative_median_error"],
        ),
    )

    legacy_max_gap = legacy_meta["legacy_front_gap"].get("max")
    lookahead_signal = {
        "legacy_max_nonnegative_front_gap_m": legacy_max_gap,
        "suggests_finite_lookahead_near_50m": (
            legacy_max_gap is not None and 45.0 <= legacy_max_gap <= 50.5
        ),
        "reason": (
            "The legacy nonnegative front-gap maximum is close to 50 m. "
            "That is a strong forensic clue for a finite forward candidate-search radius, "
            "but the candidate comparison decides whether it explains the parity gap."
        ),
    }

    report = {
        "sprint": "1E_A9_2_LEADER_GAP_FORENSICS",
        "purpose": (
            "Identify the exact interaction-definition mismatch between the established "
            "Sprint-1B calibration_interactions.parquet and the A9 common extractor "
            "before changing any SUMO behavior parameter."
        ),
        "legacy": legacy_meta,
        "forensic_signal": lookahead_signal,
        "candidates": candidate_results,
        "ranked_candidates": [
            {
                "candidate": r["candidate"],
                "rows": r["rows"],
                "max_relative_median_error": r["distribution_error"]["max_relative_median_error"],
                "mean_relative_median_error": r["distribution_error"]["mean_relative_median_error"],
                "within_10pct_all_medians": r["distribution_error"]["within_10pct_all_medians"],
                "leader_id_exact_match_fraction": r["row_level_parity"].get(
                    "leader_id_exact_match_fraction"
                ),
                "matched_fraction_of_legacy": r["row_level_parity"].get(
                    "matched_fraction_of_legacy"
                ),
            }
            for r in ranked
        ],
        "best_candidate": ranked[0] if ranked else None,
        "decision_rules": [
            "If a candidate reaches <=10% max median error and has strong row/leader parity where leader IDs exist, use that exact rule for the next common observed-vs-SUMO metric.",
            "If the 50 m lookahead candidate wins clearly, restore the original finite lookahead instead of changing tau/minGap.",
            "If no candidate reaches 10%, inspect the actual legacy leader_id and gap rows directly; do not add persona-specific parameters yet.",
            "No SUMO simulations and no validation data are used in this forensic sprint.",
        ],
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a9_2_leader_gap_forensics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
