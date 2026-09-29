from __future__ import annotations

from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

from streetlab_phase1.sumo_stage_a import (
    StageAConfig,
    load_persona_priors,
    load_behavior_dna,
)
from streetlab_phase1.sumo_boundary_fidelity import (
    _canonical_clean_with_lat,
    build_boundary_demand,
)
from streetlab_phase1.sumo_metric_parity import (
    reconstruct_interactions,
)

PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}

OBS_X_MIN_M = 65.0
OBS_X_MAX_M = 235.0

# A9.3's best global capacity/spacing compromise. This remains a diagnostic
# baseline, not a frozen calibration.
GLOBAL_BASE_TAU_S = 0.50
PROVISIONAL_MIN_GAP_M = 0.50

# Enough repeated interaction observations to estimate a per-vehicle lower-tail
# following preference without letting one or two frames dominate.
MIN_INTERACTIONS_PER_VEHICLE = 20


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
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
    }


def _epsilon_squared_kruskal(h: float, n: int, k: int) -> float | None:
    if n <= k or k < 2:
        return None
    # Common epsilon-squared effect-size estimator for Kruskal-Wallis.
    return float(max(0.0, (h - k + 1) / (n - k)))


def _canonical_observed_interactions(clean: pd.DataFrame) -> pd.DataFrame:
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

    x = all_interactions[
        all_interactions["vehicle_class"].isin(PRIMARY_CLASSES)
        & (all_interactions["follower_x_front_m"] >= OBS_X_MIN_M)
        & (all_interactions["follower_x_front_m"] <= OBS_X_MAX_M)
        & (all_interactions["front_gap_m"] >= 0)
        & all_interactions["time_headway_s"].notna()
        & (all_interactions["time_headway_s"] >= 0)
        & (all_interactions["follower_speed_mps"] > 0.05)
    ].copy()

    # Equilibrium-style tau proxy used only as a seed diagnostic:
    #   gap ~= minGap + tau * speed
    # => tau ~= headway - minGap/speed
    x["tau_proxy_s"] = (
        x["time_headway_s"]
        - PROVISIONAL_MIN_GAP_M / x["follower_speed_mps"]
    )
    # Negative/very large values are dynamic/transient states and are not useful
    # as direct seed evidence. Keep them in raw metrics, exclude from tau proxy.
    x["tau_proxy_valid_s"] = x["tau_proxy_s"].where(
        (x["tau_proxy_s"] >= 0.05) & (x["tau_proxy_s"] <= 3.0)
    )
    return x


def _vehicle_evidence(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (vid, cls), g in interactions.groupby(["vehicle_id", "vehicle_class"], sort=False):
        th = g["time_headway_s"].dropna().to_numpy(float)
        gap = g["front_gap_m"].dropna().to_numpy(float)
        tp = g["tau_proxy_valid_s"].dropna().to_numpy(float)
        sp = g["follower_speed_mps"].dropna().to_numpy(float)

        rows.append({
            "vehicle_id": str(vid),
            "vehicle_class": int(cls),
            "interaction_n": int(len(g)),
            "time_headway_p10_s": float(np.quantile(th, 0.10)) if len(th) else np.nan,
            "time_headway_p20_s": float(np.quantile(th, 0.20)) if len(th) else np.nan,
            "time_headway_median_s": float(np.median(th)) if len(th) else np.nan,
            "front_gap_p10_m": float(np.quantile(gap, 0.10)) if len(gap) else np.nan,
            "front_gap_median_m": float(np.median(gap)) if len(gap) else np.nan,
            "tau_proxy_p20_s": float(np.quantile(tp, 0.20)) if len(tp) >= 5 else np.nan,
            "tau_proxy_median_s": float(np.median(tp)) if len(tp) >= 5 else np.nan,
            "median_speed_mps": float(np.median(sp)) if len(sp) else np.nan,
            "tau_proxy_valid_n": int(len(tp)),
        })

    d = pd.DataFrame(rows)
    d["evidence_eligible"] = (
        (d["interaction_n"] >= MIN_INTERACTIONS_PER_VEHICLE)
        & d["tau_proxy_p20_s"].notna()
    )
    return d


def _attach_persona_and_dna(
    vehicle: pd.DataFrame,
    demand: pd.DataFrame,
    dna: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    d = vehicle.copy()
    d["vehicle_id"] = d["vehicle_id"].astype(str)

    dem = demand.copy()
    dem["vehicle_id"] = dem["vehicle_id"].astype(str)
    persona_cols = ["vehicle_id", "vehicle_class"]
    if "persona" in dem.columns:
        persona_cols.append("persona")
    d = d.merge(
        dem[persona_cols].drop_duplicates("vehicle_id"),
        on=["vehicle_id", "vehicle_class"],
        how="left",
    )

    dna_meta = {
        "spacing_preference_column": None,
        "speed_preference_column": None,
        "available_columns": [],
    }

    if dna is not None and not dna.empty:
        z = dna.copy()
        dna_meta["available_columns"] = [str(c) for c in z.columns]

        id_col = _pick_col(z, ["vehicle_id", "vehicle_number", "id"])
        spacing_col = _pick_col(
            z,
            [
                "spacing_preference",
                "spacing_pref",
                "spacing_preference_score",
                "dna_spacing_preference",
            ],
        )
        speed_col = _pick_col(
            z,
            [
                "speed_preference",
                "speed_pref",
                "speed_preference_score",
                "dna_speed_preference",
            ],
        )

        dna_meta["spacing_preference_column"] = (
            str(spacing_col) if spacing_col is not None else None
        )
        dna_meta["speed_preference_column"] = (
            str(speed_col) if speed_col is not None else None
        )

        if id_col is not None:
            keep = [id_col]
            ren = {id_col: "vehicle_id"}
            if spacing_col is not None:
                keep.append(spacing_col)
                ren[spacing_col] = "dna_spacing_preference"
            if speed_col is not None:
                keep.append(speed_col)
                ren[speed_col] = "dna_speed_preference"

            z = z[keep].rename(columns=ren).copy()
            z["vehicle_id"] = z["vehicle_id"].astype(str)
            d = d.merge(z.drop_duplicates("vehicle_id"), on="vehicle_id", how="left")

    return d, dna_meta


def _persona_analysis(eligible: pd.DataFrame) -> list[dict]:
    out = []

    for cls, class_name in PRIMARY_CLASSES.items():
        c = eligible[eligible["vehicle_class"] == cls].copy()
        personas = sorted(
            [p for p in c.get("persona", pd.Series(dtype=object)).dropna().unique()]
        )

        groups = []
        group_values = []
        for p in personas:
            g = c[c["persona"] == p]
            vals = g["tau_proxy_p20_s"].dropna().to_numpy(float)
            groups.append({
                "persona": str(p),
                "vehicles": int(len(g)),
                "tau_proxy_p20_s": _q(vals),
                "time_headway_p20_s": _q(g["time_headway_p20_s"]),
                "front_gap_median_m": _q(g["front_gap_median_m"]),
                "dna_spacing_preference": (
                    _q(g["dna_spacing_preference"])
                    if "dna_spacing_preference" in g.columns
                    else {"n": 0}
                ),
            })
            if len(vals):
                group_values.append(vals)

        h = pval = eps = None
        min_group_n = min((len(x) for x in group_values), default=0)
        if len(group_values) >= 2 and min_group_n >= 5:
            h, pval = kruskal(*group_values)
            h, pval = float(h), float(pval)
            eps = _epsilon_squared_kruskal(
                h,
                int(sum(len(x) for x in group_values)),
                len(group_values),
            )

        # This flag is only a modeling signal, not an inferential validation.
        persona_signal = bool(
            len(group_values) >= 2
            and min_group_n >= 20
            and pval is not None
            and pval < 0.01
            and eps is not None
            and eps >= 0.03
        )

        out.append({
            "vehicle_class": cls,
            "vehicle_class_name": class_name,
            "eligible_vehicles": int(len(c)),
            "persona_groups": groups,
            "kruskal_h": h,
            "kruskal_p": pval,
            "epsilon_squared": eps,
            "persona_structure_signal": persona_signal,
        })

    return out


def _continuous_analysis(eligible: pd.DataFrame) -> list[dict]:
    out = []
    for cls, class_name in PRIMARY_CLASSES.items():
        c = eligible[eligible["vehicle_class"] == cls].copy()

        row = {
            "vehicle_class": cls,
            "vehicle_class_name": class_name,
            "eligible_vehicles": int(len(c)),
            "spacing_preference_vs_tau_proxy": None,
            "spacing_preference_vs_headway_p20": None,
        }

        if "dna_spacing_preference" in c.columns:
            for target, label in [
                ("tau_proxy_p20_s", "spacing_preference_vs_tau_proxy"),
                ("time_headway_p20_s", "spacing_preference_vs_headway_p20"),
            ]:
                z = c[["dna_spacing_preference", target]].dropna()
                if len(z) >= 20:
                    rho, p = spearmanr(
                        z["dna_spacing_preference"].to_numpy(float),
                        z[target].to_numpy(float),
                    )
                    row[label] = {
                        "n": int(len(z)),
                        "spearman_rho": float(rho),
                        "p_value": float(p),
                        "continuous_structure_signal": bool(
                            len(z) >= 100 and abs(float(rho)) >= 0.25 and float(p) < 0.01
                        ),
                    }
        out.append(row)
    return out


def _seed_table(
    eligible: pd.DataFrame,
    persona_analysis: list[dict],
) -> pd.DataFrame:
    rows = []
    signal_by_class = {
        r["vehicle_class"]: r["persona_structure_signal"]
        for r in persona_analysis
    }

    for cls, class_name in PRIMARY_CLASSES.items():
        c = eligible[eligible["vehicle_class"] == cls].copy()
        personas = sorted(
            [p for p in c.get("persona", pd.Series(dtype=object)).dropna().unique()]
        )

        if personas:
            for p in personas:
                g = c[c["persona"] == p]
                raw = float(g["tau_proxy_p20_s"].median()) if len(g) else np.nan
                raw_clip = float(np.clip(raw, 0.35, 1.20)) if np.isfinite(raw) else np.nan
                # Conservative shrinkage toward the best global capacity baseline.
                # A11 can test these without jumping directly to the full empirical value.
                shrunk = (
                    float(np.clip(
                        GLOBAL_BASE_TAU_S + 0.50 * (raw_clip - GLOBAL_BASE_TAU_S),
                        0.35,
                        1.00,
                    ))
                    if np.isfinite(raw_clip)
                    else np.nan
                )
                rows.append({
                    "vehicle_class": cls,
                    "vehicle_class_name": class_name,
                    "persona": str(p),
                    "vehicles": int(len(g)),
                    "persona_structure_signal_for_class": bool(signal_by_class.get(cls)),
                    "raw_empirical_tau_proxy_seed_s": raw_clip,
                    "shrunk_tau_seed_s": shrunk,
                    "provisional_min_gap_m": PROVISIONAL_MIN_GAP_M,
                })

    return pd.DataFrame(rows)


def run_longitudinal_heterogeneity_audit(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    models = project_root / "data" / "models"

    clean = _canonical_clean_with_lat(processed / "calibration_clean.parquet")
    priors = load_persona_priors(models / "sumo_persona_priors.csv")
    dna = load_behavior_dna(processed / "calibration_behavior_dna.parquet")

    cfg = StageAConfig()
    demand = build_boundary_demand(clean, priors, dna, cfg)

    interactions = _canonical_observed_interactions(clean)
    vehicle = _vehicle_evidence(interactions)
    vehicle, dna_meta = _attach_persona_and_dna(vehicle, demand, dna)

    eligible = vehicle[vehicle["evidence_eligible"]].copy()

    persona = _persona_analysis(eligible)
    continuous = _continuous_analysis(eligible)
    seeds = _seed_table(eligible, persona)

    persona_signal_classes = [
        r["vehicle_class"] for r in persona if r["persona_structure_signal"]
    ]
    continuous_signal_classes = []
    for r in continuous:
        tests = [
            r.get("spacing_preference_vs_tau_proxy"),
            r.get("spacing_preference_vs_headway_p20"),
        ]
        if any(t and t.get("continuous_structure_signal") for t in tests):
            continuous_signal_classes.append(r["vehicle_class"])

    if len(persona_signal_classes) >= 2:
        recommended_next_model = "PERSONA_TAU_HETEROGENEITY"
    elif len(continuous_signal_classes) >= 2:
        recommended_next_model = "CONTINUOUS_DNA_TAU_HETEROGENEITY"
    elif persona_signal_classes or continuous_signal_classes:
        recommended_next_model = "HYBRID_CLASS_SPECIFIC_HETEROGENEITY"
    else:
        recommended_next_model = "EMPIRICAL_CLASS_MIXTURE_NOT_PERSONA"

    out_models = models
    out_models.mkdir(parents=True, exist_ok=True)
    out_processed = processed

    vehicle.to_parquet(
        out_processed / "a10_vehicle_longitudinal_evidence.parquet",
        index=False,
    )
    seeds.to_csv(
        out_models / "a10_persona_tau_seeds.csv",
        index=False,
    )

    report = {
        "sprint": "1E_A10_LONGITUDINAL_HETEROGENEITY_AUDIT",
        "purpose": (
            "Determine whether the now-validated global tau/minGap failure should "
            "be addressed with persona-level tau, continuous Behaviour-DNA tau, "
            "or a simpler empirical class mixture before running another SUMO search."
        ),
        "canonical_interaction_metric": {
            "lookahead_front_front_m": 50.0,
            "per_vehicle_lateral_safety_expansion_m": 0.20,
            "pairwise_body_clearance_threshold_m": 0.40,
            "observed_window_m": [OBS_X_MIN_M, OBS_X_MAX_M],
        },
        "baseline_for_seeding_only": {
            "global_tau_s": GLOBAL_BASE_TAU_S,
            "provisional_min_gap_m": PROVISIONAL_MIN_GAP_M,
            "note": (
                "These are not frozen calibration parameters; they are the best "
                "global capacity/spacing compromise from A9.3 and are used only "
                "as shrinkage anchors for the next experiment."
            ),
        },
        "evidence": {
            "canonical_interaction_rows": int(len(interactions)),
            "vehicles_total": int(len(vehicle)),
            "vehicles_eligible_min_20_interactions": int(len(eligible)),
            "eligibility_fraction": float(len(eligible) / len(vehicle)) if len(vehicle) else None,
            "tau_proxy_definition": "time_headway - 0.50 / follower_speed",
            "tau_proxy_per_vehicle_statistic": "20th percentile of valid instantaneous proxies",
            "warning": (
                "The tau proxy is an equilibrium-style calibration seed only. "
                "Dynamic microscopic observations are not assumed to equal SUMO tau exactly."
            ),
        },
        "dna_columns": dna_meta,
        "persona_analysis": persona,
        "continuous_dna_analysis": continuous,
        "persona_signal_classes": persona_signal_classes,
        "continuous_signal_classes": continuous_signal_classes,
        "recommended_next_model": recommended_next_model,
        "seed_file": "data/models/a10_persona_tau_seeds.csv",
        "vehicle_evidence_file": "data/processed/a10_vehicle_longitudinal_evidence.parquet",
        "decision_rules": [
            "Do not infer that personas deserve longitudinal parameters just because personas exist; require class-level structure in observed lower-tail following behavior.",
            "Prefer persona tau only where persona groups have adequate sample sizes and non-trivial separation.",
            "Prefer continuous DNA mapping where spacing_preference has a stronger monotonic relationship with observed following behavior than persona labels do.",
            "Keep minGap at 0.50 m provisionally in the next test; low-speed standstill-gap evidence remains too sparse to justify persona-specific minGap.",
            "The next SUMO sprint should test only the supported heterogeneity form against capacity + canonical spacing + speed simultaneously.",
            "Validation period remains untouched.",
        ],
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a10_longitudinal_heterogeneity_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
