from __future__ import annotations

from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PRIMARY_CLASSES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}

EXACT_FEATURE_ALIASES = {
    "spacing_preference",
    "spacing_pref",
    "spacing_preference_score",
    "dna_spacing_preference",
    "normalized_spacing_preference",
    "relative_spacing_preference",
}

POSITIVE_HINTS = (
    "spacing",
    "headway_pref",
    "gap_pref",
    "following_distance",
    "following_gap",
)

NEGATIVE_HINTS = (
    "time_headway_s",
    "front_gap_m",
    "tau_proxy",
    "leader",
    "interaction",
    "simulated",
    "simulation",
    "sumo",
)

SKIP_FILE_HINTS = (
    "interaction",
    "simulation",
    "sumo",
    "report",
    "validation",
    "a10_vehicle_longitudinal_evidence",
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _pick_id_col(df: pd.DataFrame) -> str | None:
    lut = {_norm(c): c for c in df.columns}
    for name in ("vehicle_id", "vehicle_number", "veh_id", "id"):
        if name in lut:
            return lut[name]
    return None


def _read_tabular(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _feature_column_score(name: str) -> tuple[int, str]:
    n = _norm(name)
    if n in EXACT_FEATURE_ALIASES:
        return 100, "exact_spacing_preference_alias"

    if any(bad in n for bad in NEGATIVE_HINTS):
        return -100, "excluded_target_or_simulation_like_column"

    if "spacing" in n and ("pref" in n or "score" in n or "norm" in n):
        return 80, "strong_spacing_feature_name"

    if any(h in n for h in POSITIVE_HINTS):
        return 50, "spacing_related_feature_name"

    return 0, "not_spacing_related"


def discover_candidate_sources(project_root: Path) -> list[dict]:
    roots = [
        project_root / "data" / "processed",
        project_root / "data" / "models",
    ]
    candidates = []

    for base in roots:
        if not base.exists():
            continue

        files = sorted(
            list(base.rglob("*.parquet")) + list(base.rglob("*.csv"))
        )

        for path in files:
            lname = path.name.lower()
            if any(h in lname for h in SKIP_FILE_HINTS):
                continue

            df = _read_tabular(path)
            if df is None or df.empty:
                continue

            id_col = _pick_id_col(df)
            if id_col is None:
                continue

            unique_ids = df[id_col].astype(str).nunique(dropna=True)
            row_count = len(df)
            unique_ratio = unique_ids / max(row_count, 1)

            # Continuous DNA should be vehicle-level or nearly vehicle-level.
            vehicle_level_like = (
                100 <= unique_ids <= 5000
                and unique_ratio >= 0.50
                and row_count <= 10000
            )

            if not vehicle_level_like:
                continue

            numeric_cols = [
                c for c in df.columns
                if c != id_col and pd.api.types.is_numeric_dtype(df[c])
            ]

            for col in numeric_cols:
                score, reason = _feature_column_score(str(col))
                if score <= 0:
                    continue

                # Prefer calibration / normalized / behavior feature sources.
                fn = path.name.lower()
                file_bonus = 0
                if "calibration" in fn:
                    file_bonus += 10
                if "normalized" in fn or "normalised" in fn:
                    file_bonus += 10
                if "behavior" in fn or "behaviour" in fn or "dna" in fn:
                    file_bonus += 10
                if "vehicle_features" in fn:
                    file_bonus += 5

                candidates.append({
                    "file": str(path.relative_to(project_root)),
                    "id_column": str(id_col),
                    "feature_column": str(col),
                    "feature_name_normalized": _norm(col),
                    "name_score": int(score),
                    "file_bonus": int(file_bonus),
                    "total_discovery_score": int(score + file_bonus),
                    "reason": reason,
                    "rows": int(row_count),
                    "unique_vehicle_ids": int(unique_ids),
                    "unique_id_ratio": float(unique_ratio),
                })

    return sorted(
        candidates,
        key=lambda x: (-x["total_discovery_score"], x["file"], x["feature_column"]),
    )


def _load_candidate(project_root: Path, meta: dict) -> pd.DataFrame:
    path = project_root / meta["file"]
    df = _read_tabular(path)
    if df is None:
        raise RuntimeError(f"Could not read {path}")

    id_col = meta["id_column"]
    feature_col = meta["feature_column"]

    x = pd.DataFrame({
        "vehicle_id": df[id_col].astype(str),
        "feature_value": pd.to_numeric(df[feature_col], errors="coerce"),
    }).dropna()

    # If duplicate vehicle rows exist, use median only for this discovery audit.
    x = x.groupby("vehicle_id", as_index=False)["feature_value"].median()
    return x


def _correlation_report(
    evidence: pd.DataFrame,
    feature: pd.DataFrame,
    meta: dict,
) -> dict:
    d = evidence.copy()
    d["vehicle_id"] = d["vehicle_id"].astype(str)
    d = d.merge(feature, on="vehicle_id", how="inner")

    class_rows = []
    signal_classes = []

    for cls, name in PRIMARY_CLASSES.items():
        c = d[
            (d["vehicle_class"] == cls)
            & d["evidence_eligible"].astype(bool)
        ].copy()

        tests = {}
        for target in ("tau_proxy_p20_s", "time_headway_p20_s", "front_gap_median_m"):
            z = c[["feature_value", target]].dropna()
            if len(z) >= 20 and z["feature_value"].nunique() >= 5:
                rho, p = spearmanr(
                    z["feature_value"].to_numpy(float),
                    z[target].to_numpy(float),
                )
                rho, p = float(rho), float(p)
                signal = bool(len(z) >= 100 and abs(rho) >= 0.25 and p < 0.01)
                tests[target] = {
                    "n": int(len(z)),
                    "spearman_rho": rho,
                    "p_value": p,
                    "signal": signal,
                }
            else:
                tests[target] = {
                    "n": int(len(z)),
                    "spearman_rho": None,
                    "p_value": None,
                    "signal": False,
                }

        if tests["tau_proxy_p20_s"]["signal"] or tests["time_headway_p20_s"]["signal"]:
            signal_classes.append(cls)

        class_rows.append({
            "vehicle_class": cls,
            "vehicle_class_name": name,
            "eligible_joined_vehicles": int(len(c)),
            "tests": tests,
        })

    # Cross-class standardized test: z-score within class to avoid class-level
    # scale differences masquerading as a continuous preference relationship.
    pooled = d[d["evidence_eligible"].astype(bool)].copy()
    pooled_rows = []
    if len(pooled):
        for col in ("feature_value", "tau_proxy_p20_s", "time_headway_p20_s"):
            pooled[f"z_{col}"] = pooled.groupby("vehicle_class")[col].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan
            )

        for target in ("tau_proxy_p20_s", "time_headway_p20_s"):
            z = pooled[[f"z_feature_value", f"z_{target}"]].dropna()
            if len(z) >= 50:
                rho, p = spearmanr(
                    z["z_feature_value"].to_numpy(float),
                    z[f"z_{target}"].to_numpy(float),
                )
                pooled_rows.append({
                    "target": target,
                    "n": int(len(z)),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                })

    strength = 0.0
    for cr in class_rows:
        for target in ("tau_proxy_p20_s", "time_headway_p20_s"):
            t = cr["tests"][target]
            if t["spearman_rho"] is not None:
                strength += abs(t["spearman_rho"]) * min(t["n"], 300) / 300.0

    return {
        "source": meta,
        "joined_vehicle_count": int(d["vehicle_id"].nunique()),
        "signal_classes": signal_classes,
        "class_analysis": class_rows,
        "pooled_within_class_standardized": pooled_rows,
        "evidence_strength_score": float(strength),
    }


def run_recovery(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"

    evidence_path = processed / "a10_vehicle_longitudinal_evidence.parquet"
    if not evidence_path.exists():
        raise FileNotFoundError(
            "A10 evidence file not found. Run phase1_sumo_longitudinal_heterogeneity_audit.py first."
        )

    evidence = pd.read_parquet(evidence_path).copy()
    required = {
        "vehicle_id", "vehicle_class", "evidence_eligible",
        "tau_proxy_p20_s", "time_headway_p20_s", "front_gap_median_m",
    }
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"A10 evidence file missing columns: {sorted(missing)}")

    discovered = discover_candidate_sources(project_root)

    analyses = []
    for meta in discovered[:20]:
        try:
            feat = _load_candidate(project_root, meta)
            analyses.append(_correlation_report(evidence, feat, meta))
        except Exception as exc:
            analyses.append({
                "source": meta,
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "evidence_strength_score": -1.0,
                "signal_classes": [],
            })

    valid = [a for a in analyses if a.get("status") != "ERROR"]
    ranked = sorted(
        valid,
        key=lambda a: (
            -len(a.get("signal_classes", [])),
            -a.get("evidence_strength_score", 0.0),
            -a["source"]["total_discovery_score"],
        ),
    )

    best = ranked[0] if ranked else None

    if best is None:
        recommendation = "NO_CONTINUOUS_DNA_SOURCE_FOUND"
    elif len(best.get("signal_classes", [])) >= 2:
        recommendation = "CONTINUOUS_DNA_TAU_MAPPING_SUPPORTED"
    elif len(best.get("signal_classes", [])) == 1:
        recommendation = "HYBRID_ONE_CLASS_CONTINUOUS_OTHER_CLASSES_EMPIRICAL_MIXTURE"
    else:
        recommendation = "CONTINUOUS_DNA_NOT_SUPPORTED_USE_EMPIRICAL_CLASS_MIXTURE"

    report = {
        "sprint": "1E_A10_1_CONTINUOUS_DNA_RECOVERY",
        "purpose": (
            "Recover the actual continuous spacing-preference feature from earlier "
            "StreetLab calibration artifacts and test it against A10 longitudinal "
            "evidence before defaulting to a non-persona empirical class mixture."
        ),
        "why_needed": (
            "A10 did not actually test continuous Behaviour DNA because the loaded "
            "calibration_behavior_dna.parquet contained only vehicle_id, vehicle_class "
            "and persona. Persona structure was weak, but continuous DNA remains unresolved."
        ),
        "candidate_sources_found": int(len(discovered)),
        "candidate_sources": discovered[:20],
        "analyses": ranked,
        "best_candidate": best,
        "recommendation": recommendation,
        "decision_rules": [
            "Do not treat A10's missing continuous-DNA columns as evidence that continuous DNA has no longitudinal signal.",
            "Use only vehicle-level calibration/normalized feature sources; interaction or simulation outputs are excluded to avoid target leakage.",
            "A continuous mapping is supported when at least two core vehicle classes show |rho| >= 0.25 with p < 0.01 and n >= 100 for tau-proxy/headway evidence.",
            "If no valid feature source is found, derive the stable spacing_preference explicitly from the existing context-normalized vehicle features before running another SUMO model.",
            "If a valid feature is found but has weak longitudinal correlation, proceed with an empirical within-class tau mixture rather than persona labels.",
            "Validation remains untouched."
        ],
        "validation_untouched": True,
    }

    out = processed / "sprint1e_a10_1_continuous_dna_recovery.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
