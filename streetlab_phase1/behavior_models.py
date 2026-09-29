from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


PRIMARY_CLASSES = [1, 2, 6]
CLASS_NAMES = {1: "MOTORCYCLE", 2: "CAR", 6: "AUTO_RICKSHAW"}

# Deliberately excludes acceleration/braking because Sprint 1B.6 showed
# split drift even after context normalisation.
DNA_FEATURES = [
    "median_speed_ratio_to_stream",
    "median_headway_context_ratio",
    "median_side_clearance_context_ratio",
    "median_lateral_activity_excess_mps",
]

DNA_NAMES = {
    "median_speed_ratio_to_stream": "speed_preference",
    "median_headway_context_ratio": "spacing_preference",
    "median_side_clearance_context_ratio": "side_clearance_preference",
    "median_lateral_activity_excess_mps": "lateral_assertiveness_proxy",
}


@dataclass(frozen=True)
class ContestConfig:
    primary_classes: tuple[int, ...] = (1, 2, 6)
    max_components: int = 5
    random_state: int = 42
    n_init: int = 20
    reg_covar: float = 1e-5
    clip_low_q: float = 0.01
    clip_high_q: float = 0.99
    min_bic_improvement: float = 10.0
    min_validation_loglik_gain: float = 0.05
    min_calibration_silhouette: float = 0.20
    min_validation_silhouette: float = 0.15
    min_cluster_share: float = 0.08
    min_validation_confidence: float = 0.75
    max_median_profile_drift_scaled: float = 0.25
    max_single_profile_drift_scaled: float = 0.75


def _finite_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan)


def fit_robust_scaler(cal: pd.DataFrame, features: list[str], cfg: ContestConfig) -> dict:
    params: dict[str, dict[str, float]] = {}
    for f in features:
        x = _finite_numeric(cal[f])
        med = float(x.median())
        q1 = float(x.quantile(0.25))
        q3 = float(x.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 1e-9:
            sd = float(x.std(ddof=0))
            iqr = sd if np.isfinite(sd) and sd > 1e-9 else 1.0
        lo = float(x.quantile(cfg.clip_low_q))
        hi = float(x.quantile(cfg.clip_high_q))
        params[f] = {"median": med, "scale_iqr": iqr, "clip_low": lo, "clip_high": hi}
    return params


def transform_with_scaler(df: pd.DataFrame, features: list[str], scaler: dict) -> np.ndarray:
    cols = []
    for f in features:
        p = scaler[f]
        x = _finite_numeric(df[f]).fillna(p["median"]).clip(p["clip_low"], p["clip_high"])
        z = (x - p["median"]) / p["scale_iqr"]
        cols.append(z.to_numpy(dtype=float))
    return np.column_stack(cols)


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    unique = np.unique(labels)
    if len(unique) <= 1 or len(unique) >= len(X):
        return None
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return None


def _shares(labels: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(labels.astype(int), minlength=k).astype(float) / max(len(labels), 1)


def score_gmms(X_cal: np.ndarray, X_val: np.ndarray, cfg: ContestConfig) -> tuple[pd.DataFrame, dict[int, GaussianMixture]]:
    rows = []
    models: dict[int, GaussianMixture] = {}
    max_k = min(cfg.max_components, max(1, len(X_cal) // 20))
    for k in range(1, max_k + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=cfg.n_init,
            random_state=cfg.random_state,
            reg_covar=cfg.reg_covar,
        )
        gm.fit(X_cal)
        cal_labels = gm.predict(X_cal)
        val_labels = gm.predict(X_val)
        cal_share = _shares(cal_labels, k)
        val_share = _shares(val_labels, k)
        val_conf = float(gm.predict_proba(X_val).max(axis=1).mean())
        rows.append({
            "k": k,
            "bic": float(gm.bic(X_cal)),
            "aic": float(gm.aic(X_cal)),
            "calibration_avg_loglik": float(gm.score(X_cal)),
            "validation_avg_loglik": float(gm.score(X_val)),
            "calibration_silhouette": _safe_silhouette(X_cal, cal_labels),
            "validation_silhouette": _safe_silhouette(X_val, val_labels),
            "min_calibration_cluster_share": float(cal_share.min()),
            "min_validation_cluster_share": float(val_share.min()),
            "validation_mean_assignment_confidence": val_conf,
        })
        models[k] = gm
    return pd.DataFrame(rows), models


def profile_personas(
    cal_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    X_cal: np.ndarray,
    X_val: np.ndarray,
    model: GaussianMixture,
    features: list[str],
) -> tuple[pd.DataFrame, dict]:
    cal_labels = model.predict(X_cal)
    val_labels = model.predict(X_val)
    cal_prob = model.predict_proba(X_cal).max(axis=1)
    val_prob = model.predict_proba(X_val).max(axis=1)
    k = model.n_components

    rows = []
    scaled_drifts = []
    for j in range(k):
        c_mask = cal_labels == j
        v_mask = val_labels == j
        persona = f"P{j+1}"
        base = {
            "persona": persona,
            "cluster_index": j,
            "calibration_n": int(c_mask.sum()),
            "validation_n": int(v_mask.sum()),
            "calibration_share": float(c_mask.mean()),
            "validation_share": float(v_mask.mean()),
            "calibration_mean_assignment_confidence": float(cal_prob[c_mask].mean()) if c_mask.any() else None,
            "validation_mean_assignment_confidence": float(val_prob[v_mask].mean()) if v_mask.any() else None,
        }
        for idx, f in enumerate(features):
            cal_med_scaled = float(np.median(X_cal[c_mask, idx])) if c_mask.any() else np.nan
            val_med_scaled = float(np.median(X_val[v_mask, idx])) if v_mask.any() else np.nan
            drift = abs(cal_med_scaled - val_med_scaled) if np.isfinite(cal_med_scaled) and np.isfinite(val_med_scaled) else np.nan
            if np.isfinite(drift):
                scaled_drifts.append(float(drift))
            base[f"{DNA_NAMES[f]}__calibration_median"] = float(_finite_numeric(cal_raw.loc[c_mask, f]).median()) if c_mask.any() else None
            base[f"{DNA_NAMES[f]}__validation_median"] = float(_finite_numeric(val_raw.loc[v_mask, f]).median()) if v_mask.any() else None
            base[f"{DNA_NAMES[f]}__profile_drift_scaled"] = float(drift) if np.isfinite(drift) else None
        rows.append(base)

    meta = {
        "median_profile_drift_scaled": float(np.median(scaled_drifts)) if scaled_drifts else None,
        "max_profile_drift_scaled": float(np.max(scaled_drifts)) if scaled_drifts else None,
        "calibration_min_cluster_share": float(_shares(cal_labels, k).min()),
        "validation_min_cluster_share": float(_shares(val_labels, k).min()),
        "validation_mean_assignment_confidence": float(val_prob.mean()),
    }
    return pd.DataFrame(rows), meta


def decide_persona_support(scores: pd.DataFrame, profiles_meta: dict, chosen_k: int, cfg: ContestConfig) -> dict:
    k1 = scores.loc[scores["k"] == 1].iloc[0]
    chosen = scores.loc[scores["k"] == chosen_k].iloc[0]
    bic_improvement = float(k1["bic"] - chosen["bic"])
    val_gain = float(chosen["validation_avg_loglik"] - k1["validation_avg_loglik"])
    cal_sil = chosen["calibration_silhouette"]
    val_sil = chosen["validation_silhouette"]

    checks = {
        "multiple_components": bool(chosen_k >= 2),
        "strong_bic_improvement": bool(bic_improvement >= cfg.min_bic_improvement),
        "holdout_loglik_gain": bool(val_gain >= cfg.min_validation_loglik_gain),
        "calibration_separation": bool(pd.notna(cal_sil) and float(cal_sil) >= cfg.min_calibration_silhouette),
        "validation_separation": bool(pd.notna(val_sil) and float(val_sil) >= cfg.min_validation_silhouette),
        "calibration_cluster_support": bool(profiles_meta["calibration_min_cluster_share"] >= cfg.min_cluster_share),
        "validation_cluster_support": bool(profiles_meta["validation_min_cluster_share"] >= cfg.min_cluster_share),
        "validation_assignment_confidence": bool(profiles_meta["validation_mean_assignment_confidence"] >= cfg.min_validation_confidence),
        "median_profile_stability": bool(profiles_meta["median_profile_drift_scaled"] <= cfg.max_median_profile_drift_scaled),
        "worst_profile_stability": bool(profiles_meta["max_profile_drift_scaled"] <= cfg.max_single_profile_drift_scaled),
    }
    supported = all(checks.values())
    return {
        "chosen_k_by_calibration_bic": int(chosen_k),
        "bic_improvement_vs_k1": bic_improvement,
        "validation_loglik_gain_vs_k1": val_gain,
        "checks": checks,
        "persona_evidence_supported": supported,
        "representation_decision": "HYBRID_PERSONA_PLUS_CONTINUOUS_DNA" if supported else "CONTINUOUS_DNA_ONLY",
    }


def attach_dna_and_personas(df: pd.DataFrame, X: np.ndarray, model: GaussianMixture, features: list[str]) -> pd.DataFrame:
    out = df[["vehicle_id", "vehicle_class"]].copy().reset_index(drop=True)
    labels = model.predict(X)
    probs = model.predict_proba(X)
    out["persona"] = [f"P{i+1}" for i in labels]
    out["persona_confidence"] = probs.max(axis=1)
    for i, f in enumerate(features):
        out[f"dna_{DNA_NAMES[f]}"] = X[:, i]
    return out


def run_class_contest(cal: pd.DataFrame, val: pd.DataFrame, vehicle_class: int, cfg: ContestConfig) -> dict:
    ca = cal[cal["vehicle_class"].astype(int) == int(vehicle_class)].copy().reset_index(drop=True)
    va = val[val["vehicle_class"].astype(int) == int(vehicle_class)].copy().reset_index(drop=True)
    if len(ca) < 100 or len(va) < 100:
        raise ValueError(f"Class {vehicle_class} has insufficient samples for Sprint 1C: cal={len(ca)}, val={len(va)}")

    scaler = fit_robust_scaler(ca, DNA_FEATURES, cfg)
    X_cal = transform_with_scaler(ca, DNA_FEATURES, scaler)
    X_val = transform_with_scaler(va, DNA_FEATURES, scaler)
    scores, models = score_gmms(X_cal, X_val, cfg)
    chosen_k = int(scores.loc[scores["bic"].idxmin(), "k"])
    model = models[chosen_k]
    profiles, pmeta = profile_personas(ca, va, X_cal, X_val, model, DNA_FEATURES)
    decision = decide_persona_support(scores, pmeta, chosen_k, cfg)
    cal_dna = attach_dna_and_personas(ca, X_cal, model, DNA_FEATURES)
    val_dna = attach_dna_and_personas(va, X_val, model, DNA_FEATURES)

    scores.insert(0, "vehicle_class", int(vehicle_class))
    scores.insert(1, "vehicle_class_name", CLASS_NAMES.get(int(vehicle_class), str(vehicle_class)))
    profiles.insert(0, "vehicle_class", int(vehicle_class))
    profiles.insert(1, "vehicle_class_name", CLASS_NAMES.get(int(vehicle_class), str(vehicle_class)))

    return {
        "vehicle_class": int(vehicle_class),
        "vehicle_class_name": CLASS_NAMES.get(int(vehicle_class), str(vehicle_class)),
        "calibration_n": int(len(ca)),
        "validation_n": int(len(va)),
        "scaler": scaler,
        "scores": scores,
        "profiles": profiles,
        "profile_meta": pmeta,
        "decision": decision,
        "model": model,
        "calibration_dna": cal_dna,
        "validation_dna": val_dna,
    }


def build_overall_report(results: list[dict], cfg: ContestConfig) -> dict:
    per_class = []
    for r in results:
        per_class.append({
            "vehicle_class": r["vehicle_class"],
            "vehicle_class_name": r["vehicle_class_name"],
            "calibration_n": r["calibration_n"],
            "validation_n": r["validation_n"],
            **r["decision"],
            "profile_stability": r["profile_meta"],
        })
    supported = sum(bool(x["persona_evidence_supported"]) for x in per_class)
    if supported == len(per_class):
        overall = "HYBRID_PERSONA_PLUS_CONTINUOUS_DNA"
    elif supported == 0:
        overall = "CONTINUOUS_DNA_ONLY"
    else:
        overall = "CLASS_SPECIFIC_HYBRID"
    return {
        "sprint": "1C_BEHAVIOR_MODEL_CONTEST",
        "candidate_features": [DNA_NAMES[f] for f in DNA_FEATURES],
        "excluded_from_persona_model": [
            "acceleration/braking residuals (split drift after context normalisation)",
            "TTC (safety outcome/context variable, not a stable personality trait)",
            "vehicle dimensions (physics, not behaviour)",
            "local density (environment, not behaviour)",
        ],
        "model_a": "K=1 multivariate Gaussian per vehicle class",
        "model_b": "K=2..5 Gaussian mixture personas, K selected from calibration BIC",
        "model_c": "continuous robust-scaled Behaviour DNA retained for every vehicle",
        "decision_rule": "Use personas only if they improve holdout density fit AND remain separated, supported and profile-stable in validation. Continuous DNA is always retained.",
        "overall_representation_decision": overall,
        "classes_supporting_personas": supported,
        "classes_tested": len(per_class),
        "per_class": per_class,
        "config": asdict(cfg),
    }


def save_contest_outputs(results: list[dict], report: dict, processed_dir: Path, model_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    score_frames = [r["scores"] for r in results]
    profile_frames = [r["profiles"] for r in results]
    cal_dna_frames = [r["calibration_dna"] for r in results]
    val_dna_frames = [r["validation_dna"] for r in results]

    pd.concat(score_frames, ignore_index=True).to_csv(processed_dir / "behavior_model_contest_scores.csv", index=False)
    pd.concat(profile_frames, ignore_index=True).to_csv(processed_dir / "persona_profiles.csv", index=False)
    pd.concat(cal_dna_frames, ignore_index=True).to_parquet(processed_dir / "calibration_behavior_dna.parquet", index=False)
    pd.concat(val_dna_frames, ignore_index=True).to_parquet(processed_dir / "validation_behavior_dna.parquet", index=False)

    scalers = {}
    for r in results:
        cls = str(r["vehicle_class"])
        scalers[cls] = r["scaler"]
        dump(r["model"], model_dir / f"behavior_gmm_class_{cls}.joblib")
    (model_dir / "behavior_scalers.json").write_text(json.dumps(scalers, indent=2), encoding="utf-8")
    (processed_dir / "sprint1c_model_contest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
