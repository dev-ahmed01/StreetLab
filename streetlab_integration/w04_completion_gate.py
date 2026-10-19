"""Fixed, fail-closed six-phase StreetLab research prototype acceptance gate.

This does not approve real-site deployment, independent class labels or Chennai
transfer. Run on existing saved evidence; no more manual reviewer stages.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .w04_holdout_intake import read_json, digest, validate_lock

NAMES = (
    "Original source provenance",
    "Frozen W04 policy",
    "May25 historical regression",
    "Full-cohort autonomous object audit",
    "Native synthetic SUMO and Chennai evidence",
    "Research-prototype final handoff",
)
READY = "RESEARCH_PROTOTYPE_COMPLETE_WITH_LIMITATIONS"
BLOCKED = "RESEARCH_PROTOTYPE_EVIDENCE_INCOMPLETE"


def newest(folder: Path, pattern: str, filename: str):
    found = [p / filename for p in folder.glob(pattern)
             if p.is_dir() and not p.is_symlink()
             and (p / filename).is_file()
             and not (p / filename).is_symlink()]
    return max(found, key=lambda p: p.parent.stat().st_mtime) if found else None


def sources(root: Path):
    """Find a coherent evidence chain, not necessarily the newest stale copy."""
    folder = root / "artifacts"
    def candidates(pattern, filename):
        return sorted(
            (p / filename for p in folder.glob(pattern)
             if p.is_dir() and not p.is_symlink()
             and (p / filename).is_file() and not (p / filename).is_symlink()),
            key=lambda p: p.parent.stat().st_mtime, reverse=True,
        )
    lock_dir = None
    for file in candidates("w04_policy_lock_*", "policy_lock.json"):
        protocol = file.parent / "holdout_protocol.json"
        if not protocol.is_file() or protocol.is_symlink():
            continue
        try:
            validate_lock(read_json(protocol), read_json(file), digest(file))
            lock_dir = file.parent
            break
        except (ValueError, OSError, KeyError, TypeError):
            continue

    runs = candidates("w04_may25_crossdate_*", "cross_date_summary.json")
    replays = candidates("w04_may25_forensics_*", "regression_forensics.json")
    summary = runs[0] if runs else None
    forensics = replays[0] if replays else None
    found = False
    for candidate in runs:
        pre = candidate.parent / "preflight.json"
        if not pre.is_file() or pre.is_symlink():
            continue
        for receipt in replays:
            doc = valid_read(receipt)
            if (doc and matching_hash(candidate, doc.get("saved_regression_summary_sha256"))
                and matching_hash(pre, doc.get("preflight_sha256"))):
                summary, forensics, found = candidate, receipt, True
                break
        if found:
            break

    audits = candidates("w04_autonomous_research_*", "automated_object_audit.json")
    audit = audits[0] if audits else None
    if summary and forensics:
        for candidate in audits:
            doc = valid_read(candidate)
            if (doc and matching_hash(summary, doc.get("original_saved_regression_summary_sha256"))
                and matching_hash(forensics, doc.get("original_source_forensics_sha256"))):
                audit = candidate
                break

    studies = candidates("research_submission_*", "submission_evidence.json")
    submission = studies[0] if studies else None
    for candidate in studies:
        doc = valid_read(candidate)
        if (doc and (doc.get("native_sumo") or {}).get("status") == "SUMO_EXECUTED_SYNTHETIC_ONLY"
            and doc.get("production_release_approved") is False):
            submission = candidate
            break
    return dict(
        policy_lock=lock_dir / "policy_lock.json" if lock_dir else None,
        protocol=lock_dir / "holdout_protocol.json" if lock_dir else None,
        summary=summary,
        preflight=summary.parent / "preflight.json" if summary else None,
        forensics=forensics, audit=audit, submission=submission,
    )

def valid_read(path):
    if path is None:
        return None
    try:
        return read_json(path)
    except (ValueError, OSError):
        return None


def matching_hash(path, expected):
    return bool(path and isinstance(expected, str) and len(expected) == 64
                and digest(path) == expected)


def evaluate(paths):
    p = {key: valid_read(value) for key, value in paths.items()}
    lock, proto = p["policy_lock"], p["protocol"]
    summary, pre = p["summary"], p["preflight"]
    forensic, audit, study = p["forensics"], p["audit"], p["submission"]
    frozen = False
    if lock and proto and paths["policy_lock"]:
        try:
            validate_lock(proto, lock, digest(paths["policy_lock"]))
            frozen = True
        except (ValueError, KeyError, TypeError, OSError):
            pass

    source = bool(
        summary and pre and forensic
        and pre.get("status") == "W04_MAY25_CROSSDATE_PREFLIGHT_VERIFIED_REGRESSION_ONLY"
        and forensic.get("original_video_and_FLUID_SHA_verified_from_saved_batch") is True
        and forensic.get("original_saved_scores_recomputed_from_tracks_and_FLUID") is True
        and pre.get("source_video_sha256") == summary.get("source_video_sha256")
        and pre.get("source_truth_sha256") == summary.get("original_fluid_sha256")
        and pre.get("model_tree_sha256") == summary.get("frozen_w04_model_sha256")
        and matching_hash(paths["summary"], forensic.get("saved_regression_summary_sha256"))
        and matching_hash(paths["preflight"], forensic.get("preflight_sha256"))
        and forensic.get("saved_regression_batch_sha256")
            == summary.get("original_saved_batch_sha256")
    )
    regression = bool(
        source and frozen
        and summary.get("status") == "W04_MAY25_CROSSDATE_REGRESSION_RECORDED_NOT_BLIND_HOLDOUT"
        and summary.get("cohort_role") == "PREVIOUSLY_USED_MAY25_REGRESSION_REFERENCE"
        and summary.get("selected_policy") == lock.get("selected_policy")
        and summary.get("source_window") == [60, 179]
        and summary.get("decoded_frames") == 150
        and summary.get("evaluation_frames") == 120
        and summary.get("not_a_blind_or_untouched_holdout") is True
        and summary.get("independent_holdout_pass") is False
        and summary.get("production_release_approved") is False
        and forensic.get("status")
            == "W04_MAY25_REAL_REGRESSION_REPLAYED_WITH_REVIEW_QUEUE_NOT_BLIND"
        and forensic.get("independent_holdout_pass") is False
    )
    heavy = ((summary or {}).get("per_class_fluid_label_diagnostics") or {}).get("HEAVY_VEHICLE") or {}
    autonomous = bool(
        regression and audit
        and audit.get("status") == "W04_MAY25_AUTOMATED_LABEL_RELATIVE_OBJECT_AUDIT_COMPLETE"
        and matching_hash(paths["summary"], audit.get("original_saved_regression_summary_sha256"))
        and matching_hash(paths["forensics"], audit.get("original_source_forensics_sha256"))
        and audit.get("HEAVY_wrong_class_matched_points") == heavy.get("class_errors_on_spatial_matches")
        and audit.get("original_saved_FLUID_label_F1") == summary.get("truth_label_f1")
        and audit.get("source_video_frame_window") == [60, 179]
        and audit.get("no_new_detector_inference") is True
        and audit.get("no_FLUID_relabeling") is True
        and audit.get("physical_object_labels_independently_adjudicated") is False
        and audit.get("independent_holdout_pass") is False
        and audit.get("production_release_approved") is False
    )
    sumo = bool(
        study
        and study.get("research_workflow_status")
            == "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION"
        and (study.get("native_sumo") or {}).get("status") == "SUMO_EXECUTED_SYNTHETIC_ONLY"
        and ((study.get("dataset_summaries") or {}).get("chennai") or {})
            .get("holdout_generalization_pass") is False
        and study.get("external_site_transfer_validated") is False
        and study.get("production_release_approved") is False
    )
    flags = [source, frozen, regression, autonomous, sumo]
    flags.append(all(flags))
    phases = [
        {"phase": i + 1, "name": name,
         "status": "COMPLETE" if flags[i] else "BLOCKED"}
        for i, name in enumerate(NAMES)
    ]
    return {
        "status": READY if all(flags) else BLOCKED,
        "phase_count_fixed": 6,
        "phases_completed": sum(flags),
        "phases": phases,
        "research_historical_regression_f1": summary.get("truth_label_f1") if regression else None,
        "original_source_bytes_rehashed_in_this_report": False,
        "independent_holdout_pass": False,
        "chennai_real_world_generalization_pass": False,
        "native_sumo_real_world_calibration_pass": False,
        "physical_vehicle_ground_truth_adjudicated": False,
        "production_release_approved": False,
        "missing_evidence_requires_new_manual_vehicle_reviews": False,
        "scope": "Reproducible research prototype, not a calibrated field deployment",
    }


def markdown(result):
    lines = [
        "# StreetLab: fixed six-phase completion verdict", "",
        f"Status: **{result['status']}**. Completed: "
        f"**{result['phases_completed']}/6**.", "",
        "| Phase | Requirement | Status |", "|---:|---|---|",
    ]
    lines.extend(
        f"| {x['phase']} | {x['name']} | {x['status']} |"
        for x in result["phases"])
    lines.extend([
        "", "May25 is a previously inspected regression sample, not an "
        "untouched holdout. Historical FLUID taxonomy remains unadjudicated.",
        "Chennai generalization failed; SUMO models a synthetic network only.",
        "", "**Independent holdout: not passed. Production: not approved.**", "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Finite research acceptance without reviewer loops")
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    target = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if target.exists():
        parser.error("Refusing to overwrite existing research acceptance evidence")
    result = evaluate(sources(root))
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    target.mkdir(parents=True, exist_ok=False)
    with (target / "completion_verdict.json").open("x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    (target / "completion_verdict.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "completed": result["phases_completed"],
                      "total": 6, "report": str(target / "completion_verdict.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
