"""M8 strict release-candidate acceptance (read-only, no execution or invention).

CI can verify program behavior with synthetic data. Genuine Windows inference,
independent field records, and external approval CANNOT be inferred from CI.
Never sets production_release_approved true.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.primary_runner import sha, verified_run
from streetlab_integration.worker import job_report
from streetlab_integration.workspace_read import project_workspace
from streetlab_integration.reconstruction import latest_reconstruction, _json_bytes
from streetlab_integration.site_baseline import latest_baseline, _folder as baseline_dir
from streetlab_integration.site_sumo import evaluate_tripinfo
from streetlab_integration.scenario_experiments import latest_scenario, _folder as scenario_dir
from streetlab_integration.site_scenarios import (
    experiment_model,completed_trip_times,paired_scorecard,
)
from streetlab_integration.maintenance import check_db
from streetlab_integration.field_acceptance import verify_field_records

MAX_TRIPINFO=120_000_000


def _verify_m4(store: VideoStore,project_id: str,baseline: dict) -> dict:
    """Recalculate the M4 QA verdict from SHA-checked actual SUMO tripinfo."""
    run=baseline["runtime"]
    if run["status"]=="NOT_EXECUTED":
        raise ValueError("Actual native M4 SUMO baseline is unexecuted")
    model=baseline["model"]
    folder=baseline_dir(store,project_id,baseline["revision"])/"runtime"
    trip=folder/"tripinfo.xml"
    if trip.is_symlink() or not trip.is_file() or trip.stat().st_size>MAX_TRIPINFO:
        raise ValueError("M4 actual tripinfo is missing or too large")
    derived=evaluate_tripinfo(model,trip.read_bytes())
    fields=("status","real_site_sumo_allowed","simulated_trips_completed",
            "manual_count_demand","completion_ratio","per_movement","thresholds")
    if run.get("model_revision")!=baseline["revision"] or any(run.get(k)!=derived[k] for k in fields):
        raise ValueError("M4 recorded QA outcome does not reconcile to its hashed tripinfo")
    if derived["status"]!="BASELINE_FIDELITY_CHECKED" or not derived["real_site_sumo_allowed"]:
        raise ValueError("Native baseline does not pass provisional movement travel-time QA")
    return {"status":derived["status"],
            "tripinfo_sha256":sha(trip),
            "manual_count_demand":derived["manual_count_demand"],
            "completion_ratio":derived["completion_ratio"],
            "movement_count":len(derived["per_movement"]),
            "model_revision":baseline["revision"]}


def _verify_m5(store: VideoStore, project_id: str, base: dict, experiment: dict) -> dict:
    """Recompute all 18 paired experimental tripinfos and scorecard from original bytes."""
    runtime=experiment["runtime"]
    if runtime["status"]=="NOT_EXECUTED":
        raise ValueError("No actual native M5 scenario has been executed")
    proposal=experiment["proposal"]
    if proposal["baseline_revision"]!=base["revision"]:
        raise ValueError("M5 scenario belongs to another M4 baseline")
    folder=scenario_dir(store,project_id,experiment["revision"])/"runtime"
    first=baseline_dir(store,project_id,base["revision"])/"runtime"/"site.net.xml"
    second=folder/"baseline.net.xml"
    if sha(first)!=sha(second):
        raise ValueError("M5 paired baseline network differs from M4 verified native baseline")
    pairs=[]
    levels=proposal["sensitivity"]["demand_multipliers"]
    seeds=proposal["sensitivity"]["seeds"]
    if len(levels)!=3 or seeds!=[42,43,44]:
        raise ValueError("M5 sensitivity grid differs from the released bounded design")
    for label,mult in zip(("low","mid","high"),levels):
        demand=experiment_model(base["model"],proposal,mult,changed=False)
        for seed in seeds:
            observed={}
            for side in ("baseline","scenario"):
                file=folder/f"{label}_{side}_s{seed}.tripinfo.xml"
                if file.is_symlink() or not file.is_file() or file.stat().st_size>MAX_TRIPINFO:
                    raise ValueError("M5 paired native tripinfo is missing or exceeds size budget")
                observed["baseline" if side=="baseline" else "intervention"]=(
                    completed_trip_times(demand,file.read_bytes()))
            pairs.append({"seed":seed,"demand_multiplier":mult,**observed})
    derived=paired_scorecard(pairs)
    for key in ("status","paired_runs","comparable_runs","all_conditions_comparable",
                "mean_paired_simulation_difference_s",
                "min_paired_simulation_difference_s",
                "max_paired_simulation_difference_s",
                "simulated_improvement_in_every_condition","records"):
        if runtime.get(key)!=derived[key]:
            raise ValueError("M5 recorded comparison contradicts actual native SUMO tripinfos")
    if derived["status"]!="SIMULATED_COMPARISON_ELIGIBLE":
        raise ValueError("M5 experiment has incomplete/censored trip conditions")
    return {"status":derived["status"],
            "paired_comparisons":derived["paired_runs"],
            "real_world_effect_verified":False,
            "experiment_revision":experiment["revision"]}


def evaluate_release(
    store: VideoStore,project_id: str, *,
    model_dir: Path | None=None,
    frozen_provenance: Path | None=None,
    field_dir: Path | None=None,
    full_video_sha: bool=False,
) -> dict:
    """Read-only formal readiness matrix. No field or production release is inferred."""
    project=store.project(project_id)
    checks={}
    def record(key: str, status: str, reason: str, evidence: dict | None=None):
        checks[key]={"status":status,"reason":reason}
        if evidence is not None:
            checks[key]["evidence"]=evidence
    try:
        check_db(store.db)
        ws=project_workspace(store,project_id)
        record("project_evidence","PASS","M1–M7 receipts and current project relationships verified")
    except (ValueError,OSError,KeyError,TypeError,sqlite3.Error,json.JSONDecodeError) as exc:
        # Internal paths/errors must not appear in exported public acceptance results.
        record("project_evidence","FAIL","Project source or stored evidence failed integrity verification")
        ws=None
    if ws is None:
        for key in ("original_video_sha","frozen_worker_model","survey_geometry",
                    "native_baseline","native_scenarios","field_records"):
            record(key,"BLOCKED","Upstream source/evidence integrity must be fixed first")
    else:
        obs=ws["stages"]["observation"]
        source=obs["source"]
        if not source:
            record("original_video_sha","NEEDS_DATA","Upload genuine original video footage")
        elif not full_video_sha:
            record("original_video_sha","NEEDS_VERIFICATION",
                   "Run acceptance CLI with --verify-source-sha to rehash the full original video")
        else:
            try:
                with store.connection() as conn:
                    row=conn.execute("SELECT * FROM sources WHERE project_id=?",
                                     (project_id,)).fetchone()
                src=store._dict(row)
                path=store.source_path(src)
                actual=sha(path)
                if actual!=src["sha256"]:
                    raise ValueError("Original video digest changed")
                record("original_video_sha","PASS","All original source video bytes rehashed",
                       {"video_sha256":actual,"size_bytes":src["size_bytes"]})
            except (ValueError,OSError):
                record("original_video_sha","FAIL","Full source media SHA-256 does not match immutable ingestion")
        latest=obs["last_verified_job"]
        if not latest:
            record("frozen_worker_model","NEEDS_DATA",
                   "Execute the real native inference worker on Windows and verify its M1/M2 receipts")
        elif model_dir is None or frozen_provenance is None:
            record("frozen_worker_model","NEEDS_VERIFICATION",
                   "Supply --model-dir and --frozen-provenance to check frozen W04 model bytes")
        else:
            try:
                from streetlab_integration.worker import pinned_model
                model_sha,provenance_sha=pinned_model(model_dir,frozen_provenance)
                job=store.job(latest["job_id"])
                manifest=verified_run(store.root/"runs"/job["id"],source["sha256"])
                if manifest["model_tree_sha256"]!=model_sha:
                    raise ValueError("Executed model differs from current pinned W04")
                record("frozen_worker_model","PASS",
                       "Pinned W04 model/provenance tree rehashed and matched executed M2 worker",
                       {"model_tree_sha256":model_sha,
                        "frozen_provenance_sha256":provenance_sha,
                        "job_id":job["id"]})
            except (ValueError,OSError,ImportError,KeyError):
                record("frozen_worker_model","FAIL","Model/provenance hashes do not match original tracked run")
        geo=ws["stages"]["geometry"]
        if geo["status"]=="GEOMETRY_REVIEWED":
            record("survey_geometry","PASS",
                   "M3 actual source-pixel metric transform and independent check reviewed",
                   {"spatial_revision":geo["revision"],"metric_qa":geo["surveyed_local_meters"]})
        else:
            record("survey_geometry","NEEDS_DATA",
                   "Measured local-meter correspondences and manually reviewed movements required")
        baseline=latest_baseline(store,project_id)
        if baseline.get("status")=="NOT_CONFIGURED":
            record("native_baseline","NEEDS_DATA","Build M4 from separate physical counts and independent holdout")
        elif ws["stages"]["baseline"]["status"]=="STALE_GEOMETRY":
            record("native_baseline","BLOCKED","M4 baseline references an obsolete M3 geometry revision")
        elif baseline["runtime"]["status"]=="NOT_EXECUTED":
            record("native_baseline","NEEDS_DATA","Execute original site baseline with real SUMO/netconvert")
        else:
            try:
                assessment=_verify_m4(store,project_id,baseline)
                record("native_baseline","PASS",
                       "M4 actual SUMO XML rescored and reconciled to recorded QA",
                       assessment)
            except (ValueError,OSError,KeyError,TypeError):
                record("native_baseline","FAIL",
                       "SUMO baseline heuristic quality or receipt-to-tripinfo consistency failed")
        scenario=latest_scenario(store,project_id)
        if scenario.get("status")=="NOT_CONFIGURED":
            record("native_scenarios","NEEDS_DATA","Create and execute an operationally reviewed M5 intervention")
        elif ws["stages"]["scenarios"]["status"]=="STALE_BASELINE":
            record("native_scenarios","BLOCKED","M5 scenario bound to an old/ineligible M4 baseline")
        elif scenario["runtime"]["status"]=="NOT_EXECUTED":
            record("native_scenarios","NEEDS_DATA","Execute the 18 real SUMO paired sensitivity runs")
        elif checks["native_baseline"]["status"]!="PASS":
            record("native_scenarios","BLOCKED","Baseline must be rescored first")
        else:
            try:
                assessment=_verify_m5(store,project_id,baseline,scenario)
                record("native_scenarios","PASS","All 18 hashed actual SUMO tripinfos rescored against M5 results",
                       assessment)
            except (ValueError,OSError,KeyError,TypeError):
                record("native_scenarios","FAIL",
                       "Experiment result or one of 18 paired tripinfo outputs disagrees with receipt")
        if field_dir is None:
            record("field_records","NEEDS_DATA",
                   "Supply three independent real field records and their SHA-bound reviewer attestation")
        elif baseline.get("status")=="NOT_CONFIGURED":
            record("field_records","BLOCKED","Field records must reference a configured M4 baseline")
        else:
            try:
                outcome=verify_field_records(field_dir,project_id,baseline)
                record("field_records","PASS",
                       "Human-attested survey, physical census and trip-level holdout reconcile to M4",
                       outcome)
            except (ValueError,OSError,KeyError,TypeError):
                record("field_records","FAIL",
                       "Missing, mismatched or untrusted original field record data")
    completed=all(c["status"]=="PASS" for c in checks.values())
    return {
        "schema_version":1,
        "project":project,
        "acceptance_status":("TECHNICAL_REPRODUCIBILITY_COMPLETE_OPERATOR_ATTESTED"
                             if completed else "RELEASE_BLOCKED"),
        "technical_acceptance_checks_passed":completed,
        "production_release_approved":False,
        "real_world_intervention_effect_verified":False,
        "independent_field_observer_identity_authenticated":False,
        "checks":checks,
        "blocked_by":[key for key,value in checks.items() if value["status"]!="PASS"],
        "limits":[
            "No execution in CI can substitute for a real Windows source video and local pinned model",
            "M4 physical survey and M5 simulation QA are heuristic, not accreditation",
            "Operator field attestation is NOT independent scientific or legal authentication",
            "Full public production release requires operator oversight, local security and field review",
            "No observed intervention effect or causal benefit is established by any release gate",
        ],
    }
