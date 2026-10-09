"""M7 deterministic, minimal, auditable site evidence reports.

Provides the human-readable study summary and an offline-verifiable ZIP with
small approved receipts ONLY. Deliberately excludes private/raw video,
source-native tracking rows, calibrated per-frame track coordinates, photo
overlays and high-volume SUMO output. A checksum is NOT a digital signature.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.worker import job_report
from streetlab_integration.primary_runner import verified_run
from streetlab_integration.workspace_read import project_workspace
from streetlab_integration.reconstruction import latest_reconstruction, _json_bytes
from streetlab_integration.site_baseline import latest_baseline
from streetlab_integration.scenario_experiments import latest_scenario

MAX_EVIDENCE_FILE=2*1024*1024
MAX_TOTAL_BYTES=12*1024*1024
ZIP_TIMESTAMP=(2024,1,1,0,0,0)
KNOWN_PATH=re.compile(r"^[A-Za-z0-9_.\-/]{1,180}$")


class ReportError(ValueError):
    pass


def _read_small(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ReportError("Report evidence file missing or untrusted")
    if path.stat().st_size>MAX_EVIDENCE_FILE:
        raise ReportError("Metadata receipt exceeds report export safety budget")
    return path.read_bytes()


def _evidence(store: VideoStore, project_id: str, workspace: dict) -> dict[str,bytes]:
    root=store.root
    files={}
    obs=workspace["stages"]["observation"]
    verified=obs["last_verified_job"]
    if verified:
        job=store.job(verified["job_id"])
        source=store.source(job["source_id"])
        observed=job_report(store,job["id"])
        manifest=verified_run(root/"runs"/job["id"],source["sha256"])
        if observed["source_tracking_sha256"]!=verified["source_tracking_sha256"]:
            raise ReportError("M1 tracking receipt changed while packaging export")
        files["evidence/m1_observation_receipt.json"]=_read_small(
            root/"observations"/observed["source_tracking_sha256"]/"report.json")
        files["evidence/m2_source_run_manifest.json"]=_read_small(
            root/"runs"/job["id"]/"manifest.json")
    spatial=latest_reconstruction(store,project_id)
    if spatial.get("status")!="NOT_CONFIGURED":
        rev=spatial["revision"]
        folder=root/"projects"/project_id/"reconstructions"/rev
        files["evidence/m3_site_model.json"]=_read_small(folder/"site_model.json")
        files["evidence/m3_geometry_quality.json"]=_read_small(folder/"quality.json")
    baseline=latest_baseline(store,project_id)
    if baseline.get("status")!="NOT_CONFIGURED":
        rev=baseline["revision"]
        folder=root/"projects"/project_id/"baselines"/rev
        files["evidence/m4_baseline_input.json"]=_read_small(folder/"site_input.json")
        files["evidence/m4_baseline_quality.json"]=_read_small(folder/"quality.json")
        if baseline["runtime"]["status"]!="NOT_EXECUTED":
            files["evidence/m4_sumo_run_receipt.json"]=_read_small(folder/"runtime"/"run_receipt.json")
    experiment=latest_scenario(store,project_id)
    if experiment.get("status")!="NOT_CONFIGURED":
        rev=experiment["revision"]
        folder=root/"projects"/project_id/"scenarios"/rev
        files["evidence/m5_scenario_proposal.json"]=_read_small(folder/"proposal.json")
        files["evidence/m5_scenario_quality.json"]=_read_small(folder/"quality.json")
        if experiment["runtime"]["status"]!="NOT_EXECUTED":
            files["evidence/m5_paired_experiment_receipt.json"]=_read_small(
                folder/"runtime"/"experiment_receipt.json")
    return files


def report_dict(store: VideoStore, project_id: str) -> dict:
    """Construct report using verified project read view without source media."""
    overview=project_workspace(store,project_id)
    stages=overview["stages"]
    src=stages["observation"]["source"]
    job=stages["observation"]["last_verified_job"]
    geometry=stages["geometry"]
    baseline=stages["baseline"]
    experiment=stages["scenarios"]
    return {
        "schema_version":1,
        "report_kind":"SITE_PROJECT_EVIDENCE_AND_SIMULATION_ASSUMPTIONS",
        "project":overview["project"],
        "source_video":({
            "sha256":src["sha256"],"original_filename":src["original_filename"],
            "size_bytes":src["size_bytes"],"native_metadata":src["metadata"],
            "video_binary_in_export":False} if src else None),
        "observation":({
            "status":stages["observation"]["status"],
            "tracking_export_sha256":job["source_tracking_sha256"] if job else None,
            "source_tracker_identity_count":job["tracking_identity_count"] if job else None,
            "observed_native_tracking_points":job["observed_points"] if job else None,
            "verified_distinct_physical_vehicles":False,
            "physical_vehicle_count":None,
            "coordinate_system":"SOURCE_IMAGE_PIXELS"} if src else
            {"status":"NEEDS_SOURCE","coordinate_system":"SOURCE_IMAGE_PIXELS",
             "physical_vehicle_count":None}),
        "survey_geometry":{
            "status":geometry["status"],"revision":geometry["revision"],
            "metric_transform_reviewed":geometry["surveyed_local_meters"],
            "manual_allowed_movements":geometry["movement_count"],
            "ground_plane_not_gps":True,"missing_evidence":geometry["missing_evidence"]},
        "site_baseline":{
            "status":baseline["status"],"revision":baseline["revision"],
            "sumo_run_status":baseline["execution_status"],
            "reviewed_field_vehicle_count":baseline["manual_distinct_vehicle_count"],
            "scenario_gate":baseline["scenario_eligible"],
            "independent_holdout_comparison":baseline["movement_fidelity"],
            "missing_evidence":baseline["missing_evidence"],
            "baseline_qa_not_physical_validation":True},
        "scenarios":{
            "status":experiment["status"],"revision":experiment["revision"],
            "sumo_run_status":experiment["execution_status"],
            "hypothetical_intervention":experiment["intervention"],
            "paired_simulation_comparison":experiment["comparison"],
            "real_world_effect_verified":False,
            "missing_evidence":experiment["missing_evidence"]},
        "next_action":overview["next_action"],
        "research_policy":"FROZEN_W04_PRIMARY_HARD_NMS_IOS_030",
        "research_primary_unchanged":True,
        "phase2_synthetic_demo_separate":True,
        "claim_level":"SIMULATION_HYPOTHESES_ONLY_NOT_REAL_WORLD_CAUSAL_EVIDENCE",
        "limitations":[
            "M1 tracking IDs do not equal a census of distinct physical vehicles",
            "Source-pixel trajectories alone establish no real-world metric speed or turn demand",
            "Survey evidence references are operator attestations, not independently audited by software",
            "M4 baseline QA uses heuristic travel-time and completion thresholds",
            "M5 scenario comparisons are hypothetical; no observed impact is established",
            "Evidence ZIP contains selected metadata only: no original video/tracking rows/overlays",
            "SHA-256 verification detects ordinary accidental changes but does not authenticate authors",
        ],
    }


def markdown_report(report: dict) -> str:
    def v(value):
        return "Not available" if value is None else str(value)
    p=report["project"]
    obs=report["observation"]
    geo=report["survey_geometry"]
    base=report["site_baseline"]
    exp=report["scenarios"]
    comparison=exp["paired_simulation_comparison"]
    lines=[
        "# StreetLab project evidence report",
        "",
        "Project: "+p["name"],
        "Project ID: "+p["id"],
        "Research tracking policy: "+report["research_policy"],
        "",
        "## 1. Source observation",
        "Evidence state: "+obs["status"],
        "Native tracker IDs: "+v(obs.get("source_tracker_identity_count")),
        "Native tracking observations: "+v(obs.get("observed_native_tracking_points")),
        "Physical unique vehicle count: **NOT established by tracking IDs**",
        "",
        "## 2. Survey and reconstruction",
        "Evidence state: "+geo["status"],
        "Spatial revision: "+v(geo["revision"]),
        "Independent metric plane check: "+v(geo["metric_transform_reviewed"]),
        "This local plane is not GPS and does not independently confirm demand.",
        "",
        "## 3. Observed-site SUMO baseline",
        "Evidence state: "+base["status"],
        "SUMO run: "+base["sumo_run_status"],
        "Field-count physical vehicles: "+v(base["reviewed_field_vehicle_count"]),
        "Provisional scenario eligibility: "+v(base["scenario_gate"]),
        "Baseline QA thresholds are heuristic, not empirical model accreditation.",
        "",
        "## 4. Paired hypothetical scenario comparison",
        "Evidence state: "+exp["status"],
        "SUMO run: "+exp["sumo_run_status"],
        "Simulated average paired difference, seconds: "+v(
            comparison.get("mean_paired_simulation_difference_s") if comparison else None),
        "All conditions comparable: "+v(
            comparison.get("all_conditions_comparable") if comparison else None),
        "**No causal or real-road effect is verified.**",
        "",
        "## Required next action",
        report["next_action"]["reason"],
        "",
        "## Evidence and limitations",
        *("- "+x for x in report["limitations"]),
        "",
        "The archive contains selected small checksummed provenance receipts only.",
        "",
    ]
    return "\n".join(lines)


def bundle_bytes(store: VideoStore, project_id: str) -> bytes:
    report=report_dict(store,project_id)
    evidence=_evidence(store,project_id,project_workspace(store,project_id))
    entries={"report.json":_json_bytes(report),
             "report.md":markdown_report(report).encode("utf-8"),**evidence}
    if sum(len(blob) for blob in entries.values())>MAX_TOTAL_BYTES:
        raise ReportError("Project evidence exceeds the bounded metadata export limit")
    manifest={
        "schema_version":1,"project_id":project_id,
        "report_provenance":"IMMUTABLE_SOURCE_HASH_LINKED_SELECTED_METADATA_ONLY",
        "raw_footage_included":False,"native_tracking_rows_included":False,
        "sha256":{name:hashlib.sha256(data).hexdigest()
                  for name,data in sorted(entries.items())},
        "not_a_digital_signature":True,
    }
    entries["SHA256SUMS.json"]=_json_bytes(manifest)
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6,
                         allowZip64=False) as archive:
        for name in sorted(entries):
            meta=zipfile.ZipInfo(filename=name,date_time=ZIP_TIMESTAMP)
            meta.compress_type=zipfile.ZIP_DEFLATED
            meta.external_attr=0o644<<16
            archive.writestr(meta,entries[name],compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=6)
    result=out.getvalue()
    check_bundle(result,project_id)
    return result


def check_bundle(data: bytes, project_id: str | None = None) -> dict:
    """Offline verifier: whitelisted members and exact SHA values, no extraction."""
    if not isinstance(data,bytes) or len(data)>MAX_TOTAL_BYTES:
        raise ReportError("Invalid oversized report archive")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names=archive.namelist()
            if (len(names)!=len(set(names)) or not 3<=len(names)<=16
                or any(not KNOWN_PATH.fullmatch(n) or n.startswith("/")
                       or ".." in n.split("/") for n in names)
                or not {"report.json","report.md","SHA256SUMS.json"}<=set(names)):
                raise ReportError("Untrusted report package member names")
            entries={}
            for item in archive.infolist():
                if (item.file_size>MAX_EVIDENCE_FILE
                    or item.is_dir() or item.flag_bits & 0x1):
                    raise ReportError("Unsafe compressed evidence member")
                payload=archive.read(item.filename)
                if len(payload)!=item.file_size:
                    raise ReportError("Compressed evidence size mismatch")
                entries[item.filename]=payload
    except (zipfile.BadZipFile,OSError,RuntimeError,EOFError) as exc:
        raise ReportError("Report archive is damaged or unsupported") from exc
    manifest=json.loads(entries.pop("SHA256SUMS.json"))
    if (manifest.get("schema_version")!=1 or
        (project_id is not None and manifest.get("project_id")!=project_id)
        or set(manifest.get("sha256",{}))!=set(entries)):
        raise ReportError("Project report manifest binding mismatch")
    for name,expected in manifest["sha256"].items():
        if hashlib.sha256(entries[name]).hexdigest()!=expected:
            raise ReportError("Offline evidence SHA mismatch")
    report=json.loads(entries["report.json"])
    if report["project"]["id"]!=manifest["project_id"]:
        raise ReportError("Report identity does not match the manifest")
    return manifest
