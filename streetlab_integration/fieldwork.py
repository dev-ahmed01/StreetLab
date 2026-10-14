"""M9 original-field collection: blank forms, model-bound movement plan and read-only preflight.

The kit NEVER fabricates measurements. Passing field-file checks is not
independent scientific validation or release approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.reconstruction import latest_reconstruction, _json_bytes
from streetlab_integration.site_baseline import latest_baseline
from streetlab_integration.field_acceptance import (
    FieldEvidenceError, _safe_file, verify_field_records,
)
from streetlab_integration.workspace_read import project_workspace


class FieldworkError(ValueError):
    pass


ORIGINAL_FILES=("survey.json","census.csv","holdout.csv")
FIELD_FILES=ORIGINAL_FILES+("attestation.json",)


def field_plan(store: VideoStore, project_id: str) -> dict:
    """Use actual verified M3 source, never a synthetic road network."""
    ws=project_workspace(store,project_id)
    project=ws["project"]
    geo=latest_reconstruction(store,project_id)
    if geo.get("status")=="NOT_CONFIGURED" or ws["stages"]["geometry"]["status"]!="GEOMETRY_REVIEWED":
        return {"schema_version":1,"project_id":project_id,"project_name":project["name"],
                "status":"NEEDS_REVIEWED_GEOMETRY","spatial_revision":None,
                "source_video_sha256":None,"zones":[],"movements":[],
                "next_action":"Complete actual M2 video observation and independently surveyed M3 geometry.",
                "original_field_observations_created":False}
    m=geo["model"]
    movements=sorted(
        [{"from_zone":v["from_zone"],"to_zone":v["to_zone"]}
         for v in m["movements"]],
        key=lambda x:(x["from_zone"],x["to_zone"]))
    return {"schema_version":1,"project_id":project_id,"project_name":project["name"],
            "status":"PLAN_READY_FOR_INDEPENDENT_FIELD_MEASUREMENT",
            "spatial_revision":geo["revision"],
            "source_video_sha256":m["source_video_sha256"],
            "coordinate_system":"LOCAL_GROUND_PLANE_METERS_NOT_GPS",
            "zones":[{"zone_id":z["id"],"role":z["role"],
                      "reviewed_lane_count":z["lane_count"]} for z in m["zones"]],
            "movements":movements,"minimum_holdout_trips_per_movement":3,
            "next_action":"Independently measure road arms, physical movement-class vehicle census and separate trip-level holdout.",
            "original_field_observations_created":False}


def _csv(columns: list[str],rows: list[dict]) -> bytes:
    buf=io.StringIO(newline="")
    w=csv.DictWriter(buf,fieldnames=columns,lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _guide(plan: dict) -> str:
    movement_list="\n".join(
        "- "+m["from_zone"]+" -> "+m["to_zone"] for m in plan["movements"])
    return (
        "# StreetLab M9: REAL-SITE FIELD COLLECTION (EMPTY FORMS ONLY)\n\n"
        "**This directory is NOT field evidence. The templates contain ZERO measured counts,"
        " durations or ground-truth data.**\n\n"
        "Project UUID: "+plan["project_id"]+"\n"
        "M3 revision: "+plan["spatial_revision"]+"\n"
        "Source video SHA-256: "+plan["source_video_sha256"]+"\n\n"
        "## Movements to be physically reviewed\n"+movement_list+"\n\n"
        "## Original evidence procedure\n"
        "1. Survey the site in local meters, road arms, lane connectivity and signal control."
        " Use original survey notes and independent check distances.\n"
        "2. Count DISTINCT vehicles by actual movement and class during a dated physical"
        " count session. Never turn tracker IDs/detections into actual counts.\n"
        "3. Record EACH physical trip's observed travel time during a separate holdout session."
        " At least three observations per movement; do not copy SUMO/model means.\n"
        "4. Preserve originals in four separate files: survey.json, census.csv,"
        " holdout.csv and attestation.json. Do not rename template files as originals.\n"
        "5. Enter ACTUAL independent observations into /baseline, create an immutable M4"
        " revision, then verify original files reconcile with that revision.\n"
        "6. Put an actual separate reviewer and dated record in attestation.json."
        " Hash original records using Get-FileHash -Algorithm SHA256."
        " A hash cannot authenticate observer identity or prove that a real survey occurred.\n"
        "7. Run python -m streetlab_integration.fieldwork --workdir .streetlab-m5"
        " --project UUID --check ORIGINAL_FOLDER. This is only file reconciliation.\n\n"
        "## Original file headers\n"
        "census.csv: from_zone,to_zone,type_id,count\n"
        "holdout.csv: from_zone,to_zone,session_ref,travel_time_s\n\n"
        "No synthetic fixtures, personally identifying data, or original videos should"
        " be committed to GitHub. Physical release remains blocked until external review.\n"
    )


def blank_kit(plan: dict) -> dict[str,bytes]:
    if plan["status"]!="PLAN_READY_FOR_INDEPENDENT_FIELD_MEASUREMENT":
        raise FieldworkError("Complete independently reviewed M3 calibration before generating fieldwork forms")
    census=_csv(["from_zone","to_zone","type_id","count"],[
        {"from_zone":m["from_zone"],"to_zone":m["to_zone"],
         "type_id":"","count":""} for m in plan["movements"]])
    holdout=_csv(["from_zone","to_zone","session_ref","travel_time_s"],[
        {"from_zone":m["from_zone"],"to_zone":m["to_zone"],
         "session_ref":"","travel_time_s":""} for m in plan["movements"]])
    survey={
        "_template_status":"BLANK_FORM_NOT_ORIGINAL_EVIDENCE",
        "project_id":plan["project_id"],"spatial_revision":plan["spatial_revision"],
        "coordinate_system":"LOCAL_GROUND_PLANE_METERS_NOT_GPS",
        "center_world_m":None,"arms":[{"zone_id":z["zone_id"],
        "outer_world_m":None,"lane_count":None,"speed_mps":None,
        "measurement_ref":None} for z in plan["zones"]],
        "connections":[],"control":None,"review":None,
    }
    att={
        "_template_status":"BLANK_ATTESTATION_NOT_A_SIGNED_RECORD",
        "schema_version":1,"project_id":plan["project_id"],
        "baseline_revision":None,
        "source_video_sha256":plan["source_video_sha256"],
        "original_file_sha256":{name:None for name in ORIGINAL_FILES},
        "data_origin":"UNVERIFIED_DRAFT_NOT_FIELD_EVIDENCE",
        "contains_synthetic_or_simulation_generated_observations":None,
        "holdout_collected_without_using_sumo_results":None,
        "real_world_intervention_effect_observed":None,
        "field_data_collector":None,"independent_reviewer":None,
        "survey_date_and_site_reference":None,"review_record_reference":None,
    }
    return {
        "plan.json":_json_bytes(plan),
        "survey.template.json":_json_bytes(survey),
        "census.template.csv":census,
        "holdout.template.csv":holdout,
        "attestation.template.json":_json_bytes(att),
        "FIELD_COLLECTION_GUIDE.md":_guide(plan).encode("utf-8"),
    }


def create_field_kit(store: VideoStore,project_id: str,destination: Path) -> dict:
    """Create new blank fieldwork directory atomically; never overwrite originals."""
    files=blank_kit(field_plan(store,project_id))
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FieldworkError("Refusing to overwrite existing fieldwork files")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise FieldworkError("Output parent must already exist and cannot be a symlink")
    stage=Path(tempfile.mkdtemp(prefix=".streetlab-fieldkit-",dir=destination.parent))
    try:
        for name,data in files.items():
            (stage/name).write_bytes(data)
        (stage/"TEMPLATE_SHA256SUMS.json").write_bytes(_json_bytes(
            {name:hashlib.sha256(data).hexdigest() for name,data in files.items()}))
        os.replace(stage,destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"status":"BLANK_KIT_CREATED_NOT_FIELD_EVIDENCE",
            "project_id":project_id,"files":sorted([*files,"TEMPLATE_SHA256SUMS.json"]),
            "original_field_observations_created":False,"release_approved":False}


def original_file_hashes(directory: Path) -> dict:
    """Hash bounded supplied originals. This never creates an attestation."""
    if directory.is_symlink() or not directory.is_dir():
        raise FieldworkError("Original field records directory absent or untrusted")
    return {"status":"CHECKSUMS_ONLY_NOT_VALIDATION",
            "original_file_sha256":{
                name:hashlib.sha256(_safe_file(directory,name)).hexdigest()
                for name in ORIGINAL_FILES},
            "observer_identity_authenticated":False,"release_approved":False}


def check_fieldwork(store: VideoStore, project_id: str, directory: Path) -> dict:
    """Read-only file/model consistency; no scientific truth or production approval."""
    plan=field_plan(store,project_id)
    if plan["status"]!="PLAN_READY_FOR_INDEPENDENT_FIELD_MEASUREMENT":
        return {"status":"NEEDS_REVIEWED_GEOMETRY","field_records_reconciled":False,
                "release_approved":False}
    baseline=latest_baseline(store,project_id)
    if baseline.get("status")=="NOT_CONFIGURED":
        return {"status":"NEEDS_M4_BASELINE","field_records_reconciled":False,
                "release_approved":False}
    if baseline["model"]["spatial_revision"]!=plan["spatial_revision"]:
        return {"status":"STALE_M4_BASELINE","field_records_reconciled":False,
                "release_approved":False}
    if directory.is_symlink() or not directory.is_dir():
        return {"status":"NEEDS_ORIGINAL_RECORDS",
                "missing_files":list(FIELD_FILES),
                "field_records_reconciled":False,"release_approved":False}
    missing=[n for n in FIELD_FILES if not (directory/n).is_file() or (directory/n).is_symlink()]
    if missing:
        return {"status":"NEEDS_ORIGINAL_RECORDS","missing_files":missing,
                "field_records_reconciled":False,"release_approved":False}
    try:
        receipt=verify_field_records(directory,project_id,baseline)
    except (FieldEvidenceError,ValueError,OSError,TypeError,KeyError,json.JSONDecodeError) as exc:
        return {"status":"FIELD_EVIDENCE_CONTRADICTION",
                "reason":str(exc) if isinstance(exc,FieldEvidenceError)
                else "Stored baseline and original field records could not be reconciled",
                "field_records_reconciled":False,"release_approved":False}
    return {"status":"FILES_RECONCILED_OPERATOR_ATTESTED_ONLY",
            "field_records_reconciled":True,
            "release_approved":False,
            "baseline_revision":baseline["revision"],
            "original_file_sha256":receipt["original_file_sha256"],
            "observer_identity_authenticated":False,
            "real_world_intervention_effect_verified":False}


def main() -> None:
    parser=argparse.ArgumentParser(description="StreetLab fieldwork kit and independent original-file preflight")
    parser.add_argument("--workdir",default=".streetlab-m5")
    parser.add_argument("--project",required=True)
    operations=parser.add_mutually_exclusive_group(required=True)
    operations.add_argument("--plan",action="store_true")
    operations.add_argument("--init",metavar="NEW_DIRECTORY",type=Path)
    operations.add_argument("--hashes",metavar="ORIGINAL_DIRECTORY",type=Path)
    operations.add_argument("--check",metavar="ORIGINAL_DIRECTORY",type=Path)
    args=parser.parse_args()
    store=VideoStore(args.workdir)
    if args.plan: output=field_plan(store,args.project)
    elif args.init is not None: output=create_field_kit(store,args.project,args.init)
    elif args.hashes is not None: output=original_file_hashes(args.hashes)
    else: output=check_fieldwork(store,args.project,args.check)
    print(json.dumps(output,indent=2))
    if output["status"] in {"NEEDS_REVIEWED_GEOMETRY","NEEDS_M4_BASELINE",
                            "STALE_M4_BASELINE","NEEDS_ORIGINAL_RECORDS",
                            "FIELD_EVIDENCE_CONTRADICTION"}:
        raise SystemExit(2)


if __name__=="__main__":
    main()
