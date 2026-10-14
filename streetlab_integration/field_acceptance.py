"""M8 independent field-record reconciliation against a frozen M4 site model.

Reads operator-provided local field records only from trusted CLI arguments;
no HTTP upload, auto-generated field records, or demo-fixture promotion.
SHA pins catch later file changes; no software can prove a human survey true.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

from streetlab_integration.primary_runner import sha

MAX_FIELD_FILE=2_000_000
REQUIRED=("survey.json","census.csv","holdout.csv","attestation.json")


class FieldEvidenceError(ValueError):
    pass


def _safe_file(folder: Path, name: str) -> bytes:
    p=folder/name
    if p.is_symlink() or not p.is_file() or p.stat().st_size>MAX_FIELD_FILE:
        raise FieldEvidenceError("A bounded, local original field record is absent or untrusted")
    return p.read_bytes()


def _note(v: object,label: str) -> str:
    if not isinstance(v,str) or not 12<=len(v.strip())<=250:
        raise FieldEvidenceError(f"{label} needs a dated 12–250 character reference")
    return v.strip()


def _rows(data: bytes, columns: tuple[str,...], limit: int) -> list[dict]:
    try:
        decoded=data.decode("utf-8-sig")
        reader=csv.DictReader(io.StringIO(decoded,newline=""))
        if reader.fieldnames!=list(columns):
            raise FieldEvidenceError("Field CSV header does not match the required schema")
        rows=list(reader)
    except (UnicodeDecodeError,csv.Error) as exc:
        raise FieldEvidenceError("Field CSV cannot be read as UTF-8") from exc
    if not rows or len(rows)>limit or any(None in d or any(v is None for v in d.values()) for d in rows):
        raise FieldEvidenceError("Field CSV is empty, too large or has inconsistent rows")
    return rows


def _sha256(v: object) -> bool:
    return isinstance(v,str) and re.fullmatch(r"[0-9a-f]{64}",v) is not None


def verify_field_records(folder: Path,project_id: str,baseline: dict) -> dict:
    """Cross-reconcile original field observation rows to the exact M4 revision.

    Returns operator-attested evidence status, NEVER independent scientific proof.
    """
    if not isinstance(folder,Path) or folder.is_symlink() or not folder.is_dir():
        raise FieldEvidenceError("Supply a real local directory of field observation records")
    files={name:_safe_file(folder,name) for name in REQUIRED}
    try:
        att=json.loads(files["attestation.json"])
        survey=json.loads(files["survey.json"])
    except (ValueError,UnicodeDecodeError) as exc:
        raise FieldEvidenceError("Survey and attestation must be valid JSON documents") from exc
    if not isinstance(att,dict) or not isinstance(survey,dict):
        raise FieldEvidenceError("Field survey and attestation must be objects")
    model=baseline["model"]
    expected={
        "survey.json":hashlib.sha256(files["survey.json"]).hexdigest(),
        "census.csv":hashlib.sha256(files["census.csv"]).hexdigest(),
        "holdout.csv":hashlib.sha256(files["holdout.csv"]).hexdigest(),
    }
    if (att.get("schema_version")!=1 or att.get("project_id")!=project_id or
        att.get("baseline_revision")!=baseline["revision"] or
        att.get("source_video_sha256")!=model["source_video_sha256"] or
        att.get("original_file_sha256")!=expected):
        raise FieldEvidenceError("Field attestations are not SHA-bound to this exact video and M4 baseline")
    if (att.get("data_origin")!="REAL_FIELD_RECORDS_OPERATOR_ATTESTED" or
        att.get("contains_synthetic_or_simulation_generated_observations") is not False or
        att.get("holdout_collected_without_using_sumo_results") is not True or
        att.get("real_world_intervention_effect_observed") is not False):
        raise FieldEvidenceError("No synthetic field fixture or modeled comparison may be attested as observed")
    collector=_note(att.get("field_data_collector"),"data collector")
    reviewer=_note(att.get("independent_reviewer"),"independent reviewer")
    _note(att.get("survey_date_and_site_reference"),"site/date")
    _note(att.get("review_record_reference"),"review record")
    if collector.casefold()==reviewer.casefold():
        raise FieldEvidenceError("Two distinct reviewer identities are necessary for field cross-check")
    if (survey.get("project_id")!=project_id or
        survey.get("baseline_revision")!=baseline["revision"] or
        survey.get("coordinate_system")!="LOCAL_GROUND_PLANE_METERS_NOT_GPS" or
        survey.get("center_world_m")!=model["center_world_m"] or
        survey.get("arms")!=model["arms"] or
        survey.get("connections")!=model["connections"] or
        survey.get("control")!=model["control"]):
        raise FieldEvidenceError("Field survey geometry, lane or signals differ from the frozen M4 model")
    census=_rows(files["census.csv"],("from_zone","to_zone","type_id","count"),200)
    measured={}
    for d in census:
        key=(d["from_zone"],d["to_zone"],d["type_id"])
        try:
            n=int(d["count"])
        except (TypeError,ValueError) as exc:
            raise FieldEvidenceError("Physical census counts must be integers") from exc
        if str(n)!=d["count"] or n<0 or n>20000 or key in measured:
            raise FieldEvidenceError("Census has invalid, negative or duplicate movement-class count")
        measured[key]=n
    supplied={(d["from_zone"],d["to_zone"],d["type_id"]):d["count"]
              for d in model["demand"]}
    if measured!=supplied:
        raise FieldEvidenceError("Individual physical census movement-class counts contradict M4")
    # Real trip-level holdout, not a mean copied from the M4 model or SUMO.
    hold=_rows(files["holdout.csv"],
               ("from_zone","to_zone","session_ref","travel_time_s"),100000)
    trips={}
    observed_sessions=set()
    for row in hold:
        key=(row["from_zone"],row["to_zone"])
        session=_note(row["session_ref"],"trip-level holdout session")
        if session==model["window"]["survey_session_ref"]:
            raise FieldEvidenceError("Holdout must use an independent observed session")
        observed_sessions.add(session)
        try:
            value=float(row["travel_time_s"])
        except (TypeError,ValueError) as exc:
            raise FieldEvidenceError("Holdout travel times must be finite seconds") from exc
        if not math.isfinite(value) or not 1<=value<=3600:
            raise FieldEvidenceError("Invalid physical holdout travel time")
        trips.setdefault(key,[]).append((session,value))
    holdouts={(h["from_zone"],h["to_zone"]):h for h in model["holdout"]}
    if set(trips)!=set(holdouts):
        raise FieldEvidenceError("Trip-level holdout must cover every reviewed movement")
    aggregate=[]
    for key,h in sorted(holdouts.items()):
        values=trips[key]
        if len(values)!=h["samples"] or any(s!=h["session_ref"] for s,_ in values):
            raise FieldEvidenceError("Trip-level samples/sessions differ from frozen independent holdout")
        mean=sum(v for _,v in values)/len(values)
        if abs(mean-h["mean_travel_time_s"])>.02:
            raise FieldEvidenceError("Trip-level observed mean is inconsistent with M4 source holdout")
        aggregate.append({"from_zone":key[0],"to_zone":key[1],
                          "observed_trip_samples":len(values),
                          "observed_mean_s":round(mean,5)})
    return {
        "status":"ORIGINAL_RECORDS_RECONCILED_OPERATOR_ATTESTED",
        "project_id":project_id,"baseline_revision":baseline["revision"],
        "original_file_sha256":expected,
        "attestation_file_sha256":hashlib.sha256(files["attestation.json"]).hexdigest(),
        "field_data_collector":collector,"independent_reviewer":reviewer,
        "independent_holdout_session_count":len(observed_sessions),
        "trip_level_holdout":aggregate,"physical_census_total":sum(measured.values()),
        "externally_authenticated_observers":False,
        "real_world_effect_verified":False,
        "limitations":[
            "SHA binding proves file consistency, not field measurement truth or independence",
            "Collector/reviewer identities are text attestations and not digitally authenticated",
            "Original field records can be fraudulent even if their checksums match",
            "Measured before/intervention outcomes still require independent validation",
        ],
    }
