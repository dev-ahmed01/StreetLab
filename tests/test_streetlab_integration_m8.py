"""M8 release acceptance tests; every road fixture is synthetic software data.

The field-format test uses artificial attestation text purely to check schema
reconciliation. It DOES NOT demonstrate a genuine independent field study.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from test_streetlab_integration_m3 import local,site
from test_streetlab_integration_m4 import field_fixture
from test_streetlab_integration_m5 import publish_fake_passing_m4,scenario_input
from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.primary_runner import sha
from streetlab_integration.reconstruction import reconstruct
from streetlab_integration.site_baseline import save_baseline,run_baseline
from streetlab_integration.scenario_experiments import create_scenario,run_scenario
from streetlab_integration.release_acceptance import evaluate_release,_verify_m4,_verify_m5
from streetlab_integration.field_acceptance import verify_field_records,FieldEvidenceError
from streetlab_integration.release_cli import environment,human_markdown
from streetlab_integration.maintenance import check_db


def artificial_field_records(directory,project_id,baseline):
    """Artificial software contract fixture; never a real operator measurement."""
    directory.mkdir()
    model=baseline["model"]
    survey={
        "project_id":project_id,"baseline_revision":baseline["revision"],
        "coordinate_system":model["coordinate_system"],
        "center_world_m":model["center_world_m"],
        "arms":model["arms"],"connections":model["connections"],
        "control":model["control"],
    }
    (directory/"survey.json").write_text(json.dumps(survey),encoding="utf-8")
    with (directory/"census.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["from_zone","to_zone","type_id","count"])
        writer.writeheader()
        for d in model["demand"]:
            writer.writerow({key:d[key] for key in writer.fieldnames})
    with (directory/"holdout.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["from_zone","to_zone","session_ref","travel_time_s"])
        writer.writeheader()
        for h in model["holdout"]:
            for i in range(h["samples"]):
                writer.writerow({"from_zone":h["from_zone"],"to_zone":h["to_zone"],
                                 "session_ref":h["session_ref"],
                                 "travel_time_s":h["mean_travel_time_s"]})
    file_sha={filename:sha(directory/filename)
              for filename in ["survey.json","census.csv","holdout.csv"]}
    att={"schema_version":1,"project_id":project_id,"baseline_revision":baseline["revision"],
         "source_video_sha256":model["source_video_sha256"],
         "original_file_sha256":file_sha,
         "data_origin":"REAL_FIELD_RECORDS_OPERATOR_ATTESTED",
         "contains_synthetic_or_simulation_generated_observations":False,
         "holdout_collected_without_using_sumo_results":True,
         "real_world_intervention_effect_observed":False,
         "field_data_collector":"Artificial software fixture collector, dated October 2026",
         "independent_reviewer":"Separate synthetic fixture reviewer, October 2026",
         "survey_date_and_site_reference":"Synthetic unit testing site only, 2026 October",
         "review_record_reference":"Synthetic fixture review, NOT a scientific field record"}
    (directory/"attestation.json").write_text(json.dumps(att),encoding="utf-8")
    return directory


def test_release_without_observation_stays_blocked(tmp_path):
    db=VideoStore(tmp_path/"store")
    project=db.create_project("Proposed junction")
    result=evaluate_release(db,project["id"])
    assert result["acceptance_status"]=="RELEASE_BLOCKED"
    assert result["production_release_approved"] is False
    assert result["real_world_intervention_effect_verified"] is False
    assert result["checks"]["original_video_sha"]["status"]=="NEEDS_DATA"
    assert result["checks"]["frozen_worker_model"]["status"]=="NEEDS_DATA"
    assert result["checks"]["field_records"]["status"]=="NEEDS_DATA"
    assert "Production release approved: **NO**" in human_markdown(result)


def test_release_rehashes_entire_original_video_without_inference(local):
    db,project,job=local
    result=evaluate_release(db,project["id"],full_video_sha=True)
    assert result["checks"]["project_evidence"]["status"]=="PASS"
    assert result["checks"]["original_video_sha"]["status"]=="PASS"
    assert result["checks"]["original_video_sha"]["evidence"]["video_sha256"]==(
        db.source(job["source_id"])["sha256"])
    assert result["checks"]["frozen_worker_model"]["status"]=="NEEDS_VERIFICATION"
    assert result["checks"]["survey_geometry"]["status"]=="NEEDS_DATA"
    assert result["acceptance_status"]=="RELEASE_BLOCKED"


def test_release_detects_changed_source_media(local):
    db,project,job=local
    filename=db.source_path(db.source(job["source_id"]))
    old=filename.read_bytes()
    filename.write_bytes(b"X"+old[1:])
    result=evaluate_release(db,project["id"],full_video_sha=True)
    assert result["checks"]["original_video_sha"]["status"]=="FAIL"
    assert result["production_release_approved"] is False


def test_field_records_reconcile_counts_and_independent_raw_trip_means(local,tmp_path):
    db,project,job=local
    geo=reconstruct(db,project["id"],job["id"],site())
    baseline=save_baseline(db,project["id"],geo["revision"],field_fixture())
    folder=artificial_field_records(tmp_path/"field",project["id"],baseline)
    receipt=verify_field_records(folder,project["id"],baseline)
    assert receipt["status"]=="ORIGINAL_RECORDS_RECONCILED_OPERATOR_ATTESTED"
    assert receipt["physical_census_total"]==5
    assert receipt["trip_level_holdout"][0]["observed_trip_samples"]==5
    assert receipt["externally_authenticated_observers"] is False
    assert receipt["real_world_effect_verified"] is False
    assert evaluate_release(db,project["id"],field_dir=folder)["checks"]["field_records"]["status"]=="PASS"
    # A software contract cannot authenticate whether the artificial field
    # records were collected in the real world; this test never claims it did.
    data=json.loads((folder/"attestation.json").read_text())
    data["contains_synthetic_or_simulation_generated_observations"]=True
    (folder/"attestation.json").write_text(json.dumps(data))
    with pytest.raises(FieldEvidenceError,match="No synthetic"):
        verify_field_records(folder,project["id"],baseline)


def test_field_holdout_does_not_accept_copied_model_mean_or_different_counts(local,tmp_path):
    db,project,job=local
    geo=reconstruct(db,project["id"],job["id"],site())
    baseline=save_baseline(db,project["id"],geo["revision"],field_fixture())
    folder=artificial_field_records(tmp_path/"fields",project["id"],baseline)
    # Mutate one physical count; even if hashes were recomputed by another
    # local actor, reconciliation must reject the altered raw observations.
    (folder/"census.csv").write_text(
        (folder/"census.csv").read_text().replace(",5\n",",8\n"))
    att=json.loads((folder/"attestation.json").read_text())
    att["original_file_sha256"]["census.csv"]=sha(folder/"census.csv")
    (folder/"attestation.json").write_text(json.dumps(att))
    with pytest.raises(FieldEvidenceError,match="contradict"):
        verify_field_records(folder,project["id"],baseline)
    folder2=artificial_field_records(tmp_path/"fields2",project["id"],baseline)
    raw=(folder2/"holdout.csv").read_text()
    (folder2/"holdout.csv").write_text(raw.replace(",70.0\n",",120.0\n",1))
    att2=json.loads((folder2/"attestation.json").read_text())
    att2["original_file_sha256"]["holdout.csv"]=sha(folder2/"holdout.csv")
    (folder2/"attestation.json").write_text(json.dumps(att2))
    with pytest.raises(FieldEvidenceError,match="mean"):
        verify_field_records(folder2,project["id"],baseline)


def test_baseline_receipt_qa_is_recomputed_not_trusted(local,monkeypatch):
    db,project,geo,base=publish_fake_passing_m4(local,monkeypatch)
    baseline=__import__("streetlab_integration.site_baseline",fromlist=["verified_baseline"]).verified_baseline(
        db,project["id"],base["revision"])
    assert _verify_m4(db,project["id"],baseline)["status"]=="BASELINE_FIDELITY_CHECKED"
    receipt=db.root/"projects"/project["id"]/"baselines"/base["revision"]/"runtime"/"run_receipt.json"
    data=json.loads(receipt.read_text())
    data["completion_ratio"]=.951
    receipt.write_text(json.dumps(data))
    result=evaluate_release(db,project["id"])
    assert result["checks"]["native_baseline"]["status"]=="FAIL"
    assert result["technical_acceptance_checks_passed"] is False


def test_model_tree_pin_requires_real_frozen_provenance(local,tmp_path):
    db,project,job=local
    bogus=tmp_path/"model";bogus.mkdir()
    metadata=tmp_path/"provenance.json";metadata.write_text("{}")
    report=evaluate_release(db,project["id"],model_dir=bogus,
                            frozen_provenance=metadata)
    assert report["checks"]["frozen_worker_model"]["status"]=="FAIL"


def test_real_sumo_release_reconciliation_stays_blocked_without_field(tmp_path,local):
    """Real SUMO and two baseline runs. The holdout is software-generated
    test fixture; do not use this test as evidence of actual road validation.
    """
    import shutil
    if not (shutil.which("sumo") and shutil.which("netconvert")):
        pytest.skip("Real SUMO binaries unavailable")
    db,project,job=local
    geo=reconstruct(db,project["id"],job["id"],site())
    fixture=field_fixture()
    old=save_baseline(db,project["id"],geo["revision"],fixture)
    first=run_baseline(db,project["id"],old["revision"])
    if first["completion_ratio"]<.95:
        pytest.skip("Synthetic base SUMO needs completed trip fixture")
    fixture["holdout"][0]["mean_travel_time_s"]=first["per_movement"][0]["simulated_mean_s"]
    base=save_baseline(db,project["id"],geo["revision"],fixture)
    completed=run_baseline(db,project["id"],base["revision"])
    assert completed["status"]=="BASELINE_FIDELITY_CHECKED"
    scenario=create_scenario(db,project["id"],base["revision"],scenario_input())
    run_scenario(db,project["id"],scenario["revision"])
    result=evaluate_release(db,project["id"],full_video_sha=True)
    assert result["checks"]["original_video_sha"]["status"]=="PASS"
    assert result["checks"]["native_baseline"]["status"]=="PASS"
    assert result["checks"]["native_scenarios"]["status"]=="PASS"
    assert result["checks"]["native_scenarios"]["evidence"]["paired_comparisons"]==9
    assert result["checks"]["field_records"]["status"]=="NEEDS_DATA"
    assert result["checks"]["frozen_worker_model"]["status"]=="NEEDS_VERIFICATION"
    assert result["acceptance_status"]=="RELEASE_BLOCKED"
    assert result["production_release_approved"] is False
