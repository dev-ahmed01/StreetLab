"""M5 software contract and real SUMO integration tests.

All numerical road/site fixtures are SYNTHETIC. The native-engine acceptance
below manufactures a *software-only reference travel time* from SUMO itself;
it explicitly does NOT validate independent physical road observations.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient
import pytest

pytest.importorskip("streetlab_integration.scenario_experiments")

from test_streetlab_integration_m3 import site, local
from test_streetlab_integration_m4 import field_fixture
from streetlab_integration.reconstruction import reconstruct
from streetlab_integration.site_baseline import (
    save_baseline, verified_baseline, run_baseline,
)
from streetlab_integration.site_scenarios import (
    ScenarioError, validate_scenario, experiment_model, completed_trip_times,
    paired_scorecard,
)
from streetlab_integration.scenario_experiments import (
    create_scenario, verified_scenario, latest_scenario, run_scenario,
)


def scenario_input(kind="APPROACH_SPEED_LIMIT"):
    return {
        "intervention":(
            {"kind":"APPROACH_SPEED_LIMIT","zone_id":"west_in","speed_mps":7.5}
            if kind=="APPROACH_SPEED_LIMIT" else
            {"kind":"FIXED_SIGNAL_PLAN","phase_durations_s":[25,4]}
        ),
        "assumptions":{
            "decision_question":"Whether the changed rule could affect paired simulated junction travel times",
            "feasibility_evidence_ref":"Site manager hypothetical permit and sign installation review record",
            "reviewer":"Researcher attested design feasibility review on independent software fixture",
            "provenance":"HYPOTHETICAL_REVIEWED",
        },
        "demand_multipliers":[0.9,1.0,1.1],
    }


def publish_fake_passing_m4(local,monkeypatch, *, signalized=False):
    """Only synthetic test harness; cannot count as a real field baseline."""
    db,project,job=local
    geo=reconstruct(db,project["id"],job["id"],site())
    inputs=field_fixture()
    if signalized:
        inputs["control"]={
            "kind":"FIXED_TIME_SIGNAL",
            "measurement_ref":"Synthetic reviewed phase source 2026-10-01",
            "link_index_review_ref":"Synthetic link index review 2026-10-01",
            "phases":[{"duration_s":30,"state":"G"},{"duration_s":3,"state":"r"}],
        }
    packaged=save_baseline(db,project["id"],geo["revision"],inputs)
    monkeypatch.setattr("streetlab_integration.site_baseline.shutil.which",
                        lambda _: "software-fixture-tool")
    def fake_m4_command(args,**kw):
        where=Path(kw["cwd"])
        if "--node-files" in args:
            (where/"site.net.xml").write_text("<net/>")
        else:
            (where/"summary.xml").write_text("<summary/>")
            (where/"tripinfo.xml").write_text(
                '<tripinfos>'+''.join(
                    f'<tripinfo id="flow_000.{i}" duration="70"/>' for i in range(5)
                )+'</tripinfos>')
        return subprocess.CompletedProcess(args,0,"","")
    monkeypatch.setattr("streetlab_integration.site_baseline.subprocess.run",
                        fake_m4_command)
    assert run_baseline(db,project["id"],packaged["revision"])["status"]=="BASELINE_FIDELITY_CHECKED"
    return db,project,geo,packaged


@pytest.fixture
def admitted(local,monkeypatch):
    return publish_fake_passing_m4(local,monkeypatch)


def test_site_gate_refuses_unexecuted_and_bad_baseline(local):
    db,project,job=local
    geo=reconstruct(db,project["id"],job["id"],site())
    packaged=save_baseline(db,project["id"],geo["revision"],field_fixture())
    with pytest.raises(ScenarioError,match="without a successful M4"):
        create_scenario(db,project["id"],packaged["revision"],scenario_input())


@pytest.mark.parametrize("edit,error",[
    (lambda d:d["intervention"].update({"speed_mps":12}),"only allows reductions"),
    (lambda d:d["intervention"].update({"speed_mps":1}),"only allows reductions"),
    (lambda d:d["intervention"].update({"zone_id":"unseen"}),"must exist"),
    (lambda d:d["intervention"].update({"zone_id":"east_out"}),"APPROACH"),
    (lambda d:d["intervention"].update({"kind":"ROAD_CLOSURE"}),"closures need verified detours"),
    (lambda d:d["assumptions"].update({"provenance":"MEASURED_EFFECT"}),"HYPOTHETICAL_REVIEWED"),
    (lambda d:d.update({"demand_multipliers":[1,1,1]}),"low"),
    (lambda d:d.update({"demand_multipliers":[0.5,1,2.0]}),"low"),
])
def test_assumptions_demand_and_unsupported_interventions_fail(admitted,edit,error):
    db,project,_,baseline=admitted
    input_=scenario_input();edit(input_)
    with pytest.raises((ScenarioError,ValueError),match=error):
        validate_scenario(input_,verified_baseline(db,project["id"],baseline["revision"]))


def test_change_does_not_mutate_reviewed_M4_site(admitted):
    db,project,_,baseline=admitted
    imported=verified_baseline(db,project["id"],baseline["revision"])
    proposal=validate_scenario(scenario_input(),imported)
    original=deepcopy(imported["model"])
    unchanged=experiment_model(original,proposal,.9,False)
    modified=experiment_model(original,proposal,.9,True)
    assert original==imported["model"]
    assert unchanged["arms"][0]["speed_mps"]==10
    assert modified["arms"][0]["speed_mps"]==7.5
    assert unchanged["demand"][0]["count"]==5
    assert modified["demand"][0]["count"]==5
    low=experiment_model(original,proposal,1.1,True)
    assert low["demand"][0]["count"]==6
    assert proposal["source_video_sha256"]==original["source_video_sha256"]


def test_signal_plan_requires_reviewed_original_phase_program(local,monkeypatch):
    db,project,_,baseline=publish_fake_passing_m4(local,monkeypatch,signalized=True)
    model=verified_baseline(db,project["id"],baseline["revision"])
    proposal=validate_scenario(scenario_input("FIXED_SIGNAL_PLAN"),model)
    alt=experiment_model(model["model"],proposal,1.0,True)
    assert [p["duration_s"] for p in alt["control"]["phases"]]==[25.,4.]
    assert [p["state"] for p in alt["control"]["phases"]]==["G","r"]
    original=model["model"]["control"]["phases"]
    assert [p["duration_s"] for p in original]==[30.,3.]
    bad=scenario_input("FIXED_SIGNAL_PLAN")
    bad["intervention"]["phase_durations_s"]=[20]
    with pytest.raises(ScenarioError,match="every existing signal phase"):
        validate_scenario(bad,model)


def test_tripinfo_censoring_and_nine_pair_required(admitted):
    db,project,_,baseline=admitted
    model=verified_baseline(db,project["id"],baseline["revision"])["model"]
    valid=completed_trip_times(model,(
        "<tripinfos>"+''.join(
            f'<tripinfo id="flow_000.{i}" duration="60"/>' for i in range(5)
        )+"</tripinfos>").encode())
    assert valid["completion_ratio"]==1.0
    assert valid["mean_trip_duration_s"]==60
    partial=completed_trip_times(model,b'<tripinfos><tripinfo id="flow_000.0" duration="50"/></tripinfos>')
    assert partial["completion_ratio"]==.2
    with pytest.raises(ScenarioError,match="nine"):
        paired_scorecard([{"baseline":valid,"intervention":valid,"seed":42,
                          "demand_multiplier":1.0}])
    pairs=[{"baseline":valid,"intervention":dict(valid,mean_trip_duration_s=50),
            "seed":seed,"demand_multiplier":scale}
           for scale in [.9,1,1.1] for seed in [42,43,44]]
    good=paired_scorecard(pairs)
    assert good["status"]=="SIMULATED_COMPARISON_ELIGIBLE"
    assert good["mean_paired_simulation_difference_s"]==-10
    assert good["simulated_improvement_in_every_condition"]
    assert good["real_world_outcome_verified"] is False
    pairs[0]["baseline"]=partial
    poor=paired_scorecard(pairs)
    assert poor["status"]=="INDETERMINATE_INCOMPLETE_TRIPS"
    assert poor["mean_paired_simulation_difference_s"] is None


def test_http_proposal_rejects_unobserved_site_and_does_not_simulate(local):
    db,project,job=local
    from streetlab_phase2.api import create_app
    client=TestClient(create_app(service=object(),observation_workdir=db.root))
    assert client.get("/scenarios").status_code==200
    assert client.get("/api/projects/"+project["id"]+"/scenario").json()["status"]=="NOT_CONFIGURED"
    resp=client.post("/api/projects/"+project["id"]+"/scenario",json={
        "baseline_revision":"0"*64,"scenario":scenario_input()})
    assert resp.status_code in (409,422)
    assert "href=\"/scenarios\"" in client.get("/").text


def test_immutable_proposals_http_and_fake_paired_Sumo_execution(admitted,monkeypatch):
    db,project,geo,baseline=admitted
    from streetlab_phase2.api import create_app
    client=TestClient(create_app(service=object(),observation_workdir=db.root))
    url="/api/projects/"+project["id"]+"/scenario"
    resp=client.post(url,json={"baseline_revision":baseline["revision"],
                               "scenario":scenario_input()})
    assert resp.status_code==201,resp.text
    revision=resp.json()["revision"]
    assert resp.json()["runtime"]["status"]=="NOT_EXECUTED"
    assert client.get(url+"/"+revision+"/results").status_code==422
    assert client.get(url+"/"+revision).status_code==200
    assert latest_scenario(db,project["id"])["revision"]==revision
    assert create_scenario(db,project["id"],baseline["revision"],scenario_input())["revision"]==revision
    # Emulate only the subprocess boundary. The result validates orchestration
    # and persistence; it must never be presented as real SUMO acceptance.
    monkeypatch.setattr("streetlab_integration.scenario_experiments._binary",
                        lambda which,folder:"synthetic-"+which)
    def runner(args,cwd,*,deadline):
        folder=Path(cwd)
        if args[0]=="synthetic-netconvert":
            (folder/"scenario.net.xml").write_text("<net/>")
            return subprocess.CompletedProcess(args,0,"","")
        assert args[0]=="synthetic-sumo"
        trip=args[args.index("--tripinfo-output")+1]
        summary=args[args.index("--summary-output")+1]
        routes=args[args.index("--route-files")+1]
        flow=ET.parse(folder/routes).find("flow")
        expected=int(flow.get("number"))
        duration=70 if "_baseline_" in trip else 60
        (folder/trip).write_text(
            "<tripinfos>"+''.join(
                f'<tripinfo id="flow_000.{i}" duration="{duration}"/>' for i in range(expected)
            )+"</tripinfos>")
        (folder/summary).write_text("<summary/>")
        return subprocess.CompletedProcess(args,0,"","")
    monkeypatch.setattr("streetlab_integration.scenario_experiments._call",runner)
    output=run_scenario(db,project["id"],revision)
    assert output["status"]=="SIMULATED_COMPARISON_ELIGIBLE"
    assert output["mean_paired_simulation_difference_s"]==-10
    assert output["paired_runs"]==9
    assert output["real_world_outcome_verified"] is False
    assert len(output["files"])==42
    assert client.get(url+"/"+revision+"/results").status_code==200
    assert client.get(url+"/"+revision+"/files/scenario.net.xml").status_code==200
    assert client.get(url+"/"+revision+"/files/../../README").status_code in (404,422)
    with pytest.raises(ScenarioError,match="already has immutable"):
        run_scenario(db,project["id"],revision)
    # Evidence SHA changes fail closed.
    runtime=db.root/"projects"/project["id"]/"scenarios"/revision/"runtime"
    receipt=runtime/"experiment_receipt.json"
    original=receipt.read_bytes()
    altered=json.loads(original)
    altered["scenario_revision"]="0"*64
    receipt.write_text(json.dumps(altered),encoding="utf-8")
    with pytest.raises(ScenarioError,match="does not match proposal"):
        verified_scenario(db,project["id"],revision)
    receipt.write_bytes(original)
    (runtime/"scenario.net.xml").write_text("tampered")
    with pytest.raises(ScenarioError,match="SHA integrity"):
        verified_scenario(db,project["id"],revision)


def test_cross_site_source_binding_and_proposal_immutability(admitted):
    db,project,geo,baseline=admitted
    new=create_scenario(db,project["id"],baseline["revision"],scenario_input())
    second=scenario_input()
    second["intervention"]["speed_mps"]=6.0
    newer=create_scenario(db,project["id"],baseline["revision"],second)
    assert newer["revision"]!=new["revision"]
    assert verified_scenario(db,project["id"],new["revision"])["revision"]==new["revision"]
    assert latest_scenario(db,project["id"])["revision"]==newer["revision"]
    other=db.create_project("Different spatial source")
    with pytest.raises(Exception):
        verified_scenario(db,other["id"],new["revision"])


@pytest.mark.parametrize("signalized",[False,True])
def test_real_sumo_paired_experiment_synthetic_fixture(local,signalized):
    """Actual SUMO, but holdout is derived from synthetic simulation -- NOT independent."""
    import shutil
    if not (shutil.which("sumo") and shutil.which("netconvert")):
        pytest.skip("Actual SUMO/netconvert not available on CI host")
    db,project,job=local
    geo=reconstruct(db,project["id"],job["id"],site())
    fixture=field_fixture()
    if signalized:
        fixture["control"]={
            "kind":"FIXED_TIME_SIGNAL",
            "measurement_ref":"Synthetic timed phase review 2026-10-01",
            "link_index_review_ref":"Synthetic lane link index review 2026-10-01",
            "phases":[{"duration_s":30,"state":"G"},{"duration_s":3,"state":"r"}],
        }
    initial=save_baseline(db,project["id"],geo["revision"],fixture)
    prior=run_baseline(db,project["id"],initial["revision"])
    duration=prior["per_movement"][0]["simulated_mean_s"]
    if not duration or prior["completion_ratio"]<.95:
        pytest.skip("Synthetic site baseline failed to complete all vehicles")
    # For toolchain integration ONLY, choose software generated reference
    # to open the M4 gate. This is NOT an independent empirical holdout.
    fixture["holdout"][0]["mean_travel_time_s"]=duration
    good=save_baseline(db,project["id"],geo["revision"],fixture)
    assert run_baseline(db,project["id"],good["revision"])["status"]=="BASELINE_FIDELITY_CHECKED"
    created=create_scenario(db,project["id"],good["revision"],
                            scenario_input("FIXED_SIGNAL_PLAN" if signalized else "APPROACH_SPEED_LIMIT"))
    output=run_scenario(db,project["id"],created["revision"])
    assert output["paired_runs"]==9
    assert len(output["records"])==9
    assert output["real_world_outcome_verified"] is False
    assert output["run_engine"]=="ACTUAL_SUMO_PAIRED_BASELINE_INTERVENTION"
    assert verified_scenario(db,project["id"],created["revision"])["runtime"]["status"]==output["status"]
