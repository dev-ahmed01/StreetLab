"""M4 acceptance: M1-M4 provenance, explicit physical counts, local SUMO
compiler contracts, standalone bounded run and independent baseline QA.

Every road measurement in this test is SYNTHETIC SOFTWARE FIXTURE DATA;
passing tests does not demonstrate real-world survey or simulation fidelity.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient
import pytest

pytest.importorskip("streetlab_integration.site_baseline")
from test_streetlab_integration_m3 import site, local

from streetlab_integration.reconstruction import reconstruct
from streetlab_integration.site_inputs import SiteInputError,validate_site
from streetlab_integration.site_sumo import render_sumo,evaluate_tripinfo
from streetlab_integration.site_baseline import (
    save_baseline,verified_baseline,latest_baseline,run_baseline,
)


def field_fixture():
    ref="Field reviewed survey reference 2026-10-01; test-only fixture"
    hold="Distinct independent travel-time count on 2026-10-02"
    return {
        "center_world_m":[15,10],
        "arms":[
            {"zone_id":"west_in","outer_world_m":[-45,10],
             "lane_count":2,"speed_mps":10,"measurement_ref":ref},
            {"zone_id":"east_out","outer_world_m":[75,10],
             "lane_count":1,"speed_mps":10,"measurement_ref":ref}],
        "connections":[{"from_zone":"west_in","to_zone":"east_out","from_lane":0,"to_lane":0}],
        "control":{"kind":"UNSIGNALIZED_CONFIRMED","measurement_ref":ref},
        "window":{"duration_s":600,"survey_session_ref":"Field census time-window 2026-10-01"},
        "vehicle_types":[{"id":"motorbike","vclass":"motorcycle","length_m":2.1,
            "min_gap_m":1.1,"accel_mps2":2.5,"decel_mps2":4.0,
            "max_speed_mps":17,"sigma":0.4,"measurement_ref":ref}],
        "demand":[{"from_zone":"west_in","to_zone":"east_out",
                   "type_id":"motorbike","count":5,
                   "provenance":"FIELD_COUNT_DISTINCT_VEHICLES",
                   "measurement_ref":"Physical distinct vehicle count census log 2026-10-01"}],
        "holdout":[{"from_zone":"west_in","to_zone":"east_out",
                    "mean_travel_time_s":70,"samples":5,
                    "session_ref":"Independent sample collection 2026-10-02",
                    "measurement_ref":hold}],
        "review":{"reviewer":"Survey review record signed by two separate field observers",
                  "network_evidence_ref":ref,
                  "demand_evidence_ref":"Physical census results file 2026-10-01",
                  "holdout_evidence_ref":hold,"holdout_independent":True},
    }


@pytest.fixture
def ready(local):
    db,project,job=local
    spatial=reconstruct(db,project["id"],job["id"],site())
    package=save_baseline(db,project["id"],spatial["revision"],field_fixture())
    return db,project,spatial,package


def test_m4_compiler_uses_real_local_meters_and_declared_flow(ready):
    db,project,geo,package=ready
    assert package["quality"]["status"]=="REVIEWED_INPUTS_UNEXECUTED"
    assert package["quality"]["real_site_sumo_allowed"] is False
    assert package["runtime"]["status"]=="NOT_EXECUTED"
    assert package["quality"]["manual_distinct_vehicle_count"]==5
    folder=db.root/"projects"/project["id"]/"baselines"/package["revision"]
    nodes=ET.parse(folder/"nodes.nod.xml")
    edges=ET.parse(folder/"edges.edg.xml")
    links=ET.parse(folder/"connections.con.xml")
    routes=ET.parse(folder/"routes.rou.xml")
    assert len(nodes.findall("node"))==3
    assert nodes.find("./node[@id='j']").get("x")=="15.0"
    assert len(edges.findall("edge"))==2
    assert edges.find("./edge[@id='e_west_in']").get("numLanes")=="2"
    assert links.find("connection").get("fromLane")=="0"
    assert routes.find("flow").get("number")=="5"
    assert routes.find("flow").get("departLane")=="best"
    assert "source_to_world.csv" not in {p.name for p in folder.iterdir()}
    assert latest_baseline(db,project["id"])["revision"]==package["revision"]


@pytest.mark.parametrize("edit,part",[
    (lambda x:x["demand"][0].update({"provenance":"TRACKING_ID_COUNT"}),"Tracker IDs"),
    (lambda x:x["demand"][0].update({"count":1.2}),"integer"),
    (lambda x:x["connections"][0].update({"from_lane":3}),"Lane link"),
    (lambda x:x["arms"][0].update({"lane_count":3}),"conflicts"),
    (lambda x:x["arms"][0].update({"outer_world_m":[15,10]}),"15"),
    (lambda x:x["holdout"][0].update({"session_ref":"Field census time-window 2026-10-01"}),"differ"),
    (lambda x:x["review"].update({"holdout_independent":False}),"independent"),
    (lambda x:x["control"].update({"kind":"UNVERIFIED_SIGNAL"}),"Control"),
])
def test_critical_missing_or_guessed_evidence_fails_closed(ready,edit,part):
    db,project,geo,package=ready
    data=field_fixture();edit(data)
    with pytest.raises(SiteInputError,match=part):
        validate_site(data,geo)


def test_signal_program_link_index_and_missing_state_guard(ready):
    _,_,spatial,_=ready
    data=field_fixture()
    data["control"]={"kind":"FIXED_TIME_SIGNAL",
       "measurement_ref":"Timed survey of actual fixed-phase signal 2026-10-01",
       "link_index_review_ref":"Road engineer signed lane link index reference 2026-10-01",
       "phases":[{"duration_s":40,"state":"G"},{"duration_s":20,"state":"r"}]}
    accepted,_=validate_site(data,spatial)
    xml_files=render_sumo(accepted)
    assert "signals.tll.xml" in xml_files
    link=ET.fromstring(xml_files["connections.con.xml"]).find("connection")
    assert link.get("tl")=="j" and link.get("linkIndex")=="0"
    phases=ET.fromstring(xml_files["signals.tll.xml"]).findall("./tlLogic/phase")
    assert [x.get("state") for x in phases]==["G","r"]
    data["control"]["phases"][0]["state"]="GG"
    with pytest.raises(SiteInputError,match="sorted lane-link"):
        validate_site(data,spatial)


def test_holdout_qa_pass_fails_empty_and_rejects_unexpected_trips(ready):
    model=ready[-1]["model"]
    xml=b'<tripinfos>'+b''.join(
        f'<tripinfo id="flow_000.{i}" duration="70"/>'.encode()
        for i in range(5))+b'</tripinfos>'
    good=evaluate_tripinfo(model,xml)
    assert good["status"]=="BASELINE_FIDELITY_CHECKED"
    assert good["real_site_sumo_allowed"] is True
    assert good["completion_ratio"]==1
    assert good["per_movement"][0]["relative_error"]==0
    sparse=evaluate_tripinfo(model,b'<tripinfos><tripinfo id="flow_000.0" duration="100"/></tripinfos>')
    assert sparse["status"]=="BASELINE_NEEDS_REVIEW"
    assert sparse["real_site_sumo_allowed"] is False
    assert evaluate_tripinfo(model,b"<tripinfos/>")["real_site_sumo_allowed"] is False
    with pytest.raises(SiteInputError,match="Unexpected"):
        evaluate_tripinfo(model,b'<tripinfos><tripinfo id="intruder.0" duration="70"/></tripinfos>')
    with pytest.raises(SiteInputError,match="duplicate"):
        evaluate_tripinfo(model,b'<tripinfos><tripinfo id="flow_000.0" duration="70"/><tripinfo id="flow_000.0" duration="70"/></tripinfos>')


def test_cli_runtime_is_immutable_and_evidence_hash_verified(ready,monkeypatch):
    db,project,spatial,package=ready
    monkeypatch.setattr("streetlab_integration.site_baseline.shutil.which",lambda _: "trusted-tool")
    calls=[]
    def fake_run(args,**kwargs):
        calls.append(args[0])
        work=Path(kwargs["cwd"])
        if args[0]=="trusted-tool" and "--node-files" in args:
            (work/"site.net.xml").write_text("<net/>")
        else:
            (work/"summary.xml").write_text("<summary/>")
            (work/"tripinfo.xml").write_text(
                '<tripinfos>'+''.join(
                    f'<tripinfo id="flow_000.{i}" duration="70"/>' for i in range(5)
                )+'</tripinfos>')
        return subprocess.CompletedProcess(args,0,"","")
    monkeypatch.setattr("streetlab_integration.site_baseline.subprocess.run",fake_run)
    result=run_baseline(db,project["id"],package["revision"])
    assert len(calls)==2
    assert result["status"]=="BASELINE_FIDELITY_CHECKED"
    assert latest_baseline(db,project["id"])["runtime"]["real_site_sumo_allowed"]
    with pytest.raises(SiteInputError,match="already executed"):
        run_baseline(db,project["id"],package["revision"])
    folder=db.root/"projects"/project["id"]/"baselines"/package["revision"]
    (folder/"runtime"/"tripinfo.xml").write_text("<tampered/>")
    with pytest.raises(SiteInputError,match="integrity"):
        verified_baseline(db,project["id"],package["revision"])


def test_immutable_revisions_and_cross_project_rejection(ready):
    db,project,geo,package=ready
    model=field_fixture()
    model["vehicle_types"][0]["sigma"]=.3
    second=save_baseline(db,project["id"],geo["revision"],model)
    assert second["revision"]!=package["revision"]
    assert verified_baseline(db,project["id"],package["revision"])["revision"]==package["revision"]
    other=db.create_project("Cross-project attempt")
    with pytest.raises(Exception):
        verified_baseline(db,other["id"],package["revision"])
    assert latest_baseline(db,project["id"])["revision"]==second["revision"]


def test_m4_http_persists_and_exports_but_does_not_run_sumo(local):
    db,project,job=local
    from streetlab_phase2.api import create_app
    client=TestClient(create_app(service=object(),observation_workdir=db.root))
    base="/api/projects/"+project["id"]+"/baseline"
    assert client.get("/baseline").status_code==200
    assert client.get(base).json()["status"]=="NOT_CONFIGURED"
    assert client.post(base,json={"spatial_revision":"0"*64,
                                  "site":field_fixture()}).status_code in (409,422)
    geo=reconstruct(db,project["id"],job["id"],site())
    response=client.post(base,json={"spatial_revision":geo["revision"],
                                    "site":field_fixture()})
    assert response.status_code==201,response.text
    rev=response.json()["revision"]
    assert response.json()["runtime"]["real_site_sumo_allowed"] is False
    assert client.get(base).json()["revision"]==rev
    assert client.get(base+"/"+rev).status_code==200
    assert b"<nodes>" in client.get(base+"/"+rev+"/files/nodes.nod.xml").content
    assert client.get(base+"/"+rev+"/files/../../readme").status_code in (404,422)
    assert client.get(base+"/"+rev+"/files/tripinfo.xml").status_code==422
    assert "href=\"/baseline\"" in client.get("/").text


def test_m3_missing_checkpoints_never_package_real_sumo(local):
    db,project,job=local
    incomplete=reconstruct(db,project["id"],job["id"],site(False))
    with pytest.raises(SiteInputError,match="M3"):
        save_baseline(db,project["id"],incomplete["revision"],field_fixture())
