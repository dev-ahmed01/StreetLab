"""M3 acceptance: real project/job binding, manual homography, immutable world
trajectories, movement ambiguity, poor scale checks and honest SUMO refusal.

All point sources in these tests are synthetic geometric/software fixtures.
They do not establish physical-world reconstruction accuracy.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

pytest.importorskip("streetlab_integration.reconstruction")

from streetlab_integration import video_jobs, worker
from streetlab_integration.primary_runner import sha
from streetlab_integration.reconstruction import (
    ReconstructionError, _movement_candidate, _transform, _homography,
    latest_reconstruction, reconstruct, validate_model, verified_reconstruction,
)
from streetlab_integration.video_jobs import VideoStore


VIDEO_BYTES=b"\x00\x00\x00\x18ftypisom"+b"\x00"*64
META={"fps":10.0,"frames":50,"width":320,"height":240,"duration_seconds":5.0}


def four_anchors():
    return [{"pixel":p,"world_m":m,"provenance":"OBSERVED_MANUAL"}
            for p,m in zip([[0,0],[100,0],[100,100],[0,100]],
                           [[0,0],[10,0],[10,10],[0,10]])]


def site(check=True):
    return {
        "scale_basis":"Measured tape controls + surveyed local reference; manually attested",
        "anchors":four_anchors(),
        "checks":[{"pixel":[50,50],"world_m":[5,5],"provenance":"OBSERVED_MANUAL"}]
                 if check else [],
        "zones":[
            {"id":"west_in","role":"APPROACH","label":"Westbound approach",
             "polygon_pixels":[[0,0],[32,0],[32,30],[0,30]],
             "lane_count":2,"provenance":"OBSERVED_MANUAL"},
            {"id":"east_out","role":"EXIT","label":"Eastern exit",
             "polygon_pixels":[[65,0],[100,0],[100,30],[65,30]],
             "lane_count":None,"provenance":"OBSERVED_MANUAL"},
        ],
        "movements":[{"from_zone":"west_in","to_zone":"east_out",
                      "provenance":"OBSERVED_MANUAL"}],
    }


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setattr(video_jobs, "profile_video", lambda _:dict(META))
    db=VideoStore(tmp_path/"product")
    project=db.create_project("Actual site sample")
    db.save_source(project["id"],"test.mp4",io.BytesIO(VIDEO_BYTES))
    job=db.queue(project["id"],last_frame=4)
    monkeypatch.setattr(worker,"pinned_model",lambda *a:("a"*64,"b"*64))
    def fake_runner(video, folder, model, config, fps, source_hash, model_hash, progress):
        folder.mkdir(parents=True)
        tracks=folder/"primary_ios030.txt"
        # Two distinct tracker IDs; this synthetic source says nothing about
        # distinct physical vehicles or tracking recall.
        with tracks.open("w",newline="") as f:
            out=csv.writer(f)
            for frame,x in enumerate([10,20,50,75,90]):
                out.writerow([frame,7,x,10,10,8,x,10,10,8,0,.9,10,8])
                out.writerow([frame,8,120,40,10,8,120,40,10,8,3,.85,10,8])
        payload={"schema_version":1,
                 "status":"STREETLAB_M2_OBSERVED_AUTO_PIXEL_ONLY",
                 "source_video_sha256":source_hash,"model_tree_sha256":model_hash,
                 "files":{"primary_ios030.txt":sha(tracks)},"overlay_files":[]}
        (folder/"manifest.json").write_text(json.dumps(payload))
        progress(5)
        return payload
    assert worker.run_once(db,Path("unused"),Path("unused"),"cpu-1",runner=fake_runner)
    assert db.job(job["id"])["status"]=="SUCCEEDED"
    return db,project,job


def test_known_affine_homography_is_exact_but_requires_separate_checkpoint():
    h=_homography(four_anchors())
    assert _transform(h,(25,75))==pytest.approx((2.5,7.5),abs=1e-8)
    model,quality,transform=validate_model(site(),width=320,height=240)
    assert quality["metric_transform_checked"] is True
    assert quality["status"]=="GUIDED_GEOMETRY_REVIEWED"
    assert quality["real_site_sumo_allowed"] is False
    assert model["coordinate_system"]=="LOCAL_GROUND_PLANE_METERS_NOT_GPS"
    assert transform is not None
    _,poor,no_transform=validate_model(site(check=False),width=320,height=240)
    assert poor["status"]=="NEEDS_DATA"
    assert poor["metric_transform_checked"] is False
    assert no_transform is None
    assert "independently measured check point" in ";".join(poor["missing_evidence"])


@pytest.mark.parametrize("edit,fragment",[
    (lambda m:m.update({"anchors":four_anchors()[:3]}),"four"),
    (lambda m:m["anchors"].__setitem__(2,dict(
       pixel=[200,0],world_m=[20,0],provenance="OBSERVED_MANUAL")),"Degenerate"),
    (lambda m:m["checks"].__setitem__(0,dict(
       pixel=[50,50],world_m=[12,13],provenance="OBSERVED_MANUAL")),""),
    (lambda m:m["zones"][0].update({"polygon_pixels":[[0,0],[40,30],[0,30],[40,0]]}),
      "polygon"),
    (lambda m:m["movements"][0].update({"to_zone":"west_in"}),"Movements"),
    (lambda m:m["anchors"][0].update({"provenance":"ASSUMED"}),"OBSERVED_MANUAL"),
])
def test_geometric_invalidity_and_inferred_scale_are_not_accepted(edit,fragment):
    model=site();edit(model)
    if fragment=="":
        _,q,h=validate_model(model,width=320,height=240)
        assert h is None and not q["metric_transform_checked"]
        assert q["status"]=="NEEDS_DATA"
    else:
        with pytest.raises(ReconstructionError,match=fragment):
            validate_model(model,width=320,height=240)


def test_zone_movement_candidates_are_not_physical_counts():
    m,q,h=validate_model(site(),width=320,height=240)
    match=_movement_candidate(
        m["zones"],m["movements"],[(0,10,10),(1,25,10),(2,50,10),(3,80,10)])
    assert match=={"status":"TRACK_ID_CANDIDATE_ONLY",
                   "movement":["west_in","east_out"]}
    ambiguous=_movement_candidate(m["zones"],m["movements"],[(0,150,10)])
    assert ambiguous["status"]=="INSUFFICIENT_TRACK"
    overlapping=site()
    overlapping["zones"][1]["polygon_pixels"]=[[15,0],[90,0],[90,30],[15,30]]
    m,_,_=validate_model(overlapping,width=320,height=240)
    unresolved=_movement_candidate(m["zones"],m["movements"],[(0,20,10),(1,80,10)])
    assert unresolved["status"]=="AMBIGUOUS_OVERLAP"


def test_full_m1_m2_m3_revision_persistence_and_export(local):
    db,project,job=local
    result=reconstruct(db,project["id"],job["id"],site())
    assert result["quality"]["metric_transform_checked"] is True
    assert result["quality"]["mapped_observations"]==10
    assert result["quality"]["real_site_sumo_allowed"] is False
    assert result["quality"]["traffic_demand_confirmed"] is False
    assert result["quality"]["tracked_id_candidate_movements"]["TRACK_ID_CANDIDATE_ONLY"]==1
    first_revision=result["revision"]
    assert latest_reconstruction(VideoStore(db.root),project["id"])["revision"]==first_revision
    folder=db.root/"projects"/project["id"]/"reconstructions"/first_revision
    with (folder/"source_to_world.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==10
    assert float(rows[0]["x_local_m"])==pytest.approx(1.0,abs=.001)
    assert float(rows[0]["y_local_m"])==pytest.approx(1.0,abs=.001)
    assert "speed_mps" not in rows[0]
    assert {r["provenance"] for r in rows}=={"CALIBRATED"}
    original=(folder/"source_to_world.csv").read_bytes()
    same=reconstruct(db,project["id"],job["id"],site())
    assert same["revision"]==first_revision
    assert (folder/"source_to_world.csv").read_bytes()==original
    updated=site()
    updated["zones"][0]["label"]="West entrance surveyed"
    second=reconstruct(db,project["id"],job["id"],updated)
    assert second["revision"]!=first_revision
    assert (folder/"source_to_world.csv").read_bytes()==original
    assert latest_reconstruction(db,project["id"])["revision"]==second["revision"]


def test_unchecked_scale_stores_draft_without_world_meters(local):
    db,project,job=local
    result=reconstruct(db,project["id"],job["id"],site(False))
    assert result["quality"]["status"]=="NEEDS_DATA"
    assert result["quality"]["mapped_observations"]==0
    folder=db.root/"projects"/project["id"]/"reconstructions"/result["revision"]
    assert not (folder/"source_to_world.csv").exists()
    assert "real_site_sumo_allowed" in result["quality"]


def test_tampering_and_cross_project_job_are_rejected(local):
    db,project,job=local
    outsider=db.create_project("Other Site")
    with pytest.raises(ReconstructionError,match="this project"):
        reconstruct(db,outsider["id"],job["id"],site())
    result=reconstruct(db,project["id"],job["id"],site())
    world=(db.root/"projects"/project["id"]/"reconstructions"/result["revision"]
           /"source_to_world.csv")
    world.write_text("tampered")
    with pytest.raises(ReconstructionError,match="integrity"):
        verified_reconstruction(db,project["id"],result["revision"])


def test_http_editor_and_world_export_require_verification(local):
    from streetlab_phase2.api import create_app
    db,project,job=local
    client=TestClient(create_app(service=object(),observation_workdir=db.root))
    homepage=client.get("/")
    assert homepage.status_code==200
    assert 'href="/calibration"' in homepage.text
    ui=client.get("/calibration")
    assert ui.status_code==200
    assert "frameCanvas" in ui.text
    assert "independent" in ui.text
    url="/api/projects/"+project["id"]+"/reconstruction"
    assert client.get(url).json()["status"]=="NOT_CONFIGURED"
    draft=client.post(url,json={"job_id":job["id"],"model":site(False)})
    assert draft.status_code==201
    assert draft.json()["quality"]["status"]=="NEEDS_DATA"
    assert client.get(url+"/export").status_code==422
    good=client.post(url,json={"job_id":job["id"],"model":site()})
    assert good.status_code==201
    response=client.get(url+"/export")
    assert response.status_code==200
    assert "x_local_m" in response.text
    assert "speed_mps" not in response.text
    assert client.get(url).json()["quality"]["real_site_sumo_allowed"] is False
    assert client.get("/api/jobs/"+job["id"]+"/source-frame?frame=400").status_code==422


def test_no_geometry_does_not_fabricate_coordinates(local):
    db,project,job=local
    raw={"anchors":[],"checks":[],"zones":[],"movements":[],"scale_basis":""}
    result=reconstruct(db,project["id"],job["id"],raw)
    assert result["quality"]["metric_transform_checked"] is False
    assert result["quality"]["status"]=="NEEDS_DATA"
    assert result["quality"]["mapped_observations"]==0
    assert result["quality"]["real_site_sumo_allowed"] is False



def test_source_frame_viewer_decodes_actual_video_without_invented_image(tmp_path, monkeypatch):
    """Real OpenCV decoding contract, from a generated AVI, not road-accuracy data."""
    cv2=pytest.importorskip("cv2")
    path=tmp_path/"frame_source.avi"
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10.0,(320,240))
    if not writer.isOpened():
        pytest.skip("No MJPG encoder on this CI host")
    for i in range(5):
        image=np.zeros((240,320,3),dtype=np.uint8)
        image[:,:,:]=[20+i*5,40+i*7,80+i*9]
        writer.write(image)
    writer.release()
    store=VideoStore(tmp_path/"real_media")
    project=store.create_project("Real decoder integration")
    source=store.save_source(project["id"],"source.avi",io.BytesIO(path.read_bytes()))
    assert source["metadata"]["frames"]==5
    job=store.queue(project["id"],last_frame=4)
    monkeypatch.setattr(worker,"pinned_model",lambda *a:("a"*64,"b"*64))

    def fake_runner(video,folder,model,config,fps,source_sha,model_sha,progress):
        folder.mkdir(parents=True)
        export=folder/"primary_ios030.txt"
        with export.open("w",newline="") as out:
            writer=csv.writer(out)
            for n in range(5):
                writer.writerow([n,7,50+n,60,20,10,50+n,60,20,10,0,.8,20,10])
        (folder/"manifest.json").write_text(json.dumps({
            "schema_version":1,"status":"STREETLAB_M2_OBSERVED_AUTO_PIXEL_ONLY",
            "source_video_sha256":source_sha,"model_tree_sha256":model_sha,
            "files":{"primary_ios030.txt":sha(export)},"overlay_files":[]}))
        progress(5)
    assert worker.run_once(store,Path("unused"),Path("unused"),"cpu-2",runner=fake_runner)
    assert store.job(job["id"])["status"]=="SUCCEEDED"
    from streetlab_phase2.api import create_app
    client=TestClient(create_app(service=object(),observation_workdir=store.root))
    image_response=client.get("/api/jobs/"+job["id"]+"/source-frame?frame=2")
    assert image_response.status_code==200
    assert image_response.headers["content-type"].startswith("image/jpeg")
    assert image_response.headers["x-streetlab-source-frame"]=="2"
    decoded=cv2.imdecode(np.frombuffer(image_response.content,dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    assert decoded.shape[:2]==(240,320)
    assert client.get("/api/jobs/"+job["id"]+"/source-frame?frame=5").status_code==422
