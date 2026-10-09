"""W04 visual reviewer kit synthetic tests: no actual video needed in CI."""
from __future__ import annotations
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
PACK=SCRIPTS/"phase3_w04_visual_review_pack.py"
CONS=SCRIPTS/"phase3_w04_review_consensus.py"
if not PACK.is_file() or not CONS.is_file():
    pytest.skip("Visual review tools omitted by sparse checkout",
                allow_module_level=True)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))


def import_tool(file,name):
    spec=importlib.util.spec_from_file_location(name,file)
    assert spec and spec.loader
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


pack=import_tool(PACK,"w04_visual_pack")
cons=import_tool(CONS,"w04_consensus")


def observation(frame,klass="MOTORCYCLE",tid=22,confidence=.8,ambiguous=False):
    return {"frame":frame,"vehicle_class":klass,"hybrid_id":tid,
            "x1":20.,"y1":30.,"x2":40.,"y2":50.,
            "confidence":confidence,
            "cross_class_control_conflict":bool(ambiguous),
            "competing_raw_class_hypotheses":bool(ambiguous)}


def test_source_frame_sampling_is_deterministic_and_not_fabricated():
    items=[observation(frame) for frame in range(10778,10790)]
    for x in items:
        if x["frame"]==10785:
            x["cross_class_control_conflict"]=True
    chosen=pack.choose_frames(items)
    assert 1<=len(chosen)<=5
    assert {x["frame"] for x in chosen}.issubset(
        {x["frame"] for x in items})
    assert chosen==sorted(chosen,key=lambda x:x["frame"])
    assert 10785 in {x["frame"] for x in chosen}


def test_anonymous_case_codes_do_not_expose_original_class_in_reviewer_html(tmp_path):
    candidates=[]
    records=[]
    for number in range(7):
        klass="HEAVY_VEHICLE" if number<5 else "MOTORCYCLE"
        tid=100+number
        candidates.append({"vehicle_class":klass,"hybrid_id":tid})
        for frame in (10760+number*5,10761+number*5,10762+number*5):
            records.append(observation(frame,klass,tid))
    cases=pack.build_cases(records,candidates)
    assert len(cases)==7 and {x["case_id"] for x in cases}=={
        "S%02d"%i for i in range(1,8)}
    pictures={}
    assets=tmp_path/"assets"
    assets.mkdir()
    for case in cases:
        for obs in case["sampled"]:
            stem=f'{case["case_id"]}_{obs["frame"]}'
            pictures[(case["case_id"],obs["frame"])]={
                tag:"assets/"+stem+"_"+tag+".jpg"
                for tag in ("scene","target","compare")}
            for value in pictures[(case["case_id"],obs["frame"])].values():
                (tmp_path/value).write_bytes(b"\xff\xd8\xff\xd9")
    html_doc=pack.html_packet("R01",cases,pictures)
    assert "case S01" not in html_doc.lower()
    assert "Case S01" in html_doc
    assert "reviewer" in html_doc
    assert "model score" in html_doc
    assert "data-reviewer" in html_doc
    assert "HEAVY_VEHICLE" in html_doc  # dropdown class choices, not per-case labels
    assert 'hybrid_track_id' not in html_doc
    assert 'candidate_class_internal_only' not in html_doc
    manifest=pack.package(tmp_path,cases,
                          {"batch_sha256":"original-sample"},
                          {"source_original_batch_sha256":"original-sample"},
                          pictures,video_sha="frozen-video")
    assert manifest["case_count"]==7
    with zipfile.ZipFile(tmp_path/"REVIEWER_R01_ONLY.zip") as z:
        assert "index.html" in z.namelist()
        assert not any("INTERNAL" in x for x in z.namelist())
        assert all(x.startswith("assets/") or x=="index.html" for x in z.namelist())
    with zipfile.ZipFile(tmp_path/"REVIEWER_R02_ONLY.zip") as z:
        h=z.read("index.html").decode()
        assert 'data-reviewer="R02"' in h
        assert 'data-reviewer="R01"' not in h


def test_source_crop_and_primary_comparison_is_image_backed():
    import cv2
    frame=np.zeros((120,200,3),dtype=np.uint8)
    event=observation(10777)
    primary=[dict(x1=15.,y1=25.,x2=45.,y2=55.)]
    scene,crop,comp=pack.prepare_image(frame,event,primary,
                                       cv2=cv2,width=200,height=120,
                                       scale_width=200)
    assert scene.shape==(120,200,3)
    assert crop.ndim==3 and comp.shape==crop.shape
    assert int(np.sum(crop))>0
    assert not np.array_equal(crop,comp)
    with pytest.raises(ValueError,match="size"):
        pack.prepare_image(frame,event,[],cv2=cv2,width=201,height=120)


def test_reviews_require_separate_complete_ids_and_no_unsupported_values(tmp_path):
    manifest=tmp_path/"INTERNAL_case_manifest.json"
    manifest.write_text(json.dumps({
        "status":"W04_BLIND_TWO_REVIEWER_VISUAL_PACKET_NOT_PRODUCTION",
        "eligible_for_production":False,
        "reviewers_required":["R01","R02"],
        "case_count":7,"no_auto_FLUID_changes":True,
        "case_mapping":[{
            "case_id":f"S{i:02d}","candidate_class_internal_only":"MOTORCYCLE",
            "hybrid_track_id_internal_only":i
        } for i in range(1,8)]
    }))
    r1=tmp_path/"r1.csv"
    r2=tmp_path/"r2.csv"
    def filled(reviewer):
        return (",".join(cons.HEADERS)+"\n"
                +''.join(f'S{i:02d},{reviewer},YES,MOTORCYCLE,UNTRACKED,source crop checked\n'
                          for i in range(1,8)))
    r1.write_text(filled("R01"))
    r2.write_text(filled("R02"))
    summary=cons.run(tmp_path,r1,r2,tmp_path/"consensus")
    assert summary["case_count"]==7
    assert summary["consensus_possible_untracked_count"]==7
    assert not summary["physical_vehicle_automatically_promoted"]
    assert not summary["flUID_truth_modified"]
    assert (tmp_path/"consensus"/"requires_adjudication.csv").is_file()
    with pytest.raises(FileExistsError):
        cons.run(tmp_path,r1,r2,tmp_path/"consensus")
    r2.write_text(filled("R01"))
    with pytest.raises(ValueError,match="swapped|unknown"):
        cons.run(tmp_path,r1,r2,tmp_path/"other")


def test_disagreement_becomes_explicit_adjudication(tmp_path):
    a={"vehicle_visible":"YES","physical_class":"CAR","primary_relation":"UNTRACKED"}
    b={"vehicle_visible":"YES","physical_class":"BUS","primary_relation":"ALREADY_TRACKED"}
    code,needs=cons.evaluate_case(a,b)
    assert needs and code.startswith("DISAGREEMENT")
    c={"vehicle_visible":"UNCLEAR","physical_class":"UNKNOWN","primary_relation":"UNCLEAR"}
    assert cons.evaluate_case(c,c)[1]
    assert cons.evaluate_case(a,a)==(
        "CONSENSUS_POSSIBLE_UNTRACKED_PHYSICAL_OBJECT",False)
