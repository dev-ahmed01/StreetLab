"""Synthetic distinct-vehicle counterexamples for the frozen holdout harness."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from streetlab_phase3.video import correspondence_holdout as H

SOURCE=Path(__file__).resolve().parents[1]/"scripts"/"phase3_correspondence_holdout.py"
if not SOURCE.is_file():
    pytest.skip("Correspondence holdout entry point not in sparse checkout",
                allow_module_level=True)
spec=importlib.util.spec_from_file_location("phase3_heldout",SOURCE)
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def obj(frame,ident,klass="CAR",x=0.):
    return {"frame":frame,"id":ident,"vehicle_class":klass,
            "x1":x,"y1":0.,"x2":x+30.,"y2":30.,"confidence":.9}


def native(frame,ident,klass=0,x=0.):
    cx=x+15
    return [frame,ident,cx,15,30,30,cx,15,30,30,klass,.9,30,30]


def write_export(path, records):
    with path.open("w",newline="") as stream:
        csv.writer(stream).writerows(records)


def test_fixed_120_frame_cohort_parser_rejects_fake_ids_and_out_of_window(tmp_path):
    path=tmp_path/"good.txt"
    write_export(path,[native(0,3),native(2,3)])
    items,metadata=H.read_tracks(path,0,119)
    assert metadata["rows"]==2 and len(items)==2
    with pytest.raises(ValueError,match=">=120"):
        H.read_tracks(path,0,118)
    write_export(path,[native(0,3),native(0,3)])
    with pytest.raises(ValueError,match="duplicated"):
        H.read_tracks(path,0,119)
    write_export(path,[native(0,-1)])
    with pytest.raises(ValueError,match="identities"):
        H.read_tracks(path,0,119)


def test_nearby_distinct_negative_controls_included_when_geometry_rejected():
    shadow={i:[obj(i,7,"MOTORCYCLE",0)] for i in range(10,14)}
    primary={i:[obj(i,3,"CAR",36)] for i in range(10,14)}
    pairs=H.build_pair_inventory(primary,shadow)
    assert len(pairs)==1
    assert pairs[0]["prediction"]=="NEARBY_NEGATIVE_CONTROL"
    assert pairs[0]["source_frames_sampled"]==[10,12,13]


def test_repeated_distinct_overlap_is_a_false_merge_risk_not_proof():
    shadow={i:[obj(i,7,"MOTORCYCLE",4)] for i in range(10,15)}
    primary={i:[obj(i,3,"CAR",0)] for i in range(10,15)}
    result=H.build_pair_inventory(primary,shadow)
    assert len(result)==1
    assert result[0]["prediction"]=="DUPLICATE_HYPOTHESIS"
    result[0]["case_id"]="H001"
    r1={"H001":{"physical_relation":"DISTINCT_PHYSICAL_OBJECT"}}
    r2={"H001":{"physical_relation":"DISTINCT_PHYSICAL_OBJECT"}}
    score=H.score_consensus(result,r1,r2)
    assert score["false_merge_on_annotated_distinct"]==1
    assert score["distinct_false_link_fraction"]==1
    assert score["non_promoting_diagnostic_gate_passed"] is False
    assert score["eligible_for_production"] is False


def test_two_overlapping_primary_ids_must_abstain():
    shadow={i:[obj(i,7,"MOTORCYCLE",10)] for i in range(8,13)}
    primary={i:[obj(i,3,"CAR",0),obj(i,4,"CAR",11)] for i in range(8,13)}
    result=H.build_pair_inventory(primary,shadow)
    assert len(result)==2
    assert {p["prediction"] for p in result}=={"AMBIGUOUS_MULTIPLE_PRIMARY_IDS"}


def test_deterministic_truth_blind_stratification_and_cap():
    entries=[]
    for i in range(60):
        entries.append({
            "shadow_class":"MOTORCYCLE","shadow_id":i,"primary_id":i+200,
            "prediction":("DUPLICATE_HYPOTHESIS" if i<25
                          else "WEAK_GEOMETRIC_RELATION" if i<35
                          else "NEARBY_NEGATIVE_CONTROL")})
    chosen=H.select_blind_pairs(entries,digest_hex="new_video_hash",limit=30)
    assert len(chosen)==30
    assert len({p["case_id"] for p in chosen})==30
    assert len({p["prediction"] for p in chosen})==3
    assert [p["case_id"] for p in chosen]==[f"H{i:03d}" for i in range(1,31)]


def test_reviewers_cannot_omit_cases_or_forge_identity(tmp_path):
    path=tmp_path/"r01.csv"
    with path.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(H.HEADERS)
        w.writerow(["H001","R01","SAME_PHYSICAL_OBJECT",""])
        w.writerow(["H002","R01","DISTINCT_PHYSICAL_OBJECT",""])
    result=H.parse_reviews(path,"R01",{"H001","H002"})
    assert len(result)==2
    with pytest.raises(ValueError,match="independent|invalid"):
        H.parse_reviews(path,"R02",{"H001","H002"})
    with pytest.raises(ValueError,match="ALL"):
        H.parse_reviews(path,"R01",{"H001","H002","H003"})


def test_hard_balanced_gate_needs_independently_labeled_distinct_objects():
    cases=[]
    a,b={},{}
    for i in range(30):
        case=f"H{i+1:03d}"
        relation="SAME_PHYSICAL_OBJECT" if i<10 else "DISTINCT_PHYSICAL_OBJECT"
        mode="DUPLICATE_HYPOTHESIS" if i<10 else "NEARBY_NEGATIVE_CONTROL"
        cases.append({"case_id":case,"prediction":mode})
        a[case]={"physical_relation":relation}
        b[case]={"physical_relation":relation}
    result=H.score_consensus(cases,a,b)
    assert result["non_promoting_diagnostic_gate_passed"] is True
    assert result["reviewed_same"]==10 and result["reviewed_distinct"]==20
    assert result["false_merge_on_annotated_distinct"]==0
    assert not result["eligible_for_production"]
    b["H030"]={"physical_relation":"UNCLEAR"}
    rejected=H.score_consensus(cases,a,b)
    assert rejected["non_promoting_diagnostic_gate_passed"] is False
    assert rejected["disagreed_or_unclear"]==1


def test_lock_pins_module_hash_and_rejects_dev_video(tmp_path):
    zpath=tmp_path/"dev.zip"
    with zipfile.ZipFile(zpath,"w") as z:
        z.writestr("audit_summary.json",json.dumps({
            "eligible_for_production":False,
            "all_191_unmatched_observations":191,
            "original_primary_modified":False}))
    lock=tmp_path/"LOCK.json"
    record=entry.freeze(zpath,lock)
    assert record["status"]==entry.LOCK_VERSION
    assert entry.verify_lock(lock)["lock_sha256"]==record["lock_sha256"]
    tampered=json.loads(lock.read_text())
    tampered["locked_params"]["same_extent_iou_min"]=.1
    lock.write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match="Frozen"):
        entry.verify_lock(lock)


def test_reviewer_packet_isolation_and_immutable_holdout_eval(tmp_path):
    # Synthesize >minimum agreed votes; no real-video claim or access.
    dev=tmp_path/"dev.zip"
    with zipfile.ZipFile(dev,"w") as z:
        z.writestr("audit_summary.json",json.dumps({
            "eligible_for_production":False,"all_191_unmatched_observations":191,
            "original_primary_modified":False}))
    lock=tmp_path/"lock.json"
    frozen=entry.freeze(dev,lock)
    packet=tmp_path/"review"
    packet.mkdir()
    cases=[{"case_id":f"H{i+1:03d}",
            "prediction":"DUPLICATE_HYPOTHESIS" if i<10 else "NEARBY_NEGATIVE_CONTROL"}
           for i in range(30)]
    manifest={"status":entry.STATUS,"eligible_for_production":False,
              "lock_sha256":frozen["lock_sha256"],
              "source_is_distinct_from_W04_by_SHA":True,
              "holdout_video_sha256":"synthetic-different-video",
              "selected_pairs":cases}
    (packet/"INTERNAL_manifest.json").write_text(json.dumps(manifest))
    with (packet/"SHA256SUMS.txt").open("w") as f:
        h=H.sha(packet/"INTERNAL_manifest.json")
        f.write(h+"  INTERNAL_manifest.json\n")
    reviewers=[]
    for reviewer in entry.REVIEWERS:
        dest=tmp_path/(reviewer+".csv")
        with dest.open("w",newline="") as f:
            writer=csv.writer(f);writer.writerow(H.HEADERS)
            for i in range(30):
                writer.writerow([f"H{i+1:03d}",reviewer,
                                 "SAME_PHYSICAL_OBJECT" if i<10 else "DISTINCT_PHYSICAL_OBJECT",""])
        reviewers.append(dest)
    result=entry.evaluate(lock,packet,reviewers[0],reviewers[1],tmp_path/"result")
    assert result["minimum_class_balance_verified"]
    assert result["non_promoting_diagnostic_gate_passed"]
    assert result["production_eligible"] is False
    with pytest.raises(FileExistsError):
        entry.evaluate(lock,packet,reviewers[0],reviewers[1],tmp_path/"result")


def test_direct_cli_help_works_without_pythonpath(tmp_path):
    env=os.environ.copy();env.pop("PYTHONPATH",None)
    run=subprocess.run([sys.executable,str(SOURCE),"--help"],
                       cwd=tmp_path,env=env,text=True,capture_output=True,timeout=30)
    assert run.returncode==0,run.stderr
    assert "prepare" in run.stdout and "evaluate" in run.stdout


def test_prepare_generates_separate_blinded_packets_without_truth_or_primary_edits(
        tmp_path,monkeypatch):
    dev=tmp_path/"original_dev.zip"
    with zipfile.ZipFile(dev,"w") as z:
        z.writestr("audit_summary.json",json.dumps({
            "eligible_for_production":False,"all_191_unmatched_observations":191,
            "original_primary_modified":False}))
    lock=tmp_path/"lock.json"
    entry.freeze(dev,lock)
    video=tmp_path/"never_used_camera.mp4"
    video.write_bytes(b"independent-test-video-not-genuine-footage")
    ppath=tmp_path/"primary.txt"
    spath=tmp_path/"shadow.txt"
    original=[]
    shadow=[]
    for frame in range(5):
        for objid in range(16):
            pos=objid*43.
            original.append(native(frame,objid,0,pos))
            shadow.append(native(frame,100+objid,3,pos+3.))
    write_export(ppath,original)
    write_export(spath,shadow)
    before=ppath.read_bytes()

    def fake_render(source,pairs,folder,**kwargs):
        folder.mkdir()
        images={}
        for pair in pairs:
            for frame in pair["source_frames_sampled"][:3]:
                for suffix in ("scene","detail"):
                    filename=f"{pair['case_id']}_f{frame}_{suffix}.jpg"
                    (folder/filename).write_bytes(b"\xff\xd8\xff\xd9")
                    images[(pair["case_id"],frame,suffix)]="images/"+filename
        return images

    monkeypatch.setattr(entry,"draw_review_assets",fake_render)
    out=tmp_path/"packet"
    report=entry.prepare(lock,video,ppath,spath,0,119,out)
    assert report["selected_blind_pairs"]>=30
    assert ppath.read_bytes()==before
    with zipfile.ZipFile(out/"REVIEWER_R01_ONLY.zip") as z:
        filenames=z.namelist()
        assert "index.html" in filenames
        html_page=z.read("index.html").decode()
        assert "DUPLICATE_HYPOTHESIS" not in html_page
        assert "NEARBY_NEGATIVE_CONTROL" not in html_page
        assert "shadow_id" not in html_page
        assert "data-reviewer=\"R01\"" in html_page
        assert "INTERNAL_manifest.json" not in filenames
    with zipfile.ZipFile(out/"REVIEWER_R02_ONLY.zip") as z:
        assert "data-reviewer=\"R02\"" in z.read("index.html").decode()
    manifest=json.loads((out/"INTERNAL_manifest.json").read_text())
    assert manifest["all_unlabeled_candidate_pairs"]>=len(manifest["selected_pairs"])
    assert manifest["source_is_distinct_from_W04_by_SHA"] is True
    assert all(row["physically_verified"] is False for row in manifest["selected_pairs"])
    with pytest.raises(FileExistsError):
        entry.prepare(lock,video,ppath,spath,0,119,out)
