"""One-shot preregistration, blinded offline packet and independent scoring.

CLI:
  lock    --development-audit ZIP --out LOCK.json
  prepare --lock LOCK.json --video NEW.mp4 --primary PRIMARY.txt
          --shadow SHADOW.txt --first N --last M --out NEW_DIR
  evaluate --lock LOCK.json --packet-dir NEW_DIR
           --reviewer-01 R01.csv --reviewer-02 R02.csv --out REPORT_DIR

The lock MUST be committed/archived before accessing new-video labels.
The tool never writes to production trackers, input truth or video files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import os
import shutil
import sys
import tempfile

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0,str(_ROOT))
from streetlab_phase3.video import correspondence_holdout as H
from streetlab_phase3.video.physical_object_correspondence import (
    IOU_SIMILAR, IOS_CONTAINED, MAX_PART_AREA_RATIO,
    MIN_TEMPORAL_SUPPORT_FRAMES, MIN_OBSERVATION_COVERAGE,
)

STATUS="PHASE3_BLIND_HELDOUT_CORRESPONDENCE_NOT_PRODUCTION"
LOCK_VERSION="PHASE3_CORRESPONDENCE_LOCK_V1"
CODE1=Path(H.__file__).resolve()
CODE2=_ROOT/"streetlab_phase3"/"video"/"physical_object_correspondence.py"
REVIEWERS=("R01","R02")


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("utf-8")

def locking_hash(record):
    return hashlib.sha256(canonical({
        key:value for key,value in record.items() if key!="lock_sha256"
    })).hexdigest()

def freeze(development_audit: Path, out: Path):
    if out.exists():
        raise FileExistsError("Frozen preregistration must never overwrite a previous lock")
    if not development_audit.is_file():
        raise FileNotFoundError("Immutable original W04 audit ZIP required")
    with __import__("zipfile").ZipFile(development_audit) as z:
        summary=json.loads(z.read("audit_summary.json"))
        if (summary.get("eligible_for_production") is not False or
            summary.get("all_191_unmatched_observations")!=191 or
            summary.get("original_primary_modified") is not False):
            raise ValueError("Development anchor is not the original W04 correspondence audit")
    value={
        "status":LOCK_VERSION,
        "eligible_for_production":False,
        "development_audit_sha256":H.sha(development_audit),
        "development_source_video_sha256_exclusion":H.W04_VIDEO_SHA,
        "algorithm_sha256":H.sha(CODE1),
        "geometric_implementation_sha256":H.sha(CODE2),
        "scoring_and_packet_cli_sha256":H.sha(Path(__file__).resolve()),
        "locked_params":{
            "same_extent_iou_min":IOU_SIMILAR,
            "part_whole_ios_min":IOS_CONTAINED,
            "part_whole_max_area_ratio":MAX_PART_AREA_RATIO,
            "temporal_min_frames":MIN_TEMPORAL_SUPPORT_FRAMES,
            "temporal_min_shadow_coverage":MIN_OBSERVATION_COVERAGE,
            "negative_control_proximity_diagonals":H.NEAR_DIAGONALS,
            "sample_pairs_max":H.MAX_REVIEW_PAIRS,
            "minimum_same":H.MIN_CONFIRMED_SAME,
            "minimum_distinct":H.MIN_CONFIRMED_DISTINCT,
            "minimum_evaluation_frames":H.MIN_EVALUATION_FRAMES,
            "max_distinct_false_links":0,
        },
        "holdout_data_inspected_at_lock_creation":False,
        "requires_two_reviewers":True,
        "claims_to_avoid":"A locked protocol is not proof the footage was unseen or reviewers independent."
    }
    value["lock_sha256"]=locking_hash(value)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2)
    return value

def verify_lock(lock: Path):
    value=json.loads(lock.read_text(encoding="utf-8"))
    if (value.get("status")!=LOCK_VERSION
        or value.get("eligible_for_production") is not False
        or value.get("lock_sha256")!=locking_hash(value)
        or value.get("algorithm_sha256")!=H.sha(CODE1)
        or value.get("geometric_implementation_sha256")!=H.sha(CODE2)
        or value.get("scoring_and_packet_cli_sha256")!=H.sha(Path(__file__).resolve())
        or value.get("development_source_video_sha256_exclusion")!=H.W04_VIDEO_SHA
        or value.get("locked_params",{}).get("sample_pairs_max")!=H.MAX_REVIEW_PAIRS):
        raise ValueError("Frozen holdout protocol or implementation code has changed")
    return value

def draw_review_assets(video: Path, pairs: list[dict], folder: Path, *,
                       cv2_module=None):
    """Reviewer sees real frames and A/B boxes; never predicted duplicate class."""
    if cv2_module is None:
        import cv2 as cv2_module
    cv2=cv2_module
    needed={}
    for item in pairs:
        for frame in item["source_frames_sampled"][:3]:
            needed.setdefault(frame,[]).append(item)
    if not needed:
        raise ValueError("No candidate source-video frames to render")
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot decode holdout video")
    folder.mkdir()
    artifacts={}
    try:
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total and max(needed)>=total:
            raise ValueError("Holdout annotations exceed actual source frame count")
        for frame in sorted(needed):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES,frame):
                raise ValueError("Video backend cannot seek to labelled source frame")
            ok,pixels=cap.read()
            if not ok or abs(float(cap.get(cv2.CAP_PROP_POS_FRAMES))-(frame+1))>.51:
                raise ValueError("Holdout video frame decode/index alignment failure")
            for case in needed[frame]:
                a=case["_shadow_boxes"][frame]
                b=case["_primary_boxes"][frame]
                source=pixels.copy()
                h,w=source.shape[:2]
                for box,label,color in ((a,"A",(0,220,255)),(b,"B",(200,255,0))):
                    x1,y1,x2,y2=(int(round(box[k])) for k in ("x1","y1","x2","y2"))
                    if x1<0 or y1<0 or x2>w or y2>h:
                        raise ValueError("Exported boxes out of actual video image bounds")
                    cv2.rectangle(source,(x1,y1),(x2,y2),color,3)
                    cv2.putText(source,label,(max(0,x1),max(18,y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX,.8,color,2)
                scale=min(1.,1280/w)
                scene=cv2.resize(source,(round(w*scale),round(h*scale)))
                x1=max(0,int(min(a["x1"],b["x1"])-100))
                y1=max(0,int(min(a["y1"],b["y1"])-100))
                x2=min(w,int(max(a["x2"],b["x2"])+100))
                y2=min(h,int(max(a["y2"],b["y2"])+100))
                crop=source[y1:y2,x1:x2]
                for style,image in (("scene",scene),("detail",crop)):
                    name=f'{case["case_id"]}_f{frame}_{style}.jpg'
                    target=folder/name
                    ok,jpg=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,90])
                    if not ok:
                        raise ValueError("Cannot encode original source pixels")
                    target.write_bytes(jpg.tobytes())
                    artifacts[(case["case_id"],frame,style)]="images/"+name
    finally:
        cap.release()
    return artifacts

def reviewer_html(reviewer: str, pairs: list[dict], images: dict) -> str:
    cases=[]
    for item in pairs:
        figures=[]
        for frame in item["source_frames_sampled"][:3]:
            for style in ("scene","detail"):
                image=images[(item["case_id"],frame,style)]
                figures.append("<figure><img src='"+html.escape(image,quote=True)+
                               "' loading='lazy' alt='Real source video "+style+
                               "'><figcaption>Frame "+str(frame)+
                               " · "+style+"</figcaption></figure>")
        cases.append('<section class="case" data-id="'+item["case_id"]+'">'
                     '<h2>Case '+item["case_id"]+'</h2><p>Compare object A (yellow) '
                     'with object B (cyan). Are they the <em>same physical '
                     'vehicle</em> or <em>two distinct vehicles</em>?</p>'
                     '<div class="images">'+''.join(figures)+'</div>'
                     '<label>Physical relationship<select class="vote">'
                     '<option value="">Select…</option>'
                     '<option value="SAME_PHYSICAL_OBJECT">Same physical object</option>'
                     '<option value="DISTINCT_PHYSICAL_OBJECT">Distinct physical objects</option>'
                     '<option value="UNCLEAR">Unclear / cannot determine</option>'
                     '</select></label><label>Notes<textarea class="notes" rows="2">'
                     '</textarea></label></section>')
    js="""
const cols=['case_id','reviewer_id','physical_relation','notes'];
function q(x){return '"'+String(x).replaceAll('"','""')+'"';}
function exportCSV(){
 let lines=[cols.join(',')];
 for(let section of document.querySelectorAll('.case')){
  const vote=section.querySelector('.vote').value;
  if(!vote){alert('Please review '+section.dataset.id);return;}
  const vals=[section.dataset.id,document.body.dataset.reviewer,vote,section.querySelector('.notes').value];
  lines.push(vals.map(q).join(','));
 }
 const blob=new Blob([lines.join('\\r\\n')+'\\r\\n'],{type:'text/csv'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download='holdout_'+document.body.dataset.reviewer+'.csv';a.click();
 URL.revokeObjectURL(a.href);
}
document.getElementById('export').addEventListener('click',exportCSV);
"""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Independent holdout pair review</title><style>'
            'body{font:16px/1.5 system-ui;background:#f4f6fa;color:#182332;margin:0}'
            'main{max-width:1200px;margin:24px auto;padding:0 20px}'
            'section{background:white;padding:20px;margin:20px 0;border:1px solid #ccc;border-radius:12px}'
            '.images{display:grid;grid-template-columns:1fr 1fr;gap:8px}'
            'img{width:100%;max-height:380px;object-fit:contain;background:#111}'
            'figure{margin:0}figcaption{color:#667;font-size:13px}'
            'label{display:block;margin-top:14px}select,textarea{width:100%;padding:9px}'
            'button{padding:14px;background:#123;color:white;border:0;border-radius:8px;cursor:pointer}'
            '</style></head><body data-reviewer="'+reviewer+'"><main>'
            '<h1>Independent source-video correspondence review</h1>'
            '<p>Do not consult the other reviewer or internal manifest. '
            'A/B are neutral names, not model classes or truth labels. '
            'Review nearby distinct vehicles carefully; choose unclear if occluded.</p>'
            +''.join(cases)+'<button id="export">Export independent CSV</button>'
            '</main><script>'+js+'</script></body></html>')

def _pair_frame_boxes(primary: dict,shadow: dict,pairs:list[dict]) -> list[dict]:
    # Attach geometry for images, then remove from final blinded case metadata.
    for item in pairs:
        a={}
        b={}
        for frame in item["source_frames_sampled"]:
            s=next((x for x in shadow.get(frame,[]) if
                    x["id"]==item["shadow_id"] and
                    x["vehicle_class"]==item["shadow_class"]),None)
            p=next((x for x in primary.get(frame,[]) if
                    x["id"]==item["primary_id"]),None)
            if s is None or p is None:
                raise ValueError("Blinded frame list not grounded in actual tracked pair")
            a[frame]=s
            b[frame]=p
        item["_shadow_boxes"]=a
        item["_primary_boxes"]=b
    return pairs

def prepare(lock:Path,video:Path,primary_path:Path,shadow_path:Path,
            first:int,last:int,out:Path):
    if out.exists():
        raise FileExistsError("A prior heldout result must not be overwritten")
    frozen=verify_lock(lock)
    if first<0 or last-first+1<H.MIN_EVALUATION_FRAMES:
        raise ValueError("Unseen interval too short")
    video_sha=H.sha(video)
    if video_sha==H.W04_VIDEO_SHA:
        raise ValueError("W04 development source is NOT unseen footage")
    primary,pmeta=H.read_tracks(primary_path,first,last)
    shadow,smeta=H.read_tracks(shadow_path,first,last)
    if pmeta["sha256"]==smeta["sha256"]:
        raise ValueError("Primary and shadow cannot be byte-identical evidence")
    candidates=H.build_pair_inventory(primary,shadow)
    if not candidates:
        raise ValueError("No near-object pair candidates in heldout source; cannot validate")
    selected=H.select_blind_pairs(candidates,digest_hex=video_sha)
    if len(selected)<H.MIN_CONFIRMED_DISTINCT+H.MIN_CONFIRMED_SAME:
        raise ValueError("Insufficient candidate pairs for predeclared sample minimum")
    _pair_frame_boxes(primary,shadow,selected)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+out.name+".stage-",dir=out.parent))
    try:
        imgs=draw_review_assets(video,selected,stage/"images")
        internal=[]
        for case in selected:
            record={k:v for k,v in case.items() if not k.startswith("_")}
            internal.append(record)
        manifest={
            "status":STATUS,"eligible_for_production":False,
            "holdout_video_sha256":video_sha,
            "source_interval":[first,last],"lock_sha256":frozen["lock_sha256"],
            "primary_tracking":pmeta,"shadow_tracking":smeta,
            "all_unlabeled_candidate_pairs":len(candidates),
            "selected_pairs":internal,
            "selection_is_truth_blind":True,
            "source_is_distinct_from_W04_by_SHA":True,
            "genuinely_unseen_provenance_not_independently_authenticated":True,
            "no_FLUID_changes":True,"no_primary_edits":True,
        }
        (stage/"INTERNAL_manifest.json").write_text(
            json.dumps(manifest,indent=2),encoding="utf-8")
        for reviewer in REVIEWERS:
            csv_path=stage/("TEMPLATE_"+reviewer+".csv")
            with csv_path.open("x",encoding="utf-8",newline="") as f:
                writer=csv.writer(f);writer.writerow(H.HEADERS)
                for case in selected:
                    writer.writerow([case["case_id"],reviewer,"",""])
            with zipfile.ZipFile(stage/("REVIEWER_"+reviewer+"_ONLY.zip"),
                                 "x",compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("index.html",reviewer_html(reviewer,selected,imgs))
                for picture in sorted((stage/"images").glob("*.jpg")):
                    z.write(picture,arcname="images/"+picture.name)
        with (stage/"SHA256SUMS.txt").open("x",encoding="utf-8") as f:
            for file in sorted(stage.iterdir()):
                if file.is_file() and file.name!="SHA256SUMS.txt":
                    f.write(H.sha(file)+"  "+file.name+"\n")
        if out.exists():
            raise FileExistsError("Output directory appeared while creating review")
        os.replace(stage,out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"status":STATUS,"selected_blind_pairs":len(selected),
            "video_sha256":video_sha,"output":str(out),"eligible_for_production":False}

def evaluate(lock:Path,packet:Path,r1:Path,r2:Path,out:Path):
    if out.exists():
        raise FileExistsError("Immutable original holdout score exists")
    frozen=verify_lock(lock)
    manifest_path=packet/"INTERNAL_manifest.json"
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    if (m.get("status")!=STATUS or m.get("eligible_for_production") is not False
        or m.get("lock_sha256")!=frozen["lock_sha256"]
        or m.get("source_is_distinct_from_W04_by_SHA") is not True
        or m.get("holdout_video_sha256")==H.W04_VIDEO_SHA):
        raise ValueError("Untrusted or development-cohort holdout evidence")
    checks=(packet/"SHA256SUMS.txt").read_text(encoding="utf-8")
    for line in checks.splitlines():
        h,name=line.split("  ",1)
        if Path(name).name!=name or H.sha(packet/name)!=h:
            raise ValueError("Frozen reviewer packet checksum mismatch")
    items=m["selected_pairs"]
    cases={x["case_id"] for x in items}
    if len(cases)!=len(items) or len(cases)<H.MIN_CONFIRMED_SAME+H.MIN_CONFIRMED_DISTINCT:
        raise ValueError("Insufficient uniquely selected holdout cases")
    a=H.parse_reviews(r1,"R01",cases)
    b=H.parse_reviews(r2,"R02",cases)
    score=H.score_consensus(items,a,b)
    summary={
        **{k:v for k,v in score.items() if k!="details"},
        "status":"PHASE3_HOLDOUT_PAIR_DIAGNOSTICS_NOT_PRODUCTION",
        "locked_protocol_sha256":frozen["lock_sha256"],
        "manifest_sha256":H.sha(manifest_path),
        "reviewer_01_sha256":H.sha(r1),
        "reviewer_02_sha256":H.sha(r2),
        "holdout_video_sha256":m["holdout_video_sha256"],
        "production_eligible":False,
        "confidence_warning":"Agreement alone cannot establish reviewer independence or whole-video precision.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+out.name+".stage-",dir=out.parent))
    try:
        with (stage/"pairwise_diagnostic.csv").open("x",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(score["details"][0]));writer.writeheader()
            writer.writerows(score["details"])
        (stage/"holdout_diagnostic.json").write_text(
            json.dumps(summary,indent=2),encoding="utf-8")
        with (stage/"SHA256SUMS.txt").open("x",encoding="utf-8") as f:
            for file in sorted(stage.iterdir()):
                if file.name!="SHA256SUMS.txt":
                    f.write(H.sha(file)+"  "+file.name+"\n")
        if out.exists():
            raise FileExistsError("Existing holdout result appeared")
        os.replace(stage,out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="step",required=True)
    lock=subs.add_parser("lock")
    lock.add_argument("--development-audit",type=Path,required=True)
    lock.add_argument("--out",type=Path,required=True)
    prep=subs.add_parser("prepare")
    prep.add_argument("--lock",type=Path,required=True)
    prep.add_argument("--video",type=Path,required=True)
    prep.add_argument("--primary",type=Path,required=True)
    prep.add_argument("--shadow",type=Path,required=True)
    prep.add_argument("--first",type=int,required=True)
    prep.add_argument("--last",type=int,required=True)
    prep.add_argument("--out",type=Path,required=True)
    ev=subs.add_parser("evaluate")
    ev.add_argument("--lock",type=Path,required=True)
    ev.add_argument("--packet-dir",type=Path,required=True)
    ev.add_argument("--reviewer-01",type=Path,required=True)
    ev.add_argument("--reviewer-02",type=Path,required=True)
    ev.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.step=="lock":
        result=freeze(a.development_audit,a.out)
    elif a.step=="prepare":
        result=prepare(a.lock,a.video,a.primary,a.shadow,a.first,a.last,a.out)
    else:
        result=evaluate(a.lock,a.packet_dir,a.reviewer_01,a.reviewer_02,a.out)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
