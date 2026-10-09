"""W04 blind two-reviewer visual kit from verified original source video.

Primary IoS.30 ByteTrack rows remain immutable. NO inference, tracking,
new FLUID annotations or automatic vehicle-count promotion.
Works directly with existing Windows directories: original 25-policy
continuous batch and the already-executed unified hybrid replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.holdout_validation import verify_batch
from phase3_w04_shadow_review_queue import (
    START, END, build_queue, digest, parse_raw, parse_tracks,
)

STATUS = "W04_BLIND_TWO_REVIEWER_VISUAL_PACKET_NOT_PRODUCTION"
PRIMARY_SHA = "6c695e384e05421ec25dbe450f00a83fb6a72a659a2d162a1122d67c7d0ce418"
EXPECTED_VIDEO_SHA = "57105b564ff3f9c68d88e6df05790654a49a48f626b1adaf39b62ea61c262ce0"
REVIEW_COUNTS = {"HEAVY_VEHICLE": 5, "MOTORCYCLE": 2}
COLUMNS = ("case_id", "reviewer_id", "vehicle_visible", "physical_class",
           "primary_relation", "notes")
CLASSES = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE",
           "AUTO_RICKSHAW", "BICYCLE", "PEDESTRIAN", "OTHER", "UNKNOWN", "NA")
RELATIONS = ("UNTRACKED", "ALREADY_TRACKED", "DUPLICATE_SHADOW", "UNCLEAR", "NA")
VISIBLE = ("YES", "NO", "UNCLEAR")


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda:stream.read(1024*1024), b""):
            h.update(piece)
    return h.hexdigest()


def source_evidence(batch_dir: Path, replay_dir: Path, video: Path,
                    *, verify_video: bool = True):
    """Check same original W04 source, raw cache, all 25 tracks and both lanes."""
    info=verify_batch(batch_dir,verify_originals=True)
    report=info["report"]
    origin=info["provenance"]
    if (report.get("candidate_count")!=25
        or report.get("evaluated_first_frame")!=START
        or report.get("evaluated_last_frame")!=END
        or report.get("same_window_T000") is not None
        or origin.get("video_sha256")!=EXPECTED_VIDEO_SHA):
        raise ValueError("Source is not the frozen W04 development batch")
    if not video.is_file():
        raise FileNotFoundError("Original 4K source video missing")
    if verify_video and sha(video)!=origin["video_sha256"]:
        raise ValueError("Original MP4 source video SHA differs from W04 provenance")
    if not verify_video and video.resolve()!=Path(origin["video_path"]).resolve():
        raise ValueError("Video SHA skip only permitted on the source provenance path")
    policies={x["name"]:x for x in report["policies"]}
    primary_path=batch_dir/policies["hard_nms_ios_0.30"]["tracks"]
    original_bytes=primary_path.read_bytes()
    if digest(original_bytes)!=PRIMARY_SHA:
        raise ValueError("Original authoritative track changed")
    replay_report=json.loads((replay_dir/"unified_hybrid_report.json").read_text(encoding="utf-8"))
    if (replay_report.get("source_original_batch_sha256")!=info["batch_sha256"]
        or replay_report.get("source_video_sha256_verified")!=origin["video_sha256"]
        or replay_report.get("source_fluid_sha256_verified")!=origin["fluid_sha256"]
        or replay_report.get("source_original_raw_sha256")!=report["raw_pre_global_merge_boxes_sha256"]
        or replay_report.get("original_control_reproduction",{}).get("successful") is not True
        or replay_report.get("eligible_for_production") is not False):
        raise ValueError("Hybrid replay provenance/control mismatch")
    modes={x["name"]:x for x in replay_report["experimental_results"]}
    if len(modes)!=4 or "rare_iou05" not in modes or "control_ios030" not in modes:
        raise ValueError("Incomplete unified hybrid experiment")
    for mode,entry in modes.items():
        track=replay_dir/entry["tracks"]
        sidecar=replay_dir/entry["class_evidence"]
        if (not track.is_file() or not sidecar.is_file()
            or sha(track)!=entry["tracks_sha256"]
            or sha(sidecar)!=entry["class_evidence_sha256"]):
            raise ValueError("One of four hybrid track or class sidecar hashes mismatched")
    if (replay_dir/modes["control_ios030"]["tracks"]).read_bytes()!=original_bytes:
        raise ValueError("Hybrid control not byte-identical to primary tracker")
    primary=parse_tracks(original_bytes,label="original",
                         expected_rows=policies["hard_nms_ios_0.30"]["evaluation_confirmed_rows"])
    rare=modes["rare_iou05"]
    rare_track=parse_tracks((replay_dir/rare["tracks"]).read_bytes(),
                            label="rare",expected_rows=rare["evaluation_confirmed_rows"])
    raw_path=batch_dir/"original_pre_global_merge_boxes.jsonl"
    raw_bytes=raw_path.read_bytes()
    if digest(raw_bytes)!=report["raw_pre_global_merge_boxes_sha256"]:
        raise ValueError("Original raw detector SHA mismatch")
    original_raw=parse_raw(raw_bytes,count=report["raw_tile_boxes"])
    proposals,tracklets=build_queue(primary,rare_track,original_raw)
    selected=[x for x in tracklets if x["eligible_for_human_review"]]
    by_class={cls:sum(x["vehicle_class"]==cls for x in selected) for cls in REVIEW_COUNTS}
    if (len(proposals)!=191 or len(tracklets)!=40 or len(selected)!=7
        or by_class!=REVIEW_COUNTS):
        raise ValueError("Unexpected W04 shadow case cohort: do not silently redefine review")
    return info,replay_report,primary,proposals,selected


def choose_frames(observations: list[dict], maximum: int = 5) -> list[dict]:
    """Deterministic, diverse same-object observations, never hallucinated frames."""
    frames={x["frame"]:x for x in observations}
    if not frames:
        raise ValueError("A review case must have actual shadow observations")
    ordered=sorted(frames)
    choices=[ordered[0],ordered[-1],ordered[len(ordered)//2],
             max(ordered,key=lambda f:(frames[f]["confidence"],-f)),
             max(ordered,key=lambda f:(
                 frames[f]["cross_class_control_conflict"],
                 frames[f]["competing_raw_class_hypotheses"],
                 frames[f]["confidence"],-f))]
    # Extra spread if high confidence and ambiguity coincide with first/last.
    for f in ordered:
        choices.append(f)
    result=[]
    seen=set()
    for f in choices:
        if f not in seen:
            result.append(frames[f])
            seen.add(f)
        if len(result)>=maximum:
            break
    return sorted(result,key=lambda x:x["frame"])


def build_cases(proposals: list[dict], selected: list[dict]):
    relevant={(x["vehicle_class"],x["hybrid_id"]) for x in selected}
    grouped={}
    for item in proposals:
        k=(item["vehicle_class"],item["hybrid_id"])
        if k in relevant:
            grouped.setdefault(k,[]).append(item)
    if len(grouped)!=7:
        raise ValueError("Review queue source observations incomplete")
    # Blind codes are not ordered by vehicle class/confidence/review priority.
    keys=sorted(grouped,key=lambda k: hashlib.sha256(
        ("blind-w04"+str(k)).encode()).hexdigest())
    result=[]
    for i,key in enumerate(keys,1):
        result.append({
            "case_id":f"S{i:02d}",
            "candidate_class":key[0],
            "hybrid_track_id":key[1],
            "all_observations":len(grouped[key]),
            "sampled":choose_frames(grouped[key]),
            "review_state":"UNVERIFIED_PENDING_TWO_INDEPENDENT_REVIEWS",
            "physical_vehicle_confirmed":False,
        })
    return result


def crop_bounds(item: dict, width: int, height: int,
                side: int = 680) -> tuple[int,int,int,int]:
    cx=(item["x1"]+item["x2"])/2
    cy=(item["y1"]+item["y2"])/2
    box_side=max(item["x2"]-item["x1"],item["y2"]-item["y1"])
    length=min(max(side,int(box_side*3)),min(width,height))
    left=max(0,min(int(round(cx-length/2)),width-length))
    top=max(0,min(int(round(cy-length/2)),height-length))
    return left,top,left+length,top+length


def prepare_image(frame, item: dict, primary_rows: list[dict], *,
                  cv2, width: int, height: int, scale_width: int = 1152):
    """One unbiased target and one neutral primary comparison; no class labels."""
    if frame.shape[:2]!=(height,width):
        raise ValueError("Decoded original 4K frame size is not 3840×2160")
    target=(int(round(item["x1"])),int(round(item["y1"])),
            int(round(item["x2"])),int(round(item["y2"])))
    if (target[0]<0 or target[1]<0 or target[2]>width or target[3]>height
        or target[2]<=target[0] or target[3]<=target[1]):
        raise ValueError("Candidate geometry outside source video")
    left,top,right,bottom=crop_bounds(item,width,height)
    crop=frame[top:bottom,left:right].copy()
    rect=(target[0]-left,target[1]-top,target[2]-left,target[3]-top)
    cv2.rectangle(crop,rect[:2],rect[2:],(0,255,255),3)
    scene=cv2.resize(frame,(scale_width,round(height*scale_width/width)))
    fx=scale_width/width
    fy=scene.shape[0]/height
    cv2.rectangle(scene,(round(target[0]*fx),round(target[1]*fy)),
                  (round(target[2]*fx),round(target[3]*fy)),(0,255,255),2)
    comparison=crop.copy()
    for other in primary_rows:
        x1,y1,x2,y2=(round(other[p])-o for p,o in (
            ("x1",left),("y1",top),("x2",left),("y2",top)))
        if x2<0 or y2<0 or x1>right-left or y1>bottom-top:
            continue
        cv2.rectangle(comparison,(x1,y1),(x2,y2),(230,230,230),2)
    # Reinforce target after primary comparators so reviewers know ROI.
    cv2.rectangle(comparison,rect[:2],rect[2:],(0,255,255),3)
    return scene,crop,comparison


def _write_jpg(cv2, path: Path, frame):
    ok,buf=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),91])
    if not ok:
        raise ValueError("OpenCV failed to encode local evidence frame")
    path.write_bytes(buf.tobytes())


def render_frames(video: Path, cases: list[dict], primary: dict, stage: Path,
                  *, cv2_module=None, dimensions=(3840,2160)):
    """Seek once, read successive source frames, verify exact absolute indices."""
    if cv2_module is None:
        import cv2 as cv2_module
    cv2=cv2_module
    needed={}
    for case in cases:
        for obs in case["sampled"]:
            needed.setdefault(obs["frame"],[]).append((case["case_id"],obs))
    if not needed:
        raise ValueError("No original source frames selected")
    assets=stage/"assets"
    assets.mkdir()
    first=min(needed)
    last=max(needed)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot decode the original MP4")
    built={}
    try:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES,first):
            raise ValueError("Cannot seek to W04 source frame")
        for frame_no in range(first,last+1):
            ok,frame=cap.read()
            pos=float(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ok or abs(pos-(frame_no+1))>.51:
                raise ValueError(f"Original source video seek/decode gap at frame {frame_no}")
            if frame_no not in needed:
                continue
            for case_id,item in needed[frame_no]:
                scene,crop,compare=prepare_image(frame,item,primary.get(frame_no,[]),
                                                  cv2=cv2,width=dimensions[0],
                                                  height=dimensions[1])
                files={}
                for label,pixels in (("scene",scene),("target",crop),("compare",compare)):
                    name=f"{case_id}_f{frame_no}_{label}.jpg"
                    _write_jpg(cv2,assets/name,pixels)
                    files[label]="assets/"+name
                built[(case_id,frame_no)]=files
    finally:
        cap.release()
    return built


def csv_template(reviewer: str, cases: list[dict]) -> str:
    output=io.StringIO()
    writer=csv.writer(output,lineterminator="\n")
    writer.writerow(COLUMNS)
    for case in cases:
        writer.writerow([case["case_id"],reviewer,"","","",""])
    return output.getvalue()


def html_packet(reviewer: str, cases: list[dict], pictures: dict) -> str:
    """Static offline HTML; no server/network required, local CSV export only."""
    cards=[]
    for case in cases:
        cid=case["case_id"]
        figures=[]
        for frame in case["sampled"]:
            n=frame["frame"]
            asset=pictures[(cid,n)]
            figures.append(
                '<div class="samples"><div class="frame-name">Source frame '+str(n)+'</div>'
                '<div class="shots">'
                +''.join('<figure><img loading="lazy" src="'+html.escape(asset[x],quote=True)
                         +'" alt="Source visual '+x+'"><figcaption>'+title+'</figcaption></figure>'
                         for x,title in (("scene","Scene context — target in yellow"),
                                         ("target","Target crop"),
                                         ("compare","Primary tracker outlines in gray; target yellow")))
                +'</div></div>')
        cards.append('<section class="case" data-case="'+cid+'"><h2>Case '+cid+'</h2>'
            +'<p>Source-frame visuals only. The detector class, model score, and candidate tracker ID are intentionally hidden.</p>'
            +''.join(figures)
            +'<div class="answers">'
            +_select("vehicle_visible",["",*VISIBLE],"Is a physical object visible at the target?")
            +_select("physical_class",["",*CLASSES],"Physical class from the image (not detector output)")
            +_select("primary_relation",["",*RELATIONS],"Relation to the existing primary tracks")
            +'<label>Notes / uncertainty<textarea name="notes" rows="2"></textarea></label>'
            +'</div></section>')
    js=r"""
const reviewed = document.querySelectorAll('.case');
const columns=['case_id','reviewer_id','vehicle_visible','physical_class','primary_relation','notes'];
function quote(s){ return '"' + String(s).replaceAll('"','""') + '"'; }
function exportReview(){
  const lines=[columns.join(',')];
  for(const element of reviewed){
    const get=n=>element.querySelector('[name="'+n+'"]').value;
    const visible=get('vehicle_visible'), cls=get('physical_class'), relation=get('primary_relation');
    if(!visible || !cls || !relation){alert('Complete each question for case '+element.dataset.case);return;}
    if(visible==='NO' && (cls!=='NA'||relation!=='NA')){
      alert('Use class NA and relation NA when no object is visible: '+element.dataset.case);return;
    }
    if(visible==='YES' && (cls==='NA'||relation==='NA')){
      alert('Choose an image-derived class and primary relation for '+element.dataset.case);return;
    }
    const vals=[element.dataset.case,document.body.dataset.reviewer,visible,cls,relation,get('notes')];
    lines.push(vals.map(quote).join(','));
  }
  const blob=new Blob([lines.join('\r\n')+'\r\n'],{type:'text/csv;charset=utf-8'});
  const link=document.createElement('a');
  link.href=URL.createObjectURL(blob);
  link.download='W04_'+document.body.dataset.reviewer+'_blind_review.csv';
  link.click();
  URL.revokeObjectURL(link.href);
}
document.getElementById('export').addEventListener('click',exportReview);
"""
    style="""
*{box-sizing:border-box}body{background:#f6f7f8;color:#1e293b;font:16px/1.55 system-ui;margin:0}
main{max-width:1230px;margin:0 auto;padding:20px}header{padding:22px 0 14px}
h1{margin:0;font-size:28px}h2{font-size:21px;margin:0}
.note{color:#475569}.case{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:21px;margin:18px 0}
.samples{margin-top:14px}.frame-name{font-weight:650;margin-bottom:7px}.shots{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px}
figure{margin:0;min-width:0}img{width:100%;height:auto;max-height:420px;object-fit:contain;background:#222}
figcaption{color:#64748b;font-size:13px}.answers{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:17px}
label{font-size:14px;font-weight:600}select,textarea{width:100%;border:1px solid #94a3b8;border-radius:7px;background:white;padding:10px;font:inherit;margin-top:7px}
.answers label:last-child{grid-column:span 3}button{border:0;background:#0f172a;color:white;padding:14px 22px;font-weight:700;border-radius:9px;cursor:pointer}
footer{position:sticky;bottom:0;background:#f6f7f8;padding:12px;border-top:1px solid #cbd5e1}
@media(max-width:900px){.shots{grid-template-columns:1fr 1fr}.shots figure:first-child{grid-column:span 2}.answers{grid-template-columns:1fr}.answers label:last-child{grid-column:span 1}}
"""
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>W04 Independent Visual Review</title><style>'+style+'</style>'
            '<body data-reviewer="'+html.escape(reviewer,quote=True)+'"><main><header>'
            '<h1>Independent physical-vehicle review</h1><p class="note">Reviewer '
            +html.escape(reviewer)+' · 7 anonymous W04 cases · Original source frames</p>'
            '<p>Yellow outlines the proposed target. Gray outlines in the third view show '
            'existing primary tracks. Do not guess the detector label. Record only what '
            'the images support; choose UNCLEAR when necessary. Do not consult the other reviewer.</p>'
            '</header>'+''.join(cards)
            +'<footer><button id="export" type="button">Export my independent CSV</button>'
            '<span class="note"> Save this CSV privately; do not share it with the other reviewer.</span>'
            '</footer></main><script>'+js+'</script></body></html>')


def _select(field: str, choices: list[str], question: str) -> str:
    return ('<label>'+html.escape(question)+'<select name="'+field+'">'
            +''.join('<option value="'+html.escape(value,quote=True)+'">'
                     +html.escape(value or "Select...")+'</option>'
                     for value in choices)
            +'</select></label>')


def package(stage: Path, cases: list[dict], originals: dict, replay: dict,
            pictures: dict, *, video_sha: str):
    # Only the internal manifest can reveal model class/ID; never include it
    # in reviewer-specific ZIP archives.
    identity={
        "status":STATUS,"eligible_for_production":False,
        "video_sha256":video_sha,
        "primary_track_sha256":PRIMARY_SHA,
        "original_batch_sha256":originals["batch_sha256"],
        "hybrid_report_source_original_batch_sha256":replay["source_original_batch_sha256"],
        "evaluated_frames":[START,END],
        "reviewers_required":["R01","R02"],
        "case_count":len(cases),
        "source_frames_are_zero_indexed":True,
        "case_mapping":[{
            "case_id":c["case_id"],
            "candidate_class_internal_only":c["candidate_class"],
            "hybrid_track_id_internal_only":c["hybrid_track_id"],
            "sampled_frame_numbers":[x["frame"] for x in c["sampled"]],
            "all_unmatched_observations":c["all_observations"],
            "review_state":c["review_state"],
            "physical_vehicle_confirmed":False
        } for c in cases],
        "no_auto_class_changes":True,"no_auto_FLUID_changes":True,
        "limitations":"Source imagery is development W04, not independent holdout."
    }
    (stage/"INTERNAL_case_manifest.json").write_text(
        json.dumps(identity,indent=2),encoding="utf-8")
    (stage/"REVIEW_PROTOCOL.md").write_text(
        "# W04 independent physical review\n\n"
        "Deliver the R01 and R02 ZIPs to separate reviewers. The ZIPs must "
        "not include INTERNAL_case_manifest.json or each other's CSV. "
        "Reviewers open index.html offline in a browser, inspect overview, "
        "crop and neutral primary-track comparison, then export a CSV. "
        "Both must independently choose visible object, visually observed "
        "physical class, relationship to existing primary track or UNCLEAR. "
        "Do not discuss cases before exporting. Return CSVs only to adjudicator.\n\n"
        "Class/ID/prediction confidence intentionally blinded. Source video "
        "frames are zero-indexed. No image is a new truth label. No production "
        "or FLUID changes without independent source-grounded adjudication.\n",
        encoding="utf-8")
    with (stage/"INTERNAL_SHA256SUMS.txt").open("x",encoding="utf-8") as f:
        for file in sorted((stage/"assets").glob("*.jpg")):
            f.write(sha(file)+"  assets/"+file.name+"\n")
        f.write(sha(stage/"INTERNAL_case_manifest.json")+"  INTERNAL_case_manifest.json\n")
    for reviewer in ("R01","R02"):
        html_text=html_packet(reviewer,cases,pictures)
        (stage/("TEMPLATE_"+reviewer+".csv")).write_text(
            csv_template(reviewer,cases),encoding="utf-8")
        with zipfile.ZipFile(stage/("REVIEWER_"+reviewer+"_ONLY.zip"),
                             "x",compression=zipfile.ZIP_DEFLATED,
                             compresslevel=5) as archive:
            archive.writestr("index.html",html_text)
            for pic in sorted((stage/"assets").glob("*.jpg")):
                archive.write(pic,arcname="assets/"+pic.name)
    return identity


def run(batch_dir: Path, replay_dir: Path, video: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError("Immutable visual-review output already exists")
    info,replay,primary,proposals,selected=source_evidence(
        batch_dir,replay_dir,video)
    cases=build_cases(proposals,selected)
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+output_dir.name+".stage-",
                                dir=output_dir.parent))
    try:
        picture_files=render_frames(video,cases,primary,stage)
        package(stage,cases,info,replay,picture_files,
                video_sha=info["provenance"]["video_sha256"])
        if output_dir.exists():
            raise FileExistsError("Output created during image generation")
        os.replace(stage,output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {
        "status":STATUS,"review_cases":len(cases),
        "source_frame_renders":len(picture_files),
        "reviewer_packets":["REVIEWER_R01_ONLY.zip","REVIEWER_R02_ONLY.zip"],
        "output":str(output_dir),"eligible_for_production":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir",type=Path,required=True)
    parser.add_argument("--replay-dir",type=Path,required=True)
    parser.add_argument("--video",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(args.batch_dir,args.replay_dir,args.video,
                         args.output_dir),indent=2))


if __name__=="__main__":
    main()
