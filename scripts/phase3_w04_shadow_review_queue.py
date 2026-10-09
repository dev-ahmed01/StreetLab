"""Read-only W04 two-lane rare vehicle evidence queue.

Consumes *previously executed* raw-tile and ByteTrack output ZIPs.
The original IoS.30 14-column track is authoritative. Candidate observations
are separate, unverified review evidence, never production measurements.

Pairs tracks by same-frame/same-class maximum-cardinality IoU>=.50;
verifies original control track byte parity and all four policy SHA files.
Records original raw-box support and all cross-class ambiguities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

STATUS = "W04_SHADOW_RARE_EVIDENCE_ONLY_NO_PRODUCTION"
FIRST, START, END = 10660, 10750, 10950
CLASSES = {0: "CAR", 1: "BUS", 2: "HEAVY_VEHICLE", 3: "MOTORCYCLE"}
RARE = (2, 3)
MATCH_IOU = .50
HIGH_CONF = .50
MIN_STREAK = 3
MIN_HIGH_CONF_FRAMES = 2


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_zip(z: zipfile.ZipFile, filename: str) -> bytes:
    if (filename.startswith("/") or "\\" in filename
        or len(PurePosixPath(filename).parts) != 1
        or filename.startswith(".")):
        raise ValueError("Unsafe or nested evidence file name")
    return z.read(filename)


def validate_zip(z: zipfile.ZipFile):
    names = z.namelist()
    if len(names) != len(set(names)) or any(
        p.startswith("/") or ".." in PurePosixPath(p).parts for p in names
    ):
        raise ValueError("Duplicate or unsafe zip paths")


def box_iou(a, b) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    intersection = max(0., x2-x1)*max(0., y2-y1)
    aa = (a["x2"]-a["x1"])*(a["y2"]-a["y1"])
    bb = (b["x2"]-b["x1"])*(b["y2"]-b["y1"])
    return intersection/(aa+bb-intersection) if aa+bb>intersection else 0.


def parse_tracks(data: bytes, *, label: str, expected_rows: int):
    frames = defaultdict(list)
    seen = set()
    last_frame = START
    total = 0
    for rec in csv.reader(io.StringIO(data.decode("utf-8-sig"))):
        if len(rec) != 14:
            raise ValueError(f"{label}: expected genuine 14-column track export")
        floats = tuple(float(x) for x in rec)
        if any(not math.isfinite(x) for x in floats):
            raise ValueError(f"{label}: nonfinite coordinate or class")
        frame, tid, cls = int(floats[0]), int(floats[1]), int(floats[10])
        if (float(frame) != floats[0] or float(tid) != floats[1]
            or float(cls) != floats[10] or cls not in CLASSES
            or not START <= frame <= END or tid < 0 or frame < last_frame
            or (frame, tid) in seen):
            raise ValueError(f"{label}: invalid frame/class/ID or unsorted/duplicate rows")
        cx, cy, w, h, conf = (floats[i] for i in (2, 3, 4, 5, 11))
        if w<=0 or h<=0 or not 0<=conf<=1:
            raise ValueError(f"{label}: invalid geometry or detector confidence")
        seen.add((frame,tid))
        last_frame=frame
        frames[frame].append({
            "frame":frame,"id":tid,"cls":cls,"class":CLASSES[cls],
            "x1":cx-w/2,"y1":cy-h/2,"x2":cx+w/2,"y2":cy+h/2,
            "confidence":conf
        })
        total += 1
    if total != expected_rows:
        raise ValueError(f"{label}: track row count vs manifest mismatch")
    return frames


def parse_raw(data: bytes, *, count: int):
    rows=defaultdict(list)
    all_keys=set()
    total=0
    for line in data.splitlines():
        item=json.loads(line)
        frame=item["frame"]
        if (type(frame) is not int or frame<FIRST or frame>END
            or item["vehicle_class"] not in (*CLASSES.values(),"BICYCLE","PEDESTRIAN")
            or (frame,item["source_index"]) in all_keys):
            raise ValueError("Corrupt original source-frame raw detector boxes")
        key=(frame,item["source_index"])
        all_keys.add(key)
        for v in ("x1","y1","x2","y2","confidence"):
            if not isinstance(item[v],(float,int)) or not math.isfinite(item[v]):
                raise ValueError("Nonfinite source raw box")
        if (item["x2"]<=item["x1"] or item["y2"]<=item["y1"]
            or not 0<=item["confidence"]<=1):
            raise ValueError("Invalid source raw box dimensions")
        rows[frame].append(item)
        total+=1
    if total != count or total != 21104:
        raise ValueError("Source raw box count does not match frozen W04")
    return rows


def choose_matches(original: list[dict], hybrid: list[dict]):
    """Maximize valid one-to-one IoU>=.5 pairs before aggregate IoU."""
    n,m=len(original),len(hybrid)
    if not n or not m:
        return set()
    ious=np.array([[box_iou(a,b) for b in hybrid] for a in original],
                  dtype=float)
    # Unmatched is free; each valid pairing earns >1, preventing weak links
    # from displacing two valid pairs merely to maximize raw total IoU.
    costs=np.zeros((n+m,n+m),dtype=float)
    costs[:n,:m]=np.where(ious>=MATCH_IOU,-(2.+ious),10.)
    rows,cols=linear_sum_assignment(costs)
    return {int(j) for i,j in zip(rows,cols)
            if i<n and j<m and ious[i,j]>=MATCH_IOU}


def longest_streak(frames):
    frames=sorted(set(frames))
    longest = current = 0
    prior = None
    for f in frames:
        current = current+1 if prior is not None and f == prior+1 else 1
        longest = max(longest,current)
        prior = f
    return longest


def build_queue(control: dict, hybrid: dict, raw: dict):
    novel=[]
    grouped=defaultdict(list)
    for frame in range(START,END+1):
        primary=control.get(frame,[])
        contender=hybrid.get(frame,[])
        for klass in RARE:
            originals=[x for x in primary if x["cls"]==klass]
            proposals=[x for x in contender if x["cls"]==klass]
            matched=choose_matches(originals,proposals)
            others=[x for x in primary if x["cls"]!=klass]
            for i, p in enumerate(proposals):
                if i in matched:
                    continue
                cross=sorted(
                    ((box_iou(x,p),x["class"]) for x in others),
                    key=lambda x:(-x[0],x[1])
                )
                overlap,other=(cross[0] if cross else (0.,""))
                support=[x for x in raw.get(frame,[])
                         if box_iou(x,p)>=MATCH_IOU]
                same=[x for x in support if x["vehicle_class"]==p["class"]]
                different=[x for x in support if x["vehicle_class"]!=p["class"]]
                record={
                    "frame":frame, "hybrid_id":p["id"],"vehicle_class":p["class"],
                    "x1":round(p["x1"],5),"y1":round(p["y1"],5),
                    "x2":round(p["x2"],5),"y2":round(p["y2"],5),
                    "confidence":round(p["confidence"],6),
                    "cross_class_control_iou":round(overlap,6),
                    "cross_class_control_label":other,
                    "cross_class_control_conflict":overlap>=MATCH_IOU,
                    "same_class_raw_supported":bool(same),
                    "same_class_raw_max_conf":round(
                        max((b["confidence"] for b in same),default=0.),6),
                    "competing_raw_class_hypotheses":bool(different),
                    "competing_raw_labels":"|".join(
                        sorted({b["vehicle_class"] for b in different})),
                    "physically_confirmed":False
                }
                novel.append(record)
                grouped[(klass,p["id"])].append(record)
    tracklets=[]
    for (klass,track_id),events in sorted(grouped.items()):
        total=len(events)
        highs=sum(x["confidence"]>=HIGH_CONF for x in events)
        streak=longest_streak(x["frame"] for x in events)
        crosses=sum(x["cross_class_control_conflict"] for x in events)
        raw_competes=sum(x["competing_raw_class_hypotheses"] for x in events)
        raw_same=sum(x["same_class_raw_supported"] for x in events)
        queue=(streak>=MIN_STREAK and highs>=MIN_HIGH_CONF_FRAMES)
        # Every queue item is unverified; competing raw class hypotheses may
        # indicate classification ambiguity even if primary did not overlap.
        level=("needs_cross_class_adjudication" if crosses or raw_competes
               else "lower_class_ambiguity")
        tracklets.append({
            "vehicle_class":CLASSES[klass],"hybrid_id":track_id,
            "unmatched_frame_rows":total,
            "first_frame":min(x["frame"] for x in events),
            "last_frame":max(x["frame"] for x in events),
            "longest_consecutive_unmatched_frames":streak,
            "high_confidence_rows":highs,
            "mean_confidence":round(sum(x["confidence"] for x in events)/total,5),
            "cross_class_control_conflict_rows":crosses,
            "competing_raw_class_rows":raw_competes,
            "same_class_raw_supported_rows":raw_same,
            "eligible_for_human_review":queue,
            "class_ambiguity_category":level,
            "review_state":"PENDING_HUMAN_REVIEW" if queue else "BELOW_REVIEW_THRESHOLD",
            "physical_vehicle_confirmed":False
        })
    priority={"lower_class_ambiguity":0,"needs_cross_class_adjudication":1}
    tracklets.sort(key=lambda x:(
        not x["eligible_for_human_review"],
        priority[x["class_ambiguity_category"]],
        -x["high_confidence_rows"],-x["longest_consecutive_unmatched_frames"],
        x["vehicle_class"],x["hybrid_id"]
    ))
    return novel,tracklets


def analyze(baseline_zip: Path, hybrid_zip: Path):
    with zipfile.ZipFile(baseline_zip) as b, zipfile.ZipFile(hybrid_zip) as h:
        validate_zip(b)
        validate_zip(h)
        original_report_bytes=read_zip(b,"batch_report.json")
        report=json.loads(original_report_bytes)
        unified=json.loads(read_zip(h,"unified_hybrid_report.json"))
        if (report.get("candidate_count")!=25
            or report.get("raw_tile_boxes")!=21104
            or report.get("first_decoded_frame")!=FIRST
            or report.get("evaluated_first_frame")!=START
            or report.get("evaluated_last_frame")!=END
            or report.get("same_window_T000") is not None
            or unified.get("source_frames")!=[FIRST,END]
            or unified.get("evaluation_frames")!=[START,END]
            or unified.get("eligible_for_production") is not False
            or unified.get("original_control_reproduction",{}).get("successful") is not True
            or unified.get("source_original_batch_sha256")!=digest(original_report_bytes)
            or unified.get("source_original_raw_sha256")!=report.get("raw_pre_global_merge_boxes_sha256")):
            raise ValueError("Original SHA-anchored continuous W04 cohort mismatch")
        results={x["name"]:x for x in unified["experimental_results"]}
        expected_modes={"control_ios030","rare_iou05",
                        "rare_center_gate","rare_cross_tile_center_gate"}
        if len(results)!=4 or set(results)!=expected_modes:
            raise ValueError("Incomplete four-policy unified hybrid comparison")
        original_reference=next(
            x for x in report["policies"] if x["name"]=="hard_nms_ios_0.30")
        original_control=read_zip(b,original_reference["tracks"])
        if digest(original_control)!=original_reference["tracks_sha256"]:
            raise ValueError("Frozen original IoS .30 primary track was altered")
        source_tracks={}
        for mode in expected_modes:
            item=results[mode]
            content=read_zip(h,item["tracks"])
            sidecar=read_zip(h,item["class_evidence"])
            if (digest(content)!=item["tracks_sha256"]
                or digest(sidecar)!=item["class_evidence_sha256"]):
                raise ValueError("Candidate or class sidecar evidence SHA changed")
            # Check *all* four track files and evidence CSV, not only the pair
            tracks=parse_tracks(content,label=mode,
                                expected_rows=item["evaluation_confirmed_rows"])
            with io.StringIO(sidecar.decode("utf-8-sig")) as f:
                sidecar_rows=list(csv.DictReader(f))
            if len(sidecar_rows)!=item["sidecar_evidence_rows"]:
                raise ValueError("Class sidecar row count mismatch")
            source_tracks[mode]=tracks
        if read_zip(h,results["control_ios030"]["tracks"])!=original_control:
            raise ValueError("Primary authoritative track control is NOT byte-identical")
        raw_bytes=read_zip(b,"original_pre_global_merge_boxes.jsonl")
        if digest(raw_bytes)!=report["raw_pre_global_merge_boxes_sha256"]:
            raise ValueError("Original raw-tile cache SHA changed")
        raw=parse_raw(raw_bytes,count=report["raw_tile_boxes"])
        novel,tracklets=build_queue(
            source_tracks["control_ios030"],source_tracks["rare_iou05"],raw)
        eligible=[x for x in tracklets if x["eligible_for_human_review"]]
        counts=Counter(x["vehicle_class"] for x in novel)
        review_counts=Counter(x["vehicle_class"] for x in eligible)
        summary={
            "status":STATUS,"eligible_for_production":False,
            "primary_tracks_immutable":True,"primary_track_sha256":digest(original_control),
            "raw_tiled_evidence_sha256":digest(raw_bytes),
            "original_batch_sha256":digest(original_report_bytes),
            "source_hybrid_zip_sha256":digest(hybrid_zip.read_bytes()),
            "original_control_metrics_reproduced":True,
            "source_frame_cohort":[FIRST,END],
            "evaluation_frame_cohort":[START,END],
            "matching":"same-source-frame and same-vehicle-class maximum-cardinality Hungarian; IoU>=0.50",
            "review_rule":{
                "min_consecutive_unmatched_frames":MIN_STREAK,
                "min_confidence":HIGH_CONF,
                "min_high_confidence_rows":MIN_HIGH_CONF_FRAMES,
                "raw_detector_support_iou":MATCH_IOU
            },
            "unmatched_shadow_observations":len(novel),
            "unmatched_by_class":dict(counts),
            "shadow_tracklets":len(tracklets),
            "eligible_review_tracklets":len(eligible),
            "eligible_review_by_class":dict(review_counts),
            "review_candidates_with_competing_raw_classes":sum(
                x["competing_raw_class_rows"]>0 for x in eligible),
            "review_candidates_without_all_raw_same_class_support":sum(
                x["same_class_raw_supported_rows"]<x["unmatched_frame_rows"]
                for x in eligible),
            "needs_independent_physical_adjudication":True,
            "explanation":(
                "Non-promoted review hypotheses. Matching to a primary same-class "
                "track is NOT proof of vehicle novelty or recall. Primary IoS.30 "
                "tracker remains authoritative; no original track or FLUID class "
                "may be replaced by shadow observations."
            ),
            "limitations":(
                "These are same-video hypothesis matches, not physically labeled "
                "true positives. Some may be duplicate, missed-association or "
                "mislabeled original vehicles. No unseen holdout, original T000, "
                "p95 throughput evidence or production approval."
            )
        }
        return summary,novel,tracklets


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-batch-zip",type=Path,required=True)
    parser.add_argument("--unified-replay-zip",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Immutable review output already exists")
    result,observations,queue=analyze(
        args.original_batch_zip,args.unified_replay_zip)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+args.output_dir.name+".stage-",
                                dir=args.output_dir.parent))
    try:
        for name,rows in (("shadow_observations.csv",observations),
                          ("shadow_tracklet_review.csv",queue),
                          ("human_review_queue.csv",
                           [x for x in queue if x["eligible_for_human_review"]])):
            if not rows:
                raise ValueError("Expected real W04 shadow evidence missing")
            with (stage/name).open("x",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (stage/"shadow_lane_report.json").write_text(
            json.dumps(result,indent=2),encoding="utf-8")
        # Output SHA manifest is over the complete immutable rendered reports.
        with (stage/"SHA256SUMS.txt").open("x",encoding="utf-8") as f:
            for path in sorted(stage.iterdir()):
                if path.name!="SHA256SUMS.txt":
                    f.write(digest(path.read_bytes())+"  "+path.name+"\n")
        if args.output_dir.exists():
            raise FileExistsError("Output unexpectedly exists")
        os.replace(stage,args.output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps({"status":result["status"],
                      "unmatched":result["unmatched_shadow_observations"],
                      "review_candidates":result["eligible_review_tracklets"],
                      "report":str(args.output_dir),
                      "production_promotion":False},indent=2))


if __name__=="__main__":
    main()
