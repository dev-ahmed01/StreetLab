"""Locked, non-promoting correspondence evaluation on never-used traffic footage.

Uses only native 14-column tracker exports, never FLUID, predicted class as
truth, or reviewer votes as input to duplicate predictions. Pair-level
labels are SAME_PHYSICAL_OBJECT / DISTINCT_PHYSICAL_OBJECT / UNCLEAR.
False-link rate is explicitly tested against distinct nearby vehicles.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from .physical_object_correspondence import (
    overlap_evidence, IOU_SIMILAR, IOS_CONTAINED,
    MAX_PART_AREA_RATIO, MIN_TEMPORAL_SUPPORT_FRAMES,
    MIN_OBSERVATION_COVERAGE,
)

CLASSES={0:"CAR",1:"BUS",2:"HEAVY_VEHICLE",3:"MOTORCYCLE"}
LABELS={"SAME_PHYSICAL_OBJECT","DISTINCT_PHYSICAL_OBJECT","UNCLEAR"}
HEADERS=("case_id","reviewer_id","physical_relation","notes")
PROTOCOL="PHASE3_HELDOUT_CORRESPONDENCE_V1_NOT_PRODUCTION"
W04_VIDEO_SHA="57105b564ff3f9c68d88e6df05790654a49a48f626b1adaf39b62ea61c262ce0"
MIN_EVALUATION_FRAMES=120
MIN_CONFIRMED_SAME=10
MIN_CONFIRMED_DISTINCT=20
MAX_REVIEW_PAIRS=120
NEAR_DIAGONALS=1.5

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda:stream.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()

def read_tracks(path: Path, first: int, last: int) -> tuple[dict[int,list[dict[str,Any]]],dict]:
    """Strict schema/fps-independent absolute-frame tracks. No fake IDs."""
    if type(first) is not int or type(last) is not int or first<0 or last-first+1<MIN_EVALUATION_FRAMES:
        raise ValueError("Holdout must preregister >=120 source evaluation frames")
    if not path.is_file():
        raise FileNotFoundError(path)
    frames=defaultdict(list)
    seen=set()
    count=0
    prior=first
    classes=Counter()
    with path.open(encoding="utf-8-sig",newline="") as f:
        for row in csv.reader(f):
            if len(row)!=14:
                raise ValueError("Native Geo-trax requires 14 numeric columns without a header")
            v=[float(x) for x in row]
            if not all(math.isfinite(x) for x in v):
                raise ValueError("Nonfinite tracking values")
            frame,tid,klass=(int(v[i]) for i in (0,1,10))
            if (frame!=v[0] or tid!=v[1] or klass!=v[10] or tid<0
                or klass not in CLASSES or not first<=frame<=last
                or frame<prior or (frame,tid) in seen):
                raise ValueError("Track frames/classes/identities invalid, unsorted or duplicated")
            cx,cy,w,h,conf=v[2],v[3],v[4],v[5],v[11]
            if w<=0 or h<=0 or not 0<=conf<=1:
                raise ValueError("Invalid source-pixel box or confidence")
            seen.add((frame,tid))
            prior=frame
            frames[frame].append({
                "frame":frame,"id":tid,"vehicle_class":CLASSES[klass],
                "x1":cx-w/2,"y1":cy-h/2,"x2":cx+w/2,"y2":cy+h/2,
                "confidence":conf,
            })
            classes[CLASSES[klass]]+=1
            count+=1
    if not count:
        raise ValueError("Source tracking export has no object evidence")
    return dict(frames),{"sha256":sha(path),"rows":count,"classes":dict(classes)}

def broad_nearby(shadow: dict, primary: dict) -> bool:
    """Candidate inventory extends beyond overlap to include close distinct cars."""
    a=(shadow["x1"],shadow["y1"],shadow["x2"],shadow["y2"])
    b=(primary["x1"],primary["y1"],primary["x2"],primary["y2"])
    ca=((a[0]+a[2])/2,(a[1]+a[3])/2)
    cb=((b[0]+b[2])/2,(b[1]+b[3])/2)
    da=math.hypot(a[2]-a[0],a[3]-a[1])
    db=math.hypot(b[2]-b[0],b[3]-b[1])
    return math.dist(ca,cb)<=NEAR_DIAGONALS*max(da,db)

def build_pair_inventory(primary: Mapping[int,list[dict]],
                         shadow: Mapping[int,list[dict]]) -> list[dict[str,Any]]:
    """Fixed candidates selected before human labels. No truth-assisted ranking."""
    all_pair=defaultdict(list)
    group_frames=defaultdict(set)
    for frame,shadows in shadow.items():
        existing=primary.get(frame,[])
        for s in shadows:
            skey=(s["vehicle_class"],s["id"])
            group_frames[skey].add(frame)
            for p in existing:
                if not broad_nearby(s,p):
                    continue
                evidence=overlap_evidence(s,p)
                key=(s["vehicle_class"],s["id"],p["id"])
                all_pair[key].append((frame,evidence,s,p))
    pairs=[]
    for (klass,sid,pid),events in all_pair.items():
        fs=sorted({v[0] for v in events})
        strong=[e for e in events if e[1]["relation"] in (
            "SAME_EXTENT_OVERLAP","PART_WHOLE_OVERLAP")]
        strong_fs=sorted({e[0] for e in strong})
        total=len(group_frames[(klass,sid)])
        frac=len(strong_fs)/total
        # Pair prediction is an association hypothesis only. A tie with
        # another primary will be marked AMBIGUOUS after all keys are ranked.
        duplicate=(len(strong_fs)>=MIN_TEMPORAL_SUPPORT_FRAMES
                   and frac>=MIN_OBSERVATION_COVERAGE)
        bucket=("DUPLICATE_HYPOTHESIS" if duplicate else
                "WEAK_GEOMETRIC_RELATION" if strong else
                "NEARBY_NEGATIVE_CONTROL")
        sampled=sorted(set([fs[0],fs[len(fs)//2],fs[-1],
                            max(events,key=lambda x:x[1]["ios"])[0]]))
        pairs.append({
            "shadow_class":klass,"shadow_id":sid,"primary_id":pid,
            "cooccurring_near_frames":len(fs),
            "strong_overlap_frames":len(strong_fs),
            "shadow_frames":total,
            "strong_fraction":round(frac,6),
            "prediction":bucket,
            "source_frames_sampled":sampled,
            "best_ios":max(e[1]["ios"] for e in events),
            "class_agrees_on_overlap":all(
                s["vehicle_class"]==p["vehicle_class"] for _,_,s,p in events),
            "physically_verified":False,
        })
    # Treat multiple strong associations for one shadow ID as an abstention.
    by_shadow=Counter(
        (p["shadow_class"],p["shadow_id"])
        for p in pairs if p["prediction"]=="DUPLICATE_HYPOTHESIS")
    for p in pairs:
        if (p["prediction"]=="DUPLICATE_HYPOTHESIS"
            and by_shadow[(p["shadow_class"],p["shadow_id"])]>1):
            p["prediction"]="AMBIGUOUS_MULTIPLE_PRIMARY_IDS"
    pairs.sort(key=lambda p:(
        p["prediction"],p["shadow_class"],p["shadow_id"],p["primary_id"]))
    return pairs

def select_blind_pairs(pairs: list[dict], *, digest_hex: str,
                       limit: int = MAX_REVIEW_PAIRS) -> list[dict]:
    """Deterministic fixed-stratum sampling without knowing physical labels."""
    if type(limit) is not int or not 12<=limit<=MAX_REVIEW_PAIRS:
        raise ValueError("Invalid frozen holdout review size")
    buckets=defaultdict(list)
    for p in pairs:
        buckets[p["prediction"]].append(p)
    for bucket,entries in buckets.items():
        entries.sort(key=lambda p:hashlib.sha256(
            (digest_hex+"|"+str(p["shadow_class"])+"|"+str(p["shadow_id"])
             +"|"+str(p["primary_id"])).encode()).hexdigest())
    names=("DUPLICATE_HYPOTHESIS","AMBIGUOUS_MULTIPLE_PRIMARY_IDS",
           "WEAK_GEOMETRIC_RELATION","NEARBY_NEGATIVE_CONTROL")
    take={name:[] for name in names}
    # Round-robin across strata ensures diverse collisions, nearby distinct
    # controls and uncertain boxes rather than class/ID confidence cherry-pick.
    while sum(map(len,take.values()))<limit:
        chosen=False
        for name in names:
            if (len(take[name])<len(buckets[name])
                and sum(map(len,take.values()))<limit):
                take[name].append(buckets[name][len(take[name])])
                chosen=True
        if not chosen:
            break
    selected=[item for name in names for item in take[name]]
    # Hide model prediction from reviewers; map is kept in INTERNAL manifest.
    selected.sort(key=lambda p:hashlib.sha256(
        ("blind|"+digest_hex+"|"+str(p["shadow_class"])+str(p["shadow_id"])
         +str(p["primary_id"])).encode()).hexdigest())
    for i,entry in enumerate(selected,1):
        entry["case_id"]=f"H{i:03d}"
    return selected

def parse_reviews(path: Path, reviewer: str, cases: set[str]):
    data={}
    with path.open(encoding="utf-8-sig",newline="") as f:
        r=csv.DictReader(f)
        if r.fieldnames!=list(HEADERS):
            raise ValueError("Holdout review schema must use frozen four columns")
        for row in r:
            case=row["case_id"]
            if (case not in cases or case in data or row["reviewer_id"]!=reviewer
                or row["physical_relation"] not in LABELS
                or row["notes"] is None or len(row["notes"])>2000):
                raise ValueError("Unknown, duplicate, missing or invalid independent judgment")
            data[case]=row
    if set(data)!=cases:
        raise ValueError("Both reviewers must independently annotate ALL cases")
    return data

def score_consensus(rows: list[dict], review1: dict, review2: dict) -> dict:
    """Predictions were frozen before reviewers. Uncertain votes are abstentions."""
    cm=Counter()
    detail=[]
    for item in rows:
        case=item["case_id"]
        a=review1[case]["physical_relation"]
        b=review2[case]["physical_relation"]
        consensus=a if a==b and a!="UNCLEAR" else None
        prediction=item["prediction"]
        auto_link=prediction=="DUPLICATE_HYPOTHESIS"
        if consensus is not None:
            cm["adjudicated"]+=1
            cm["same" if consensus=="SAME_PHYSICAL_OBJECT" else "distinct"]+=1
            if auto_link and consensus=="DISTINCT_PHYSICAL_OBJECT":
                cm["false_merge"]+=1
            if auto_link and consensus=="SAME_PHYSICAL_OBJECT":
                cm["true_duplicate_link"]+=1
            if not auto_link and consensus=="SAME_PHYSICAL_OBJECT":
                cm["missed_duplicate_link"]+=1
            if auto_link:
                cm["positive_predictions"]+=1
        else:
            cm["disagreed_or_unclear"]+=1
        detail.append({"case_id":case,
                       "proposed_relation":prediction,
                       "reviewer1":a,"reviewer2":b,
                       "agreed_relation":consensus or "REQUIRES_ADJUDICATION",
                       "would_link_if_enabled":auto_link,
                       "known_distinct_false_merge":bool(
                           auto_link and consensus=="DISTINCT_PHYSICAL_OBJECT"),
                       "production_eligible":False})
    minimum=cm["same"]>=MIN_CONFIRMED_SAME and cm["distinct"]>=MIN_CONFIRMED_DISTINCT
    no_disputes=cm["disagreed_or_unclear"]==0
    zero_false=cm["false_merge"]==0
    diagnostics_ok=minimum and no_disputes and zero_false
    return {
        "case_count":len(rows),"reviewed_same":cm["same"],
        "reviewed_distinct":cm["distinct"],
        "disagreed_or_unclear":cm["disagreed_or_unclear"],
        "false_merge_on_annotated_distinct":cm["false_merge"],
        "duplicate_true_links":cm["true_duplicate_link"],
        "duplicate_missed_links":cm["missed_duplicate_link"],
        "pair_sensitivity_on_annotated_same":(
            cm["true_duplicate_link"]/cm["same"] if cm["same"] else None),
        "distinct_false_link_fraction":(
            cm["false_merge"]/cm["distinct"] if cm["distinct"] else None),
        "minimum_class_balance_verified":minimum,
        "non_promoting_diagnostic_gate_passed":diagnostics_ok,
        "eligible_for_production":False,
        "release_gates_unverified":[
            "independence_of_human_reviews",
            "unseen_video_provenance_beyond_sha",
            "whole_video_object_count_accuracy",
            "primary_tracker_identity_nonregression",
            "CPU_p95_full_pipeline",
        ],
        "details":detail,
    }
