"""Conservative physical-object correspondence *evidence*, NOT identity merging.

Re-evaluates a shadow hypothesis against every primary tracker box in the same
source frame, including different detector classes. Uses spatial containment,
whole-extent overlap and repeated primary-ID support, never FLUID labels or
human-review answers. A correspondence is *not* proof of the same object.

Thresholds are structural, fixed constants, NOT optimized on W04 reviewers.
No source updates, no promotion, no replacement of primary track IDs/classes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Mapping, Sequence, Any

IOU_SIMILAR = .50
IOS_CONTAINED = .90
MAX_PART_AREA_RATIO = .65
MIN_TEMPORAL_SUPPORT_FRAMES = 3
MIN_OBSERVATION_COVERAGE = .50
KINDS = ("SAME_EXTENT_OVERLAP", "PART_WHOLE_OVERLAP")


def _number(obj: Mapping[str, Any], field: str) -> float:
    value=obj[field]
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("Invalid finite numeric box field: "+field)
    return float(value)


def _box(obj: Mapping[str,Any]) -> tuple[float,float,float,float]:
    xy=tuple(_number(obj,key) for key in ("x1","y1","x2","y2"))
    if xy[0]<0 or xy[1]<0 or xy[2]<=xy[0] or xy[3]<=xy[1]:
        raise ValueError("Invalid positive pixel-coordinate box")
    return xy


def overlap_evidence(shadow: Mapping[str,Any], primary: Mapping[str,Any]) -> dict[str,Any]:
    """Whole/part evidence including class-agnostic spatial containment.

    IoS / IoU alone are not reliable when adjacent vehicles overlap; the
    smaller-box center also must be inside the larger box for part-whole.
    """
    a=_box(shadow)
    b=_box(primary)
    left=max(a[0],b[0])
    top=max(a[1],b[1])
    right=min(a[2],b[2])
    bottom=min(a[3],b[3])
    area_i=max(0,right-left)*max(0,bottom-top)
    area_a=(a[2]-a[0])*(a[3]-a[1])
    area_b=(b[2]-b[0])*(b[3]-b[1])
    iou=area_i/(area_a+area_b-area_i)
    ios=area_i/min(area_a,area_b)
    ratio=min(area_a,area_b)/max(area_a,area_b)
    smaller=a if area_a<=area_b else b
    larger=b if area_a<=area_b else a
    center=((smaller[0]+smaller[2])/2,(smaller[1]+smaller[3])/2)
    center_inside=larger[0]<=center[0]<=larger[2] and larger[1]<=center[1]<=larger[3]
    if iou>=IOU_SIMILAR:
        kind="SAME_EXTENT_OVERLAP"
    elif ios>=IOS_CONTAINED and ratio<=MAX_PART_AREA_RATIO and center_inside:
        kind="PART_WHOLE_OVERLAP"
    else:
        kind="NO_STRONG_GEOMETRIC_RELATION"
    return {"relation":kind,"iou":round(iou,6),
            "ios":round(ios,6),"smaller_larger_area_ratio":round(ratio,6),
            "smaller_center_inside_larger":center_inside,
            "class_agrees":str(shadow["vehicle_class"])==str(primary["vehicle_class"]),
            "same_physical_object_proven":False}


def _id(obj: Mapping[str,Any], field:str) -> int:
    value=obj[field]
    if type(value) is not int or value<0:
        raise ValueError("Original tracker IDs must be genuine nonnegative integers")
    return value


def compare_shadow_to_primary(
    shadow_observations: Sequence[dict[str,Any]],
    primary_by_frame: Mapping[int,Sequence[dict[str,Any]]]
) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    """Return one observation per shadow row, complete candidate pairs, tracklets.

    Never suppresses a row or invents a primary ID for an unpaired hypothesis.
    Pairs of different detector classes are retained as class-ambiguous
    correspondence evidence, not class corrections.
    """
    observations=[]
    pairs=[]
    grouped=defaultdict(list)
    seen=set()
    for item in shadow_observations:
        frame=_id(item,"frame")
        shadow_id=_id(item,"shadow_id")
        klass=str(item["vehicle_class"])
        if not klass:
            raise ValueError("Shadow class missing")
        key=(frame,klass,shadow_id)
        if key in seen:
            raise ValueError("Repeated shadow tracker ID/class on one source frame")
        seen.add(key)
        _box(item)
        eligible=[]
        all_primary=primary_by_frame.get(frame,())
        primary_seen=set()
        for primary in all_primary:
            pid=_id(primary,"primary_id")
            if pid in primary_seen:
                raise ValueError("Duplicated original primary tracker ID in source frame")
            primary_seen.add(pid)
            if _id(primary,"frame")!=frame:
                raise ValueError("Primary tracker row belongs to another source frame")
            metrics=overlap_evidence(item,primary)
            if metrics["relation"] not in KINDS:
                continue
            score=(2. if metrics["relation"]=="SAME_EXTENT_OVERLAP" else 1.)+metrics["ios"]
            record={"frame":frame,"shadow_id":shadow_id,"shadow_class":klass,
                    "primary_id":pid,"primary_class":str(primary["vehicle_class"]),
                    "spatial_relation":metrics["relation"],
                    "iou":metrics["iou"],"ios":metrics["ios"],
                    "area_ratio":metrics["smaller_larger_area_ratio"],
                    "class_agrees":metrics["class_agrees"],
                    "rank_score":round(score,6),
                    "physical_object_confirmed":False}
            eligible.append(record)
        eligible.sort(key=lambda r:(-r["rank_score"],-r["iou"],r["primary_id"]))
        pairs.extend(eligible)
        obs={"frame":frame,"shadow_id":shadow_id,"shadow_class":klass,
             "plausible_primary_count":len(eligible),
             "best_primary_id_spatial_only":(eligible[0]["primary_id"] if eligible else None),
             "best_primary_class":(eligible[0]["primary_class"] if eligible else ""),
             "best_spatial_relation":(eligible[0]["spatial_relation"] if eligible else "NO_SPATIAL_PAIR"),
             "best_iou":(eligible[0]["iou"] if eligible else 0.),
             "best_ios":(eligible[0]["ios"] if eligible else 0.),
             "same_physical_vehicle_confirmed":False,
             "counts_as_new_vehicle":False}
        observations.append(obs)
        grouped[(klass,shadow_id)].append(obs)
    # Choose repeated primary IDs using *all* plausible spatial pairs, not
    # merely the strongest per-frame box, to avoid false temporal fragmentation
    # when an occluding third vehicle briefly wins a geometric tie.
    pair_by_shadow=defaultdict(list)
    for p in pairs:
        pair_by_shadow[(p["shadow_class"],p["shadow_id"])].append(p)
    tracklets=[]
    for (klass,shadow_id),rows in sorted(grouped.items()):
        count=len(rows)
        pids=defaultdict(list)
        for pair in pair_by_shadow[(klass,shadow_id)]:
            pids[pair["primary_id"]].append(pair)
        ranked=[]
        for pid,prs in pids.items():
            fs=sorted({p["frame"] for p in prs})
            ranked.append({
                "primary_id":pid,"matched_frame_count":len(fs),
                "matched_fraction":round(len(fs)/count,5),
                "longest_consecutive_frames":_consecutive(fs),
                "different_class_frame_count":sum(not p["class_agrees"] for p in prs),
                "part_whole_frames":sum(p["spatial_relation"]=="PART_WHOLE_OVERLAP" for p in prs),
                "whole_overlap_frames":sum(p["spatial_relation"]=="SAME_EXTENT_OVERLAP" for p in prs),
                "mean_ios":round(sum(p["ios"] for p in prs)/len(prs),5),
            })
        ranked.sort(key=lambda p:(-p["matched_frame_count"],
                                  -p["longest_consecutive_frames"],
                                  -p["mean_ios"],p["primary_id"]))
        primary=ranked[0] if ranked else None
        temporal=bool(
            primary and primary["matched_frame_count"]>=MIN_TEMPORAL_SUPPORT_FRAMES
            and primary["matched_fraction"]>=MIN_OBSERVATION_COVERAGE
        )
        alternatives=[
            r for r in ranked[1:]
            if r["matched_frame_count"]>=MIN_TEMPORAL_SUPPORT_FRAMES
            and r["matched_fraction"]>=MIN_OBSERVATION_COVERAGE
        ]
        if temporal and alternatives:
            status="MULTIPLE_PRIMARY_OBJECT_HYPOTHESES"
        elif temporal:
            status="REPEATED_GEOMETRIC_PRIMARY_CORRESPONDENCE"
        elif ranked:
            status="WEAK_OR_DISCONTINUOUS_SPATIAL_CORRESPONDENCE"
        else:
            status="NO_SPATIAL_CORRESPONDENCE"
        tracklets.append({
            "vehicle_class":klass,"shadow_id":shadow_id,
            "shadow_observation_frames":count,
            "spatially_paired_frames":sum(x["plausible_primary_count"]>0 for x in rows),
            "repeated_correspondence_status":status,
            "primary_id_hypothesis":primary["primary_id"] if primary else None,
            "primary_support_frames":primary["matched_frame_count"] if primary else 0,
            "primary_support_fraction":primary["matched_fraction"] if primary else 0.,
            "primary_consecutive_support":primary["longest_consecutive_frames"] if primary else 0,
            "primary_part_whole_frames":primary["part_whole_frames"] if primary else 0,
            "primary_cross_class_frames":primary["different_class_frame_count"] if primary else 0,
            "independent_alternative_primary_hypotheses":len(alternatives),
            "physical_object_duplicate_verified":False,
            "eligible_for_count":False,
            "requires_physical_review_for_truth":True,
        })
    return observations,pairs,tracklets


def _consecutive(frames: Sequence[int]) -> int:
    if not frames:
        return 0
    longest=current=1
    prev=frames[0]
    for frame in frames[1:]:
        current=current+1 if frame==prev+1 else 1
        longest=max(longest,current)
        prev=frame
    return longest
