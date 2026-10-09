"""Reproducible W04 part-whole correspondence audit from immutable source ZIPs.

Uses all 191 rare-IoU unmatched observations, not only seven reviewer cases.
The geometry module is generic, runs without FLUID/4K video or new inference,
and never consumes reviewer answers as features. Human votes are appended
*only after* correspondence scoring for a descriptive seven-case sanity check.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0,str(_ROOT))

from streetlab_phase3.video.physical_object_correspondence import (
    compare_shadow_to_primary, IOU_SIMILAR, IOS_CONTAINED,
    MAX_PART_AREA_RATIO, MIN_TEMPORAL_SUPPORT_FRAMES, MIN_OBSERVATION_COVERAGE)
from phase3_w04_shadow_review_queue import analyze, digest, parse_tracks

STATUS="W04_CROSS_CLASS_PART_WHOLE_CORRESPONDENCE_OBSERVATION_NOT_PRODUCTION"
START,END=10750,10950
PRIMARY_SHA="6c695e384e05421ec25dbe450f00a83fb6a72a659a2d162a1122d67c7d0ce418"


def zip_bytes(z: zipfile.ZipFile, name: str):
    if (not name or name.startswith(".") or name.startswith("/")
        or "\\" in name or len(PurePosixPath(name).parts)!=1):
        raise ValueError("Unsafe evidence ZIP member")
    return z.read(name)


def _read_csv(data: bytes) -> list[dict[str,str]]:
    reader=csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if not reader.fieldnames:
        raise ValueError("Expected named evidence columns")
    rows=list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError("Missing or malformed evidence CSV")
    return rows


def verify_shadow_archive(path: Path, audited: list[dict], eligible: list[dict]):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(".." in PurePosixPath(n).parts for n in names):
            raise ValueError("Unsafe shadow archive")
        checks=zip_bytes(z,"SHA256SUMS.txt").decode()
        for line in checks.splitlines():
            expected,name=line.split("  ",1)
            if digest(zip_bytes(z,name))!=expected:
                raise ValueError("Shadow reviewer source checksum mismatch: "+name)
        summary=json.loads(zip_bytes(z,"review_summary.json"))
        rows=_read_csv(zip_bytes(z,"all_191_unmatched_observations.csv"))
        selected=_read_csv(zip_bytes(z,"human_review_queue_7.csv"))
        if (len(rows)!=191 or len(selected)!=7 or len(audited)!=191
            or summary.get("unmatched_rare_class_observations")!=191
            or summary.get("eligible_human_review_ids")!=7
            or summary.get("primary_201_frame_track_unchanged") is not True):
            raise ValueError("W04 191/7 source cohort changed")
        keys={(int(r["frame"]),r["vehicle_class"],int(r["hybrid_tracker_id"]))
              for r in rows}
        actual={(r["frame"],r["vehicle_class"],r["hybrid_id"]) for r in audited}
        if len(keys)!=191 or keys!=actual:
            raise ValueError("Shadow source ZIP no longer represents audited raw tracks")
        candidate_keys={(r["vehicle_class"],int(r["hybrid_tracker_id"]))
                        for r in selected}
        verified_keys={(r["vehicle_class"],r["hybrid_id"])
                       for r in eligible if r["eligible_for_human_review"]}
        if len(candidate_keys)!=7 or candidate_keys!=verified_keys:
            raise ValueError("Human shadow-review candidate source drift")
        return {"archive_sha256":digest(path.read_bytes()),
                "191_row_manifest_verified":True,
                "seven_review_queue_keys_verified":True}


def verify_consensus(path: Path, actual_cases: set[tuple[str,int]]):
    """Provenance check only. Do not read these votes into geometry decisions."""
    with zipfile.ZipFile(path) as z:
        entries=z.namelist()
        if len(entries)!=len(set(entries)) or any(".." in PurePosixPath(n).parts for n in entries):
            raise ValueError("Unsafe completed human-review archive")
        for line in zip_bytes(z,"SHA256SUMS.txt").decode().splitlines():
            expected,name=line.split("  ",1)
            if digest(zip_bytes(z,name))!=expected:
                raise ValueError("Completed review audit checksum drift")
        summary=json.loads(zip_bytes(z,"consensus_summary.json"))
        votes=_read_csv(zip_bytes(z,"case_level_consensus.csv"))
        if (summary.get("case_count")!=7 or summary.get("exact_three_field_agreements")!=7
            or summary.get("already_tracked_in_both")!=7
            or summary.get("untracked_in_both")!=0
            or summary.get("independence_of_reviewers_verified") is not False
            or summary.get("eligible_for_production") is not False):
            raise ValueError("Not the frozen seven-case recorded W04 consensus")
        mapped={(r["shadow_candidate_class_internal"],int(r["shadow_track_id_internal"]))
                for r in votes}
        if len(votes)!=7 or len(mapped)!=7 or mapped!=actual_cases:
            raise ValueError("Reconstructed case association differs from shadow cohort")
        for r in votes:
            if (r["reviewer_01_primary_relation"]!="ALREADY_TRACKED"
                or r["reviewer_02_primary_relation"]!="ALREADY_TRACKED"
                or r["new_physical_vehicle_verified"].lower()!="false"):
                raise ValueError("Unexpected original human review vote")
        return {"audit_zip_sha256":digest(path.read_bytes()),
                "recorded_review_agreement":7,
                "reviewer_independence_authenticated":False,
                "original_internal_case_manifest_available":False}


def generate(original_zip: Path, hybrid_zip: Path, shadow_zip: Path,
             consensus_zip: Path) -> tuple[dict,list[dict],list[dict],list[dict],list[dict]]:
    # The previous W04 unified scorer/sha audit independently checks 25 source
    # tracks, all 4 hybrid/sidecar hashes and the exact primary control bytes.
    source, unmatched, candidates=analyze(original_zip,hybrid_zip)
    if (source.get("primary_track_sha256")!=PRIMARY_SHA
        or source.get("unmatched_shadow_observations")!=191
        or source.get("eligible_review_tracklets")!=7):
        raise ValueError("W04 original primary control/cohort mismatch")
    eligible=[x for x in candidates if x["eligible_for_human_review"]]
    archive=verify_shadow_archive(shadow_zip,unmatched,eligible)
    reviewed_keys={(x["vehicle_class"],x["hybrid_id"]) for x in eligible}
    votes=verify_consensus(consensus_zip,reviewed_keys)
    with zipfile.ZipFile(hybrid_zip) as zip_source:
        report=json.loads(zip_bytes(zip_source,"unified_hybrid_report.json"))
        control=next(x for x in report["experimental_results"] if x["name"]=="control_ios030")
        control_bytes=zip_bytes(zip_source,control["tracks"])
        if digest(control_bytes)!=PRIMARY_SHA:
            raise ValueError("Primary track source changed after full original verification")
        primary_tracks=parse_tracks(control_bytes,label="frozen primary",
                                    expected_rows=control["evaluation_confirmed_rows"])
    by_frame={}
    for frame,existing in primary_tracks.items():
        by_frame[frame]=[{"frame":frame,"primary_id":r["id"],
                          "vehicle_class":r["class"],
                          "x1":r["x1"],"y1":r["y1"],"x2":r["x2"],"y2":r["y2"]}
                         for r in existing]
    shadow=[{"frame":r["frame"],"shadow_id":r["hybrid_id"],
             "vehicle_class":r["vehicle_class"],
             "x1":r["x1"],"y1":r["y1"],"x2":r["x2"],"y2":r["y2"]}
            for r in unmatched]
    obs,pairs,tracklets=compare_shadow_to_primary(shadow,by_frame)
    # Human votes are joined POST-HOC to measure how strongly the geometry
    # *independently* explains the reviewed decisions. No tuning/threshold update.
    reviewed=[]
    for t in tracklets:
        key=(t["vehicle_class"],t["shadow_id"])
        if key in reviewed_keys:
            reviewed.append({
                **t,"reviewer_01_primary_relation":"ALREADY_TRACKED",
                "reviewer_02_primary_relation":"ALREADY_TRACKED",
                "consensus_case_mapping_reconstructed_not_signed":True,
                "geometry_is_independent_of_human_consensus":True})
    if len(reviewed)!=7 or len(obs)!=191:
        raise ValueError("Geometry reviewed-cohort mismatch")
    status_count=Counter(t["repeated_correspondence_status"] for t in tracklets)
    result={
        "status":STATUS,"eligible_for_production":False,
        "original_primary_byte_identical":True,"primary_track_sha256":PRIMARY_SHA,
        "unified_original_sha256":source["source_hybrid_zip_sha256"],
        "shadow_archive":archive,"human_recorded_consensus":votes,
        "evaluated_zero_indexed_frames":[START,END],
        "geometric_rule":{"same_extent_iou_min":IOU_SIMILAR,
                          "part_whole_ios_min":IOS_CONTAINED,
                          "part_whole_area_ratio_max":MAX_PART_AREA_RATIO,
                          "pair_temporal_frames_min":MIN_TEMPORAL_SUPPORT_FRAMES,
                          "primary_id_observation_fraction_min":MIN_OBSERVATION_COVERAGE},
        "shadow_observation_count":len(obs),
        "all_shadow_tracklet_count":len(tracklets),
        "reviewed_candidate_count":len(reviewed),
        "all_observations_with_strong_primary_geometry":sum(
            x["plausible_primary_count"]>0 for x in obs),
        "all_tracklet_evidence_status_counts":dict(status_count),
        "reviewed_tracklet_evidence_status_counts":dict(
            Counter(x["repeated_correspondence_status"] for x in reviewed)),
        "new_physical_vehicles_inferred":0,
        "primary_track_edits":0,"FLUID_ground_truth_edits":0,
        "no_automatic_duplicate_deletion":True,
        "limitations":(
            "IoU/IoS and temporal correspondence are uncalibrated, "
            "class-agnostic hypotheses; close/occluded vehicles may overlap. "
            "Seven W04 reviews were already-known results, and no reviewer-negative "
            "holdout exists: this is a diagnostic replay, not sensitivity or "
            "specificity validation. Case mapping reconstructed, procedural reviewer "
            "independence not authenticated. No unseen physical truth, CPU p95 "
            "or T000 source-aligned baseline."
        )
    }
    return result,obs,pairs,tracklets,reviewed


def write_csv(path: Path, entries: list[dict]):
    if not entries:
        raise ValueError("Cannot publish an empty W04 evidence table")
    with path.open("x",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-batch-zip",type=Path,required=True)
    ap.add_argument("--unified-replay-zip",type=Path,required=True)
    ap.add_argument("--shadow-review-zip",type=Path,required=True)
    ap.add_argument("--recorded-consensus-zip",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args(argv)
    if args.output_dir.exists():
        ap.error("Immutable correspondence output already exists")
    report,obs,pairs,tracklets,reviewed=generate(
        args.original_batch_zip,args.unified_replay_zip,
        args.shadow_review_zip,args.recorded_consensus_zip)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+args.output_dir.name+".stage-",
                                dir=args.output_dir.parent))
    try:
        write_csv(stage/"all_191_observation_correspondence.csv",obs)
        write_csv(stage/"all_plausible_primary_box_pairs.csv",pairs)
        write_csv(stage/"all_40_tracklet_correspondence.csv",tracklets)
        write_csv(stage/"seven_reviewed_cases_geometry_vs_votes.csv",reviewed)
        (stage/"correspondence_summary.json").write_text(
            json.dumps(report,indent=2),encoding="utf-8")
        with (stage/"SHA256SUMS.txt").open("x",encoding="utf-8") as f:
            for file in sorted(stage.iterdir()):
                if file.name!="SHA256SUMS.txt":
                    f.write(digest(file.read_bytes())+"  "+file.name+"\n")
        if args.output_dir.exists():
            raise FileExistsError("Cannot overwrite an existing W04 audit")
        os.replace(stage,args.output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps({
        "status":STATUS,"all_observations":len(obs),
        "reviewed_case_count":len(reviewed),
        "geometry_status_counts":report["reviewed_tracklet_evidence_status_counts"],
        "output":str(args.output_dir),"eligible_for_production":False
    },indent=2))


if __name__=="__main__":
    main()
