"""Fail-closed independent R01/R02 agreement for W04 rare vehicle review.

Does not edit FLUID truth, primary ByteTrack tracks, candidate class, or
production counts. Unclear or disagreement cases require human adjudication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import os
import shutil
import tempfile

VISIBLE={"YES","NO","UNCLEAR"}
CLASSES={"CAR","BUS","HEAVY_VEHICLE","MOTORCYCLE",
         "AUTO_RICKSHAW","BICYCLE","PEDESTRIAN","OTHER","UNKNOWN","NA"}
RELATIONS={"UNTRACKED","ALREADY_TRACKED","DUPLICATE_SHADOW","UNCLEAR","NA"}
HEADERS=("case_id","reviewer_id","vehicle_visible",
         "physical_class","primary_relation","notes")
STATUS="W04_TWO_INDEPENDENT_HUMAN_REVIEWS_NOT_PHYSICAL_TRUTH_PROMOTION"


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read_review(path: Path, expected: set[str], reviewer: str):
    """Exactly one canonical vote per case, no omission, extras or duplicate ID."""
    if not path.is_file():
        raise FileNotFoundError(f"Review CSV absent: {path}")
    votes={}
    with path.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=list(HEADERS):
            raise ValueError("Review CSV headers differ from frozen schema")
        for record in reader:
            key=record["case_id"]
            if (key not in expected or key in votes
                or record["reviewer_id"]!=reviewer):
                raise ValueError("Missing, duplicate, swapped or unknown reviewer/case ID")
            visible=record["vehicle_visible"]
            cls=record["physical_class"]
            relation=record["primary_relation"]
            notes=record["notes"]
            if (visible not in VISIBLE or cls not in CLASSES or
                relation not in RELATIONS or notes is None or
                len(notes)>2500):
                raise ValueError(f"Incomplete/invalid human review fields for {key}")
            if visible=="NO" and (cls!="NA" or relation!="NA"):
                raise ValueError("No visible object requires NA class/relation")
            if visible=="YES" and (cls=="NA" or relation=="NA"):
                raise ValueError("Visible object requires physical class/relation")
            votes[key]=record
    if set(votes)!=expected:
        raise ValueError("Reviewer submission does not contain every case")
    return votes


def evaluate_case(a: dict,b: dict) -> tuple[str,bool]:
    """Cautious categories, never claim independent ground-truth additions."""
    keys=("vehicle_visible","physical_class","primary_relation")
    if any(a[key]!=b[key] for key in keys):
        return "DISAGREEMENT_REQUIRES_ADJUDICATION",True
    if a["vehicle_visible"]=="UNCLEAR":
        return "BOTH_UNCLEAR_REQUIRES_ADJUDICATION",True
    if a["vehicle_visible"]=="NO":
        return "CONSENSUS_OBJECT_NOT_VISIBLE",False
    if (a["physical_class"]=="UNKNOWN" or
        a["primary_relation"]=="UNCLEAR"):
        return "CONSENSUS_UNCERTAIN_REQUIRES_ADJUDICATION",True
    if a["primary_relation"]=="UNTRACKED":
        return "CONSENSUS_POSSIBLE_UNTRACKED_PHYSICAL_OBJECT",False
    if a["primary_relation"]=="ALREADY_TRACKED":
        return "CONSENSUS_ALREADY_IN_PRIMARY",False
    return "CONSENSUS_DUPLICATE_SHADOW",False


def run(packet_dir: Path, reviewer_01: Path, reviewer_02: Path,
        output_dir: Path):
    if output_dir.exists():
        raise FileExistsError("Immutable two-reviewer consensus output exists")
    manifest=packet_dir/"INTERNAL_case_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Original internal case manifest missing")
    obj=json.loads(manifest.read_text(encoding="utf-8"))
    if (obj.get("status")!="W04_BLIND_TWO_REVIEWER_VISUAL_PACKET_NOT_PRODUCTION"
        or obj.get("eligible_for_production") is not False
        or obj.get("reviewers_required")!=["R01","R02"]
        or obj.get("case_count")!=7
        or obj.get("no_auto_FLUID_changes") is not True):
        raise ValueError("Unrecognized research-only seven-case W04 manifest")
    case_map=obj["case_mapping"]
    if len(case_map)!=7:
        raise ValueError("Incomplete W04 reviewer case manifest")
    by_id={x["case_id"]:x for x in case_map}
    if len(by_id)!=7 or set(by_id)!={"S%02d"%i for i in range(1,8)}:
        raise ValueError("Case IDs or identities changed")
    r1=read_review(reviewer_01,set(by_id),"R01")
    r2=read_review(reviewer_02,set(by_id),"R02")
    rows=[]
    needs=[]
    statuses={}
    for case_id in sorted(by_id):
        first,second=r1[case_id],r2[case_id]
        label,requires_adjudication=evaluate_case(first,second)
        statuses[label]=statuses.get(label,0)+1
        base=by_id[case_id]
        record={
            "case_id":case_id,
            "candidate_class_internal_only":base["candidate_class_internal_only"],
            "hybrid_track_id_internal_only":base["hybrid_track_id_internal_only"],
            "reviewer_01_visible":first["vehicle_visible"],
            "reviewer_02_visible":second["vehicle_visible"],
            "reviewer_01_class":first["physical_class"],
            "reviewer_02_class":second["physical_class"],
            "reviewer_01_relation":first["primary_relation"],
            "reviewer_02_relation":second["primary_relation"],
            "consensus_status":label,
            "requires_separate_adjudication":requires_adjudication,
            "physical_vehicle_confirmed_for_production":False,
        }
        rows.append(record)
        if requires_adjudication:
            needs.append(record)
    summary={
        "status":STATUS,"eligible_for_production":False,
        "source_case_manifest_sha256":digest(manifest),
        "reviewer_01_csv_sha256":digest(reviewer_01),
        "reviewer_02_csv_sha256":digest(reviewer_02),
        "reviewers":2,"case_count":7,
        "case_status_counts":statuses,
        "independent_exact_agreement_cases":sum(
            not x["consensus_status"].startswith("DISAGREEMENT")
            for x in rows),
        "adjudication_required_count":len(needs),
        "consensus_possible_untracked_count":statuses.get(
            "CONSENSUS_POSSIBLE_UNTRACKED_PHYSICAL_OBJECT",0),
        "flUID_truth_modified":False,
        "primary_tracks_modified":False,
        "physical_vehicle_automatically_promoted":False,
        "limit":"Consensus indicates reviewer agreement, not independent ground-truth certification. "
                "Never modify original FLUID or production counts from this output."
    }
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+output_dir.name+".stage-",dir=output_dir.parent))
    try:
        for filename,records in (
            ("case_level_consensus.csv",rows),
            ("requires_adjudication.csv",needs)):
            fields=list(rows[0])
            with (stage/filename).open("x",encoding="utf-8",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
        (stage/"consensus_summary.json").write_text(
            json.dumps(summary,indent=2),encoding="utf-8")
        with (stage/"SHA256SUMS.txt").open("x",encoding="utf-8") as f:
            for file in sorted(stage.iterdir()):
                if file.name!="SHA256SUMS.txt":
                    f.write(digest(file)+"  "+file.name+"\n")
        if output_dir.exists():
            raise FileExistsError("Consensus destination appeared while writing")
        os.replace(stage,output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir",type=Path,required=True)
    parser.add_argument("--reviewer-01",type=Path,required=True)
    parser.add_argument("--reviewer-02",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(args.packet_dir,args.reviewer_01,
                         args.reviewer_02,args.output_dir),indent=2))


if __name__=="__main__":
    main()
