"""Blind physical-object adjudication for the historical May-25 review cohort.

Unlike the engineering contact sheet this makes NEW source-video crops with just
a neutral target circle and randomized sample codes. Never expose the sealed
FLUID/model metadata to reviewers. No model inference or retuning is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from .w04_holdout_intake import digest, read_json
from .w04_visual_review_packet import (
    PACKET_STATUS, ReviewError, check_sources, make_plan,
)

STATUS = "W04_MAY25_PHYSICAL_BLIND_REVIEW_PREPARED_NOT_ADJUDICATED"
ADJ_STATUS = "W04_MAY25_TWO_REVIEWER_SELECTED_SAMPLE_DIAGNOSTIC_NOT_BLIND_HOLDOUT"
CLASSES = (
    "passenger_car", "light_commercial_vehicle", "heavy_goods_vehicle",
    "passenger_bus", "two_wheeler", "three_wheeler", "bicycle",
    "pedestrian", "fixed_object", "not_visible", "uncertain",
)
REVIEW_FIELDS = [
    "sample_code", "reviewer_code", "physical_object_type", "marker_on_object",
    "visibility", "confidence", "reviewed_without_FLUID_or_model_labels",
    "notes",
]
VISIBILITY = {"clear", "partially_occluded", "heavily_occluded", "uncertain"}
CONFIDENCE = {"high", "medium", "low"}


def _match_existing_packet(packet_dir: Path, samples: list[dict], sources: dict) -> None:
    packet = read_json(packet_dir / "review_packet_manifest.json")
    if (packet.get("status") != PACKET_STATUS or
        packet.get("independent_holdout_pass") is not False or
        packet.get("production_release_approved") is not False or
        packet.get("source_evidence", {}).get("original_video_sha256") != sources["original_video_sha256"] or
        packet.get("source_evidence", {}).get("forensics_sha256") != sources["forensics_sha256"] or
        packet.get("selected_sample_count") != len(samples)):
        raise ReviewError("Labeled engineering packet no longer matches frozen May25 source")
    frozen = packet.get("samples", [])
    if not isinstance(frozen, list) or len(frozen) != len(samples):
        raise ReviewError("Original visual packet sample coverage has changed")
    for original, sample in zip(frozen, samples):
        for name in ("sample_id", "event", "video_frame", "truth_track_id",
                     "predicted_track_id", "truth_cx", "truth_cy",
                     "predicted_cx", "predicted_cy"):
            if original.get(name) != sample.get(name):
                raise ReviewError(f"Visual packet sample correspondence changed at {name}")
        src = packet_dir / original["image_file"]
        if src.is_symlink() or not src.is_file() or digest(src) != original.get("image_sha256"):
            raise ReviewError("Existing engineering-review image hash changed")


def _blind_tile(rgb: Any, sample: dict, frame_size: tuple[int, int], code: str):
    """Visual target only; no model/FLUID label, class, track ID, frame or event."""
    from PIL import Image, ImageDraw, ImageFont
    width, height = frame_size
    if rgb is None or tuple(rgb.shape[:2]) != (height, width):
        raise ReviewError("Original decoded frame dimensions disagree with prior preflight")
    centers = [(sample[k], sample[k.replace("cx", "cy")])
               for k in ("truth_cx", "predicted_cx") if sample[k] is not None]
    if not centers:
        raise ReviewError("Review sample has no original pixel anchor")
    x = sum(p[0] for p in centers)/len(centers)
    y = sum(p[1] for p in centers)/len(centers)
    if not 0 <= x < width or not 0 <= y < height:
        raise ReviewError("Original source target falls outside decoded frame")
    crop_w, crop_h = min(850, width), min(575, height)
    left = min(max(int(round(x-crop_w/2)), 0), width-crop_w)
    top = min(max(int(round(y-crop_h/2)), 0), height-crop_h)
    original = Image.fromarray(rgb[top:top+crop_h,left:left+crop_w], mode="RGB")
    display_w, display_h = 500, 338
    scaled = original.resize((display_w,display_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (540, 400), (248,248,248))
    canvas.paste(scaled, (20, 47))
    pen = ImageDraw.Draw(canvas)
    pen.text((20, 12), f'{code}  |  Identify the object inside the ring',
             fill=(20,20,20), font=ImageFont.load_default())
    px = 20 + (x-left)*display_w/crop_w
    py = 47 + (y-top)*display_h/crop_h
    pen.ellipse((px-23,py-23,px+23,py+23), outline=(16,16,16), width=6)
    pen.ellipse((px-23,py-23,px+23,py+23), outline=(255,255,255), width=3)
    return canvas


def render_blind(*, samples: list[dict], frames: Iterable[tuple[int, Any]],
                 frame_size: tuple[int, int], output: Path,
                 rng: random.Random | None = None) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite previous blind review")
    if len(samples) < 1:
        raise ReviewError("No source-verified review samples")
    if rng is None:
        rng = random.SystemRandom()
    random_order = list(range(len(samples)))
    rng.shuffle(random_order)
    codes = {original: f"B{slot+1:03d}" for slot, original in enumerate(random_order)}
    groups: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for i, item in enumerate(samples):
        groups[item["video_frame"]].append((codes[i], item))
    output.mkdir(parents=True, exist_ok=False)
    for n in (1, 2):
        (output / f"reviewer_{n}").mkdir()
        (output / f"reviewer_{n}" / "images").mkdir()
    seen = set()
    for index, pixels in frames:
        if index not in groups:
            continue
        if index in seen:
            raise ReviewError("Duplicate source-frame decode in blind packet")
        seen.add(index)
        for code, sample in groups[index]:
            image = _blind_tile(pixels, sample, frame_size, code)
            for n in (1, 2):
                image.save(output / f"reviewer_{n}" / "images" / f"{code}.jpg",
                           format="JPEG", quality=92)
    if seen != set(groups):
        raise ReviewError(f"Missing source frames needed for blind review: {sorted(set(groups)-seen)}")
    public = sorted(codes.values())
    from PIL import Image, ImageDraw, ImageFont
    for n in (1,2):
        folder = output / f"reviewer_{n}"
        cols = 3
        rows = math.ceil(len(public)/cols)
        sheet = Image.new("RGB", (cols*540, rows*400+42), (230,232,237))
        pen = ImageDraw.Draw(sheet)
        pen.text((15,12), "OBJECT-ONLY REVIEW | Categories intentionally concealed",
                 fill=(10,10,10), font=ImageFont.load_default())
        for i, code in enumerate(public):
            with Image.open(folder/"images"/f"{code}.jpg") as tile:
                sheet.paste(tile,(i%cols*540,42+i//cols*400))
        sheet.save(folder/"blind_contact_sheet.jpg",quality=90,format="JPEG")
        with (folder/"independent_reviewer_blank.csv").open("x", newline="",
                                                            encoding="utf-8") as stream:
            w = csv.DictWriter(stream,fieldnames=REVIEW_FIELDS)
            w.writeheader()
            for code in public:
                w.writerow({"sample_code": code})
        (folder/"INSTRUCTIONS.txt").write_text(
            "Independently identify the ONE object inside the monochrome ring. "
            "Do not view earlier labeled contact sheets, project predictions, "
            "other reviewers' answers, or sealed_reference.json. "
            "Fill every row: reviewer_code (distinct for each person), "
            "physical_object_type from: "+", ".join(CLASSES)+". "
            "marker_on_object=yes/no/uncertain; visibility=clear/"
            "partially_occluded/heavily_occluded/uncertain; "
            "confidence=high/medium/low; "
            "reviewed_without_FLUID_or_model_labels=yes only if genuinely true. "
            "Keep the original MP4, FLUID and tracking files unchanged. "
            "This sample was selected for suspected error and is NOT a random cohort.\n",
            encoding="utf-8")
    sealed = []
    for i, sample in enumerate(samples):
        sealed.append({
            "sample_code": codes[i],
            "event": sample["event"],
            "video_frame": sample["video_frame"],
            "truth_track_id": sample["truth_track_id"],
            "predicted_track_id": sample["predicted_track_id"],
            "truth_class": sample["truth_class"],
            "predicted_class": sample["predicted_class"],
            "original_sample_id": sample["sample_id"],
        })
    sealed.sort(key=lambda row: row["sample_code"])
    return {"codes":public, "sealed_mapping":sealed, "sample_count":len(public)}


def prepare(*, run_dir: Path, forensics_dir: Path, old_packet_dir: Path,
            output: Path, rng=None) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing blind packet: {output}")
    video, sources, events = check_sources(run_dir=run_dir,forensics_dir=forensics_dir)
    old_manifest = read_json(old_packet_dir/"review_packet_manifest.json")
    count = old_manifest.get("selected_sample_count")
    if type(count) is not int:
        raise ReviewError("Prior review sample count missing")
    samples = make_plan(events,max_samples=count)
    _match_existing_packet(old_packet_dir,samples,sources)
    from streetlab_phase3.video.continuous_box_tracking_lab import sequential_video_rgb_frames
    native_size = tuple(sources["source_video_dimensions"])
    report = render_blind(
        samples=samples,
        frames=sequential_video_rgb_frames(video, min(p["video_frame"] for p in samples),
                         max(p["video_frame"] for p in samples),
                         expected_dimensions=native_size),
        frame_size=native_size, output=output, rng=rng,
    )
    reference = {
        "status":"SEALED_ORIGINAL_MAY25_LABELS_NOT_FOR_REVIEWER_DISTRIBUTION",
        "original_source_sha256":sources["original_video_sha256"],
        "original_fluid_sha256":sources["original_truth_sha256"],
        "forensics_sha256":sources["forensics_sha256"],
        "prior_packet_sha256":digest(old_packet_dir/"review_packet_manifest.json"),
        "sample_mapping":report["sealed_mapping"],
        "independent_holdout_pass":False, "production_release_approved":False,
    }
    with (output/"sealed_reference.json").open("x",encoding="utf-8") as stream:
        json.dump(reference,stream,indent=2)
        stream.write("\n")
    receipt = {
        "schema_version":1, "status":STATUS,
        "prepared_utc":datetime.now(timezone.utc).isoformat(),
        "original_video_sha256":sources["original_video_sha256"],
        "source_bound_forensics_sha256":sources["forensics_sha256"],
        "sealed_reference_sha256":digest(output/"sealed_reference.json"),
        "sample_count":report["sample_count"],
        "blinded_visual_classes_hidden":True,
        "reviewers_exposed_to_previous_labeled_contact_sheet_are_not_blinded":True,
        "reviewer_decisions_received":False,
        "two_independent_reviewers_verified":False,
        "independent_holdout_pass":False,
        "production_release_approved":False,
        "deliver_only_subfolders":["reviewer_1","reviewer_2"],
    }
    with (output/"blind_review_receipt.json").open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2)
        stream.write("\n")
    return receipt


def _read_reviewer(file:Path,codes:list[str]) -> tuple[str,dict[str,dict]]:
    if file.is_symlink() or not file.is_file():
        raise ReviewError(f"Independent review form missing or linked: {file}")
    with file.open("r",newline="",encoding="utf-8-sig") as stream:
        r=csv.DictReader(stream)
        if r.fieldnames != REVIEW_FIELDS:
            raise ReviewError("Reviewer form columns do not match the locked blank template")
        rows=list(r)
    if len(rows)!=len(codes) or len({r["sample_code"] for r in rows})!=len(codes):
        raise ReviewError("All original samples must have exactly one reviewer answer")
    mapping={r["sample_code"]:r for r in rows}
    if set(mapping)!=set(codes):
        raise ReviewError("Reviewer source sample IDs changed")
    ids=set()
    for r in rows:
        if r["physical_object_type"] not in CLASSES:
            raise ReviewError("Reviewer physical class absent/invalid")
        if r["marker_on_object"] not in {"yes","no","uncertain"}:
            raise ReviewError("Reviewer target attribution missing")
        if r["visibility"] not in VISIBILITY or r["confidence"] not in CONFIDENCE:
            raise ReviewError("Reviewer visibility/confidence is invalid")
        if r["reviewed_without_FLUID_or_model_labels"] != "yes":
            raise ReviewError("Blinded review cannot be attested by this reviewer")
        if not r["reviewer_code"] or len(r["reviewer_code"]) > 64:
            raise ReviewError("Reviewer_code must identify a person")
        ids.add(r["reviewer_code"])
    if len(ids)!=1:
        raise ReviewError("One form must represent one declared reviewer")
    return next(iter(ids)),mapping


def summarize(*, packet_dir: Path, reviewer1: Path, reviewer2: Path) -> dict:
    receipt=read_json(packet_dir/"blind_review_receipt.json")
    sealed=read_json(packet_dir/"sealed_reference.json")
    if (receipt.get("status")!=STATUS or
        receipt.get("sealed_reference_sha256")!=digest(packet_dir/"sealed_reference.json") or
        sealed.get("status")!="SEALED_ORIGINAL_MAY25_LABELS_NOT_FOR_REVIEWER_DISTRIBUTION" or
        receipt.get("independent_holdout_pass") is not False):
        raise ReviewError("Sealed blind packet is inconsistent or promoted")
    mapping=sealed["sample_mapping"]
    codes=[x["sample_code"] for x in mapping]
    if len(codes)!=receipt["sample_count"] or len(set(codes))!=len(codes):
        raise ReviewError("Sealed sample IDs were altered")
    id1,one=_read_reviewer(reviewer1,codes)
    id2,two=_read_reviewer(reviewer2,codes)
    if id1==id2:
        raise ReviewError("Reviewer identities must be distinct")
    outcomes=[]
    for row in mapping:
        code=row["sample_code"]
        a,b=one[code],two[code]
        agreed=(a["physical_object_type"]==b["physical_object_type"] and
                a["marker_on_object"]==b["marker_on_object"])
        decisive=(agreed and a["marker_on_object"]=="yes" and
                  a["physical_object_type"] not in {"uncertain","not_visible"} and
                  a["confidence"]!="low" and b["confidence"]!="low")
        outcomes.append({
            "sample_code":code,"original_sample_id":row["original_sample_id"],
            "event":row["event"],"truth_track_id":row["truth_track_id"],
            "predicted_track_id":row["predicted_track_id"],
            "FLUID_label":row["truth_class"],"model_label":row["predicted_class"],
            "reviewer_1_type":a["physical_object_type"],
            "reviewer_2_type":b["physical_object_type"],
            "reviewer_type_and_target_agree":agreed,
            "agreed_visible_type":a["physical_object_type"] if decisive else None,
            "candidate_diagnostic_only_not_relabeling":True,
        })
    by_group=defaultdict(list)
    for row in outcomes:
        key=(row["event"],str(row["truth_track_id"]),str(row["predicted_track_id"]))
        by_group[key].append(row)
    groups=[]
    for key,observations in sorted(by_group.items()):
        labels={x["agreed_visible_type"] for x in observations if x["agreed_visible_type"]}
        groups.append({
            "event":key[0],"truth_track_id":key[1],"predicted_track_id":key[2],
            "reviewed_frames":len(observations),
            "decisive_agreed_labels":sorted(labels),
            "physical_track_type_consistent_across_decisive_frames":len(labels)==1 and bool(labels),
            "track_level_label_candidate":next(iter(labels)) if len(labels)==1 else None,
            "cannot_infer_census_level_accuracy_from_biased_selected_samples":True,
        })
    result={
        "schema_version":1, "status":ADJ_STATUS,
        "original_review_packet_sha256":digest(packet_dir/"blind_review_receipt.json"),
        "reviewer_1_file_sha256":digest(reviewer1),
        "reviewer_2_file_sha256":digest(reviewer2),
        "reviewer_identifiers_distinct_and_blindness_self_attested":True,
        "actual_reviewer_independence_externally_verified":False,
        "physical_labels_independently_confirmed_by_external_adjudicator":False,
        "sample_count":len(outcomes),
        "agreed_class_and_marker_count":sum(x["reviewer_type_and_target_agree"] for x in outcomes),
        "decisive_agreed_visible_type_count":sum(x["agreed_visible_type"] is not None for x in outcomes),
        "track_groups":groups,"selected_samples":outcomes,
        "original_FLUID_labels_modified":False,
        "model_retrained_or_retuned":False,
        "regression_F1_recomputed_or_promoted":False,
        "independent_holdout_pass":False,"production_release_approved":False,
    }
    return result


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description="Prepare blind May25 object review or compare completed reviewers")
    sub=p.add_subparsers(dest="mode",required=True)
    prep=sub.add_parser("prepare")
    for name in ("run-dir","forensics-dir","old-packet-dir","output-dir"):
        prep.add_argument("--"+name,required=True,type=Path)
    review=sub.add_parser("summarize")
    for name in ("packet-dir","reviewer-1","reviewer-2","output"):
        review.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args(argv)
    if args.mode=="prepare":
        result=prepare(run_dir=args.run_dir,forensics_dir=args.forensics_dir,
                       old_packet_dir=args.old_packet_dir,output=args.output_dir)
        print(json.dumps({"status":result["status"],"samples":result["sample_count"],
                          "reviewer_1":str(args.output_dir/"reviewer_1"),
                          "reviewer_2":str(args.output_dir/"reviewer_2"),
                          "independent_holdout_pass":False},indent=2))
    else:
        if args.output.exists():
            raise FileExistsError("Refusing overwrite of previous adjudication")
        result=summarize(packet_dir=args.packet_dir,reviewer1=args.reviewer_1,
                         reviewer2=args.reviewer_2)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            json.dump(result,stream,indent=2)
            stream.write("\n")
        print(json.dumps({"status":result["status"],
                          "decisive_selected_samples":result["decisive_agreed_visible_type_count"],
                          "independent_holdout_pass":False,"output":str(args.output)},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
