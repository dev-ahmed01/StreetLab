"""Synthetic-only privacy/consensus tests for the May25 label-blind review layer."""
from __future__ import annotations

import csv
import json
import random

import numpy as np
from PIL import Image
import pytest

from streetlab_integration import w04_blind_physical_review as blind
from streetlab_integration.w04_holdout_intake import digest


def samples():
    return [
        {
            "sample_id": "R001", "event": "HEAVY_WRONG_CLASS",
            "video_frame": 60, "truth_track_id": "36", "predicted_track_id": "29",
            "truth_class": "HEAVY_VEHICLE", "predicted_class": "CAR",
            "truth_cx": 68., "truth_cy": 55.,
            "predicted_cx": 69., "predicted_cy": 55.,
        },
        {
            "sample_id": "R002", "event": "HEAVY_WRONG_CLASS",
            "video_frame": 144, "truth_track_id": "36", "predicted_track_id": "29",
            "truth_class": "HEAVY_VEHICLE", "predicted_class": "CAR",
            "truth_cx": 70., "truth_cy": 55.,
            "predicted_cx": 71., "predicted_cy": 55.,
        },
        {
            "sample_id": "R003", "event": "BUS_PRED_UNMATCHED",
            "video_frame": 150, "truth_track_id": None, "predicted_track_id": "110",
            "truth_class": None, "predicted_class": "BUS",
            "truth_cx": None, "truth_cy": None,
            "predicted_cx": 55., "predicted_cy": 30.,
        },
    ]


def prepared(tmp_path):
    output=tmp_path/"blind"
    original=blind.render_blind(
        samples=samples(), rng=random.Random(24),
        frames=[(n,np.zeros((100,120,3),dtype=np.uint8)) for n in (60,144,150)],
        frame_size=(120,100), output=output,
    )
    sealed={
        "status":"SEALED_ORIGINAL_MAY25_LABELS_NOT_FOR_REVIEWER_DISTRIBUTION",
        "sample_mapping":original["sealed_mapping"],
        "independent_holdout_pass":False,
    }
    (output/"sealed_reference.json").write_text(json.dumps(sealed))
    receipt={
        "status":blind.STATUS,
        "sample_count":3,
        "sealed_reference_sha256":digest(output/"sealed_reference.json"),
        "independent_holdout_pass":False,
    }
    (output/"blind_review_receipt.json").write_text(json.dumps(receipt))
    return output,original


def filled(template,output,reviewer,types,blind_attest="yes"):
    with template.open(newline="",encoding="utf-8") as fh:
        src=list(csv.DictReader(fh))
    for row in src:
        row.update({
            "reviewer_code":reviewer,
            "physical_object_type":types[row["sample_code"]],
            "marker_on_object":"yes",
            "visibility":"clear", "confidence":"high",
            "reviewed_without_FLUID_or_model_labels":blind_attest,
        })
    with output.open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=blind.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(src)
    return output


def test_blind_packets_only_show_neutral_object_targets_and_bare_sample_codes(tmp_path):
    packet,original=prepared(tmp_path)
    assert original["sample_count"]==3
    assert sorted(original["codes"])==["B001","B002","B003"]
    assert sorted(x["original_sample_id"] for x in original["sealed_mapping"])==[
        "R001","R002","R003",
    ]
    assert (packet/"reviewer_1"/"INSTRUCTIONS.txt").is_file()
    assert (packet/"reviewer_2"/"blind_contact_sheet.jpg").is_file()
    for n in (1,2):
        reviewer=packet/f"reviewer_{n}"
        with (reviewer/"independent_reviewer_blank.csv").open(newline="") as fp:
            answers=list(csv.DictReader(fp))
        assert [x["sample_code"] for x in answers]==["B001","B002","B003"]
        assert not any(x["physical_object_type"] for x in answers)
        assert "HEAVY_WRONG_CLASS" not in (reviewer/"INSTRUCTIONS.txt").read_text()
        assert not (reviewer/"sealed_reference.json").exists()
        for code in original["codes"]:
            with Image.open(reviewer/"images"/f"{code}.jpg") as crop:
                assert crop.size==(540,400)
    assert (packet/"sealed_reference.json").is_file()
    with pytest.raises(FileExistsError):
        blind.render_blind(samples=samples(),frames=[],frame_size=(120,100),
                           output=packet,rng=random.Random(24))


def test_two_completed_blinded_forms_yield_diagnostic_selected_track_consensus(tmp_path):
    packet,result=prepared(tmp_path)
    ids=result["codes"]
    classes={c:"light_commercial_vehicle" for c in ids}
    f1=filled(packet/"reviewer_1"/"independent_reviewer_blank.csv",
              tmp_path/"a.csv","independent_A",classes)
    f2=filled(packet/"reviewer_2"/"independent_reviewer_blank.csv",
              tmp_path/"b.csv","independent_B",classes)
    result=blind.summarize(packet_dir=packet,reviewer1=f1,reviewer2=f2)
    assert result["sample_count"]==3
    assert result["agreed_class_and_marker_count"]==3
    assert result["decisive_agreed_visible_type_count"]==3
    assert len(result["track_groups"])==2
    assert any(x["reviewed_frames"]==2 for x in result["track_groups"])
    assert result["actual_reviewer_independence_externally_verified"] is False
    assert result["physical_labels_independently_confirmed_by_external_adjudicator"] is False
    assert result["original_FLUID_labels_modified"] is False
    assert result["independent_holdout_pass"] is False


def test_disagreement_retained_not_auto_reconciled(tmp_path):
    packet,result=prepared(tmp_path)
    ids=result["codes"]
    a=filled(packet/"reviewer_1"/"independent_reviewer_blank.csv",
             tmp_path/"a.csv","A",{x:"passenger_car" for x in ids})
    b=filled(packet/"reviewer_2"/"independent_reviewer_blank.csv",
             tmp_path/"b.csv","B",{x:"heavy_goods_vehicle" for x in ids})
    output=blind.summarize(packet_dir=packet,reviewer1=a,reviewer2=b)
    assert output["decisive_agreed_visible_type_count"]==0
    assert all(g["track_level_label_candidate"] is None for g in output["track_groups"])
    assert output["independent_holdout_pass"] is False


@pytest.mark.parametrize("bad",["same_reviewer","no_blind_attestation","unknown_class","sample_id_swap","source_seal"])
def test_two_person_contract_fails_closed(tmp_path,bad):
    packet,result=prepared(tmp_path)
    ids=result["codes"]
    options={x:"passenger_car" for x in ids}
    f1=filled(packet/"reviewer_1"/"independent_reviewer_blank.csv",
              tmp_path/"a.csv","A",options)
    reviewer2="A" if bad=="same_reviewer" else "B"
    f2=filled(packet/"reviewer_2"/"independent_reviewer_blank.csv",
              tmp_path/"b.csv",reviewer2,options,
              blind_attest="no" if bad=="no_blind_attestation" else "yes")
    if bad in ("unknown_class","sample_id_swap"):
        with f2.open(newline="") as fh:
            rows=list(csv.DictReader(fh))
        if bad=="unknown_class":
            rows[0]["physical_object_type"]="invented_class"
        else:
            rows[0]["sample_code"]="B999"
        with f2.open("w",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=blind.REVIEW_FIELDS)
            w.writeheader();w.writerows(rows)
    if bad=="source_seal":
        r=json.loads((packet/"sealed_reference.json").read_text())
        r["sample_mapping"][0]["predicted_class"]="FAKE"
        (packet/"sealed_reference.json").write_text(json.dumps(r))
    with pytest.raises(blind.ReviewError):
        blind.summarize(packet_dir=packet,reviewer1=f1,reviewer2=f2)


def test_blind_packet_missing_decode_or_changed_native_size_fails(tmp_path):
    for tag,frames in (
        ("missing",[(60,np.zeros((100,120,3),dtype=np.uint8))]),
        ("bad_shape",[(60,np.zeros((90,120,3),dtype=np.uint8))]),
    ):
        with pytest.raises(blind.ReviewError):
            blind.render_blind(samples=samples(),frames=frames,frame_size=(120,100),
                               output=tmp_path/tag,rng=random.Random(21))
