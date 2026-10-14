"""Synthetic security, matching, duplicate and queue tests for read-only shadow lane."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from copy import deepcopy
import pytest

TOOL = Path(__file__).resolve().parents[1]/"scripts"/"phase3_w04_shadow_review_queue.py"
if not TOOL.exists():
    pytest.skip("Shadow-lane analysis omitted by sparse checkout",
                allow_module_level=True)
spec=importlib.util.spec_from_file_location("w04_shadow_queue", TOOL)
assert spec and spec.loader
shadow=importlib.util.module_from_spec(spec)
spec.loader.exec_module(shadow)


def obs(frame,tid,cls,x0=0.,score=.9):
    return {"frame":frame,"id":tid,"cls":cls,"class":shadow.CLASSES[cls],
            "x1":x0,"y1":0.,"x2":x0+10.,"y2":10.,"confidence":score}


def raw(frame,cls,x0=0.,score=.9):
    return {"frame":frame,"source_index":frame,"vehicle_class":cls,
            "x1":x0,"y1":0.,"x2":x0+10.,"y2":10.,"confidence":score}


def test_hungarian_maximum_cardinality_does_not_assume_novelty():
    a=[obs(10750,1,2,x0=0.),obs(10750,2,2,x0=5.)]
    b=[obs(10750,3,2,x0=0.),obs(10750,4,2,x0=5.),
       obs(10750,5,2,x0=90.)]
    assert shadow.choose_matches(a,b)=={0,1}
    assert shadow.choose_matches([],b)==set()


def test_three_frame_persistent_highconfidence_candidate_goes_to_review_only():
    trial={}
    raw_stream={}
    for frame in (10750,10751,10752):
        trial[frame]=[obs(frame,33,3,score=.72)]
        raw_stream[frame]=[raw(frame,"MOTORCYCLE")]
    before=deepcopy(trial)
    proposals,queue=shadow.build_queue({},trial,raw_stream)
    assert trial==before
    assert len(proposals)==3
    assert len(queue)==1
    assert queue[0]["eligible_for_human_review"] is True
    assert queue[0]["physical_vehicle_confirmed"] is False
    assert queue[0]["class_ambiguity_category"]=="lower_class_ambiguity"


def test_raw_class_ambiguity_flagged_even_without_primary_overlap():
    frame=10750
    trial={frame:[obs(frame,7,2)]}
    raw_stream={frame:[raw(frame,"HEAVY_VEHICLE"),
                       raw(frame,"CAR")]}
    proposals,queue=shadow.build_queue({},trial,raw_stream)
    assert proposals[0]["same_class_raw_supported"]
    assert proposals[0]["competing_raw_class_hypotheses"]
    assert proposals[0]["competing_raw_labels"]=="CAR"
    assert queue[0]["class_ambiguity_category"]=="needs_cross_class_adjudication"


def test_same_class_primary_overlapping_track_blocks_review():
    frame=10750
    original={frame:[obs(frame,1,2)]}
    trial={frame:[obs(frame,300,2)]}
    novel,queue=shadow.build_queue(original,trial,{frame:[]})
    assert novel==[] and queue==[]


def test_review_rule_does_not_convert_single_frame_confidence_to_true_vehicle():
    frame=10750
    novel,queue=shadow.build_queue({}, {frame:[obs(frame,1,3)]},
                                   {frame:[raw(frame,"MOTORCYCLE")]})
    assert len(novel)==1 and len(queue)==1
    assert not queue[0]["eligible_for_human_review"]


def test_strict_parser_rejects_duplicate_id_and_malformed_rows():
    row="10750,1,10,10,10,10,10,10,10,10,2,0.9,10,10\n"
    assert len(shadow.parse_tracks(row.encode(),label="test",expected_rows=1))==1
    with pytest.raises(ValueError,match="duplicate|invalid"):
        shadow.parse_tracks((row+row).encode(),label="test",expected_rows=2)
    with pytest.raises(ValueError,match="14-column"):
        shadow.parse_tracks(b"10750,1\n",label="test",expected_rows=1)


def test_existing_destination_refuses_to_run(tmp_path):
    (tmp_path/"out").mkdir()
    with pytest.raises(SystemExit):
        shadow.main(["--original-batch-zip",str(tmp_path/"missing.zip"),
                     "--unified-replay-zip",str(tmp_path/"missing2.zip"),
                     "--output-dir",str(tmp_path/"out")])


def test_raw_iou_and_consecutive_math():
    assert shadow.box_iou(obs(10750,1,2),obs(10750,2,2))==1.
    assert shadow.box_iou(obs(10750,1,2),obs(10750,2,2,x0=100.))==0.
    assert shadow.longest_streak([3,1,2,8,9])==3


def test_real_uploaded_frozen_archives_when_available():
    base=Path("/mnt/data")
    z1=base/"W04_CONTINUOUS_201_RESULTS.zip"
    z2=base/"W04_UNIFIED_HYBRID_REPLAY_01.zip"
    if not z1.exists() or not z2.exists():
        pytest.skip("Original W04 user evidence not bundled in CI")
    result,observations,queue=shadow.analyze(z1,z2)
    assert result["primary_tracks_immutable"] is True
    assert result["unmatched_shadow_observations"]==191
    assert result["eligible_review_tracklets"]==7
    assert len(observations)==191
    assert sum(x["eligible_for_human_review"] for x in queue)==7
    assert not any(x["physical_vehicle_confirmed"] for x in queue)
