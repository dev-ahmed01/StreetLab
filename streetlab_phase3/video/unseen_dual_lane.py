"""Unseen-video experimental dual-lane OpenVINO -> ByteTrack exporter.

ONE per-frame SAHI tiled detector pass feeds independently instantiated
primary IoS.30 and shadow rare-IoU.50 ByteTrack instances.
NO FLUID scoring, T000 comparison, W04 source re-use or production writes.

Unlike W04-only continuous matrix, accepts new fixed-camera source resolution.
Original frozen OpenVINO model tree hash MUST match W04 provenance. Exports
the two authentic 14-col full-pixel track streams needed by the *separate*
pre-registered object-correspondence holdout visual review tool.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .continuous_box_tracking_lab import as_tracker_detections, _count_ids
from .integrated_box_lab import RawBox, box_dict, merge_boxes
from .sahi_tracker_trial import rows_from_tracks
from .w04_hybrid_rare_suppression import merge_rare_preserving

STATUS="PHASE3_UNSEEN_DUAL_TRACKING_EVIDENCE_NOT_PRODUCTION"
PRIMARY_NAME="primary_ios030.txt"
SHADOW_NAME="shadow_rare_iou050.txt"
MIN_EVAL=120
MAX_EVAL=600
MIN_WARMUP=30

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda:s.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def percentile95(items: Sequence[float]) -> float:
    if not items or any(not math.isfinite(v) or v<0 for v in items):
        raise ValueError("Measured nonnegative timings required")
    return float(np.quantile(np.asarray(items,dtype=float),.95))

def policy_merge(raw:Sequence[RawBox],name:str):
    if name=="primary":
        return merge_boxes(raw,policy="hard_nms",threshold=.30,metric="ios",
                           score_floor=.15)
    if name=="shadow":
        return merge_rare_preserving(raw,mode="rare_iou05")
    raise ValueError("Policy must be original primary or rare-preserving shadow")

def dual_lane_stream(raw_stream:Iterable[tuple[int,list[RawBox],float]],
                     *,first:int,start:int,end:int,out:Path,fps:float,
                     source_provenance:dict,
                     tracker_factory:Callable[...,Any],
                     detection_class:Callable[...,Any],
                     rows_converter:Callable[...,Any]=rows_from_tracks,
                     clock:Callable[[],float]=time.perf_counter):
    """Fail closed on nonconsecutive frames, fake IDs or mutated raw boxes.

    Uses real tracker.update() once PER lane PER source frame, including all
    warmup and empty frames. Output staged atomically; original input read-only.
    """
    if out.exists():
        raise FileExistsError("Dual-lane tracking output must be unique")
    if (type(first) is not int or type(start) is not int or type(end) is not int
        or first<0 or start-first<MIN_WARMUP
        or end-start+1<MIN_EVAL or end-start+1>MAX_EVAL
        or not math.isfinite(fps) or fps<=0):
        raise ValueError("Unseen interval, warmup or source FPS outside frozen bounds")
    if source_provenance.get("eligible_for_production") is not False:
        raise ValueError("Real capture evidence must explicitly forbid promotion")
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+out.name+".stage-",dir=out.parent))
    modes=("primary","shadow")
    state={}
    samples=[]
    row_counts={name:0 for name in modes}
    detection_counts={name:0 for name in modes}
    raw_box_count=0
    frame_count=0
    try:
        for name in modes:
            tracker=tracker_factory(
                track_activation_threshold=.20,
                high_conf_det_threshold=.15,
                lost_track_buffer=45,
                minimum_consecutive_frames=2,
                frame_rate=fps)
            file=stage/(PRIMARY_NAME if name=="primary" else SHADOW_NAME)
            state[name]={"tracker":tracker,"file":file,
                         "fp":file.open("x",newline="",encoding="utf-8"),
                         "merge_time":[],"tracking_time":[]}
            state[name]["writer"]=csv.writer(state[name]["fp"])
        original_raw=stage/"original_pre_global_merge_boxes.jsonl"
        with original_raw.open("x",encoding="utf-8") as raw_file:
            for frame,boxes,detector_seconds in raw_stream:
                if type(frame) is not int or frame!=first+frame_count or frame>end:
                    raise ValueError("Detectors must emit every consecutive absolute source frame")
                if (not isinstance(detector_seconds,(float,int))
                    or not math.isfinite(detector_seconds) or detector_seconds<0):
                    raise ValueError("Invalid detector timing evidence")
                if any(p.frame!=frame for p in boxes):
                    raise ValueError("Raw source detections belong to different video frame")
                if len({p.source_index for p in boxes})!=len(boxes):
                    raise ValueError("Duplicated raw tile box source index")
                raw_box_count+=len(boxes)
                for p in boxes:
                    raw_file.write(json.dumps(box_dict(p))+"\n")
                record={"frame":frame,"detector_wall_seconds":float(detector_seconds)}
                for name in modes:
                    st=state[name]
                    t0=clock()
                    merged=policy_merge(boxes,name)
                    post_sec=clock()-t0
                    detection=as_tracker_detections(merged,detection_class)
                    t0=clock()
                    tracked=st["tracker"].update(detection)
                    track_sec=clock()-t0
                    st["merge_time"].append(post_sec)
                    st["tracking_time"].append(track_sec)
                    record[name+"_postprocess_wall_seconds"]=post_sec
                    record[name+"_tracking_wall_seconds"]=track_sec
                    if frame<start:
                        continue
                    if (len(np.asarray(tracked.xyxy))==0
                        and getattr(tracked,"tracker_id",None) is None):
                        rows=[]
                    else:
                        rows,_=rows_converter(tracked,frame)
                    if (any(len(r)!=14 or int(r[0])!=frame for r in rows)
                        or _count_ids(tracked)!=len(rows)):
                        raise ValueError("Tracker result schema/confirmed IDs invalid")
                    st["writer"].writerows(rows)
                    row_counts[name]+=len(rows)
                    detection_counts[name]+=len(detection.xyxy)
                record["sum_two_lane_wall_seconds"]=(
                    record["detector_wall_seconds"]+
                    sum(record[name+"_postprocess_wall_seconds"]+
                        record[name+"_tracking_wall_seconds"] for name in modes))
                if frame>=start:
                    samples.append(record)
                frame_count+=1
        if frame_count!=end-first+1 or len(samples)!=end-start+1:
            raise ValueError("Detector stream incomplete or warmup missing")
        for st in state.values():
            st["fp"].close()
        with (stage/"per_frame_timing.csv").open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        report={
            "status":STATUS,"eligible_for_production":False,
            "source_provenance":source_provenance,
            "first_decoded_frame":first,"first_evaluation_frame":start,
            "last_evaluation_frame":end,"warmup_frames":start-first,
            "tracking_updates_per_policy":frame_count,
            "evaluation_frames":len(samples),
            "actual_source_fps":fps,"inference_passes_per_frame":1,
            "independent_tracker_instances_per_policy":True,
            "raw_box_count":raw_box_count,"raw_sha256":sha(original_raw),
            "tracking":{},
            "detector_eval_wall_median_seconds":statistics.median(
                r["detector_wall_seconds"] for r in samples),
            "detector_eval_wall_p95_seconds":percentile95(
                [r["detector_wall_seconds"] for r in samples]),
            "two_lane_eval_wall_p95_seconds":percentile95(
                [r["sum_two_lane_wall_seconds"] for r in samples]),
            "limitations":(
                "CPU OpenVINO wall timings plus postprocess/tracker only; "
                "excludes input decoding, output writes and video SHA. "
                "Not full end-to-end CPU p95. "
                "No FLUID, human-grounded new-object or ID continuity scoring. "
                "Real fixed-camera original source and labels must be separately verified."
            )
        }
        for name in modes:
            st=state[name]
            report["tracking"][name]={
                "filename":st["file"].name,"sha256":sha(st["file"]),
                "confirmed_rows":row_counts[name],
                "evaluation_detections":detection_counts[name],
                "postprocess_wall_median_seconds":statistics.median(st["merge_time"]),
                "tracker_wall_median_seconds":statistics.median(st["tracking_time"]),
            }
        (stage/"dual_lane_report.json").write_text(
            json.dumps(report,indent=2),encoding="utf-8")
        if out.exists():
            raise FileExistsError("Refusing to overwrite completed dual-lane evidence")
        os.replace(stage,out)
        return report
    finally:
        for st in state.values():
            fp=st.get("fp")
            if fp is not None and not fp.closed:
                fp.close()
        if stage.exists():
            shutil.rmtree(stage)


def source_rgb(video:Path, first:int,end:int, *,cv2_module=None):
    """One seek then consecutive decode; variable camera width/height allowed."""
    if cv2_module is None:
        import cv2 as cv2_module
    cap=cv2_module.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot read heldout source video")
    try:
        total=int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))
        if total and end>=total:
            raise ValueError("Holdout frame range exceeds source video length")
        if not cap.set(cv2_module.CAP_PROP_POS_FRAMES,first):
            raise ValueError("Video cannot seek to first warmup frame")
        shape=None
        for frame in range(first,end+1):
            ok,img=cap.read()
            position=float(cap.get(cv2_module.CAP_PROP_POS_FRAMES))
            if not ok or not math.isfinite(position) or abs(position-(frame+1))>.51:
                raise ValueError(f"Source heldout video frame misalignment at {frame}")
            if img is None or len(img.shape)!=3 or img.shape[2]!=3:
                raise ValueError("Expected valid RGB/BGR traffic video frame")
            if shape is None:
                shape=img.shape[:2]
            if img.shape[:2]!=shape:
                raise ValueError("Holdout video resolution changed midsequence")
            yield frame,cv2_module.cvtColor(img,cv2_module.COLOR_BGR2RGB)
    finally:
        cap.release()


def run_video(video:Path,model_dir:Path,w04_provenance:Path, *,
              start:int,end:int,warmup:int,out:Path,
              tracker_factory=None,detection_class=None,
              cv2_module=None,model_loader=None,slicer=None,predictor=None):
    """Only real-image entry point; model + source SHA checked before inference."""
    if out.exists():
        raise FileExistsError("Tracking output exists")
    if not video.is_file() or not w04_provenance.is_file():
        raise FileNotFoundError("Original source video and W04 model provenance must exist")
    if warmup<MIN_WARMUP or start<warmup or end-start+1<MIN_EVAL or end-start+1>MAX_EVAL:
        raise ValueError("Holdout interval invalid or fewer than thirty warmup frames")
    from .openvino_export import hash_model_tree
    origin=json.loads(w04_provenance.read_text(encoding="utf-8"))
    model_sha=hash_model_tree(model_dir)
    if origin.get("eligible_for_promotion") is not False or model_sha!=origin.get("openvino_model_sha256"):
        raise ValueError("OpenVINO export does not match W04 frozen reference")
    video_sha=sha(video)
    if video_sha==origin.get("video_sha256"):
        raise ValueError("Cannot score W04 development video as unseen footage")
    if not model_dir.is_dir() or not list(model_dir.glob("*.xml")) or not list(model_dir.glob("*.bin")):
        raise ValueError("Frozen model XML+BIN missing")
    if cv2_module is None:
        import cv2 as cv2_module
    cap=cv2_module.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError("New video cannot be opened")
    try:
        fps=float(cap.get(cv2_module.CAP_PROP_FPS))
        width=int(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH))
        height=int(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT))
        count=int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if (not math.isfinite(fps) or fps<=0 or width<=0 or height<=0
        or (count and count<=end)):
        raise ValueError("Holdout video FPS/geometry/frame count invalid")
    if model_loader is None:
        from sahi import AutoDetectionModel
        model_loader=AutoDetectionModel.from_pretrained
    if tracker_factory is None:
        from trackers import ByteTrackTracker
        tracker_factory=ByteTrackTracker
    if detection_class is None:
        from supervision import Detections
        detection_class=Detections
    if slicer is None:
        from sahi.slicing import get_slice_bboxes
        slicer=get_slice_bboxes
    if predictor is None:
        from sahi.predict import get_prediction
        predictor=get_prediction
    from .premerge_box_capture import capture_boxes
    model=model_loader(model_type="ultralytics",model_path=str(model_dir),
                       confidence_threshold=.15,device="cpu",image_size=640)
    original_model_ref=sha(w04_provenance)
    provenance={
        "status":"EXPERIMENTAL_NEW_SOURCE_VIDEO_MODEL_FROZEN",
        "eligible_for_production":False,
        "video_path":str(video.resolve()),"video_sha256":video_sha,
        "video_width":width,"video_height":height,
        "video_fps_metadata":fps,"total_video_frames_metadata":count,
        "model_dir":str(model_dir.resolve()),"openvino_model_sha256":model_sha,
        "frozen_w04_model_reference_sha256":original_model_ref,
        "source_is_different_from_W04_SHA":True,
        "source_is_genuinely_unseen_not_independently_proven":True,
    }
    def streams():
        for frame,rgb in source_rgb(video,start-warmup,end,cv2_module=cv2_module):
            boxes,meta=capture_boxes([(frame,rgb)],model,slicer=slicer,
                                     predictor=predictor,overlap=.20,slice_size=640)
            yield frame,boxes,meta[0]["elapsed_s"]
    return dual_lane_stream(streams(),first=start-warmup,start=start,
                            end=end,out=out,fps=fps,
                            source_provenance=provenance,
                            tracker_factory=tracker_factory,
                            detection_class=detection_class)
