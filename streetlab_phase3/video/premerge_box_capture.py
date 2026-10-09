"""Isolated, bounded SAHI-compatible tiled raw-box capture (pre-global-merge).

Requires optional SAHI/OpenVINO only when actually executing inference. Uses
SAHI's own slicer and per-tile get_prediction; model-local NMS may still occur.
No tracking, ground-truth-dependent thresholding, or production mutations.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from .integrated_box_lab import ALL_CLASSES, RawBox, box_dict, evaluate_candidates

MODEL_CLASS_MAP = {
    'car':'CAR','automobile':'CAR',
    'bus':'BUS','truck':'HEAVY_VEHICLE','lorry':'HEAVY_VEHICLE',
    'motorcycle':'MOTORCYCLE','motorbike':'MOTORCYCLE','moped':'MOTORCYCLE',
    'pedestrian':'PEDESTRIAN','person':'PEDESTRIAN',
    'bicycle':'BICYCLE','cycle':'BICYCLE',
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_data(bundle: Path) -> tuple[list[int],list[dict[str,Any]],dict[str,Any]]:
    """Verify ALL entries before returning any experiment data."""
    with zipfile.ZipFile(bundle) as z:
        if any(info.filename.startswith('/') or '..' in Path(info.filename).parts
               for info in z.infolist()):
            raise ValueError("Unsafe experiment bundle path")
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('status')!='UNPROMOTED_W04_SELF_CONTAINED_RESEARCH_INPUT':
            raise ValueError('Unexpected evidence-bundle status')
        checks=manifest['files']
        for filename,digest in checks.items():
            if _sha(z.read(filename)) != digest:
                raise ValueError(f'W04 evidence SHA mismatch: {filename}')
        frames=manifest['frames']
        if (frames != list(range(10750,11351,30))
            or len({*frames})!=len(frames)):
            raise ValueError('Unexpected W04 frame positions; must rebaseline separately')
        with io.StringIO(z.read('truth/raw_fluid_selected.csv').decode('utf-8-sig')) as f:
            labels=list(csv.DictReader(f))
        ref=json.loads(z.read('audits/openvino/report.json'))
        if (ref['model_sha256']!=manifest['original_model_sha256']
            or ref['runtime_model_sha256']!=manifest['openvino_export_sha256']
            or ref['sample_frames']!=frames):
            raise ValueError('Model provenance or frame sample differs')
        # The original reference label SHA is for the 2GB original FLUID CSV;
        # the selected rows are a derived export with its OWN manifest digest.
        return frames,labels,{
            'manifest_sha256':_sha(z.read('manifest.json')),
            'archived_csv_sha256':checks['truth/raw_fluid_selected.csv'],
            'checkpoint_sha256':manifest['original_model_sha256'],
            'openvino_sha256':manifest['openvino_export_sha256'],
            'frozen_reference':ref['sliced'],
        }


def sample_rgb_frames(bundle: Path, frames: list[int],
                      *,cv2_module:Any = None) -> Iterable[tuple[int,Any]]:
    if cv2_module is None:
        import cv2 as cv2_module
    import numpy as np
    with zipfile.ZipFile(bundle) as z:
        for frame in frames:
            member=f'frames/frame_{frame:06d}.jpg'
            raw=z.read(member)
            bgr=cv2_module.imdecode(np.frombuffer(raw,dtype=np.uint8),
                                    cv2_module.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2]!=(2160,3840):
                raise ValueError(f'Invalid original W04 image dimensions for {frame}')
            yield frame,cv2_module.cvtColor(bgr,cv2_module.COLOR_BGR2RGB)


def sample_video_rgb_frames(video: Path, frames: list[int],
                            *, cv2_module:Any=None) -> Iterable[tuple[int,Any]]:
    if cv2_module is None:
        import cv2 as cv2_module
    cap=cv2_module.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Unable to open source video: {video}')
    try:
        for frame in frames:
            if not cap.set(cv2_module.CAP_PROP_POS_FRAMES,frame):
                raise RuntimeError(f'Cannot seek to requested frame {frame}')
            ok,img=cap.read()
            pos=cap.get(cv2_module.CAP_PROP_POS_FRAMES)
            if not ok or not math.isfinite(pos) or abs(pos-frame-1)>.51:
                raise RuntimeError(f'Source video frame alignment failed at {frame}')
            if img is None or img.shape[:2]!=(2160,3840):
                raise ValueError('Unexpected source video frame geometry')
            yield frame,cv2_module.cvtColor(img,cv2_module.COLOR_BGR2RGB)
    finally:
        cap.release()


def capture_boxes(
    frames:Iterable[tuple[int,Any]], model:Any, *,
    slicer:Callable[...,Any]|None=None,
    predictor:Callable[...,Any]|None=None,
    clock:Callable[[],float]|None=None,
    overlap:float=.20, slice_size:int=640,
) -> tuple[list[RawBox],list[dict[str,Any]]]:
    """Per-tile predictions before SAHI global merge, in absolute coordinates.

    The source local prediction can have already gone through YOLO model NMS.
    This is NOT an export of lower-level raw network logits.
    """
    if not 0<=overlap<.5 or not 128<=slice_size<=4096:
        raise ValueError('Invalid sample tile shape/overlap')
    if slicer is None:
        from sahi.slicing import get_slice_bboxes
        slicer=get_slice_bboxes
    if predictor is None:
        from sahi.predict import get_prediction
        predictor=get_prediction
    if clock is None:
        clock=time.perf_counter
    raw=[]; metadata=[]
    previous_frame=-1
    for frame,rgb in frames:
        if type(frame) is not int or frame<=previous_frame:
            raise ValueError('Sample frame positions must be unique and increasing')
        previous_frame=frame
        height,width=rgb.shape[:2]
        coords=slicer(image_height=height,image_width=width,
                      slice_height=slice_size,slice_width=slice_size,
                      overlap_height_ratio=overlap,overlap_width_ratio=overlap)
        if not coords:
            raise ValueError('No SAHI tiles returned')
        frame_start=clock()
        dropped_unmapped=Counter()
        per_frame=[]
        for tile_id,(x0,y0,x1,y1) in enumerate(coords):
            if not (0<=x0<x1<=width and 0<=y0<y1<=height):
                raise ValueError('Invalid absolute slice origin returned by slicer')
            # get_prediction operates on a zero-origin tile. Passing no shift
            # avoids double-counting SAHI's own shift_amount metadata.
            objects=predictor(rgb[y0:y1,x0:x1],model,verbose=0).object_prediction_list
            for ob in objects:
                klass=MODEL_CLASS_MAP.get(str(ob.category.name).strip().lower())
                if klass is None:
                    dropped_unmapped[str(ob.category.name)]+=1
                    continue
                xy=ob.bbox.to_xyxy()
                if len(xy)!=4:
                    raise ValueError('Invalid SAHI prediction XYXY dimensions')
                xmin,ymin,xmax,ymax=[float(q) for q in xy]
                conf=float(ob.score.value)
                if not all(math.isfinite(z) for z in (xmin,ymin,xmax,ymax,conf)):
                    raise ValueError('Nonfinite model box')
                if xmin < -2 or ymin < -2 or xmax > (x1-x0)+2 or ymax > (y1-y0)+2:
                    raise ValueError('Tile prediction unexpectedly already offset or outside crop')
                # Small floating-point overshoots are explicitly clipped,
                # no silently discarded boxes or confidence changes.
                xmin=max(0.,xmin);ymin=max(0.,ymin)
                xmax=min(float(x1-x0),xmax);ymax=min(float(y1-y0),ymax)
                per_frame.append(RawBox(frame,x0+xmin,y0+ymin,x0+xmax,y0+ymax,
                                        conf,klass,tile_id,len(per_frame)))
        metadata.append({
            'frame':frame,'tile_count':len(coords),'captured_box_count':len(per_frame),
            'elapsed_s':clock()-frame_start,
            'unmapped_names_count':dict(dropped_unmapped),
        })
        raw.extend(per_frame)
    return raw,metadata


def _write_json(path:Path,data:Any) -> None:
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def run_integrated_lab(
    *,bundle:Path, model_path:Path,output_dir:Path,
    video:Path|None=None, loader:Callable[...,Any]|None=None,
    slicer:Callable[...,Any]|None=None,predictor:Callable[...,Any]|None=None,
    cv2_module:Any=None,
) -> dict[str,Any]:
    """One batched bounded run; atomically save ALL 25 candidate reports."""
    if output_dir.exists():
        raise FileExistsError(f'Experiment output already exists: {output_dir}')
    if not model_path.is_dir() or not list(model_path.glob('*.xml')) or not list(model_path.glob('*.bin')):
        raise ValueError('OpenVINO model directory requires XML+BIN')
    frames, labels, evidence=bundle_data(bundle)
    if loader is None:
        from sahi import AutoDetectionModel
        loader=AutoDetectionModel.from_pretrained
    model=loader(model_type='ultralytics',model_path=str(model_path),
                 confidence_threshold=.15,device='cpu',image_size=640)
    if video is None:
        iterable=sample_rgb_frames(bundle,frames,cv2_module=cv2_module)
        source='jpeg_evidence_bundle_not_pixel_identical_video'
    else:
        if not video.is_file():
            raise FileNotFoundError(f'Original source video unavailable: {video}')
        iterable=sample_video_rgb_frames(video,frames,cv2_module=cv2_module)
        source='actual_source_video_decoded_frame_exact'
    raw, times=capture_boxes(iterable,model,slicer=slicer,predictor=predictor)
    report=evaluate_candidates(raw,labels,frames)
    report['source_frame_evidence']=source
    report['source_provenance']=evidence
    report['frame_inference']=times
    report['known_reference_from_prior_SAHI_run']={
        k:v for k,v in evidence['frozen_reference'].items()
        if k in ('truth_points','predicted_points','matched_points',
                 'recall','precision','per_class','latency_median_s')}
    report['capture_not_comparable_to_old_cached_boxes']=True
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'.stage-',
                              dir=output_dir.parent))
    try:
        with (tmp/'pre_global_merge_boxes.jsonl').open('w',encoding='utf-8') as f:
            for b in raw:
                f.write(json.dumps(box_dict(b))+'\n')
        _write_json(tmp/'matrix_report.json',report)
        _write_json(tmp/'capture_metadata.json',{
            'status':'W04_OPENVINO_PER_TILE_BOX_EVIDENCE_NOT_PRODUCTION',
            'frames':frames,'source':source,'provenance':evidence,
            'frame_inference':times,
            'raw_boxes_sha256':_sha((tmp/'pre_global_merge_boxes.jsonl').read_bytes()),
        })
        if output_dir.exists():
            raise FileExistsError('Refusing experiment overwrite')
        os.replace(tmp,output_dir)
        return report
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
