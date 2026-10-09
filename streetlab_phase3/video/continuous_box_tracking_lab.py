"""Experimental continuous pre-merge SAHI/OpenVINO -> box policies -> ByteTrack.

One actual detector inference pass per source frame is fanned out into distinct
policy-specific tracker instances. NO sparse-frame tracking, fabricated IDs,
FLUID-dependent filtering, production changes, or automatic promotion.
The model's per-tile local NMS has already occurred; this varies only global
box postprocessing. W04 is tuned evidence, never independent holdout.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .integrated_box_lab import RawBox, candidate_matrix, merge_boxes, box_dict
from .premerge_box_capture import bundle_data, capture_boxes

CANONICAL_IDS = {'CAR': 0, 'BUS': 1, 'HEAVY_VEHICLE': 2, 'MOTORCYCLE': 3}
STATUS = 'EXPERIMENTAL_CONTINUOUS_BOX_BYTETRACK_BENCHMARK_NOT_PRODUCTION'


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


@dataclass(frozen=True)
class ContinuousTrial:
    video: Path
    bundle: Path
    runtime_model: Path
    fluid_tracks: Path
    output_dir: Path
    start_frame: int
    end_frame: int
    warmup_frames: int = 90
    baseline_tracks: Path | None = None
    policy_names: tuple[str, ...] | None = None
    tile_size: int = 640
    tile_overlap: float = .20
    fps: float = 30.0

    def validate(self) -> None:
        if (type(self.start_frame) is not int or type(self.end_frame) is not int
            or type(self.warmup_frames) is not int or self.start_frame < 0
            or self.end_frame < self.start_frame or self.warmup_frames < 0):
            raise ValueError('Continuous interval or warmup invalid')
        if (not 128 <= self.tile_size <= 4096 or not 0 <= self.tile_overlap < .5
            or not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError('Invalid tile geometry or actual source FPS')
        if self.output_dir.exists():
            raise FileExistsError('Experiment output already exists')
        for path in (self.video, self.bundle, self.fluid_tracks):
            if not path.is_file():
                raise FileNotFoundError(f'Missing original source: {path}')
        if (not self.runtime_model.is_dir()
            or not list(self.runtime_model.glob('*.xml'))
            or not list(self.runtime_model.glob('*.bin'))):
            raise ValueError('Isolated OpenVINO IR model directory required')
        if self.baseline_tracks is not None and not self.baseline_tracks.is_file():
            raise FileNotFoundError('Same-window T000 track file not found')


def sequential_video_rgb_frames(video: Path, first_frame: int, end_frame: int,
                                *, cv2_module: Any = None):
    """ONE seek then sequential reads; fail on any timestamp/frame gap."""
    if type(first_frame) is not int or type(end_frame) is not int or not 0 <= first_frame <= end_frame:
        raise ValueError('Source video frame range invalid')
    if cv2_module is None:
        import cv2 as cv2_module
    cap=cv2_module.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open 4K source video: {video}')
    try:
        if not cap.set(cv2_module.CAP_PROP_POS_FRAMES,first_frame):
            raise RuntimeError(f'Cannot seek to first warmup frame {first_frame}')
        for frame in range(first_frame,end_frame+1):
            ok,img=cap.read()
            position=float(cap.get(cv2_module.CAP_PROP_POS_FRAMES))
            if not ok or not math.isfinite(position) or abs(position-(frame+1))>.51:
                raise RuntimeError(f'Continuous decode alignment failure at {frame}')
            if img is None or img.shape[:2] != (2160,3840):
                raise ValueError('Expected original 3840x2160 source video')
            yield frame,cv2_module.cvtColor(img,cv2_module.COLOR_BGR2RGB)
    finally:
        cap.release()


def policy_matrix(names: Sequence[str] | None = None) -> list[dict[str, Any]]:
    configs = candidate_matrix()
    names_by_key = {p['name']: p for p in configs}
    if names is None:
        return configs
    if not names or len(set(names)) != len(names):
        raise ValueError('Requested policy names must be nonempty and unique')
    if any(n not in names_by_key for n in names):
        raise ValueError('Policy name not in frozen 25-policy matrix')
    return [names_by_key[n] for n in names]


def as_tracker_detections(boxes: Sequence[Any], detection_class: Callable[..., Any]) -> Any:
    """Use original COCO/name mapping from Build 1, NEVER implicit class indices.

    BICYCLE and PEDESTRIAN remain in the full box evidence, but the existing
    Geo-trax four-class track export cannot represent them; do not reclassify.
    """
    selected = [p for p in boxes if p.vehicle_class in CANONICAL_IDS]
    return detection_class(
        xyxy=np.asarray([(p.x1,p.y1,p.x2,p.y2) for p in selected],
                        dtype=np.float32).reshape(-1,4),
        confidence=np.asarray([p.confidence for p in selected], dtype=np.float32),
        class_id=np.asarray([CANONICAL_IDS[p.vehicle_class] for p in selected],
                            dtype=np.int32),
    )


def _count_ids(tracks: Any) -> int:
    raw_ids=getattr(tracks,'tracker_id',None)
    if raw_ids is None:
        if len(np.asarray(tracks.xyxy)) == 0:
            return 0
        raise ValueError('Nonempty tracker results require genuine IDs')
    ids = np.asarray(raw_ids)
    if ids.ndim != 1:
        raise ValueError('Tracker IDs must be a one-dimensional vector')
    return sum(int(i) >= 0 for i in ids)


def cached_window_scorer(fluid_tracks: Path, start_frame: int, end_frame: int):
    """Load FLUID ONCE for every candidate's frozen pixel/identity evaluation.

    The identity benchmark intentionally remains original class-agnostic
    spatial matching. A separate report retains correct-class motorcycle
    recall; never conflate these two different notions of accuracy.
    """
    from streetlab_phase3.pixel_benchmark import (
        PixelBenchmark, normalize_fluid_pixel_truth)
    from streetlab_phase3.identity_benchmark import IdentityBenchmark
    from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks
    from streetlab_phase3.serialization import package_to_dict
    from streetlab_phase3.video.class_diagnostics import class_diagnostics

    selected = []
    with fluid_tracks.open('r', encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f)
        for record in reader:
            if not record.get('frame'):
                continue
            frame = int(float(record['frame']))
            if start_frame + 1 <= frame <= end_frame + 1:
                selected.append(record)
    truths=normalize_fluid_pixel_truth(selected)
    if not truths:
        raise ValueError('No supported FLUID labels in fixed evaluation interval')

    class Scorer:
        def __call__(self, *,tracks,fluid_tracks: Path,max_distance_px,
                     start_frame: int,end_frame: int):
            if (fluid_tracks.resolve()!=path.resolve() or max_distance_px!=50.
                or start_frame != left or end_frame != right):
                raise ValueError('Cached scorer input differs from fixed FLUID cohort')
            pts=load_geotrax_pixel_tracks(tracks)
            window=(left+1,right+1)
            pixel=package_to_dict(PixelBenchmark(50.).evaluate(
                pts,truths,frame_offsets=(1,),evaluation_frame_range=window))
            ident=IdentityBenchmark(50.).evaluate(
                pts,truths,frame_offsets=(1,),evaluation_frame_range=window)
            return pixel,ident.scorecard_dict()

        def class_report(self, tracks: Path):
            pts=load_geotrax_pixel_tracks(tracks)
            return class_diagnostics(pts,truths,start_frame=left,end_frame=right)

    path=fluid_tracks
    left,right=start_frame,end_frame
    return Scorer()


def stream_tracking_matrix(
    *, raw_stream: Iterable[tuple[int, Sequence[RawBox], float]],
    start_frame: int, end_frame: int, first_frame: int, output_dir: Path,
    fluid_tracks: Path, configs: Sequence[dict[str, Any]],
    tracker_factory: Callable[..., Any], detection_class: Callable[..., Any],
    rows_converter: Callable[[Any, int], tuple[list[list[object]], int]] | None = None,
    score_fn: Callable[..., tuple[dict[str, Any],dict[str, Any]]] | None = None,
    baseline_tracks: Path | None = None, fps: float = 30.,
    provenance: dict[str,Any] | None = None,
) -> dict[str, Any]:
    """Replay ONE contiguous raw detection stream into independent ByteTrack.

    No source video is decoded here. Synthetic test doubles may exercise the
    pure integration without claiming real detector/tracker performance.
    The output directory appears atomically after a complete successful run.
    """
    if (type(first_frame) is not int or type(start_frame) is not int
        or type(end_frame) is not int or not 0 <= first_frame <= start_frame <= end_frame):
        raise ValueError('Invalid contiguous frame window')
    if not configs or len({p['name'] for p in configs}) != len(configs):
        raise ValueError('Configs must be nonempty with distinct names')
    approved = {p['name']: p for p in candidate_matrix()}
    if any(p['name'] not in approved or p != approved[p['name']] for p in configs):
        raise ValueError('All tracker policies must come unmodified from frozen matrix')
    if output_dir.exists():
        raise FileExistsError('Refusing to overwrite previous tracking evidence')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Source FPS must be positive')
    if rows_converter is None:
        from .sahi_tracker_trial import rows_from_tracks
        rows_converter = rows_from_tracks
    if score_fn is None:
        score_fn = cached_window_scorer(fluid_tracks,start_frame,end_frame)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.stage-',dir=output_dir.parent))
    states = {}
    try:
        raw_path=temporary/'original_pre_global_merge_boxes.jsonl'
        raw_file=raw_path.open('w',encoding='utf-8')
        for p in configs:
            name = p['name']
            tracker = tracker_factory(
                track_activation_threshold=.20, high_conf_det_threshold=.15,
                lost_track_buffer=45, minimum_consecutive_frames=2,
                frame_rate=fps,
            )
            file = temporary / f'{name}.txt'
            fh = file.open('w',encoding='utf-8',newline='')
            states[name] = {
                'tracker': tracker, 'handle': fh, 'writer': csv.writer(fh),
                'file': file, 'detection_count': 0, 'confirmed_count': 0,
                'unconfirmed_count': 0, 'postprocess_seconds': [],
                'tracking_seconds': [], 'boxes_kept_all_classes': 0,
            }
        total_frames = 0
        warmup_actual = max(0,start_frame-first_frame)
        raw_boxes = 0
        detector_seconds = []
        clock = time.perf_counter
        for frame, raw, detector_time in raw_stream:
            if type(frame) is not int or frame != first_frame + total_frames:
                raise ValueError('Tracker requires exactly consecutive absolute video frames')
            if frame > end_frame:
                raise ValueError('Tracker stream contained out-of-window frame')
            if not math.isfinite(detector_time) or detector_time < 0:
                raise ValueError('Detector runtime must be measured and nonnegative')
            if any(p.frame != frame for p in raw):
                raise ValueError('Mixed source frames in one inference result')
            if len(set(p.source_index for p in raw)) != len(raw):
                raise ValueError('Duplicate tile prediction source indices in frame')
            detector_seconds.append(float(detector_time))
            raw_boxes += len(raw)
            for original_box in raw:
                raw_file.write(json.dumps(box_dict(original_box))+'\n')
            for p in configs:
                name = p['name']
                state = states[name]
                begin = clock()
                merged = merge_boxes(raw, **{k:v for k,v in p.items() if k!='name'})
                det = as_tracker_detections(merged,detection_class)
                state['postprocess_seconds'].append(clock()-begin)
                begin = clock()
                tracked = state['tracker'].update(det)
                state['tracking_seconds'].append(clock()-begin)
                if frame < start_frame:
                    continue
                # Some supervision versions expose tracker_id=None on an
                # entirely empty result. Do not invent an ID, and still call
                # the tracker exactly once to advance its lost-track buffer.
                if (len(np.asarray(tracked.xyxy)) == 0
                    and getattr(tracked,'tracker_id',None) is None):
                    rows,unconfirmed = [],0
                else:
                    rows,unconfirmed = rows_converter(tracked,frame)
                if any(len(row) != 14 or int(row[0]) != frame for row in rows):
                    raise ValueError('Geo-trax row schema or absolute frame violated')
                if len(rows) != _count_ids(tracked):
                    raise ValueError('Exported confirmed track IDs do not match tracker')
                if len(det.xyxy) < 0 or unconfirmed < 0:
                    raise ValueError('Invalid output detections count')
                state['writer'].writerows(rows)
                state['detection_count'] += len(det.xyxy)
                state['confirmed_count'] += len(rows)
                state['unconfirmed_count'] += unconfirmed
                state['boxes_kept_all_classes'] += len(merged)
            total_frames += 1
        if total_frames != end_frame-first_frame+1:
            raise ValueError('Missing last source frames or warmup frames in tracker stream')
        raw_file.close()
        for state in states.values():
            state['handle'].close()
        results = []
        baseline = None
        if baseline_tracks is not None:
            base_pixel,base_identity = score_fn(
                tracks=baseline_tracks, fluid_tracks=fluid_tracks,
                max_distance_px=50.,start_frame=start_frame,end_frame=end_frame)
            baseline = {'pixel':base_pixel,'identity':base_identity,
                        'source_sha256':_sha(baseline_tracks),
                        'class_diagnostics':(score_fn.class_report(baseline_tracks)
                            if hasattr(score_fn,'class_report') else None)}
        for p in configs:
            name=p['name']; state=states[name]
            pixel, identity = score_fn(
                tracks=state['file'],fluid_tracks=fluid_tracks,max_distance_px=50.,
                start_frame=start_frame,end_frame=end_frame)
            if pixel.get('frame_offset') != 1 or identity.get('frame_offset') != 1:
                raise ValueError('Scorer changed frozen FLUID frame offset')
            if (pixel.get('max_distance_px') != 50.
                or identity.get('max_distance_px') != 50.):
                raise ValueError('Scorer changed frozen FLUID matching radius')
            if baseline is not None:
                for key in ('truth_points','evaluation_frame_start',
                            'evaluation_frame_end','frame_offset','max_distance_px'):
                    if pixel.get(key) != baseline['pixel'].get(key):
                        raise ValueError(f'Baseline video cohort mismatch: {key}')
            report = {
                'name':name,'settings':p,
                'tracks':f'{name}.txt',
                'tracks_sha256':_sha(state['file']),
                'evaluation_detections':state['detection_count'],
                'evaluation_confirmed_rows':state['confirmed_count'],
                'evaluation_unconfirmed_rows':state['unconfirmed_count'],
                'evaluation_merged_boxes_all_classes':state['boxes_kept_all_classes'],
                'postprocess_median_seconds':statistics.median(state['postprocess_seconds']),
                'tracking_median_seconds':statistics.median(state['tracking_seconds']),
                'pixel':pixel,'identity':identity,
                'class_diagnostics':(score_fn.class_report(state['file'])
                    if hasattr(score_fn,'class_report') else None),
                'eligible_for_production':False,
            }
            (temporary/f'{name}.score.json').write_text(
                json.dumps(report,indent=2),encoding='utf-8')
            results.append(report)
        summary = {
            'status':STATUS, 'eligible_for_promotion':False,
            'tracking_updates_are_consecutive':True,
            'evaluated_first_frame':start_frame,'evaluated_last_frame':end_frame,
            'first_decoded_frame':first_frame,'warmup_frames_processed':warmup_actual,
            'processed_frames':total_frames,
            'evaluated_frames':end_frame-start_frame+1,
            'candidate_count':len(configs),'raw_tile_boxes':raw_boxes,
            'raw_pre_global_merge_boxes_file':raw_path.name,
            'raw_pre_global_merge_boxes_sha256':_sha(raw_path),
            'detector_median_seconds':statistics.median(detector_seconds),
            'independent_tracker_instance_per_policy':True,
            'policies':results,'same_window_T000':baseline,
            'physical_object_review_available':False,
            'unseen_footage_validated':False,
            'benchmark_is_continuous_tracking_not_sparse_W04_21':True,
            'limitations':(
                'One source frame per tracker update; independent ByteTrack '
                'instances receive frame-consistent geometry variants. '
                'FLUID class taxonomy contains known W04 disagreements. '
                'ByteTrack identity metrics are class-agnostic 50px matching, '
                'and cannot establish physical precision or annotate missing cars. '
                'Use independent two-reviewer truth before object-level claims. '
                'Timing is CPU environment dependent. NO promotion; held-out '
                'video and same-cohort T000 remain mandatory.'
            ),
        }
        (temporary/'batch_report.json').write_text(json.dumps(summary,indent=2),
                                                   encoding='utf-8')
        if provenance is not None:
            if provenance.get('eligible_for_promotion') is not False:
                raise ValueError('Source provenance must explicitly forbid promotion')
            (temporary/'source_provenance.json').write_text(
                json.dumps(provenance,indent=2),encoding='utf-8')
        if output_dir.exists():
            raise FileExistsError('Refusing to overwrite complete tracking output')
        os.replace(temporary,output_dir)
        return summary
    finally:
        if 'raw_file' in locals() and not raw_file.closed:
            raw_file.close()
        for state in states.values():
            try:
                if not state['handle'].closed:
                    state['handle'].close()
            except Exception:
                pass
        if temporary.exists():
            shutil.rmtree(temporary)


def run_continuous_tracking_lab(trial: ContinuousTrial) -> dict[str, Any]:
    """Only path that executes REAL local OpenVINO inference (no sparse JPGs)."""
    trial.validate()
    _frames,_labels,evidence=bundle_data(trial.bundle)
    from .openvino_export import hash_model_tree
    if hash_model_tree(trial.runtime_model) != evidence['openvino_sha256']:
        raise ValueError('Isolated OpenVINO runtime model is not W04-provenance exact')
    from sahi import AutoDetectionModel
    from sahi.slicing import get_slice_bboxes
    from sahi.predict import get_prediction
    from supervision import Detections
    from trackers import ByteTrackTracker
    model=AutoDetectionModel.from_pretrained(
        model_type='ultralytics',model_path=str(trial.runtime_model),
        confidence_threshold=.15,device='cpu',image_size=640)
    # The full original FLUID CSV is verified against the SAME exact bytes
    # used in W04 audit; the selected-row sidecar is not an equivalent input.
    import zipfile
    with zipfile.ZipFile(trial.bundle) as z:
        original=json.loads(z.read('audits/openvino/report.json'))
    if _sha(trial.fluid_tracks)!=original['ground_truth_sha256']:
        raise ValueError('Original FLUID CSV differs from frozen source annotation SHA')
    provenance = {
        'status':'EXPERIMENTAL_OPENVINO_FULL_SOURCE_VIDEO_PROVENANCE',
        'eligible_for_promotion':False,
        'video_path':str(trial.video),'video_sha256':_sha(trial.video),
        'fluid_path':str(trial.fluid_tracks),
        'fluid_sha256':_sha(trial.fluid_tracks),
        'bundle_sha256':_sha(trial.bundle),
        'frozen_checkpoint_sha256':evidence['checkpoint_sha256'],
        'openvino_model_sha256':evidence['openvino_sha256'],
        'source_fps_user_supplied':trial.fps,
        'video_frame_window':[trial.start_frame,trial.end_frame],
        'policies':[p['name'] for p in policy_matrix(trial.policy_names)],
    }
    first=max(0,trial.start_frame-trial.warmup_frames)
    import cv2
    def source() -> Iterable[tuple[int,Sequence[RawBox],float]]:
        for frame,rgb in sequential_video_rgb_frames(
                trial.video,first,trial.end_frame,cv2_module=cv2):
            boxes,times=capture_boxes([(frame,rgb)],model,
                                      slicer=get_slice_bboxes,predictor=get_prediction,
                                      overlap=trial.tile_overlap,slice_size=trial.tile_size)
            yield frame,boxes,times[0]['elapsed_s']
    result=stream_tracking_matrix(
        raw_stream=source(),first_frame=first,
        start_frame=trial.start_frame,end_frame=trial.end_frame,
        output_dir=trial.output_dir,fluid_tracks=trial.fluid_tracks,
        configs=policy_matrix(trial.policy_names),
        tracker_factory=ByteTrackTracker,detection_class=Detections,
        baseline_tracks=trial.baseline_tracks,fps=trial.fps,provenance=provenance,
    )
    return result