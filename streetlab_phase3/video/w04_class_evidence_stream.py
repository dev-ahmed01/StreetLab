"""Experimental non-destructive online class evidence stream for StreetLab W04.

Never overwrites the native 14-column Geo-trax class, tracker identity, frozen
FLUID labels or original detector outputs. Uses same-frame raw detections only.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping


def box_iou(a: tuple[float,float,float,float],
            b: tuple[float,float,float,float]) -> float:
    if len(a)!=4 or len(b)!=4 or not all(isfinite(x) for x in (*a,*b)):
        raise ValueError('Finite XYXY coordinates required')
    if a[2]<=a[0] or a[3]<=a[1] or b[2]<=b[0] or b[3]<=b[1]:
        raise ValueError('Positive XYXY dimensions required')
    dx=max(0., min(a[2],b[2])-max(a[0],b[0]))
    dy=max(0., min(a[3],b[3])-max(a[1],b[1]))
    inter=dx*dy
    area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area else 0.


@dataclass(frozen=True)
class ClassEvidence:
    tracker_id: int
    frame: int
    instantaneous_class: str
    stable_class: str
    stable_class_supported_by_raw: bool
    detector_classes_at_track_box: tuple[str,...]
    conflicting_detector_hypotheses: bool
    provisional_class_change: bool
    eligible_for_production: bool = False


class ClassEvidenceStream:
    """Causal two-observation hysteresis with detector support/fail-closed fallback.

    The causal state can propose one class, but the public interpretation falls
    back to instantaneous class when no same-frame raw box of that stable class
    overlaps sufficiently. The full raw instant class is always preserved.
    """

    def __init__(self, *, consecutive_required: int = 2,
                 min_box_iou: float = .5):
        if consecutive_required < 2 or not 0 < min_box_iou <= 1:
            raise ValueError('Invalid class evidence safety policy')
        self.consecutive_required=consecutive_required
        self.min_box_iou=min_box_iou
        self._states: dict[int,dict] = {}

    def observe(self, *, tracker_id: int, frame: int,
                instantaneous_class: str,
                track_box: tuple[float,float,float,float],
                raw_detections: Iterable[Mapping]) -> ClassEvidence:
        if type(tracker_id) is not int or tracker_id < 0 or type(frame) is not int or frame < 0:
            raise ValueError('Real nonnegative tracker ID and source frame required')
        if not instantaneous_class:
            raise ValueError('Missing instantaneous class')
        box_iou(track_box,track_box)
        state=self._states.get(tracker_id)
        if state is None:
            state={'class':instantaneous_class,'last_frame':frame-1,
                   'pending':None,'count':0}
            self._states[tracker_id]=state
        if frame<=state['last_frame']:
            raise ValueError('Tracker classes must arrive in increasing source frames')
        if instantaneous_class==state['class']:
            state['pending']=None
            state['count']=0
        else:
            if instantaneous_class==state['pending']:
                state['count']+=1
            else:
                state['pending']=instantaneous_class
                state['count']=1
            if state['count']>=self.consecutive_required:
                state['class']=instantaneous_class
                state['pending']=None
                state['count']=0
        state['last_frame']=frame
        support=set()
        for candidate in raw_detections:
            cls=candidate['vehicle_class']
            bb=tuple(float(candidate[k]) for k in ('x1','y1','x2','y2'))
            if box_iou(track_box,bb)>=self.min_box_iou:
                support.add(cls)
        proposed=state['class']
        backed=proposed in support
        # Keep native class when stable hypothesis has no same-frame raw evidence.
        stable=proposed if backed else instantaneous_class
        return ClassEvidence(
            tracker_id=tracker_id,frame=frame,
            instantaneous_class=instantaneous_class,stable_class=stable,
            stable_class_supported_by_raw=stable in support,
            detector_classes_at_track_box=tuple(sorted(support)),
            conflicting_detector_hypotheses=len(support)>1,
            provisional_class_change=stable!=instantaneous_class,
        )
