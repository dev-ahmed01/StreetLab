"""W04 separate class-evidence labels must be causal and detector-supported."""
from __future__ import annotations

import pytest
from streetlab_phase3.video.w04_class_evidence_stream import ClassEvidenceStream


def detection(klass):
    return {'vehicle_class':klass,'x1':0.,'y1':0.,'x2':10.,'y2':10.}


def test_two_frame_hysteresis_remains_auxiliary_and_support_constrained():
    model=ClassEvidenceStream()
    box=(0.,0.,10.,10.)
    a=model.observe(tracker_id=7,frame=100,instantaneous_class='CAR',
                    track_box=box,raw_detections=[detection('CAR')])
    b=model.observe(tracker_id=7,frame=101,instantaneous_class='MOTORCYCLE',
                    track_box=box,raw_detections=[detection('CAR'),detection('MOTORCYCLE')])
    c=model.observe(tracker_id=7,frame=102,instantaneous_class='MOTORCYCLE',
                    track_box=box,raw_detections=[detection('MOTORCYCLE')])
    assert (a.stable_class,b.stable_class,c.stable_class)==(
        'CAR','CAR','MOTORCYCLE'
    )
    assert b.provisional_class_change
    assert b.conflicting_detector_hypotheses
    assert not b.eligible_for_production


def test_absent_detector_support_for_prior_class_falls_back_to_original():
    x=ClassEvidenceStream()
    box=(0.,0.,10.,10.)
    x.observe(tracker_id=1,frame=1,instantaneous_class='CAR',track_box=box,
              raw_detections=[detection('CAR')])
    b=x.observe(tracker_id=1,frame=2,instantaneous_class='BUS',track_box=box,
                raw_detections=[detection('BUS')])
    assert b.instantaneous_class=='BUS'
    assert b.stable_class=='BUS'
    assert b.stable_class_supported_by_raw


def test_original_timing_and_id_validation():
    x=ClassEvidenceStream()
    box=(0.,0.,10.,10.)
    x.observe(tracker_id=1,frame=1,instantaneous_class='CAR',track_box=box,
              raw_detections=[detection('CAR')])
    with pytest.raises(ValueError,match='increasing'):
        x.observe(tracker_id=1,frame=1,instantaneous_class='CAR',track_box=box,
                  raw_detections=[detection('CAR')])
    with pytest.raises(ValueError):
        x.observe(tracker_id=-1,frame=2,instantaneous_class='CAR',
                  track_box=box,raw_detections=[])
