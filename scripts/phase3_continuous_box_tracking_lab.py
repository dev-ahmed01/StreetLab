"""Run one REAL continuous-video OpenVINO + 25-policy ByteTrack batch trial.

Not for sparse 21-frame JPEG input. Source MP4 and original FLUID CSV required.
No automatic production promotion; no tracker identity is fabricated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from streetlab_phase3.video.continuous_box_tracking_lab import (
    ContinuousTrial,policy_matrix,run_continuous_tracking_lab)


def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video',required=True,type=Path)
    ap.add_argument('--bundle',required=True,type=Path)
    ap.add_argument('--runtime-model',required=True,type=Path)
    ap.add_argument('--fluid-tracks',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--start-frame',required=True,type=int)
    ap.add_argument('--end-frame',required=True,type=int)
    ap.add_argument('--warmup-frames',type=int,default=90)
    ap.add_argument('--fps',type=float,default=30.)
    ap.add_argument('--baseline-tracks',type=Path)
    ap.add_argument('--policies',nargs='*',help='Known policy names; omit for all 25')
    args=ap.parse_args(argv)
    outcome=run_continuous_tracking_lab(ContinuousTrial(
        video=args.video,bundle=args.bundle,runtime_model=args.runtime_model,
        fluid_tracks=args.fluid_tracks,output_dir=args.output_dir,
        start_frame=args.start_frame,end_frame=args.end_frame,
        warmup_frames=args.warmup_frames,fps=args.fps,
        baseline_tracks=args.baseline_tracks,
        policy_names=tuple(args.policies) if args.policies is not None else None))
    print(json.dumps({
        'status':outcome['status'],'eligible_for_promotion':False,
        'processed_frames':outcome['processed_frames'],
        'candidate_count':outcome['candidate_count'],
        'independent_tracker_instance_per_policy':True,
        'report':str(args.output_dir/'batch_report.json'),
        'note':outcome['limitations'],
    },indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())