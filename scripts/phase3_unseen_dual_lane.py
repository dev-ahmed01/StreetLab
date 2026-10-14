"""Original-model frozen unseen-video OpenVINO dual-lane tracking, no FLUID truth.

Generates SHA-anchored primary_ios030.txt and shadow_rare_iou050.txt in ONE
source-video pass; use with scripts/phase3_correspondence_holdout.py prepare.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from streetlab_phase3.video.unseen_dual_lane import run_video

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--frozen-w04-source-provenance",type=Path,required=True)
    p.add_argument("--first-eval-frame",type=int,required=True)
    p.add_argument("--last-eval-frame",type=int,required=True)
    p.add_argument("--warmup-frames",type=int,default=30)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(argv)
    outcome=run_video(a.video,a.model_dir,a.frozen_w04_source_provenance,
                      start=a.first_eval_frame,end=a.last_eval_frame,
                      warmup=a.warmup_frames,out=a.output_dir)
    print(json.dumps({
        "status":outcome["status"],"eligible_for_production":False,
        "original_model_sha256":outcome["source_provenance"]["openvino_model_sha256"],
        "video_sha256":outcome["source_provenance"]["video_sha256"],
        "inference_passes_per_frame":outcome["inference_passes_per_frame"],
        "tracker_evaluation_frames":outcome["evaluation_frames"],
        "primary_export":str(a.output_dir/"primary_ios030.txt"),
        "shadow_export":str(a.output_dir/"shadow_rare_iou050.txt"),
        "two_lane_wall_p95_seconds":outcome["two_lane_eval_wall_p95_seconds"],
        "limitations":outcome["limitations"],
    },indent=2))

if __name__=="__main__":
    main()
