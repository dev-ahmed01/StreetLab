"""Run one isolated 21-frame OpenVINO pre-global-merge box experiment matrix.

No tracker, no promotions, no ground-truth-dependent filtering. Use the
original local video for exact decoded frames, or the manifest-verified JPEG
source bundle when video is unavailable. Requires SAHI+OpenVINO locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from streetlab_phase3.video.premerge_box_capture import run_integrated_lab


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path,
                   help='original W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip')
    p.add_argument('--runtime-model',required=True,type=Path,
                   help='isolated SHA-checked OpenVINO export directory from W04')
    p.add_argument('--video',type=Path,help='preferred exact source 4K AVI/MP4 (omit to use JPEG bundle)')
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args(argv)
    result=run_integrated_lab(bundle=args.bundle,model_path=args.runtime_model,
                              output_dir=args.output_dir,video=args.video)
    print(json.dumps({
        'status':result['status'],'eligible_for_promotion':False,
        'sample_frames':result['frames'],'raw_premerge_box_count':result['raw_box_count'],
        'candidate_count':result['candidate_count'],
        'source_frame_evidence':result['source_frame_evidence'],
        'output':str(args.output_dir),
        'note':result['notes'],
    },indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())