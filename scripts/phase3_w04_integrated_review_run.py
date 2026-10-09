"""One-shot W04 human-reviewed OpenVINO 25-policy box benchmark.

Pass --cached-only to reproduce the 13 reviewed-case cached A/B comparison
without OpenVINO. Omit it to run actual per-tile OpenVINO inference.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from streetlab_phase3.video.w04_integrated_review_run import run_w04_reviewed_box_batch

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',required=True,type=Path)
    ap.add_argument('--review-dir',required=True,type=Path)
    ap.add_argument('--consensus',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--video',type=Path,help='Prefer pixel-exact original source video')
    ap.add_argument('--model-dir',type=Path,help='REQUIRED for real inference: complete SHA-verified original OpenVINO export dir (the ZIP contains only XML/BIN)')
    ap.add_argument('--cached-only',action='store_true')
    a=ap.parse_args(argv)
    result=run_w04_reviewed_box_batch(bundle=a.bundle,review_dir=a.review_dir,
       consensus_file=a.consensus,output_dir=a.output_dir,video=a.video,
       model_dir=a.model_dir,cached_only=a.cached_only)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()