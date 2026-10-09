"""Compare actual W04 cached detector centers with 13 independent reviewed cases."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from streetlab_phase3.video.reviewed_cached_center_bridge import compare_reviewed_centers

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--review-dir',type=Path,required=True)
    p.add_argument('--consensus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    result=compare_reviewed_centers(bundle=a.bundle,review_dir=a.review_dir,
                                    consensus_file=a.consensus)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],
                      'case_level_center_coverage_by_backend':result['case_level_center_coverage_by_backend'],
                      'output':str(a.output),'eligible_for_promotion':False},indent=2))

if __name__=='__main__':main()