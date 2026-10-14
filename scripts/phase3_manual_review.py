"""Build blind two-reviewer cases or adjudicate two independently filled sheets.

Never edits frozen FLUID annotations, historical scores or production models.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from streetlab_phase3.video.manual_review_protocol import generate_review_pack,adjudicate_pair,evaluate_adjudicated_candidates

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare',help='Create blind 15-case source-video review folder')
    a.add_argument('--bundle',required=True,type=Path)
    a.add_argument('--output-dir',required=True,type=Path)
    a.add_argument('--crop-size',type=int,default=480)
    b=s.add_parser('adjudicate',help='Compare two independent completed reviewer CSV sheets')
    b.add_argument('--review-dir',required=True,type=Path)
    b.add_argument('--review-a',required=True,type=Path)
    b.add_argument('--review-b',required=True,type=Path)
    b.add_argument('--output',required=True,type=Path)
    c=s.add_parser('compare',help='Compare all 25 box candidates against accepted case-level review')
    c.add_argument('--review-dir',required=True,type=Path)
    c.add_argument('--consensus',required=True,type=Path)
    c.add_argument('--box-lab-dir',required=True,type=Path)
    c.add_argument('--output',required=True,type=Path)
    args=p.parse_args(argv)
    if args.action=='prepare':
        r=generate_review_pack(args.bundle,args.output_dir,crop_size=args.crop_size)
        print(json.dumps({'status':r['status'],'cases':r['case_count'],
                          'output_dir':str(args.output_dir),
                          'review_files':['reviewer_A_TEMPLATE.csv','reviewer_B_TEMPLATE.csv'],
                          'instruction':r['reviewer_instruction']},indent=2))
    elif args.action=='adjudicate':
        r=adjudicate_pair(args.review_dir,args.review_a,args.review_b,args.output)
        print(json.dumps({'status':r['status'],'decisions':r['decisions'],
                          'eligible_for_promotion':False,'output':str(args.output),
                          'caution':r['caution']},indent=2))
    if args.action=='compare':
        r=evaluate_adjudicated_candidates(pack_dir=args.review_dir,
            consensus_file=args.consensus,box_lab_dir=args.box_lab_dir,
            output=args.output)
        print(json.dumps({'status':r['status'],'candidates':r['candidate_count'],
                          'output':str(args.output),
                          'physical_precision':None,'full_frame_recall':None},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())