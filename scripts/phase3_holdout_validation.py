"""Build 4/4: verify continuous tracking evidence; lock development policy; audit holdout.

All outputs are new files, evidence only, and cannot promote the production
Geo-trax engine. Run the holdout BEFORE opening candidate metrics if possible;
independent proof of that chronology must be retained externally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from streetlab_phase3.video.holdout_validation import (
    lock_policy, validate_holdout, verify_batch)


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    inspect=sub.add_parser('inspect',help='SHA-check all Build 3 candidate tracks and reports')
    inspect.add_argument('--batch-dir',type=Path,required=True)
    inspect.add_argument('--output',type=Path,required=True)
    lock=sub.add_parser('lock',help='Predeclare a W04 development policy before unseen holdout')
    lock.add_argument('--development-dir',type=Path,required=True)
    lock.add_argument('--policy',required=True)
    lock.add_argument('--max-cpu-seconds-per-frame',type=float,default=1.)
    lock.add_argument('--output',type=Path,required=True)
    held=sub.add_parser('validate',help='Fail-closed separate-camera/footage check')
    held.add_argument('--policy-lock',type=Path,required=True)
    held.add_argument('--holdout-dir',type=Path,required=True)
    held.add_argument('--baseline-tracks',type=Path,required=True,
                      help='Unchanged same-window T000 tracks, SHA-checked and rescored')
    held.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.mode=='inspect':
        source=verify_batch(args.batch_dir)
        value={'status':'BUILD4_VERIFIED_BATCH_NOT_PRODUCTION',
               'eligible_for_production':False,
               'batch_report_sha256':source['batch_sha256'],
               'original_source_sha256_verified':source['original_source_sha256_verified'],
               'policy_count':source['policy_count'],
               'evaluated_frames':source['report']['evaluated_frames'],
               'has_same_window_baseline':source['report']['same_window_T000'] is not None,
               'note':'Data integrity only, not validation or production approval.'}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:
            json.dump(value,f,indent=2)
    elif args.mode=='lock':
        value=lock_policy(args.development_dir,args.policy,args.output,
                          max_cpu_seconds_per_frame=args.max_cpu_seconds_per_frame)
    else:
        value=validate_holdout(args.policy_lock,args.holdout_dir,args.output,
                               baseline_tracks=args.baseline_tracks)
    print(json.dumps(value,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())