# W04 — T000 recovery outcome and next experiment

9 October 2026. Research only; original data is unchanged. Draft PR #10 remains unmerged.

## Recovery outcome

The user-provided `W04_T000_FORENSIC_INVENTORY_01.json` was read and checked. Both `StreetLab` and `StreetLab-engine-trial` exist on Desktop. All six specifically tested historic Geo-trax T000 output, experiment and summary locations were **missing**. The scan visited 299 filesystem entries, reported four matched JSON files, no scan errors, and no truncation. The four matches were two earlier candidate-inventory reports and two unrelated Sprint 1E simulation baseline reports. No genuine source-aligned T000 track output was recovered.

Decision: T000 remains **unavailable in the scanned workspace**. Do not compare an older full-video score with 201-frame W04 performance. A later newly executed Geo-trax control would be a **replacement baseline**, requiring recorded model/version/settings, unchanged source SHA, absolute frame convention, exact 10750-10950 source window, +1 FLUID offset and 50-pixel matcher.

## Next: cached class-partitioned tracking

The runner `scripts/phase3_cached_class_partition_ablation.py` and PowerShell launcher `scripts/RUN_W04_CLASS_PARTITION_ABLATION.ps1` are committed for a controlled offline experiment. They verify the earlier Build 3 files and original source hashes, then reuse the 21,104 original cached detector boxes over frames 10660-10950. **No video decoding or OpenVINO inference is needed**.

Four preset policy configurations are compared: `hard_nms_ios_0.30`, `hard_nms_ios_0.50`, `hard_nms_iou_0.30`, and `raw_unmerged`. For each, four independent ByteTrack trackers are used, one per canonical vehicle class. Tracker updates occur once per frame, including empty and warmup frames. The original postprocessing policies, detector classes and all prior output files remain unchanged. A new immutable output directory receives 14-column tracks, SHA256, frozen +1/50 pixel and identity scorecards, per-class diagnostics and median timing.

Class changes for a single new tracker ID are impossible by construction. This is not a performance improvement on its own: class-partitioning can worsen genuine track fragmentation and rare-class recall. Judge overall precision and recall, motorcycle/heavy vehicle correct-class recall, identity switches and truth-track fragmentation against each corresponding original policy. Never overwrite FLUID labels or count zero class flips as proof of a better tracker.

## Windows execution

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git switch codex/phase3-engine-shootout
git pull --ff-only origin codex/phase3-engine-shootout
& ".\scripts\RUN_W04_CLASS_PARTITION_ABLATION.ps1"
```

Output: `artifacts\phase3\sahi_detector_trials\W04_cached_class_partition_ablation_01`.

The code and synthetic tests are committed, but no real Windows replay has yet been run in this chat turn. No claimed class-aware numerical improvement, no T000 comparison and no deployment authorization. Further gates remain source-aligned replacement baseline, genuinely new holdout, independently reviewed physical vehicle classes and p95 CPU timing.