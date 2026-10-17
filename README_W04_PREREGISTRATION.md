# W04 prospective holdout preregistration (after verified development replay)

**Status:** engineering policy-lock workflow, not blind validation or a production release. The historical **May 26 / W04 201-frame scores were tuned/development metrics**. Source and model SHA verification plus all 25 successfully reproduced FLUID-label scorecards allow one policy to be documented for a later, truly untouched test.

## Policy selected on *development* evidence

`hard_nms_ios_0.30` is pinned as the provisional maximum development spatial F1 choice, using contiguous identity-switch count to break ties. In the saved W04 cohort, the policy reports ~86.70% spatial precision, ~91.37% recall, ~88.97% F1 and **71** contiguous ID switches. It does not dominate all per-class recall criteria. Relative to unsuppressed raw boxes, the documented frozen study records losses in MOTORCYCLE and HEAVY_VEHICLE correct-class recall; the 201-frame cohort has no BUS truth points, so BUS recall is **not estimable** here.

The preregistration step binds the user's local `w04_score_replay_*.json` to the original 25-policy `batch_report.json` by its SHA-256, each policy's track SHA and numerical score, and the source FLUID SHA. It uses the existing authoritative `lock_policy` from Phase 3 Build 4 rather than changing holdout acceptance thresholds. The current manifest's `held_out_validation` role for May 26 does **not** make its already-tuned W04 development window untouched. The Jan 20 and May 25 cohorts have other assigned roles and are not silently claimed as blind.

## Windows command

```powershell
cd C:\Users\Admin\Desktop\StreetLab-final
git pull --ff-only origin main
$batch = 'C:\Users\Admin\Desktop\StreetLab-engine-trial\artifacts\phase3\sahi_detector_trials\W04_box_tracking_batch201_01'
$replay = 'C:\Users\Admin\Desktop\StreetLab-final\artifacts\w04_score_replay_20261018_032008_446.json'
.\scripts\STREETLAB_W04_PREREGISTER.ps1 -BatchDir $batch -Replay $replay -MaxCpuSecondsPerFrame 2.0
```

The proposed 2.0 s/frame budget is a **prospective research acceptance condition**, not an observed p95 or real-time target. The published frozen W04 detector median is ~1.245s/frame and the existing Build 4 validator only verifies a sum of stage medians; end-to-end measured CPU p95 is separately required. The script rejects a proposed budget already lower than the source's measured sum of stage medians, and never asserts that a budget is achieved on the future holdout.

A new `artifacts/w04_policy_lock_*/` contains:

- `policy_lock.json` — unchanged Phase 3 Build 4 lock contract for subsequent `validate` use.
- `holdout_protocol.json` — all W04 candidate F1s and per-class recall trade-offs bound to the frozen replay, study roles, duplicated track-byte groups, prospective gates, and explicit missing evidence.

Output directories are never reused. A failed integrity check refuses to write a policy-lock directory.

## What must happen before honest independent validation

1. Document an **untouched, independent source** before opening its results, with a distinct video and matching FLUID/human labels. A new time slice of the tuned May 26 recording does not by itself establish independence. The historical manifest does **not** establish unseen status for the Jan 20 or May 25 datasets.
2. Run the pinned detector/tracker on a continuous holdout window without policy retuning, preserving original source frame number, hashes, warmup, raw boxes and track files.
3. Obtain real same-window T000 baseline tracks where available. If missing, create a **clearly labeled new replacement baseline** rather than inventing a historical recovered T000.
4. Use `scripts/phase3_holdout_validation.py validate --policy-lock ... --holdout-dir ... --baseline-tracks ... --output ...` to execute the existing fail-closed metric and provenance gates.
5. Separately review physical objects with **two independent reviewers**, especially MOTORCYCLE, HEAVY_VEHICLE and BUS, and measure end-to-end CPU p95. Only independent audit can support a blind holdout accuracy claim.

Even passing the numeric checks does not certify a citywide calibrated traffic model or establish real-road intervention effects. The Chennai held-out generalization failure remains frozen.
