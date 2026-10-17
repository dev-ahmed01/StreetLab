# StreetLab W04 — Original-label score replay (development only)

**Stage:** original SHA-bound source and 25 saved continuous-tracking policy results → independent code-level recomputation from recorded FLUID trajectories. **It is not independent held-out validation.**

## Source boundaries

The existing May 26 W04 study (source frames **10660–10950**, warmup 90, scored **10750–10950**) was used to choose/tune candidate policies. Results in this cohort are development-only even if every score recomputes identically. The raw original MP4, FLUID truth, frozen OpenVINO model and full tracking batch are **local-only**, never to be committed to GitHub.

Previous user-local audit already established original video/model/truth SHA-256 consistency with their local provenance. This replay goes further by checking that the saved batch and all named track files/scorecards match their hashes, and then rescoring **every frozen policy track file** against the original FLUID labels using the existing `cached_window_scorer` (fixed +1 frame alignment and 50 pixel gate). It independently checks previously saved precision, recall, identity, and class-specific score fields; any disagreement fails without writing an accepted replay report. It reuses recorded tracker outputs and does **not** run detector inference.

## Windows command

After updating `main`:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-final
git pull --ff-only origin main
$batch = 'C:\Users\Admin\Desktop\StreetLab-engine-trial\artifacts\phase3\sahi_detector_trials\W04_box_tracking_batch201_01'
$readiness = 'C:\Users\Admin\Desktop\StreetLab-final\artifacts\w04_readiness_20261018_030532_026.json'
.\scripts\STREETLAB_W04_SCORE_REPLAY.ps1 -BatchDir $batch -Readiness $readiness
```

The launcher can infer the latest `w04_readiness_*.json` if `-Readiness` is omitted, but explicitly supplying the matching verified receipt is more reproducible. The output is a fresh `artifacts/w04_score_replay_*.json`; it does not overwrite prior evidence. It may take time to check 25 complete 201-frame tracker files and label matches; it does not repeat 4K inference.

If the historical batch directory lacks a saved original track or scorecard, if an older recorded FLUID path has moved, or if any hash/metric differs, the command should fail. Restore/correct access to original source bytes, **not** the frozen files or score numbers. Do not bypass the prior checks to obtain a pass.

## Next independent evaluation

After a successful W04 development replay:

1. Decide and preregister **one** frozen policy using `scripts/phase3_holdout_validation.py lock`, preserving a genuinely predeclared CPU budget and the evaluation/ontology assumptions.
2. Choose a **different untouched source video/camera** and matching original FLUID truth. A different time window of the already tuned May 26 video does not become blind by selecting it after examining results. The May 25 study is manifest-designated `regression_reference` and must not be mislabeled unseen without proof it remained unexamined.
3. Run continuous tracking with identical settings and independent same-window T000/reference baseline (or transparently named replacement), then invoke existing Build 4 `validate` with its frozen comparison and integrity checks.
4. Include independent two-reviewer physical-object adjudication for class-label discrepancies and BUS/rare-class behavior, end-to-end measured p95, and site-transfer limits. Never promote this development replay as field-calibrated prediction or production acceptance.

The original Chennai held-out generalization remains failed. This replay can strengthen reproducibility of historical FLUID development scores, not change that separate frozen outcome.
