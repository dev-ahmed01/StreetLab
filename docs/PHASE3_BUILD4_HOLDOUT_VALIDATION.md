# Phase 3 Build 4/4 — Read-only integrity + preregistered holdout validation

**Status: Implemented, NOT field-validated, NOT promoted.** User's draft PR #10 and `codex/phase3-engine-shootout` remain experimental. Zero changes to production Geo-trax, original FLUID annotations, historical P3B tracking matches, T000 or earlier W04 artifacts.

## What this stage can—and cannot—prove

Build 4 consumes *completed Build 3 continuous video runs*, not sparse 21-frame detector CSVs. It verifies: original video and raw FLUID SHA256; Build 3 provenance and continuous frame-window counts; all original per-tile box file SHA; all 25 (or explicitly selected) candidate track SHA; real 14-column Geo-trax track rows; no duplicate `(frame, tracker_id)`; the frozen +1-frame/50px matching contract; independent-policy tracker counters and original scorecard copies. A missing or modified file **fails closed**. Run `inspect` without asking for new inference.

A policy is locked using development footage *before* comparison to a different video. The selection includes model SHA, source-video SHA and immutable policy config plus explicit numerical budgets. Build 4 compares the chosen policy to an **independently supplied, SHA-verified same-window T000 track file**. It independently recomputes pixel, class and identity scoring from the original FLUID CSV and original tracker rows (not just trusting saved score JSON). One continuous 120-frame minimum and at least 30 warmup frames are required. CPU stage-median budget is predeclared; it is NOT p95 end-to-end latency. Point precision, recall, **MOTORCYCLE correct-class recall**, car correct-class recall, identity fragmentation and contiguous switches are compared separately with frozen gates.

**A numeric pass is never a production approval.** The development footage W04 is tuned and fails the unseen-video SHA comparison if reused. Different hashes still cannot prove that a new dataset was held out before development. The 15 selected Build 2 review images are W04 tuning observations; they **cannot** establish an independent holdout's physical precision or full-frame recall. Until the new video has genuine two-person blind physical-object adjudication, independently documented holdout chronology, and separate measured p95 hardware timing, Build 4 ALWAYS sets both `eligible_for_tracking_promotion` and `eligible_for_production` to `false`, with explicit unmet requirements. No identity metrics are inferred from sparse frames or fabricated IDs.

## Workflow — one development run and one independent run, then automated comparison

Run these commands *after actual Build 3 batch runs exist*. Build 3 performs OpenVINO inference on **every actual video frame**, with 25 separate ByteTrack instances but a single shared detector inference. Without a completed `batch_report.json`, do not pretend Build 4 has measured long-run identity quality.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"

# 1. Verify a completed W04 Build 3 development tracking batch
& $python scripts/phase3_holdout_validation.py inspect `
  --batch-dir "artifacts/phase3/sahi_detector_trials/W04_box_tracking_batch201_01" `
  --output "artifacts/phase3/sahi_detector_trials/W04_box_tracking_integrity_01.json"

# 2. Explicitly choose and preregister ONE development policy;
# raw_unmerged shown as a NEUTRAL EXAMPLE, not a research winner.
& $python scripts/phase3_holdout_validation.py lock `
  --development-dir "artifacts/phase3/sahi_detector_trials/W04_box_tracking_batch201_01" `
  --policy raw_unmerged `
  --max-cpu-seconds-per-frame 1.0 `
  --output "artifacts/phase3/sahi_detector_trials/W04_predeclared_policy_01.json"

# 3. AFTER collecting completely separate unseen CCTV and real T000 baseline:
& $python scripts/phase3_holdout_validation.py validate `
  --policy-lock "artifacts/phase3/sahi_detector_trials/W04_predeclared_policy_01.json" `
  --holdout-dir "artifacts/phase3/sahi_detector_trials/HOLDOUT_continuous_batch_01" `
  --baseline-tracks "C:\PATH\TO\UNCHANGED_SAME_WINDOW_T000.txt" `
  --output "artifacts/phase3/sahi_detector_trials/HOLDOUT_validation_01.json"
```

For the holdout Build 3 batch, use a distinct source video's `--video` and its corresponding *unaltered* `--fluid-tracks`, `--start-frame`, `--end-frame`, `--fps`, and same-window `--baseline-tracks`. The OpenVINO model SHA must be unchanged from the preregistered W04 model. **No dataset change by editing labels or clipping only easy frames**. Keep the original unseen video private and archive immutable SHA manifests. The chosen policy may be included in a larger fixed 25-policy run, but Build 4 only evaluates the preregistered one.

## Explicit numerical gates

| Check | Development-selected holdout requirement |
|---|---|
| Holdout duration | At least 120 consecutive evaluation frames and 30 warmup frames |
| Different source evidence | Video SHA differs from W04; raw FLUID SHA differs from W04; exact same model SHA |
| Frame alignment | Original W04 +1 frame offset and 50px scoring radius, unchanged |
| Recall | Point recall no more than 1.0 percentage point lower than T000 |
| Precision | Point precision no more than 2.0 pp below T000; NOT physical precision |
| Motorcycle recall | Correct-class MOTORCYCLE recall must not regress against T000 |
| Car recall | Correct-class CAR recall no more than 1.0 pp lower |
| Identity | Fragmentation no more than 2.0 pp higher; zero additional contiguous ID switches |
| CPU | Sum of stage medians ≤ predeclared budget (default 1.0 second/frame); still require separate p95 |
| Physical preservation | Independent blind two-reviewer holdout audit still required |
| Chronology and generalization | Independently show holdout was unobserved during development |

These bounds are conservative **research gates**, not externally validated traffic-production SLOs. Empty, missing or nonfinite required metrics block a numerical pass. No auto-selected winners, no relabeling, no production merge.

## Local verification and limitations

CI executes synthetic tampering tests for modified original source hashes, matching-cohort T000 SHA, duplicate track IDs, changed scorecards, missing warmup, reused footage, motorcycle regression and absent reviews. Synthetic passing tests **do not** establish OpenVINO throughput, ByteTrack fragmentation or physical-object recall on genuine heldout footage. The final promotion decision remains separate and human-reviewed.