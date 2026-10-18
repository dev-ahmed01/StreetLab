# StreetLab — May 25 cross-date forensic replay and physical-review candidate queue

## Why this stage exists

The user's first fully completed May 25 historical FLUID regression (original video at **3840×2160, 10 FPS**) successfully decoded **150 frames**, evaluated **120 consecutive frames** and recorded:

| Measure | May 25 regression |
|---|---:|
| Spatial precision | 90.38% |
| Spatial recall | 92.49% |
| Spatial F1 | 91.42% |
| Contiguous identity switches | 21 |
| CAR correct-class recall | 95.99% (4,391 FLUID truth points) |
| MOTORCYCLE correct-class recall | 82.44% (2,785 FLUID truth points) |
| HEAVY_VEHICLE correct-class recall | **60.89%** (716 FLUID truth points) |
| BUS | **Not measurable** — zero FLUID BUS truth points |

This performance was obtained on a historical **regression reference**, not a truly unseen holdout, using the frozen May 26 W04 model/policy. It is not production or field-calibrated evidence. The heavy-vehicle discrepancy is meaningful as a *label-level failure category*: 672/716 heavy truth points have a spatial association, but only 436 are assigned the heavy class. Therefore **236 matched heavy truth points were classified differently**, and another 44 were spatially missed.

One BUS diagnostic subtlety: in \`class_diagnostics\`, BUS \`spatial_matches\` counts associations for **BUS ground truth** (zero here), while BUS \`spatial_precision\` counts **predicted BUS** objects matched to truth of *any* class. One predicted BUS matched non-BUS truth; 184 BUS predictions were unmatched. Neither observation proves these detections are physically false buses because FLUID labels can be incomplete. The manifest must preserve that ambiguity.

May 26 W04 development F1 was ~88.97% on a **different 201-frame** sample. The May 25 F1 difference is descriptive; it is not a paired or controlled estimate of improved site generalization. Raw ID-switch totals must not be treated as directly comparable quality metrics.

## One-command Windows forensic audit

After \`git pull --ff-only origin main\`:

\`\`\`powershell
cd C:\Users\Admin\Desktop\StreetLab-final
.\scripts\STREETLAB_W04_MAY25_FORENSICS.ps1
\`\`\`

The launcher discovers the latest existing \`artifacts\w04_may25_crossdate_*\` folder containing \`cross_date_summary.json\`, plus the original \`artifacts\w04_policy_lock_*\` folder. Both are re-verified; it does not need another OpenVINO inference pass, and it does not edit May 25 or May 26 MP4/FLUID files.

### Checks before accepting forensic output

- The preflight, original saved batch, regression score and SHA-bound W04 policy lock identify exactly one unmodified \`hard_nms_ios_0.30\` candidate, original May 25 source/video hashes, fixed video 60–179 / FLUID 61–180 frames and 30 warmup frames.
- The existing Phase 3 \`verify_batch\` rehashes the original May 25 video and FLUID CSV, original raw boxes and tracking file and saved scorecard. This may read the entire original MP4 once, but performs **zero detector inference**.
- The original \`cached_window_scorer\` independently recalculates pixel, identity and class diagnostics from the saved track file and original FLUID labels using frozen +1 frame / 50px Hungarian matching; all saved source metrics and the summary must match.
- The separate human-review candidate queue is **not an independent physical-object evaluation**. It ranks at most 18 original source frames and 220 error events by fixed weights on label-level HEAVY class disagreements, unmatched HEAVY truth, BUS predictions associated to another label, unmatched BUS predictions, and MOTORCYCLE misses.

### Produced files

A unique non-overwriting \`artifacts\w04_may25_forensics_*/\` directory contains:

1. \`regression_forensics.json\`: reproducible historical May 25 score replay, original source hashes, W04-development-versus-May25 comparison (explicitly **unpaired/descriptive**), per-class recall differences, BUS ambiguity and the ranked frame manifest, plus stringent no-blind/no-production flags.
2. \`review_events.csv\`: reproducible high-priority label-level review queue with absolute source \`video_frame\`, corresponding \`fluid_frame\`, truth and predicted labels, original x/y centers, original track identities and distances when spatially paired.

Physical-object adjudication remains absent. A reviewer must inspect the actual unchanged original video frames, preferably independently by two people, before calling any annotation a true label or algorithmic misclassification. If changing video labels is contemplated, create an **independent reviewer layer**, not edits to the original FLUID file.

You can provide exact source folders when defaults differ:

\`\`\`powershell
.\scripts\STREETLAB_W04_MAY25_FORENSICS.ps1 -RunDir 'C:\Users\Admin\Desktop\StreetLab-final\artifacts\w04_may25_crossdate_YYYYMMDD_HHMMSS_mmm' -ProtocolFolder 'C:\Users\Admin\Desktop\StreetLab-final\artifacts\w04_policy_lock_YYYYMMDD_HHMMSS_mmm'
\`\`\`

If the original source is missing or any re-evaluated score disagrees, this stage should **fail**, not silently recalculate the acceptance claims or retune the W04 selected policy.

## Remaining research boundaries

- **No matched original T000** baseline exists.
- **End-to-end CPU p95** is not reported (the May 25 detector median was ~1.4s/frame; adding medians does not yield an end-to-end tail latency).
- Zero FLUID BUS truth points mean bus correctness cannot be estimated from this interval.
- May 25 and May 26 have different historical source roles, frame intervals, scene populations and class mixes.
- **Chennai independent generalization is failed** and the native SUMO research junction demonstration remains **synthetic-only**.
- **Production release remains forbidden** without new unseen source, independent labels, matched baseline and explicit field approval.
