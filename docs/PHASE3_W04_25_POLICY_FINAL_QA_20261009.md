# W04 final 25-policy evidence QA (9 October 2026)

**Completed development milestone, not a production promotion.** User-uploaded `W04_FINAL_RESULTS.zip` was subjected to a separate stdlib-only SHA/CSV/score-consistency audit after the local Windows trial.

## Evidence integrity
- Exactly **32** archive members; 25/25 per-candidate CSV SHA-256 checks passed, as did hashes of `matrix_report.json`, `reviewed_candidate_comparison.json`, `cached_review.json`, and the 1,609 raw boxes JSONL. Report/manifest cross-references, 21 exact frames, candidate row counts, per-class predicted counts and score arithmetic all reconcile.
- Archived result SHA-256: `a42f0677a05fb979974d81c2368f49e560b4000d145aab3ceb35a17738fb12aa`.
- Explicit negative test: a modified candidate CSV was rejected as `Candidate SHA mismatch: hard_nms_ios_0.30`.
- Real-video sampled-frame capture on the user workstation: 10750 through 11350 in increments of 30. This is not continuous tracking.
- Each candidate covers 13/13 manually accepted cases with any predicted center and 12/13 with a correct-class center. The remaining case is AUTO_RICKSHAW, outside the four-class model ontology.
- **Audit boundary:** original FLUID CSV, OpenVINO model and source MP4 were not included in the uploaded results ZIP; therefore checks prove internal integrity/consistency, not fresh independent inference or original FLUID metric replay.

## Interpretation and development shortlist
- **Provisional tracking lead: `hard_nms_ios_0.30`**, 86.87% FLUID-label precision, 90.39% recall, retains 5/5 frozen heavy-vehicle matches and 319/380 motorcycle matches.
- **Numerical aggregate precision leader: `weighted_fusion_ios_0.30`**, 86.98% precision at equal aggregate recall, but retains only **4/5** frozen heavy-vehicle matches. This is not evidence of physical false-negative safety or generalization.
- **Recall-sensitive alternative: `hard_nms_iou_0.30`**, 82.31% precision, 90.66% recall, 320/380 motorcycle, 5/5 heavy-vehicle matches.
- **Unsuppressed control: `raw_unmerged`**, 52.05% label precision, 90.93% recall. The IoS hard-NMS option cuts 522 vehicle detections from the 1,291 scored by FLUID, with four fewer matches (a 0.54 percentage-point recall reduction). The implied label precision improvement does not prove a physical precision improvement.
- Multiple candidate centers inside the same manual human box drop from 11/13 targets (`raw_unmerged`) to 2/13 targets (hard NMS IoS and weighted fusion IoS 0.30). Multiple centers in one review box alone **do not prove duplicated physical boxes**.
- The two human-agreed BUS objects are particularly important because there are **zero** BUS truth points in this particular frozen FLUID cohort. Do not suppress physical buses based solely on label-based precision.
- Median tiled OpenVINO inference measured across the sampled frames was **2.364 seconds/frame**, including an 11.78-second first-frame cold start. It cannot establish end-to-end tracking FPS or p95 on continuous footage.

## All 25 policies
All precision/recall figures below are *frozen FLUID label* metrics, not physical precision or tracking quality.

| Policy | Precision | Recall | FLUID matches/739 | Motorcycle matches/380 | Heavy matches/5 | All-class output boxes | Reviewed cases with >1 candidate center |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hard_nms_ios_0.30` | 86.87% | 90.39% | 668 | 319 | 5 | 969 | 2/13 |
| `hard_nms_ios_0.50` | 86.64% | 90.39% | 668 | 319 | 5 | 973 | 2/13 |
| `hard_nms_ios_0.70` | 85.53% | 90.39% | 668 | 319 | 5 | 987 | 2/13 |
| `hard_nms_iou_0.30` | 82.31% | 90.66% | 670 | 320 | 5 | 1018 | 3/13 |
| `hard_nms_iou_0.50` | 78.18% | 90.66% | 670 | 320 | 5 | 1071 | 5/13 |
| `hard_nms_iou_0.70` | 72.28% | 90.66% | 670 | 320 | 5 | 1189 | 6/13 |
| `raw_unmerged` | 52.05% | 90.93% | 672 | 322 | 5 | 1609 | 11/13 |
| `soft_gaussian_ios_0.30` | 84.68% | 90.53% | 669 | 319 | 5 | 990 | 3/13 |
| `soft_gaussian_ios_0.50` | 84.68% | 90.53% | 669 | 319 | 5 | 990 | 3/13 |
| `soft_gaussian_ios_0.70` | 84.68% | 90.53% | 669 | 319 | 5 | 990 | 3/13 |
| `soft_gaussian_iou_0.30` | 70.53% | 90.66% | 670 | 320 | 5 | 1157 | 6/13 |
| `soft_gaussian_iou_0.50` | 70.53% | 90.66% | 670 | 320 | 5 | 1157 | 6/13 |
| `soft_gaussian_iou_0.70` | 70.53% | 90.66% | 670 | 320 | 5 | 1157 | 6/13 |
| `soft_linear_ios_0.30` | 86.3% | 90.39% | 668 | 319 | 5 | 974 | 2/13 |
| `soft_linear_ios_0.50` | 86.19% | 90.39% | 668 | 319 | 5 | 977 | 2/13 |
| `soft_linear_ios_0.70` | 85.31% | 90.39% | 668 | 319 | 5 | 989 | 2/13 |
| `soft_linear_iou_0.30` | 73.22% | 90.66% | 670 | 320 | 5 | 1120 | 6/13 |
| `soft_linear_iou_0.50` | 72.75% | 90.66% | 670 | 320 | 5 | 1136 | 6/13 |
| `soft_linear_iou_0.70` | 70.16% | 90.66% | 670 | 320 | 5 | 1217 | 6/13 |
| `weighted_fusion_ios_0.30` | 86.98% | 90.39% | 668 | 319 | 4 | 968 | 2/13 |
| `weighted_fusion_ios_0.50` | 86.87% | 90.39% | 668 | 319 | 4 | 971 | 2/13 |
| `weighted_fusion_ios_0.70` | 85.31% | 90.39% | 668 | 319 | 4 | 988 | 2/13 |
| `weighted_fusion_iou_0.30` | 82.11% | 90.66% | 670 | 320 | 5 | 1020 | 3/13 |
| `weighted_fusion_iou_0.50` | 78.45% | 90.66% | 670 | 320 | 5 | 1068 | 5/13 |
| `weighted_fusion_iou_0.70` | 72.28% | 90.66% | 670 | 320 | 5 | 1189 | 6/13 |

## Required next evidence
1. Execute the existing Build 3 201-consecutive-frame trial, plus 90 prior warmup frames, through one shared OpenVINO inference stream and separate ByteTrack instances. Evaluate all 25 policies to avoid premature selection. The frozen May-26 W04 dataset remains development-only.
2. Obtain a genuinely same-window, unchanged T000 baseline track file and compare separate point recall/precision, MOTORCYCLE/CAR correct-class recall, fragmentation, ID switches and CPU stage medians. Build 4 will reject mismatched source or frame cohorts.
3. Preregister a single eligible policy *before* genuinely unseen source-video validation; preserve independent physical review and measure real p95 if production readiness is later sought.

**No winner locked, no original FLUID labels edited, no PR merge, and no production Geo-trax modification.**