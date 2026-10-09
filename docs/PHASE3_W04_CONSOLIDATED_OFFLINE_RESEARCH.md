# StreetLab W04 — consolidated offline experimental conclusions (9 Oct 2026)

**Development research ONLY. No production merge, no frozen-FLUID changes, no T000 superiority claim, no independent held-out validation.**

## Completed offline investigation

The user supplied the complete real W04 continuous benchmark ZIP. A separate container run verified **all 25 policy track hashes** plus the **21,104 raw per-tile box** SHA from the original manifest; the source contained 291 consecutive frames (10660–10950), with 90 warmup + 201 evaluation (10750–10950).

The analysis examined:
- All **25 frozen original ByteTrack policies** using their actual saved, already FLUID-scored metrics.
- **126 class-stabilization counterfactuals** across 7 real track policies: hysteresis, rolling votes, noncausal whole-ID majority and raw-detector-supported causal smoothing.
- **5 geometric merge variants × 4 classes × all 291 detector frames**, including tile-provenance, center-proximity and IoU gates.
- **63 conservative spatial tracklet-linking conditions** across 7 tracker policies.

An independently reimplemented Hard NMS IoS 0.30 reproduced the exact original **6,916 vehicle detections in the 201-frame evaluation period**. This validates the detector-box comparator's native baseline count. These newly designed variants were NOT run through real ByteTrack or the original full FLUID truth scorer in this assistant container: the original 4K video/full truth CSV and native `trackers` runtime were unavailable. Do **not** present new geometric retention or label-flip proxy numbers as new physical precision, true class recall, identity or BUS recall.

## Frozen measured trade-off

| Existing full-run policy | FLUID precision | Recall | MOTORCYCLE correct-class recall | HEAVY correct-class recall | Contiguous identity switches |
|---|---:|---:|---:|---:|---:|
| `hard_nms_ios_0.30` | **86.70%** | 91.37% | 83.32% | 79.75% | **71** |
| `hard_nms_iou_0.50` | 79.83% | 91.80% | 83.81% | **84.81%** | 132 |
| `raw_unmerged` | 55.74% | 92.95% | **85.54%** | 84.18% | 908 |

**No original suppression policy wins on class preservation, precision and identity.** This window contains zero BUS FLUID truth points; physical human reviews elsewhere in W04 document BUS annotation disagreements.

## Recommended engineering candidates

**A. Hybrid rare-vehicle geometric suppression — PRIMARY**: retain `hard_nms_ios_0.30` only for CAR/BUS (and other nonrare ontology classes), but apply **Hard NMS IoU 0.50** to MOTORCYCLE and HEAVY_VEHICLE. Against the frozen all-IoS merger, the actual raw-box counterfactual retained **293 more motorcycle** and **102 more heavy detections** across 291 frames, including **27 more high-confidence MOTORCYCLE** and **80 more high-confidence HEAVY** detections (confidence >= .5). This yielded **248 extra detector boxes over the 201 scored frames**. These could be both missed objects *and* duplicates; only paired tracker/FLUID/physical review can decide. See experimental `streetlab_phase3/video/w04_hybrid_rare_suppression.py` and synthetic guard tests.

**B. Center-distance guarded IoS for rare classes — SECONDARY**: for MOTORCYCLE/HEAVY, preserve overlapping candidates unless their center distance is <=0.35×minimum box dimension as well as IoS >=.30. This retained **379 more MOTORCYCLE and 115 more HEAVY** boxes over 291 frames (+296 201-frame detector boxes). Same original CAR/BUS handling. More permissive cross-tile+center version retained +510 201-frame boxes: useful high-recall stress control, too aggressive for provisional default.

**C. Non-destructive class-evidence stream — COMPANION**: causal 2-observation hysteresis constrained by same-frame raw detector support (IoU >=.5), while preserving *original* track IDs, XYXY boxes, and instantaneous classes in immutable evidence. On `hard_nms_ios_0.30`, the study reduced **100 observed class changes to 59**, modifying only **44/6,757** emitted class observations, all with overlapping raw-detector support for the proposed label. This is **class-stability proxy only**, not true class accuracy. Experimental `streetlab_phase3/video/w04_class_evidence_stream.py` maintains separate metadata, never silently rewrites the 14-column scored tracks.

## Rejected or deprioritized approaches

- **Unconstrained class smoothing**: 2-frame hysteresis reduced 100 to 46 changes but changed 77 labels, **33 with no corresponding raw class hypothesis** at IoU>=.5. Longer smoothing and whole-track majority improve the visual smoothness metric by forcibly changing ungrounded frame classes.
- **Majority class per ID**: zero class flips by definition, but 158 relabeled rows and 67 without same-frame raw support. It also requires future observations, making it noncausal. **Reject as proof of semantic correction.**
- **Four independent class-specific trackers as a final fix**: zero within-ID class flips by architecture, but a genuinely single motorcycle may become multiple IDs when detector class flickers. Good diagnostic ablation, not proof of superior physical identity.
- **Spatial stitching**: the lead policy's 119 native IDs yielded at most **one conservative mutual-unique merge** under 9 tested gap/distance combinations. Too little supported upside to justify substantial false-merging risk. 63 scenarios were evaluated across seven policies.
- **T000 original recovery**: final forensic inventory found original artifacts unavailable in searched paths. Any subsequent extraction must be transparently named a **new same-window replacement baseline**, never the recovered original experiment.

## Release-quality gates (not yet met)

1. Native tracker pass across the two hybrid geometric candidates and original matched baselines using the same cached 291 frames, exact FLUID CSV, +1 frame/50px, per-class motorcycle/heavy/car recall, identity switches, fragmentation, detector-class ambiguity and CPU timing. **One consolidated benchmark**, not individual small iterations.
2. Genuinely unseen other-source validation, separate two-human physical object adjudication including BUS/AUTO_RICKSHAW ontology exceptions, and source-aligned replacement Geo-trax baseline if original cannot be recovered.
3. CPU p95 and full end-to-end timing. Tiled OpenVINO had measured median **1.245 s/source 4K frame**, making detector inference the clear throughput bottleneck—not millisecond NMS.

The concrete source code and synthetic tests of the two experimental components are on this draft branch; no production code or original evidence is overwritten. Keep PR #10 draft.
