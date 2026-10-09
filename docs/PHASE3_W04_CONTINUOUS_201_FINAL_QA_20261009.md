# W04 — 201-frame continuous OpenVINO/ByteTrack QA (9 October 2026)

**Status: completed real-video *development* experiment; NO T000-aligned comparison, NO held-out validation, NO promotion.** This report records an independently inspected user-supplied `W04_CONTINUOUS_201_RESULTS.zip` archive, not synthetic CI evidence.

## Evidence and verification

- Archive SHA-256: `da5ee55a97ea1ba4ec794e1b61af4bbd816a6a9b9eaf20e3ac5aeb5a47b0cd0b` (53 members).
- Original 4K source 10660–10950: **291 consecutive decoded frames**, 90 warmup + 201 evaluation (10750–10950), **21,104** raw pre-global-merge boxes, **25 independent ByteTrack policy streams** fed by one shared per-frame tiled detector stream.
- **25/25 track SHA checks**, **25/25 policy JSON ↔ batch JSON equality checks**, raw-box SHA, original source/model SHA provenance *as reported*, every track file's 14-column row shape, no duplicate (frame, tracker ID), metric count/recomputed precision and recall, and +1-frame/50-pixel rule all independently checked. One tampered track file failed SHA validation as expected.
- Boundary: original video, entire FLUID source and original model were NOT uploaded. The audit confirms artifact consistency and reported source provenance, **not** independently rerun model inference or independently recomputed FLUID matches from the original truth. The local Build 4 `inspect` command is needed for stronger original-source hash verification.
- Source video SHA: `57105b564ff3f9c68d88e6df05790654a49a48f626b1adaf39b62ea61c262ce0`; FLUID SHA: `0c772f131b1cf137d87405d460f9c8b01b935b79ba62837111ffd4fa3ef2e70c`; OpenVINO model tree SHA: `feff81d24b5f7790d8b5ca34ec9d4641b3513ba87962ced52c7060491b0212ea`.

## Leading policies on W04 (frozen FLUID-label numbers; not physical precision)

| Policy | Point precision | Point recall | Motorcycle correct-class recall | Heavy-vehicle correct-class recall | Contiguous ID switches | Fragmented FLUID tracks /71 |
|---|---:|---:|---:|---:|---:|---:|
| `hard_nms_ios_0.30` | **86.70%** | 91.37% | 83.32% | 79.75% | **71** | **15** |
| `hard_nms_ios_0.50` | 86.68% | 91.37% | 83.32% | 79.75% | **71** | **15** |
| `weighted_fusion_ios_0.30` | 86.31% | 91.05% | 83.25% | 79.75% | 76 | **15** |
| `hard_nms_iou_0.30` | 83.64% | 91.58% | 83.55% | 79.11% | 99 | 23 |
| `soft_gaussian_ios_0.30` | 85.84% | 91.41% | 83.25% | 79.11% | 78 | 18 |
| `raw_unmerged` | 55.74% | **92.95%** | **85.54%** | **84.18%** | 908 | 55 |

**Conclusion:** `hard_nms_ios_0.30` is the **provisional W04 development leader** (best point F1, 88.97%, tied minimum contiguous switches). Versus unmerged raw boxes, it reduces 908 to 71 contiguous switches (~92%) and 55 to 15 fragmented FLUID truth tracks. However, recall falls ~1.58 percentage points overall, ~2.22 pp for correct-class MOTORCYCLE and ~4.43 pp for correct-class HEAVY_VEHICLE. No production decision is justified until class preservation is independently audited and T000 is compared on the **same window**.

There are **zero BUS ground-truth points** in this 201-frame FLUID cohort, so it cannot verify BUS recall; earlier W04 dual-review cases document real buses misclassified in FLUID elsewhere. No physical recall/precision or automatically repaired FLUID labels are claimed.

## Additional class identity instability

Track IDs may change assigned vehicle class over time (different concept from FLUID-matched ID switches). For `hard_nms_ios_0.30`, **27 of 119** distinct emitted tracker IDs took multiple class IDs, with **100** observed class transitions, of which **80** were between adjacent observed frames. Raw unmerged: **60 of 308** IDs, **183** class transitions. This may be detector class flicker or cross-class associations and is **not yet causally isolated**. Review physical examples and consider cached-box offline class-association ablations rather than treating a stable tracker ID as a stable physical vehicle category.

## CPU budget and missing gates

Shared tiled OpenVINO detector median: **1.245 seconds/source frame** (not real-time 30 FPS); `hard_nms_ios_0.30` box postprocess median ~1.097 ms and tracker update median ~4.528 ms. These are stage medians, **not** end-to-end throughput or p95 latency.

`batch_report.json.same_window_T000 = null`. The original unchanged Geo-trax/T000 baseline on **video frames 10750–10950 (FLUID 10751–10951)** remains unprovided, and no independent unseen source or full-frame human physical-object labels were evaluated. Build 4 release gates remain fail-closed; numeric gates cannot be evaluated against T000.

## Exact next work

1. On the original Windows workspace run `scripts/phase3_holdout_validation.py inspect --batch-dir artifacts/phase3/sahi_detector_trials/W04_box_tracking_batch201_01 --output artifacts/phase3/sahi_detector_trials/W04_continuous_build4_integrity_01.json` to verify current local original-source SHA and all Build 3 result artifacts.
2. Locate/generate **unchanged same-window T000 tracks**; never substitute a different window's original benchmark. Perform like-for-like recall, class, identity and timing comparisons with source hashes.
3. Run additional **offline replays on cached 291-frame raw boxes** to investigate per-ID class flips without paying for another 4K detector pass or silently rewriting physical classes. Preserve the original 25 results.
4. Only after reviewing class-preservation results and T000 baseline, pre-register ONE policy and evaluate truly untouched holdout with independent physical review and measured p95. PR #10 stays draft; no production Geo-trax, frozen FLUID or historical score changes.

**All 25 per-policy results remain in the immutable local experiment output and user-provided CSV/QA artifacts; aggregate documentation here contains no private video frames or raw tracks.**
