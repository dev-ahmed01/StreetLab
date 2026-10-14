# W04 unified hybrid real tracking verdict — 9 October 2026

**Decision: reject all three hybrids as replacements for the original primary ByteTrack feed. Keep draft PR #10 experimental and unmerged.** Original continuous 291-frame source (90 warmup + 201 evaluation, frames 10750–10950) was replayed locally on Windows with four independent tracker instances and frozen FLUID +1/50px scoring.

## Independently audited upload

- Upload ZIP SHA-256: `3664bd7f046b5a5c41692723ca0bc880db58aef3d2736515a1eaa0578510cc5d`. Previous continuous batch report exact SHA-256 `dbb3f0539a788926c630c198f383421040f57938712c0de93531e9f694c00120`; pre-global-merger raw box checksum matches earlier cache.
- **The newly replayed control track `.txt` is byte-for-byte identical to the earlier frozen `hard_nms_ios_0.30.txt`** (SHA-256 `6c695e384e05421ec25dbe450f00a83fb6a72a659a2d162a1122d67c7d0ce418`). Stronger than the script's 11 exact control-metric checks.
- All 4 `.txt` files and `.class_evidence.csv` files pass their saved SHA-256 hashes. Every policy `.score.json` matches the unified JSON. All **27,824** 14-column track rows are in the original 201 evaluation frames, with unique `(frame, tracker_id)` per mode and one class-evidence sidecar row per track row. Pixel precision and recall recomputed from recorded matched/truth/predicted counts; source FLUID itself not re-uploaded for independent end-to-end re-scoring.

## Native ByteTrack + frozen FLUID measurements

| Policy | Precision | Recall | MOTORCYCLE correct-class recall | HEAVY correct-class recall | ID switches | Fragmented FLUID tracks / 71 | Development gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Original IoS .30 | **86.70%** | 91.37% | 83.32% | 79.75% | **71** | **15** | Reproduced exactly |
| Rare IoU .50 | 84.77% | 91.78% | 83.78% | **87.97%** | 111 | 20 | **FAILED** continuity and fragmentation |
| Center-guard | 84.31% | 91.69% | 83.55% | 87.34% | 110 | 23 | **FAILED** precision/continuity/fragmentation |
| Cross-tile center-guard | 82.33% | **91.87%** | **83.97%** | 85.44% | 192 | 27 | **FAILED** precision/continuity/fragmentation |

For the strongest rare-IoU alternative, heavy correct-class matches improved **126→139** (out of 158 FLUID HEAVY points), and motorcycle correct matches **2547→2561** (out of 3057). But FLUID ID switches worsened **71→111** and fragmented tracks **15→20**, so the additional recall does not justify replacing the primary tracker. Other variants were worse. All 3 fail the previously committed development gate. The frozen W04 BUS truth has zero points, so no BUS recall conclusion is supported.

## Fresh analysis: retain extra rare evidence without poisoning stable primary IDs

The independent post-run analysis spatially paired the original and primary rare-IoU track rows with per-frame/per-class one-to-one Hungarian assignment at bbox IoU≥0.50. This is *not* full physical truth matching.

- Rare IoU produced **78 HEAVY** and **113 MOTORCYCLE** additional, unmatched track observations in the 201-frame cohort. None has a *same-class* original control track with IoU≥0.50. However, **21/78 HEAVY** and **2/113 MOTORCYCLE** candidate observations do overlap a *different-class* original control track at IoU≥0.50, warning of class ambiguity.
- Those observations span **11 HEAVY** and **29 MOTORCYCLE** hybrid track IDs. A transparent candidate-review filter (>=3 consecutive unmatched frames and >=2 frames with confidence>=0.5) retains **5 HEAVY and 2 MOTORCYCLE** IDs for visual review. These are **seven provisional review targets, not independently verified new vehicles or recall gains**.
- The hybrid leaves original CAR/BUS detection suppression unchanged but uses one shared multi-class ByteTrack association instance. Its added rare detections change the resulting overall tracks and IDs; this supports keeping rare hypotheses out of the primary association bank pending independent tests.

## Engineering direction

**Authoritative lane:** original IoS .30 merge -> original continuous ByteTrack. This preserves the byte-reproducible 71-switch/15-fragmentation reference and original 14-column classes, IDs and coordinates.

**Shadow evidence lane:** independently retain selected rare-IoU .50 HEAVY/MOTORCYCLE hypotheses. Compare against primary tracked boxes, mark cross-class overlap ambiguity, and expose only persistent high-confidence unmatched proposals to a review queue. Never inject those proposals indiscriminately into the primary tracker, assign certified identity, or call them confirmed true positives without image review. The seven ID candidates are an evidence-driven starting point; no production change yet.

**Auxiliary class-evidence lane:** if desired, show same-frame detector-supported causal class interpretation *next to*, not replacing, original tracker labels. All new labels remain explicitly provisional. Prior class-support study showed 44 supported alternate observations in lead track.

**Open release gates:** new unseen footage, independent physical-object review (including BUS/AUTO_RICKSHAW), a new source-aligned Geo-trax replacement benchmark if needed (authentic T000 original unavailable), and CPU p95. Original OpenVINO 4K median ~1.245 seconds/frame; no full pipeline performance claim.

All original evidence, production Geo-trax, and frozen FLUID remain unchanged. **The new hybrids are research rejects for primary association, not failed software execution**.