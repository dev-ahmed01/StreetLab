# StreetLab — May25 full-cohort replay and object attribution (23 October 2026)

**Status: coordinate-and-tracking replay reproduced; canonical full-source audit still pending.** This report records a new calculation against the saved complete 120-frame cohort, not a new inference run, independently adjudicated vehicle labels, an unseen holdout, or a production release. Do **not** pass this Markdown to `STREETLAB_MAY25_AUDIT_PATH` or mark the canonical `automated_object_audit.json` complete on the basis of this calculation alone.

## Source lineage and reproducibility

- Frozen W04 policy: `hard_nms_ios_0.30`; video frames **60–179**, FLUID frames **61–180**, 50-pixel class-agnostic Hungarian assignment **before** distance gating.
- Original FLUID CSV in connected Google Drive: `20250525_video_Traj.csv`, 41,521,208 bytes, SHA-256 **`ff845d13053065b01c2954a2cdf586fc070214cf896b208734a703692c74c643`**. Its byte hash was independently recomputed in this continuation before a container failure.
- The 504,559,952-byte original MP4 was materialized from connected Drive. The execution environment failed before rehashing those bytes in this continuation; **video SHA-256 cannot be newly certified here**. Previously recorded frozen reference: `7ba62d051a0fdd1dd8cb0bb91d32a37fef1ee9f16c3aeadd70e1b820dfa88c82`.
- Saved GeoTrax tracking file: 8,076 14-column records, class IDs 0 CAR, 1 BUS, 2 HEAVY_VEHICLE, 3 MOTORCYCLE. Replayed directly from the connected saved tracking file, not regenerated detections.
- The prior private derived-coordinate staging document contained **9,541 eligible coordinate rows**, including **1,649 duplicate frame/ID/class records** and **no coordinate conflicts**. Deduplication yielded **7,892 distinct scored truth points** (4,391 CAR, 2,785 MOTORCYCLE, 716 HEAVY; zero BUS), matching the previous frozen source scorecard exactly. The staging document is **not** a substitute for re-reading and verifying every row against the original CSV inside the canonical runner.
- The reconstruction used rectangular minimum-cost Hungarian assignment and applied the unchanged 50-pixel filter **after** assignment. No greedy nearest-neighbor, revised alignment, class-aware matching or retuning was introduced.

## Whole-cohort replay: counts conserve exactly

| Quantity | Recomputed |
|---|---:|
| FLUID truth points | 7,892 |
| Saved prediction points | 8,076 |
| Spatial Hungarian matches | 7,299 |
| Point precision | 90.37890044576523% |
| Point recall | 92.48606183476938% |
| Point F1 | **91.42034068136272%** |
| Distinct annotated FLUID truth track IDs | 84 |
| Distinct model-predicted track IDs | 132 |

| FLUID class | Truth points | Spatial matched | Correct-class matched | Matched to another model class |
|---|---:|---:|---:|---:|
| CAR | 4,391 | 4,325 | 4,215 | 110 |
| MOTORCYCLE | 2,785 | 2,302 | 2,296 | 6 |
| HEAVY_VEHICLE | 716 | 672 | 436 | **236** |
| BUS | 0 | 0 | 0 | 0 |

The complete non-diagonal confusion among spatially assigned observations is **HEAVY→CAR 236**, **CAR→HEAVY 109**, **CAR→BUS 1**, **MOTORCYCLE→CAR 6**. This reproduces the saved 236 HEAVY class errors and the exact earlier F1, rather than changing the scientific baseline.

## Key Phase 3 object-level result

The **236 HEAVY→CAR frame-level label disagreements correspond to precisely two distinct annotated FLUID track IDs**, not 236 distinct vehicles:

| Annotated FLUID track ID | Matched scored frames | Model class | Associated predicted ID | Disagreements |
|---|---:|---|---:|---:|
| 7 | 120 / 120 | CAR throughout | 26 | 120 |
| 36 | 116 / 116 | CAR throughout | 29 | 116 |
| **Total** | **236 / 236** | | | **236** |

Top-two **and** top-five HEAVY disagreement concentration: **100%**. The result is about annotation-relative identity aggregation only. It does **not** establish that either FLUID HEAVY label or model CAR label corresponds to the physical vehicle category.

Across the full replay, 42 matched points had a near alternative spatial association (at most 10 pixels worse and within the frozen 50-pixel gate). Nine were wrong-class matches; **zero** of the 236 HEAVY disagreements met the near-alternative criterion. This ambiguity count is based on the same saved coordinate replay and awaits upstream source audit, as above.

## BUS predictions and why accuracy is unknown

There were **185 BUS prediction-frame records belonging to 12 distinct predicted track IDs**. One BUS record was spatially associated to a FLUID CAR; **184** BUS records had no FLUID point within the post-assignment match. The three largest model BUS identities were ID 97 (57 BUS frames), ID 110 (45), and ID 81 (36). **BUS FLUID truth count was zero**; neither BUS precision against physical reality nor BUS recall can be calculated. Unmatched predictions cannot automatically be called physically false detections.

## Preservation of the six-phase contract

| Phase | Evidence-based status |
|---|---|
| 1 — Source and reproducibility | Original asset lineage and FLUID SHA confirmed; previously completed |
| 2 — Frozen development/May25 regression | Previously completed; original saved 91.42% F1 preserved |
| 3 — Autonomous full-cohort object attribution | **Numerical replay completed; canonical SHA-verifying runner/export not verified** |
| 4 — Scientific report/native SUMO | This report adds object attribution; existing native SUMO is **synthetic-only**, not a calibrated field simulator |
| 5 — Local UI/API/E2E | Existing `/research` integration remains fail-closed until source-verified `automated_object_audit.json` exists |
| 6 — Release/CI and bounded handoff | Evidence packaging PR under verification; no production permission |

### Hard scientific boundaries

Historical May25 is **not** an unseen independent holdout. May26 was tuned development, with unpaired cross-date descriptive comparison only. **Chennai field-transfer FAILED** and must remain failed. The SUMO network uses assumed geometry/demand; no measured field intervention effect, same-window independent T000, independent physical taxonomy adjudication, source-calibrated road network, end-to-end CPU p95 or field-production approval is shown. The original video source SHA and complete canonical batch/forensics hashes **must be checked before publishing the canonical audit output and enabling the UI evidence gate**.

**Next canonical acceptance step, automated only:** run the existing `streetlab_integration.w04_autonomous_diagnostics.run` with verified original video, original FLUID CSV, full tracking batch and saved forensics in a stable execution environment. Its four original-source artifacts—not this report—are the release-gated authoritative object-audit outputs. No human crop reviews or new detector inference required.
