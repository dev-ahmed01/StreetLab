# W04 completed visual review consensus — seven shadow candidates (9 October 2026)

**Status: recorded two-reviewer agreement; no newly untracked vehicle in the seven submitted case decisions. No production or FLUID changes.** This records user-uploaded blind CSV votes; it does not certify procedural reviewer independence or a new unseen-video benchmark.

## Uploaded review submissions and evidence authentication

- `W04_R01_blind_review.csv`: 7 complete, unique cases, `reviewer_id=R01`; SHA-256 `2de9aa7f9a176a0f2a9d263d370478cf2ef3002b8108f77ef7a84d269823b8e0`.
- `W04_R02_blind_review.csv`: 7 complete, unique cases, `reviewer_id=R02`; SHA-256 `1f9145b2e40c9782170287b4feddcabeb0924414bd10c9d9472908090febba3d`.
- Corresponding reviewer-only visual packet ZIP SHA-256 values are the same as the earlier preflight: R01 `ad0549ce264821fb7a55fa470ee2764bd47bace010d9baa6285d1852b15a2923`; R02 `f436d970d5424e8e5b22d30fae77968a167387d032d9dcdb53ff9034163fe727`.
- The 35 zero-indexed source-frame annotations match the previous preflight's seven anonymous case frame lists exactly, and preserve the FLUID +1 source offset. The queued seven shadow candidate tracklets and original 191 observation records passed manifest hashes.
- The **original** private `INTERNAL_case_manifest.json` was not uploaded in this conversation. Case-to-shadow identity mapping was **reconstructed and independently cross-checked** from the `blind-w04` deterministic hash sort of seven `(vehicle_class, hybrid_tracker_id)` keys and every sampled frame's membership in the exact original 191-row shadow evidence. This is not a claim to have authenticated the unavailable original manifest.

## What the reviewers actually submitted

| Anonymous case | Reconstructed hybrid candidate | R01 | R02 | Status |
|---|---|---|---|---|
| S01 | MOTORCYCLE #102 | YES, MOTORCYCLE, ALREADY_TRACKED | Same | AGREED_ALREADY_TRACKED |
| S02 | HEAVY_VEHICLE #2 | YES, HEAVY_VEHICLE, ALREADY_TRACKED | Same | AGREED_ALREADY_TRACKED |
| S03 | MOTORCYCLE #22 | YES, MOTORCYCLE, ALREADY_TRACKED | Same | AGREED_ALREADY_TRACKED |
| S04 | HEAVY_VEHICLE #122 | YES, HEAVY_VEHICLE, ALREADY_TRACKED | Same | AGREED_ALREADY_TRACKED |
| S05 | HEAVY_VEHICLE #96 | YES, HEAVY_VEHICLE, ALREADY_TRACKED | Same | AGREED_ALREADY_TRACKED |
| S06 | HEAVY_VEHICLE #137 | YES, HEAVY_VEHICLE, ALREADY_TRACKED | Same | AGREED_ALREADY_TRACKED |
| S07 | HEAVY_VEHICLE #97 | YES, HEAVY_VEHICLE, ALREADY_TRACKED | Same | AGREED_ALREADY_TRACKED |

**Exact agreement: 7/7** on visible object, reported physical class and relation to primary tracking; **5 reported heavy**, **2 reported motorcycle**, **0 reported untracked**, **0 reported missing/not visible**, and **0 explicit disagreement/unclear reviews**. **All fourteen notes cells are blank.** The reviewer IDs differ but independence of the actual work cannot be proved from the two CSVs alone.

## Scientific verdict and operational consequence

**Retain authoritative IoS.30/ByteTrack and do not add or promote any of the seven shadow IDs as a second physical vehicle.** The earlier hybrid gave marginal rare-class match gains but unacceptable contiguous identity regressions. In these seven selected examples, a reviewer-visible novel-object candidate count of **0/7** under both submitted review forms points toward duplicates, alternative detection extents and/or classification ambiguity, rather than demonstrated true omissions. No inference is warranted that all 191 unmatched observations are duplicates, or that the model has perfect rare-class recall.

The internal engineering preflight had marked small roadside candidates (S01/S03) visually uncertain and suggested possible part-to-whole alignment issues for several long white vehicle crops. That preflight is separate from the recorded reviewer votes and is not evidence of a fourth review, label correction or unquestionable object truth.

The actual user-private, SHA-anchored detailed consensus deliverable is `W04_TWO_REVIEWER_CONSENSUS_20261009.zip`, containing `consensus_summary.json`, `case_level_consensus.csv`, `case_mapping_RECONSTRUCTED.json`, `requires_adjudication.csv`, `verdict.md`, and `SHA256SUMS.txt`. Keep the private reviewer media and raw CSVs out of public GitHub.

## Future iteration discipline

**Close W04's seven selected shadow candidate novelty question**; do not spend additional model CPU on them. Before using the shadow lane as a missed-vehicle discovery mechanism, add physical-object correspondence tests that can recognize part-whole, cross-class and adjacent-extent duplicates. Do not overfit those duplicate rules to just seven W04 examples. Test them with independent physical labels and a separate unseen video plus original class/ID metrics. Historical T000 original remains unrecovered; a new comparison must be explicitly a replacement baseline. The W04 PR stays draft/unmerged. No production model, source FLUID, primary tracks or object counts are modified.