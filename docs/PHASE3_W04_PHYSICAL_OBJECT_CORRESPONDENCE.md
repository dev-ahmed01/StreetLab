# StreetLab Phase 3 — Generic cross-class physical-object correspondence (W04 diagnostic audit)

**Status:** reusable, non-destructive experimental correspondence module implemented. No automatic identity merging, no vehicle-count addition, no FLUID truth correction, no deployment. Original IoS.30 ByteTrack remains authoritative.

## Problem established by the completed W04 experiment

The native W04 hybrid IoU .50 preserved more heavy-vehicle detector hypotheses, but failed physical identity continuity compared with original `hard_nms_ios_0.30` (111 vs 71 contiguous ID switches, 20 vs 15 fragmented FLUID tracks). A subsequent SHA-verified shadow queue reported 191 rare-class unmatched track observations over 40 hybrid IDs and seven persistent review candidates. Both submitted review CSVs agreed `ALREADY_TRACKED` on all seven, but reviewer independence is not independently authenticated and the original private case manifest was reconstructed, not obtained.

The prior shadow pairing used **same-class bounding-box IoU >= .50**. A small box inside a large vehicle box can have small IoU despite covering almost its entire own area; different detector classes also prevented pairing. Thus `unmatched` did NOT mean physically new.

## Implemented generic module

`streetlab_phase3/video/physical_object_correspondence.py` implements original-source-frame **class-agnostic** box correspondence, with **predeclared structural** (not review-fitted) relationships:

| Evidence class | Geometric criterion |
|---|---|
| Same extent overlap | Bounding-box IoU >= 0.50 |
| Part-to-whole overlap | Intersection / smaller-box area (IoS) >= 0.90; smaller/larger box area <= 0.65; center of smaller lies in larger |
| No strong relation | No criterion above applies |

Candidate links are recorded for **every** primary track in the same source frame, including alternative detector classes, not just the single best match. At the tracklet level a repeated-ID hypothesis requires >=3 distinct frames linked to the same primary track, covering >=50% of shadow observations. If multiple primary IDs meet this criterion, the status explicitly reports `MULTIPLE_PRIMARY_OBJECT_HYPOTHESES` rather than arbitrarily merging them. Weak and unpaired shadow observations remain visible and unresolved.

**No spatial match is interpreted as physical identity.** Overlapping or occluded distinct vehicles can satisfy IoS; source-video inspection and independent physical annotations are needed to settle identity. No original track IDs, classes, or geometry are modified. There are no auto-duplicate deletions and zero new production counts.

## Whole-cohort W04 diagnostic outcome

An **independent, local re-analysis** of the already saved `W04_CONTINUOUS_201_RESULTS.zip`, `W04_UNIFIED_HYBRID_REPLAY_01.zip`, `W04_SHADOW_REVIEW_V2.zip` and `W04_TWO_REVIEWER_CONSENSUS_20261009.zip` used the structural geometry rules and produced a private SHA-anchored `W04_PHYSICAL_CORRESPONDENCE_AUDIT.zip`. Source original 25 track hashes, all four unified track+sidecar hashes, primary control byte identity, shadow review ZIP manifest, and consensus ZIP manifest passed.

| Result across all 191 shadow observations | Count |
|---|---:|
| Shadow observations with >=1 plausible primary geometric overlap | **149 / 191** |
| Shadow observations still without a sufficiently strong geometric overlap | **42 / 191** |
| Shadow tracklets with repeated support for one primary ID | **5 / 40** |
| Shadow tracklets with multiple qualifying primary ID hypotheses | **2 / 40** |
| Tracklets with weak/discontinuous geometry | **22 / 40** |
| Tracklets with no strong geometric correspondence | **11 / 40** |

Of **the seven** previously reviewed candidate tracklets, four have repeated single-primary geometric support, two have multiple plausible primary identities and one has weak/discontinuous support. **All seven reviewer CSVs agreed ALREADY_TRACKED**, but this outcome must remain *external diagnostic metadata*, not a training target, algorithm input, or accuracy calculation. This study has no known independently verified *new-object* negatives/positives, so no sensitivity, false-positive rate or held-out precision should be inferred.

Illustrative large-candidate matches: shadow MOTORCYCLE #102 repeatedly overlaps existing primary #85; shadow HEAVY_VEHICLE #122 overlaps primary #83. These are geometry hypotheses, not certified physical-vehicle identifications.

## Reproducibility and safeguarding user evidence

`scripts/phase3_w04_object_correspondence_audit.py` recomputes all 191 geometry comparisons from original frozen ZIP files using the generic module. It verifies original 25-policy/source controls, all hybrid sidecar hashes, original SHA, independent shadow-review ZIP checksums and the completed consensus archive. **Reviewer answers enter only AFTER geometry scoring**, solely to describe the seven already-reviewed examples. Outputs are immutable new CSV/JSON evidence files plus a SHA manifest; original data remains unchanged.

`tests/test_phase3_physical_object_correspondence.py` tests containment, unrelated adjacent boxes, overlapping-but-ambiguous vehicles, persistent single-ID evidence, multiple alternative primary IDs, false merges across tracker ID switches, no-overlap and malformed/duplicate IDs. `tests/test_phase3_w04_object_correspondence_audit.py` checks direct Windows-compatible CLI import paths, immutable output failure, unsafe ZIP paths, and optional uploaded evidence integration when private ZIPs are available. Phase 3 CI is wired to include the CLI in its sparse checkout.

## Independent holdout requirements

**Do not tune this module's thresholds on these seven already-known reviews.** Before claiming actual missed-vehicle recovery or automatically suppressing duplicate detections, test on independently annotated, previously unseen video with both (1) physically duplicate part/whole detections and (2) genuinely distinct physically close or occluded vehicles. Report per-class new-object precision/recall, false merges, ID continuity, physical object count error and end-to-end CPU p95. The W04 development window alone cannot justify promotion; no authentic original T000 same-window reference was recovered.

**The next engineering gate is a labeled external holdout**, not more W04 threshold search. Draft PR #10 remains experimental and unmerged.