# Phase 3 — Preregistered independent physical-object correspondence holdout

**Status: validation tooling complete, independent footage NOT YET PROVIDED or scored. This is a separate experimental research diagnostic, not a production release gate.** The finished W04 physical-object correspondence audit is **development only** and must never be described as independent holdout. May 25/May 26 footage previously analyzed in StreetLab also does not automatically qualify as unseen.

## Why we need this

W04 showed 149/191 previously same-class-unmatched rare-vehicle observations had a plausible broader primary geometric relation; 42 did not. Seven previously reviewed candidates were all marked already tracked, but their reviewer procedure and original case manifest were not independently authenticated. **Positive-only already-tracked cases cannot test the danger of merging two distinct vehicles close together.**

To prevent accidental identity loss, validate pair-level matching against both `SAME_PHYSICAL_OBJECT` labels and **`DISTINCT_PHYSICAL_OBJECT` nearby negative controls**, from footage not used for model/threshold selection. A plausible IoU/IoS or repeated primary ID is an *uncertainty-bearing proposal*, not ground truth.

## Code delivered

- `streetlab_phase3/video/correspondence_holdout.py`: native 14-column export parsing, frame/ID checks, class-agnostic close-pair inventory, frozen temporal geometry predictions and truth-blind stratified sampling across four kinds of pairs. Uses **unchanged** `physical_object_correspondence.overlap_evidence`. Multiple plausible primary tracks cause **AMBIGUOUS**, never automatic linking.
- `scripts/phase3_correspondence_holdout.py`: `lock`, `prepare`, `evaluate` commands, SHA-anchored source provenance, zero-indexed source frame selection, read-only OpenCV source crops of neutral A/B boxes, independently packaged offline reviewer ZIPs, exactly-complete reviewer CSV agreement and diagnostic scoring.
- `scripts/RUN_PHASE3_CORRESPONDENCE_HOLDOUT.ps1`: one launcher with three explicit modes and fail-closed missing-source checks.
- `tests/test_phase3_correspondence_holdout.py`: synthetic physically distinct nearby vehicles, dangerous overlap that would be a false merge, multi-primary ambiguity, reviewer completeness, source track strictness, development-video exclusion, code-lock integrity, direct Windows script import, and diagnostic safety gate. The Phase 3 CI suite includes all tests.

## Frozen study design (before viewing holdout)

| Parameter or minimum | Frozen value |
|---|---:|
| Same-extent bbox IoU | >= 0.50 |
| Part-whole intersection-over-smaller-area (IoS) | >= 0.90 |
| Part-whole smaller/larger area ratio | <= 0.65 |
| Per primary ID support | >= 3 separate frames and >= 50% shadow observations |
| Nearby distinct-object control inclusion | Center separation <= 1.5x max box diagonal |
| Minimum evaluation duration | 120 consecutive source-frame indices |
| Maximum deterministic review sample | 120 pair cases, stratified without annotations |
| Agreed true same-object pairs to score | >= 10 |
| Agreed true distinct-object pairs to score | >= 20 |
| False proposed links among agreed distinct pairs | 0 allowed by diagnostic filter |
| Disagreements or UNCLEAR across both reviewers | Blocks diagnostic pass |

The thresholds are *predeclared diagnostic criteria*, not empirically justified operational safety guarantees. Even observing zero false merges among twenty labeled distinct pairs leaves considerable statistical uncertainty and does **not** establish production readiness. True new-vehicle recall and whole-video vehicle count accuracy are not measured by this pair-level diagnostic.

## Source inputs required for actual execution

- A genuinely new continuous fixed-camera traffic video, not W04 and not used earlier to tune the detector/trackers. SHA-256 and capture/source details must be archived; a nonmatching SHA **alone** does not prove the footage was never seen.
- Two matching, separately generated native 14-column source-pixel track exports from exactly the same zero-indexed source frames: authoritative IoS .30 primary ByteTrack, and non-authoritative rare-IoU .50 shadow ByteTrack. The two exports cannot be byte-identical. Frame/ID/class/finiteness/geometry checks are enforced.
- At least 120 source-frame indices. The exports may contain no-object frames implicitly; the tool does not claim it verified true per-frame tracker invocation just from export rows. That claim requires the detector/tracker execution manifest.
- Two independent authorized human reviewers, and reviewed crops from the **new** source video. Reviewer forms intentionally hide tracker IDs, classes, algorithm prediction and developer-case judgments.
- The immutable W04 development audit ZIP previously supplied in this chat. It is used **only to preregister a source/code lock**, never as heldout testing truth.

## Windows commands

First **before examining the new video's labels**, place `W04_PHYSICAL_CORRESPONDENCE_AUDIT.zip` in the `StreetLab-engine-trial` root and preregister the lock:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git switch codex/phase3-engine-shootout
git pull --ff-only origin codex/phase3-engine-shootout
& '.\scripts\RUN_PHASE3_CORRESPONDENCE_HOLDOUT.ps1' -Mode Lock
```

The generated `artifacts\phase3\independent_holdout\CORRESPONDENCE_LOCK_V1.json` pins SHA-256 hashes of the two implementation modules, dev-audit ZIP and all fixed criteria. Keep the lock unchanged. **New code changes intentionally invalidate the lock**; any revision must happen before the unseen review, with a transparent new preregistration.

Only when the distinct video and paired tracker outputs are available:

```powershell
& '.\scripts\RUN_PHASE3_CORRESPONDENCE_HOLDOUT.ps1' -Mode Prepare `
  -Video 'C:\Data\independent_new_traffic.mp4' `
  -PrimaryTracks 'C:\Data\new_primary_ios030.txt' `
  -ShadowTracks 'C:\Data\new_shadow_rare_iou050.txt' `
  -First 0 -Last 299
```

This produces **separate** `REVIEWER_R01_ONLY.zip` / `REVIEWER_R02_ONLY.zip` in `artifacts\phase3\independent_holdout\PAIR_REVIEW_PACKET_01`, plus a private `INTERNAL_manifest.json`, SHA manifest and review templates. The `C:\Data` paths and 0–299 window are examples, NOT claims that files exist or validation ran. Distribute only the assigned reviewer ZIP, not the internal manifest or the other reviewer answers.

After both reviewers complete the CSVs:

```powershell
& '.\scripts\RUN_PHASE3_CORRESPONDENCE_HOLDOUT.ps1' -Mode Evaluate `
  -Reviewer01 'C:\Reviews\holdout_R01.csv' `
  -Reviewer02 'C:\Reviews\holdout_R02.csv'
```

Returns immutable `PAIR_DIAGNOSTICS_01` CSV and JSON with agreed true-same and true-distinct counts, false links among confirmed distinct objects, duplicate-link sensitivity in the annotated pair sample, ambiguity and review disagreements. Zero fields in the original video, FLUID and 14-column primary track are written.

## Remaining hard gates before any production consideration

Pair-level diagnostic safety is necessary but NOT sufficient. Additional requirements: truly independent footage with source integrity and annotation lineage; physically adjudicated same and distinct objects from multiple sources/intersections; explicit class breakdown including BUS and AUTO_RICKSHAW mapping; original primary ID-switch/fragmentation nonregression; whole-video traffic reconstruction/count and false-merge error; p50/p95 full-pipeline CPU runtime; source-aligned replacement Geo-trax baseline if justified. Original T000 remains unavailable in the scanned workspace.

**No fabricated independent result:** until video, two source-aligned real track exports and complete independent reviews exist, the only CI evidence is synthetic/unit behavior. PR #10 stays draft and unmerged.