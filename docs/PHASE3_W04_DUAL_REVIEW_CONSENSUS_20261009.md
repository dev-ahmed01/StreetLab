# W04 15-case dual-review consensus — 9 October 2026

**Development footage only. Do not use this as held-out validation, whole-frame accuracy, or corrected FLUID truth.**

The user supplied two complete, differently identified reviewer CSV files from the SHA-verified Build 2 blinded W04 15-case package. The original `adjudicate_pair` implementation verified all 15 blinded JPEGs, case IDs, reviewer field validations, 480px crop-relative boxes and the 0.50 reviewer IoU agreement threshold, and wrote an immutable local consensus sidecar. Distinct reviewer identifiers do **not** independently authenticate whether reviewers worked without collaboration.

## Actual consensus

- **15/15** marked PRESENT by both reviewers, on intentionally preselected targets.
- **13 AGREED_PRESENT**, **2 UNRESOLVED**, **0 AGREED_ABSENT**.
- Agreed class breakdown: **6 MOTORCYCLE, 2 CAR, 2 BUS, 2 HEAVY_VEHICLE, 1 AUTO_RICKSHAW**.
- **Case 2 (frame 11020):** motorcycle vs bicycle, boxes IoU **0.543**; unresolved on class.
- **Case 3 (frame 11230):** both motorcycle, boxes IoU **0.463**; unresolved on box alignment (<0.50).
- Every unresolved case stays unresolved: no replacement labels, box interpolation or automatic tie-break.

| Case | Frame | Reviewer 1 | Reviewer 2 | IoU | Decision | Nearest original FLUID label | Nearest distance |
|---:|---:|---|---|---:|---|---|---:|
| 1 | 10990 | MOTORCYCLE | MOTORCYCLE | 0.612 | MOTORCYCLE | moped | 0.00px |
| 2 | 11020 | MOTORCYCLE | BICYCLE | 0.543 | UNRESOLVED | moped | 0.00px |
| 3 | 11230 | MOTORCYCLE | MOTORCYCLE | 0.463 | UNRESOLVED | moped | 0.00px |
| 4 | 10990 | MOTORCYCLE | MOTORCYCLE | 0.762 | MOTORCYCLE | moped | 8.49px |
| 5 | 10930 | MOTORCYCLE | MOTORCYCLE | 0.747 | MOTORCYCLE | moped | 9.14px |
| 6 | 11200 | MOTORCYCLE | MOTORCYCLE | 0.506 | MOTORCYCLE | moped | 16.56px |
| 7 | 10780 | MOTORCYCLE | MOTORCYCLE | 0.802 | MOTORCYCLE | pedestrian | 0.87px |
| 8 | 11080 | MOTORCYCLE | MOTORCYCLE | 0.740 | MOTORCYCLE | moped | 113.53px |
| 9 | 11260 | AUTO_RICKSHAW | AUTO_RICKSHAW | 0.835 | AUTO_RICKSHAW | moped | 2.51px |
| 10 | 11050 | CAR | CAR | 0.760 | CAR | car | 16.03px |
| 11 | 11200 | CAR | CAR | 0.856 | CAR | car | 16.28px |
| 12 | 10960 | HEAVY_VEHICLE | HEAVY_VEHICLE | 0.947 | HEAVY_VEHICLE | car | 6.83px |
| 13 | 10840 | BUS | BUS | 0.929 | BUS | car | 1.31px |
| 14 | 10750 | BUS | BUS | 0.827 | BUS | car | 5.35px |
| 15 | 10900 | HEAVY_VEHICLE | HEAVY_VEHICLE | 0.893 | HEAVY_VEHICLE | car | 153.00px |

### Consequences

- Among the 13 agreed cases, **11** have a raw FLUID center within 50px. At the nearest center, **6** have the same normalized class and **5** have a different normalized class. These are spatial associations, not independently verified track identities. The other **2** agreed objects (cases 8 and 15) have no raw annotation center within 50px.
- Among the **12** agreed cases whose review anchor was a selected detector prediction (rather than one of the three PyTorch–OpenVINO *FLUID truth observation* comparisons), **11** visually agree with that detection's selected class. This is *targeted-case agreement*, **not a precision estimate**.
- **AUTO_RICKSHAW** at case 9 is an important ontological exception: reviewers chose AUTO_RICKSHAW, selected model class is CAR, nearest FLUID type is `moped`/MOTORCYCLE. Neither class can be presumed universally correct.
- BUS cases 13/14 and HEAVY_VEHICLE case 12 are visually agreed examples near original `car` centers. MOTORCYCLE case 7 is near `pedestrian`. Extra physical objects or center ambiguities cannot be ruled out without identity-level frame inspection.
- W04 was already used for development; these selected crops **do not measure physical precision, exhaustive detection recall, or generalization**.

### Exact local evidence and nonpromotion

- Original blind package: `W04_BLIND_REVIEW_15_02`; source bundle manifest SHA: `c4ab33ddd53ddc01aab9fa36479e3eea597b29d485366e4352fe7896179ffd56`.
- Reviewer 1 CSV SHA-256: `18358f42c90d11ec6962cc54dae9a4055a5600aff608ee3d1a1cf62696069b44`.
- Reviewer 2 CSV SHA-256: `305fc0ec6dcfa4c913635f27b84d61ca58b7bdde0656fb9705511da2d0bf943e`.
- Official dual-consensus JSON SHA-256: `64442b60d7ccf7e0faaba16bb74c15a2d339da472367a7aca6aec1f650139848`.
- JSON and original CCTV review images remain user-local (not uploaded to this repository). The review script uses `open("x")` to avoid overwriting existing experiment evidence.
- Original 21-frame four-class W04 score **670/857 precision, 670/739 recall** remains frozen; no FLUID labels or original Geo-trax, P3B or T000 source files changed.
- Candidate-wise physical comparison remains pending actual Build 1 real per-tile OpenVINO box files; Build 3 continuous ByteTrack metrics and Build 4 *unseen video*, true same-window T000 and separate blinded **holdout** physical-object review remain unverified.
- PR stays **DRAFT** and `eligible_for_promotion=false`.

## Follow-up

If both unresolved images require a definitive label, commission a **fresh blinded third reviewer or documented arbitration** for cases 2 and 3. Do not rewrite the submitted independent sheets or count a majority vote as an original two-reviewer consensus. Proceed on the 13 accepted cases while the remaining real inference and holdout prerequisites are collected.
