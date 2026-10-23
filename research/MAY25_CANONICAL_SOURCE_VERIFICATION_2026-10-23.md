# StreetLab — original May25 source-verified canonical audit (23 October 2026)

**Result:** **SOURCE-VERIFIED HISTORICAL REGRESSION AUDIT COMPLETED**. This report records execution of the existing `streetlab_integration.w04_autonomous_diagnostics` canonical audit code against original private May25 video and original FLUID CSV bytes. This is **not** an independent held-out test, independently confirmed physical vehicle taxonomy, a calibrated real-site simulator, or production approval.

## Immutable original inputs verified at execution

| Evidence | Verified SHA-256 |
|---|---|
| Original `20250525_video.mp4` (504,559,952 bytes) | `7ba62d051a0fdd1dd8cb0bb91d32a37fef1ee9f16c3aeadd70e1b820dfa88c82` |
| Original `20250525_video_Traj.csv` (41,521,208 bytes) | `ff845d13053065b01c2954a2cdf586fc070214cf896b208734a703692c74c643` |
| Frozen `batch_report.json` | `3ebac039d153a45dc33a19f4d8eed2615b0d9d49388a6722d002884423dc2ee1` |
| Frozen policy `policy_lock.json` | `e2993a6f757e3f212b111122e44e8a8fe1f61da56e692908c2b1d32565027ecb` |
| Frozen preregistration `holdout_protocol.json` | `d96d0b0b55dabe64fb0d2d7388abef9af19a933e74f92853890e2ab25d10b939` |

The original 505 MB video was decoded for its metadata; native 3840 × 2160, 10 FPS. SHA verification was performed on its *bytes*, not inferred from a file name or a Google Drive metadata field. Original policy, batch tracks, source labels and field truth were not altered.

## Canonical audit output

The canonical runner returned:

- `status=W04_MAY25_AUTOMATED_LABEL_RELATIVE_OBJECT_AUDIT_COMPLETE`
- Video frames 60–179, FLUID frames 61–180, original +1 alignment and 50-pixel class-agnostic Hungarian association
- 7,892 annotated FLUID points; 8,076 saved prediction points; 7,299 matched points
- Historical FLUID-label F1 91.42034068136272%
- 236 FLUID HEAVY classification disagreements, originating in exactly **two** annotated HEAVY tracks
- 12 unique predicted BUS-track IDs, but zero FLUID BUS truth points (so BUS physical precision and recall are **not measurable**)
- No new model inference, retuning or FLUID relabeling
- `independent_holdout_pass=false`, `production_release_approved=false`

The resulting `automated_object_audit.json` also passed the independent `streetlab_integration.research_audit_display.read_audit` saved-artifact validation check.

**Canonical audit JSON SHA-256:** `4825c574429f78b7199f812cd764ddfacac327ac32c563f73cf1f2f61ab05118`

## Six-phase fixed research completion verdict

The real unchanged frozen May25 policy lock and protocol were recovered from the private connected Drive workspace, SHA-matched to the saved preflight, and used alongside the original-source canonical audit, forensics, historical regression and successfully executed native SUMO *synthetic* evidence. Running the repository's own `w04_completion_gate` produced:

`RESEARCH_PROTOTYPE_COMPLETE_WITH_LIMITATIONS` — **6/6 research phases COMPLETE**.

This is a **research-prototype submission completion**. It is not an independent field acceptance/production certification. The original Chennai generalization gate remains failed, native SUMO uses assumed topology and demand, and original same-window T000, independent physical-label adjudication, site geometry/census calibration and measured full-pipeline p95 remain unavailable.

## Storage and reproducibility scope

The canonical output includes derived `automated_object_audit.json`, `all_truth_track_diagnostics.csv`, `predicted_track_class_stability.csv`, and `research_findings.md`. The private 505 MB MP4 and original 41 MB CSV **must not** be committed to this public repository. This public report contains only scientific counts and frozen SHA fingerprints. The original source-verified artifact is retained separately for offline submission and can be shown in the local research UI with `STREETLAB_MAY25_AUDIT_PATH` set to the extracted canonical JSON. The GitHub CI does not have the private May25 bytes and cannot rehash them from a public checkout.

The older `MAY25_FULL_COHORT_REPLAY_2026-10-23.md` remains an accurate timestamped record of an earlier *saved-coordinate-only* replay; it has not been overwritten, and was never the canonical source-backed audit.

**Do not interpret real original-file SHA verification as a genuinely unseen model holdout or as proof that the underlying FLUID label taxonomy is physically correct.**
