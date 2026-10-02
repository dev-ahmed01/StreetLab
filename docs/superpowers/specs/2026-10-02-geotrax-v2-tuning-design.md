# Geo-trax V2 Controlled Tuning Design

Date: 2026-10-02

Status: Approved architecture, pending implementation plan

Scope: Phase 3 diagnostic tuning harness only

## 1. Objective

Build a reproducible, bounded harness for tuning Geo-trax 1.5.1 motorcycle
observation recall and trajectory continuity using the May-25 and May-26 data as
development/calibration data. The harness must preserve the existing spatial
benchmark semantics and must not modify Geo-trax itself, SUMO behavior, class
mappings, Phase 1/2 logic, or existing benchmark artifacts.

The initial implementation prepares and validates the tuning workflow. It does
not automatically execute an expensive candidate sweep.

## 2. Frozen evaluation semantics

All candidate evaluation uses the existing PixelBenchmark association logic:

- frame offset: `+1`
- maximum spatial match distance: `50 px`
- benchmark-compatible filtering occurs before matching
- accepted associations remain one-to-one Hungarian matches
- no independent offset or distance search is permitted
- no class mapping changes are permitted

Identity and continuity metrics reuse the accepted point matches rather than
implementing a second matching algorithm. A predicted-ID change separated by a
large unmatched interval is not treated as a contiguous identity switch.

## 3. Selected architecture

StreetLab will orchestrate the installed Geo-trax CLI as an external,
version-pinned black box. It will create explicit experiment plans, invoke
`geotrax extract` with supported dotted configuration overrides, and evaluate
the resulting tracks with StreetLab's existing benchmark code.

This is preferred over importing private Geo-trax APIs because it preserves the
installed package's effective configuration and normal execution path. It is
also preferred over patching Geo-trax because the latter would expand the
experiment scope and make baseline equivalence harder to prove.

The harness will not use `geotrax batch`, because its visualization stage adds
substantial runtime that is unnecessary for tuning.

## 4. Verified installed capabilities

The inventory is derived from the installed Geo-trax 1.5.1 package, its bundled
configuration, the installed Ultralytics tracker implementation, and the
effective YAML metadata from the existing May-25 and May-26 extractions.

Verified detector controls include:

- `ultralytics.conf`
- `extraction.class_conf`
- `ultralytics.iou`
- `ultralytics.imgsz`
- `ultralytics.max_det`
- `ultralytics.classes`
- `ultralytics.agnostic_nms`
- `ultralytics.vid_stride`

Verified active BoT-SORT controls include:

- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `track_buffer`
- `match_thresh`
- `fuse_score`
- GMC, proximity, appearance, and ReID settings

Verified post-processing controls include `min_track_length` and `interpolate`.
SAHI configuration is exposed by Geo-trax, but the optional SAHI dependency is
not installed.

Per-class confidence is implemented by lowering the detector prediction floor
to the smallest configured class threshold and applying the relevant class
threshold before detections reach the tracker. Geo-trax class `3` is
`Motorcycle`.

In the installed BoT-SORT implementation:

- detections at or above `track_high_thresh` enter first-stage association
- detections above `track_low_thresh` but below `track_high_thresh` are
  available for low-confidence recovery
- `new_track_thresh` controls whether an eligible unmatched detection can
  initialize a new track

The bundled `lenient` configuration provides verified recall-oriented values of
`track_high_thresh=0.20`, `track_low_thresh=0.10`, and
`new_track_thresh=0.10` alongside detector confidence `0.15`. The Stage-A
tracker-entry candidate uses these verified values but applies the lower
detector threshold only to class `3`; thresholds for all other detector classes
remain at the frozen baseline.

Raw pre-tracker detections are not persisted as reusable artifacts by the
installed package. Consequently, raw-detector versus tracker-loss comparison is
explicitly deferred. The harness must not fabricate this distinction from
post-tracker data.

## 5. Baseline configuration

The frozen May-25 and May-26 extractions use:

- model: `hf://rfonod/geo-trax/geotrax_hbb_yolov8s_1920_v1.pt`
- image size: `1920`
- detector confidence: `0.25`
- detector IoU: `0.70`
- maximum detections: `1000`
- classes: `[0, 1, 2, 3]`
- class-agnostic NMS: enabled
- video stride: `1`
- tracker: BoT-SORT
- `track_high_thresh=0.25`
- `track_low_thresh=0.10`
- `new_track_thresh=0.25`
- `track_buffer=30`
- `match_thresh=0.80`
- score fusion: enabled
- sparse optical-flow GMC
- ReID: disabled
- minimum track length: `3`
- interpolation: disabled

The T000 candidate must reproduce this configuration exactly on the Stage-A
windows. Its same-window results are the reference for all guardrails and
scoring. No non-baseline candidate is eligible for consideration until T000 has
completed and passed reproducibility validation.

Baseline validation compares the effective saved Geo-trax configuration against
the frozen configuration and verifies deterministic evaluation invariants. If
an existing compatible same-window baseline is available, its files may be
validated, but they must never be overwritten. Full-video May-25/May-26 scores
are contextual evidence, not a substitute for the T000 same-window baseline.

## 6. Stage-A windows

The approved evaluation intervals remain unchanged. Each extraction begins 90
source frames before its evaluation interval to warm the online tracker. Only
the central evaluation interval contributes to metrics.

| Window | Evaluation frames, inclusive | Warm-up frames, inclusive | Purpose |
| --- | ---: | ---: | --- |
| W01 | 4850-5250 | 4760-4849 | control 1084 |
| W02 | 7600-7900 | 7510-7599 | control 1727 |
| W03 | 9150-9450 | 9060-9149 | control 2094 |
| W04 | 10750-11350 | 10660-10749 | poor 2339, control 2460 |
| W05 | 13250-13850 | 13160-13249 | poor 2705, control 2718 |
| W06 | 15150-15750 | 15060-15149 | poor 3014 and 3051 |
| W07 | 16200-16800 | 16110-16199 | poor 3189 and 3280 |

The evaluation intervals contain 3,407 frames. With 90-frame warm-up for each
window, each candidate processes 4,037 frames. Predicted track IDs are
namespaced by window for aggregate evaluation so independent tracker restarts
cannot create false cross-window identity associations.

Frame cropping must preserve original source frame numbers in the Geo-trax
output or apply a deterministic, recorded translation back to source frame
numbers before evaluation.

## 7. Stage-A candidate matrix

The initial round is a one-factor sensitivity study, except for T004, whose
detector and tracker threshold changes form one semantically necessary unit.

| ID | Name | Changes from T000 |
| --- | --- | --- |
| T000 | BASELINE | none |
| T001 | MC_CONF_020 | `extraction.class_conf={3: 0.20}` |
| T002 | MC_CONF_015 | `extraction.class_conf={3: 0.15}` |
| T003 | BUFFER_45 | BoT-SORT `track_buffer=45` |
| T004 | MC_CONF_015_TRACKER_ENTRY | class `3` confidence `0.15`; BoT-SORT `track_high_thresh=0.20`, `track_low_thresh=0.10`, `new_track_thresh=0.10` |
| T005 | IMGSZ_2304 | `ultralytics.imgsz=2304` |

T004 is intentionally coupled. Lowering only class `3` detector confidence
allows additional motorcycle detections to reach BoT-SORT, but the frozen
`track_high_thresh` and `new_track_thresh` would prevent many such detections
from entering first-stage association or starting tracks. T004 uses the exact
tracker confidence values documented by the installed recall-oriented preset.

`match_thresh` remains `0.80` throughout the initial round. A
`match_thresh=0.90` candidate is deferred to the gated second round, after the
detector and tracker-confidence sensitivity results are understood.

Second-round candidates are templates rather than preselected winners. They may
combine the best valid class-confidence/tracker-entry setting with buffer or
matching-threshold changes. They are generated only after explicit review of
T000-T005.

Excluded from the initial round:

- global confidence changes, because they would affect every class
- interpolation, because it synthesizes observations
- minimum-track-length tuning, because it is post-processing rather than
  detector recall
- SAHI, because its dependency is absent and cost is high
- ReID and tracker replacement, because they expand the search space
- heuristic trajectory merging

## 8. Raw-label and canonical evaluation

The FLUID raw label is retained separately from the canonical class mapping.
The evaluator first performs the frozen all-class Hungarian matching and then
groups truth observations and accepted matches by raw FLUID label.

The development data contains the raw labels `bus`, `car`, `moped`,
`pedestrian`, `trailer`, `tricycle`, `truck`, and `van`. Of these, `moped` is
the only observed raw label mapping to canonical `MOTORCYCLE`. Unsupported raw
labels are reported explicitly and are never silently discarded.

Per raw label, report at least:

- truth, matched, and missed points
- point recall
- truth and matched truth tracks
- track coverage
- mean and median matched fraction
- fragmentation among matched truth tracks
- mean predicted IDs per matched truth track
- mean and median longest-run fraction
- miss-gap distribution using actual consecutive truth frame numbers

Raw-label `moped` precision is not defined because Geo-trax emits canonical
classes and cannot distinguish raw FLUID motorcycle sublabels. It is represented
as `null` with an explicit reason. Canonical `MOTORCYCLE` precision remains
available and is reported.

Overall and canonical-class metrics include existing PixelBenchmark metrics,
predicted points, predicted tracks, matched and unmatched predicted tracks,
unmatched predicted-track fraction, average predictions per evaluated frame,
and identity-continuity aggregates.

## 9. Guardrails and score

Every candidate is compared with the completed T000 same-window baseline. A
candidate is invalid if any of these conditions holds:

- overall point precision is below `0.94`
- canonical CAR point recall decreases by more than `0.02` absolute
- pixel MAE exceeds `3.0 px`
- prediction growth triggers the documented track-explosion rule
- its configuration, inputs, or evaluation constants differ from its immutable
  manifest

The track-explosion rule is deterministic: invalidate when predicted-track
count exceeds `1.50x` T000 and unmatched-predicted-track fraction exceeds T000
by more than `0.10` absolute. Report predicted-point ratio and average
predictions-per-frame ratio as additional diagnostics, but do not use them as
unstated rejection criteria.

For valid candidates, normalize bounded fraction metrics directly on `[0, 1]`
and calculate:

```text
40 * moped point recall
+ 20 * moped track coverage
+ 15 * moped median matched fraction
+ 10 * moped median longest-run fraction
+ 10 * overall point precision
+  5 * overall point recall
```

The maximum score is 100. Invalid candidates have no ranking score and retain a
machine-readable list of failed guardrails.

## 10. Experiment lifecycle and immutability

The CLI provides explicit non-expensive and expensive operations:

- `inventory`: inspect installed capabilities and write the parameter inventory
- `plan-stage-a`: write and validate the window/candidate plan
- `run-candidate`: perform explicitly requested extraction for one candidate
- `evaluate-candidate`: evaluate already-produced tracks
- `summarize`: apply guardrails and produce a comparison table

Suggested artifact layout:

```text
artifacts/phase3/geotrax_tuning/
  parameter_inventory.json
  stage_a_plan.json
  experiments/
    T000_BASELINE/
      experiment.json
      windows/W01/...
      metrics/...
```

Experiment directories are created with exclusive semantics and are never
reused or overwritten. Geo-trax overwrite options are not enabled. A duplicate
experiment ID or existing output path causes a clear failure.

Each experiment manifest records:

- experiment ID and lifecycle status
- source video and truth paths
- window and warm-up bounds
- exact baseline and override settings
- effective Geo-trax configuration saved after execution
- model identifier and resolved model revision when available
- Geo-trax and Ultralytics versions
- StreetLab commit and dirty-worktree flag
- frozen evaluation constants
- commands, timestamps, per-window and total runtime
- whether detector inference was rerun
- output hashes or equivalent provenance for evaluation inputs
- metrics, score, and guardrail results

The current CLI does not expose a reusable detection cache. The inventory will
therefore distinguish parameters that conceptually affect only tracking from
the practical fact that the current execution path reruns end-to-end inference
for every candidate.

The existing Geo-trax virtual environment has a broken base-Python reference.
The harness validates executable readiness and fails with a targeted diagnostic;
it does not silently recreate the environment or change package versions.

## 11. Module responsibilities

`streetlab_phase3/geotrax_tuning.py` owns:

- immutable dataclasses for tunables, windows, candidates, manifests, metrics,
  and guardrail outcomes
- inventory construction from verified installed files
- deterministic Stage-A plan construction
- experiment-directory allocation and manifest lifecycle
- safe Geo-trax command construction
- raw-label metric aggregation using existing accepted matches
- window namespacing and aggregation
- baseline reproducibility validation
- guardrails, scoring, and summary generation

`scripts/phase3_geotrax_tune.py` is a thin CLI wrapper. It performs no expensive
action without an explicit `run-candidate` request and prints structured status
or summaries to standard output.

## 12. Error handling

The harness fails before inference when:

- Geo-trax is unavailable or its version differs from the recorded version
- the input video or truth file is unavailable
- an experiment directory already exists
- a candidate attempts to change frozen settings
- T000 has not passed before a non-baseline candidate is requested
- a requested setting is absent from the verified inventory
- output frame provenance cannot be translated to source frame numbers

Partial runs retain their immutable manifest with a failed or interrupted
status. Resuming requires a new experiment ID rather than overwriting partial
evidence.

## 13. Tests

Synthetic unit and regression tests cover:

- installed-parameter inventory representation
- fixed evaluation constants and forbidden-setting rejection
- exact Stage-A windows and 90-frame warm-up
- exact T000 baseline configuration
- T000 prerequisite enforcement
- T004 coupled class-confidence/tracker-entry settings
- raw FLUID label preservation and explicit unsupported labels
- accepted-match reuse and benchmark-compatible filtering
- actual-frame continuity for runs and gaps
- matched-truth denominator for fragmentation
- deterministic scoring and every guardrail
- deterministic track-explosion detection
- predicted-ID namespacing across windows
- unique experiment allocation and no-overwrite behavior
- manifest/config serialization and artifact paths
- CLI planning and evaluation on synthetic fixtures without inference
- unchanged existing Phase-3 benchmark tests

Tests do not run Geo-trax or require the external May-25/May-26 datasets.

## 14. Execution stages and cost

Implementation ends after generating the inventory, Stage-A plan, tests, and
documented commands. It does not run the sweep.

When later authorized:

1. repair or recreate the pinned Geo-trax environment as a separately reviewed
   operational step
2. generate inventory and Stage-A plan
3. run and validate T000
4. stop if T000 fails reproducibility
5. run T001-T005 individually
6. evaluate and summarize initial sensitivity results
7. design the gated second round, including `match_thresh`, only after review

Each candidate processes approximately 4,037 frames. Based on the observed
May-26 runtime, estimate roughly 2.7-3.0 hours per candidate, or approximately
16-18 hours for T000-T005 sequentially. Higher-resolution T005 may take longer.
Seven separate extraction invocations are required per candidate, for 42
initial-round invocations. These are planning estimates, not guaranteed
runtimes.

## 15. Explicit non-goals

This work does not:

- rerun May-25 or May-26 full-video inference during implementation
- modify Geo-trax package code or frozen extraction outputs
- change class mappings or benchmark association semantics
- tune SUMO or downstream behavior logic
- implement trajectory merging
- infer physical causes from diagnostic geometry
- claim raw detector-versus-tracker attribution without an exposed artifact
- select or run second-round combinations before initial sensitivity review
