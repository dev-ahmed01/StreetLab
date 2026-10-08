# StreetLab Phase 3 — SAHI + ByteTrack continuous-track candidate

**EXPERIMENTAL / ISOLATED / NO PRODUCTION PROMOTION**

This sprint joins two independent, documented upstream components:

- [SAHI](https://github.com/obss/sahi): full-image and sliced detection modes; the
  API returns full-frame `ObjectPrediction` boxes.
- [Roboflow Trackers](https://github.com/roboflow/trackers): modern
  `ByteTrackTracker.update(sv.Detections)` API (Apache-2.0). We deliberately
  avoid the deprecated `supervision.ByteTrack` wrapper. Unconfirmed detections
  reported as tracker ID `-1` are omitted from trajectory exports.
- [FLUID P3B benchmark](../streetlab_phase3/pixel_benchmark.py): frozen
  offset **+1**, fixed 50px matching threshold, full fixed evaluation window.

The original Geo-trax model, existing production configuration, FLUID labels
and phase 1/2 behavior calibrations are unchanged. The new trial consumes the
same aerial detector checkpoint for both modes and writes fresh files only.

## Continuous semantics

Do **not** call a multiobject tracker once every tenth sampled frame: that
would distort lifecycle and fragmentation. Tracking trials decode EVERY
consecutive source frame from `max(0,start-90)` through `end`, including
blank frames; the first 90 frames merely warm up. Only the declared central
window is exported. There is no optical stabilization, georeferencing,
interpolation, or invented identity.

Tracking preset (not a demonstrated optimal choice): detector confidence 0.15,
ByteTrack activation 0.20, high-confidence association split 0.15,
`lost_track_buffer=45`, `minimum_consecutive_frames=2`. Tracker's internal
frame-rate is deliberately fixed to 30 so the buffer counts 45 per-frame
updates regardless of the video fps; actual video frames are never skipped.
All values are in a manifest alongside the checkpoint SHA-256. This
configuration intentionally differs from the original BoT-SORT baseline, and
is a *candidate*, not a production policy.

The pipeline implements:
`raw video -> RGB -> SAHI detection -> sv.Detections -> ByteTrackTracker ->
confirmed IDs -> 14-column pixel tracks -> fixed-window FLUID scoring`.

## Real validation locally on FLUID

The 17,293-frame May-26 video, its FLUID trajectory CSV and T000 extraction
results are on your Windows machine; none are present in GitHub. Never
represent GitHub unit or public-image smoke tests as FLUID accuracy.

Create the independent experiment worktree as described in
[engine shootout](PHASE3_ENGINE_SHOOTOUT.md), then:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git fetch origin
git checkout --detach origin/codex/phase3-engine-shootout

$python = ".\.venv-sahi-audit\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python -m pip install "sahi[ultralytics]" "trackers==2.6.0" opencv-python

$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$weights = "C:\path\to\geotrax_hbb_yolov8s_1920_v1.pt"

# W04: warm up 10660–10749; evaluate 10750–11350.
& $python scripts/phase3_sahi_track.py `
  --video $video --weights $weights --fluid-tracks $truth `
  --output artifacts/phase3/sahi_tracker_trials/W04_sliced.txt `
  --start-frame 10750 --end-frame 11350 --mode sliced `
  --confidence 0.15 --image-size 1920 `
  --slice-height 640 --slice-width 640 --overlap 0.20 --device cpu

# Matched non-sliced detection mode through EXACT SAME TRACKER.
& $python scripts/phase3_sahi_track.py `
  --video $video --weights $weights --fluid-tracks $truth `
  --output artifacts/phase3/sahi_tracker_trials/W04_standard.txt `
  --start-frame 10750 --end-frame 11350 --mode standard `
  --confidence 0.15 --image-size 1920 --device cpu
```

If and ONLY if the checkpoint uses unknown numeric names (e.g. the official
four-class ONNX export), add
`--class-map config/phase3/geotrax_class_ids.json`.
Never apply that numeric map to a generic COCO checkpoint (class 0 is person).

To compare either run against **the exact same-window T000 file**, pass
`--baseline-tracks "C:\path\to\W04_T000_absolute_frame_tracks.txt"`
on a fresh run with a new `--output`. This scores the baseline and candidate
in the same fixed frame window and produces a fail-closed promotion gate.
Previously generated results are never overwritten. Exit code 2 means the
candidate failed a scoring gate, not a successful promotion.

Run W01 (4850–5250) and W05 (13250–13850) using fresh unique output paths.
The same 90-frame warm-up rule applies. The first test can be smaller (e.g.
40–100 evaluation frames) to check model compatibility and runtime cost, but
small pilots **cannot** serve as the final comparison.

## Diagnose motorcycle misses without rerunning video inference

The class diagnostic reads an EXISTING Geo-trax-compatible track export and
FLUID annotation CSV, applying the identical **fixed** +1 frame alignment
and <=50 px spatial Hungarian matching used in `PixelBenchmark`.

```powershell
$python = ".\\.venv-sahi-audit\\Scripts\\python.exe"
& $python scripts/phase3_class_diagnostics.py `
  --tracks artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01.txt `
  --fluid-tracks "C:\\Users\\Admin\\Desktop\\StreetLabData\\Video_2\\20250526_video_Traj.csv" `
  --start-frame 10750 --end-frame 10759 `
  --output artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01_classes.json
```

To compare against T000, pass its real original Geo-trax tracking file as
`--tracks` with **the same frame bounds**, saving to a separate JSON path.
This does not change or rerun either engine.

The report distinguishes **spatial recall** (did any vehicle prediction
occupy a truth location?) from **correct-class recall** (was it assigned the
right class?). In a wrong-class spatial association, the truth contributes
a class error, *not* a spatial false positive. This preserves the frozen
benchmark's class-agnostic spatial association and avoids falsely claiming
a motorcycle was detected correctly when it was labeled as a car.

### CPU tiling safety

For the 3840×2160 May-26 video, running every 640-pixel slice with
`--image-size 1920` causes redundant resizing and excessive CPU work.
Choose **`--image-size 640`** with 640×640 slices for the first SAHI pilot.
The unsliced control may retain `--image-size 1920`; this explicitly
tests two *inference configurations* with the same trained model weights,
not identical image input sizes. Keep the manifest settings visible so
accuracy and compute comparisons remain honest. Never use tiny warm-up or
10-frame identity results as production quality evidence.

## First actual May-26 W04 pilots and fair offline A/B

The user has now executed the experiments locally on the same authentic video:
3840x2160, 10 FPS, original official overhead checkpoint SHA-256
`7d462ae523b15f3679a83f74ab08aa79d2978f319ca2f0cf2892fbbb60df79da`.

| Run | Source video frames | Warm-up | Model image size | Point recall | Point precision | Elapsed |
|---|---|---:|---:|---:|---:|---:|
| Standard+ByteTrack | 10750–10759 (10) | 5 | 1920 | 246/368 = 66.85% | 246/262 = 93.89% | 32.97s / 15 processed |
| SAHI+ByteTrack | 10750–10752 (3) | 2 | 640 tile and inference | 98/113 = 86.73% | 98/127 = 77.17% | 65.89s / 5 processed |

**These two headline recalls are not comparable**: the denominator and
interval differ. The 10-frame standard per-class diagnostic showed
MOTORCYCLE: 65/186 spatial matches (34.95%), 121 spatial misses, 11
unmatched motorcycle predictions, zero spatially matched class confusions;
CAR: 171/172 spatial matches (99.42%), with 21 car→bus class mismatches.
Thus motorcycle detection coverage is the primary observed weakness, while
car/bus classification is a secondary target. Identity fragmentation
measured over three or ten frames is too short to support a tracking claim.

**Do not run more inference yet.** Both raw `.txt` exports already contain
the shared source frames 10750–10752. The new offline comparator re-scores
both on the SAME three FLUID frames, reports motorcycle recall and unmatched
predictions and refuses unmatched checkpoint hashes or tracker settings:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\\.venv-sahi-audit\\Scripts\\python.exe"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
& $python scripts/phase3_compare_existing_trials.py `
  --standard-tracks "artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01.txt" `
  --sliced-tracks "artifacts/phase3/sahi_tracker_trials/W04_sliced_smoke01.txt" `
  --fluid-tracks $truth --start-frame 10750 --end-frame 10752 `
  --output "artifacts/phase3/sahi_tracker_trials/W04_existing_3frame_comparison.json"
```

It reports both per-class and identity summaries in a new JSON, plus the
sliced-minus-standard differences.

### Paired rescue/loss and unmatched-confidence forensic report

After the first W04 comparison, an additional offline-only analysis is
available using the exact same inputs. Because results are immutable, give
the `--output` a NEW filename:

```powershell
& $python scripts/phase3_compare_existing_trials.py `
  --standard-tracks "artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01.txt" `
  --sliced-tracks "artifacts/phase3/sahi_tracker_trials/W04_sliced_smoke01.txt" `
  --fluid-tracks $truth --start-frame 10750 --end-frame 10752 `
  --output "artifacts/phase3/sahi_tracker_trials/W04_paired_forensics01.json"
```

This reports **paired** truth-observation outcomes for each class: matched
by both, standard only, SAHI only, or neither; the same accounting with
correct class required; and concrete FLUID motorcycle track IDs/video frames
rescued or lost. It separately groups unmatched prediction observations by
category, confidence ([0,.25), [.25,.5), [.5,1]) and track ID appearing on
one versus multiple frames **within this 3-frame evaluation cohort**.
An ID's short presence inside the restricted cohort is NOT proof it is an
ephemeral false positive on a longer video.

Do not treat confidence groupings as calibrated probabilities or derive a
production threshold from these three tuned frames. The purpose is to
identify whether the additional 23 unmatched SAHI predictions look like
low-confidence motorcycles, wrong-class vehicles or high-confidence false
alarms before spending CPU on a controlled, longer experiment. Raw FLUID
files and private video stay on the local machine.
 It deliberately records **unequal
warm-up (5 versus 2)** and **image-size (1920 versus 640)** as confounders;
thus even the same-three-frame comparison is exploratory, not a controlled
component-level ablation or evidence for production promotion. To isolate
the effects of tiling, later rerun both configurations on identical image
size, confidence, source window and 90-frame tracker warm-up.

`T000` original Geo-trax track export has not been located in the earlier
searched `C:\Users\Admin\Desktop\StreetLab\artifacts` directory. Do
not invent its score or use a different-window summary as a replacement.

## Precision sensitivity and unmatched proximity (zero inference)

The paired May-26 audit found 19 motorcycle truth observations recovered by
SAHI and one lost, from the SAME 10750–10752 evaluation frames. The recovered
observations include repeated detections of the same five vehicle track IDs.
Do **not** count these as 19 unique motorcycles. Sliced produced 29 unmatched
predictions: 12 motorcycle (11 with confidence <0.50), 12 car (7 with confidence
>=0.50), 4 bus, 1 heavy vehicle. Some unmatched observations may represent
annotation gaps, class mismatches, or center-association errors rather than
real-world false alarms.

A new **post-track** sensitivity audit compares preset cutoffs
`0.15,0.25,0.35,0.45,0.50,0.60,0.75` for:
- removing predictions of **all classes** below the cutoff; versus
- removing **only MOTORCYCLE** predictions below the cutoff.

The same `--start-frame`/`--end-frame` and FLUID labels are applied at
every cutoff. The first `0.15` global row reproduces the unfiltered
SAHI output. This is a counterfactual *filter on confirmed track export*;
it cannot simulate how changing YOLO/SAHI confidence changes NMS, ByteTrack
activation, or identity association. Cutoffs are preset, not fitted to labels.

The report also examines **unmatched** outputs (not verified physical false
positives). It flags prediction centers within 25 pixels of a *separately
matched* prediction, and centers within 50 pixels of any FLUID truth point,
including class differences. These are **proximity heuristics**, not IoU
measurements or proof that two boxes belong to one vehicle. Separate summary
counts for high-confidence unmatched CAR predictions help decide whether
duplicate suppression or an annotation/association review is justified.

Use the same prior inputs but a **new** output name:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\\.venv-sahi-audit\\Scripts\\python.exe"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
& $python scripts/phase3_compare_existing_trials.py `
  --standard-tracks "artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01.txt" `
  --sliced-tracks "artifacts/phase3/sahi_tracker_trials/W04_sliced_smoke01.txt" `
  --fluid-tracks $truth --start-frame 10750 --end-frame 10752 `
  --output "artifacts/phase3/sahi_tracker_trials/W04_precision_sensitivity01.json"
```

Full outputs now include `post_track_confidence_sensitivity` with 14
rows, and `sliced_unmatched_proximity` with class summaries plus
individual unmatched track IDs and center distances. This diagnostic is
run exclusively on a **three-frame tuning interval**, and all decisions
continue to say `eligible_for_promotion=false`.

**Recommended follow-up:** inspect high-confidence unmatched cars that
are *not* near an existing match or FLUID truth point and verify them
visually. Then choose a truly controlled longer A/B at equal detector
input size, 90-frame warm-up, threshold and checkpoint; avoid costly
blind parameter grids on the CPU-only 3840x2160 recording.

## W04 matched-pair and visual error review

The three-frame W04 study shows why a confidence cutoff is not a
validated solution: filtering existing SAHI tracks at >=0.50 raises
point precision from 77.17% to 91.67% but reduces motorcycle recall
from 78.57% to 55.36%. This post-track score cannot predict what
ByteTrack would do with a changed detector confidence.

The new `shadow_matching_audit` quantifies how many matches a
cardinality-first 50px assignment would gain over the original
Hungarian-then-gate matcher. It does not change the frozen evaluation.

Run these commands in the existing experimental clone:

```powershell
git pull --ff-only
$python = ".\\.venv-sahi-audit\\Scripts\\python.exe"
$truth = "C:\\Users\\Admin\\Desktop\\StreetLabData\\Video_2\\20250526_video_Traj.csv"
$video = "C:\\Users\\Admin\\Desktop\\StreetLabData\\Video_2\\20250526_video.mp4"
$std = "artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01.txt"
$sliced = "artifacts/phase3/sahi_tracker_trials/W04_sliced_smoke01.txt"
$comparison = "artifacts/phase3/sahi_tracker_trials/W04_shadow_check01.json"
& $python scripts/phase3_compare_existing_trials.py `
  --standard-tracks $std --sliced-tracks $sliced `
  --fluid-tracks $truth --start-frame 10750 --end-frame 10752 `
  --output $comparison

& $python scripts/phase3_visual_error_review.py `
  --comparison $comparison --video $video --fluid-tracks $truth `
  --standard-tracks $std --sliced-tracks $sliced `
  --output-dir "artifacts/phase3/sahi_tracker_trials/W04_visual_review01"
```

The review output is a local-only collection of raw versus annotated
crops, with an index JSON. It prioritizes higher-confidence unmatched
CAR points, MOTORCYCLE points near matches, and representative
rescued/lost truth vehicles. Proximity does not prove duplicate
detections; a shadow association is not an official benchmark score.
No model inference is rerun and none of the original evidence is
overwritten. The output is exploratory, never eligible for promotion.

## First visual adjudication and confirmed-track box audit

A manual review of 20 May-26 W04 raw-vs-overlay crops suggested distinct
hypotheses: multiple visible small motorcycles really recovered by sliced
inference, sometimes *multiple green centers on one motorcycle*, and CAR
detections placed along roofs/sections of large buses. Several unmatched
CAR markers also sit on visibly real cars; therefore unmatched != false
physical object, and no detector/class labels were overwritten. Crops
with two close motorcycles show why center-distance-only suppression
would be unsafe.

The existing 14-column candidate Geo-trax exports include the raw pixel
center and box **width and height** (columns 2–5); no new inference is
required to inspect overlap of **confirmed track boxes**. The optional
`phase3_box_forensics.py` reports, for the already unmatched observations:
- same-class IoU >=0.30 across distinct confirmed tracker IDs;
- unmatched CAR box coverage >=0.60 inside an exported BUS or HEAVY_VEHICLE
  track box; and
- high-confidence CAR (>=0.50) versus unmatched MOTORCYCLE summaries.

To run after the earlier `W04_shadow_check01.json` comparison:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\\.venv-sahi-audit\\Scripts\\python.exe"
& $python scripts/phase3_box_forensics.py `
  --comparison "artifacts/phase3/sahi_tracker_trials/W04_shadow_check01.json" `
  --sliced-tracks "artifacts/phase3/sahi_tracker_trials/W04_sliced_smoke01.txt" `
  --output "artifacts/phase3/sahi_tracker_trials/W04_confirmed_box_forensics01.json"
```

The report is new and immutable. It includes per-target track IDs,
box coordinates, and overlapping neighboring exported track boxes. This
is **not a raw SAHI NMS diagnostic**: ByteTrack has already associated
objects, so real overlaps may occur and upstream rejected duplicates are
unobservable. A CAR on a bus without a separately exported BUS track will
not be counted as containment; inspect raw CCTV regardless of overlap
count. No threshold or suppression should be promoted from these three
tuned frames.

## W04 geometry-only suppression simulation (offline, fully reversible)

The user's latest **unmatched-only** box audit flagged 11 observations:
CAR track 22 contained in a large vehicle on source frames
10750–10752; CAR 33 overlapping another CAR on 10750;
MOTORCYCLE 45 on 10750, 51 and 52 on 10751 and 52 on 10752;
and overlapping BUS observations (21 / 26). These are **track
observations**, not confirmed false-positive objects. Since earlier
crops showed multiple nearby real motorcycles and real cars on the road,
it would be unsound to remove rows simply because FLUID marked them
unmatched.

The new independent offline simulator applies geometry-only policies to
**all** confirmed source detections, not just those marked unmatched:
- MOTORCYCLE same-class overlap: greedily retain the higher-confidence
  box at IoU >=0.50 or IoU >=0.30, per frame.
- CAR containment: consider the car box for suppression if at least
  80% or 60% of its area lies inside an exported BUS/HEAVY_VEHICLE
  box at least twice its area.
- Separate conservative and exploratory combinations.

These six **post-hoc hypotheses** are applied on the existing W04
source outputs without accessing ground-truth or matching labels when
making suppression decisions. Each policy is then scored separately
against the identical frozen +1, 50px FLUID cohort. The report shows
point recall, precision, correct-class MOTORCYCLE recall, correct-class
CAR recall, unmatched predicted points and identity diagnostics.
It saves all removed `(video_frame, track_id)` decisions as evidence
but never rewrites the raw tracking outputs.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\\.venv-sahi-audit\\Scripts\\python.exe"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
& $python scripts/phase3_offline_geometry_suppression.py `
  --comparison "artifacts/phase3/sahi_tracker_trials/W04_shadow_check01.json" `
  --sliced-tracks "artifacts/phase3/sahi_tracker_trials/W04_sliced_smoke01.txt" `
  --fluid-tracks $truth `
  --output "artifacts/phase3/sahi_tracker_trials/W04_geometry_whatif01.json"
```

The JSON is immutable and SHA-256-hashed to the source track, FLUID
annotations, and comparison; choose a fresh filename to rerun. This
analysis still covers only **three frames from previously tuned May-26**
and is not a production gating experiment. Suppressing confirmed tracks
after ByteTrack cannot reconstruct what upstream NMS or ByteTrack
association would have done. A policy that increases precision yet
reduces motorcycle recall or creates temporal identity holes should
not be adopted. CAR-on-BUS suppression is particularly dangerous
when the BUS class itself is uncertain. After this diagnostic, choose
a narrower one-factor-at-a-time experiment and validate against
a longer, fixed-cohort recording before discussing promotion.

## W04 controlled scale check (completed locally, May-26 tuning only)

An additional **two warm-up frame, three evaluation frame** controlled
run on the user's original May-26 3840x2160 / 10 fps CCTV now holds
the weights (SHA-256
`7d462ae523b15f3679a83f74ab08aa79d2978f319ca2f0cf2892fbbb60df79da`),
FLUID offset (+1), 50px scoring radius, confidence 0.15, class mapping,
Roboflow ByteTrack settings, warm-up (frames 10748–10749), and scored
frames (10750–10752) constant. The recorded runs are:

| Candidate | Inference configuration | Matched / 113 | Recall | Precision | MOTORCYCLE correct-class recall | 5-frame end-to-end elapsed |
|---|---|---:|---:|---:|---:|---:|
| Standard | Full-frame resized to 640 | 55 | 48.67% | 94.83% | 0/56 = 0% | 23.53s |
| Standard | Full-frame resized to 1920 | 81 | 71.68% | 93.10% | 25/56 = 44.64% | 19.74s |
| Sliced SAHI | 640-pixel tiles; 640 model input | 98 | 86.73% | 77.17% | 44/56 = 78.57% | 65.89s |

The elapsed times are only 5-frame end-to-end pilot timings including
initialization, not stable detector throughput measurements. Full-frame
640 and 1920 both use the same detector weights; the whole-frame
downsampling erases many motorcycles at 640. Compared with the
1920 full-frame run with matched warm-up, the sliced run recovered
**19 additional matched MOTORCYCLE truth observations**, with zero
standard-only motorcycle observations in this 3-frame cohort. Those
19 are repeat observations of a smaller set of vehicles, not 19
unique motorcycles. Precision fell by **15.94 percentage points**.
This is strong evidence for a scale-related issue on W04 but NOT proof
that tiling alone, independent of effective image scale, is responsible.
The sample is only 0.3 seconds; neither tracker fragmentation nor
long-run runtime are established. Standard Geo-trax T000 was never
loaded into this comparison, so none of these numbers supersede it.

### Next, bounded detector-only spatial coverage experiment

Before paying for a full continuous 90-frame ByteTrack warm-up and
hundreds of 4K SAHI frames, predeclare a wider **21-frame stratified
detector-only** sample spread across the existing W04 tuning window
10750–11350 inclusive at step 30 (source 10 fps: 60 seconds of coverage).
Reuse the existing `phase3_sahi_audit.py` public script. Each
sampled frame is sent through standard full-frame **640** and sliced
**640** with the same checkpoint, confidence 0.15, original resolution
and class mapping. This test does **not** run ByteTrack or estimate
identity stability; it uses a class-aware, maximum-cardinality
50px detector matching method, which is intentionally different from
the class-agnostic frozen P3B tracking matcher. Do not compare its
percentage scores directly with the 3-frame P3B numbers.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$weights = "C:\Users\Admin\.cache\huggingface\hub\models--rfonod--geo-trax\snapshots\f512e0d1445e65fc2cf505d7474deccf33f11bf9\geotrax_hbb_yolov8s_1920_v1.pt"

& $python scripts/phase3_sahi_audit.py `
  --video $video --fluid-tracks $truth --weights $weights `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_spaced21_control640_01" `
  --start-frame 10750 --end-frame 11350 --sample-step 30 `
  --confidence 0.15 --image-size 640 --slice-height 640 `
  --slice-width 640 --overlap 0.20 --device cpu

# Exit code 2 means the predefined detector gate was NOT met,
# not that the report failed to save:
$report = Get-Content "artifacts/phase3/sahi_detector_trials/W04_spaced21_control640_01/report.json" -Raw | ConvertFrom-Json
$report.standard.per_class.MOTORCYCLE
$report.sliced.per_class.MOTORCYCLE
$report.standard.precision
$report.sliced.precision
$report.gate
```

It saves the original fixed sample positions, both complete per-class
detection scorecards, actual detections and per-mode latency; image
and video bytes are NOT uploaded. The `detector_gate` has deliberately
strict precision and latency limits; it may return exit code 2 even
when the run and output files are correct. Do NOT weaken those
limits or retrospectively relabel this as a generalization holdout.

After this step, choose a single **continuous** paired experiment
with equal 90-frame warm-up and a genuinely longer contiguous
evaluation window, budgeting potentially tens of minutes to more
than an hour on the user's CPU, before any claims about ID continuity.
The next truly independent evaluation must be on unseen data.
No automatic SAHI or IoU=0.30 suppression promotion.

## W04 spaced 21-frame detector-only sample — completed locally

The user executed the planned sample of source frames 10750, 10780, ...,
11350 (step 30, 10fps, 60 seconds between first and last), in the
original 3840x2160 May-26 *tuning* footage. Both use checkpoint SHA-256
`7d462ae523b15f3679a83f74ab08aa79d2978f319ca2f0cf2892fbbb60df79da`,
model image input 640, detector confidence 0.15, and same class-aware
maximum-cardinality matching gate at 50px. **No tracking is run.**

| Detection-only metric (21 sampled frames) | Standard full-640 | SAHI 640 tiles, overlap 0.20 |
|---|---:|---:|
| Total FLUID truth observations | 739 | 739 |
| Predicted observations | 377 | 860 |
| Matched truth observations | 355 | 671 |
| Overall recall | 48.04% | 90.80% |
| Overall precision | 94.16% | 78.02% |
| MOTORCYCLE correct-class matches / 380 | 12 | 321 |
| MOTORCYCLE recall | 3.16% | 84.47% |
| MOTORCYCLE precision | 100% (12/12) | 78.29% (321/410) |
| CAR recall | 96.05% | 97.46% |
| Median detector time per frame (CPU) | 0.379s | 12.055s |

Slicing improved motorcycle recall by **81.32 percentage points** over
the whole-frame-640 detection baseline, but overall precision lost
**16.14 percentage points**, and median inference latency increased
**31.8x**, far beyond the existing **5x** detector-trial budget.
Thus the frozen detector gate is **NOT PASSED** with exactly these
reasons: precision regression and latency budget exceeded.
The report was still successfully written; the CLI exited 2 by design.
No permission to promote the model, its overlap policy, or tracking.

There were **zero annotated BUS observations** in the 21 sampled FLUID
frames, but 9 standard and 13 sliced bus detections. Under this
class-aware matcher those detections are counted as unmatched,
not confirmed physically absent buses. Heavy vehicles had 5 annotated
observations; the class is also too sparse for robust conclusions.
The predictions tagged 'false_positives' in the detector-only score
represent **unmatched predictions**, not independently adjudicated
scene-level false alarms.

At 12 seconds per frame, a long continuous SAHI tracker trial would
be an expensive CPU investment. **Do not run a 90-frame-warmup,
multi-hundred-frame tracking trial yet.** Use a bounded inference-only
speed/recall trade-off probe on the **same five predeclared frames**.

### Next one-factor CPU probe: 1280px tiles (5 frames only)

Test wider, less-overlapping *input tiles* while retaining the
same model size=640, checkpoint, confidence=0.15, and fixed source frames:
10750, 10900, 11050, 11200, 11350. Wider tiles require fewer inference
calls, but the larger area is rescaled to 640 and may erase motorcycles.
This is a **candidate with unknown outcomes**, not a promised fix.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$weights = "C:\Users\Admin\.cache\huggingface\hub\models--rfonod--geo-trax\snapshots\f512e0d1445e65fc2cf505d7474deccf33f11bf9\geotrax_hbb_yolov8s_1920_v1.pt"

& $python scripts/phase3_sahi_audit.py `
  --video $video --fluid-tracks $truth --weights $weights `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_cpu_probe1280_5frames01" `
  --start-frame 10750 --end-frame 11350 --sample-step 150 `
  --confidence 0.15 --image-size 640 --slice-height 1280 `
  --slice-width 1280 --overlap 0.10 --device cpu

# The strict detector gate may return exit code 2 even when report.json
# was written correctly. Do not rerun into the same output directory.

& $python scripts/phase3_compare_cached_detector_subsets.py `
  --reference-dir "artifacts/phase3/sahi_detector_trials/W04_spaced21_control640_01" `
  --candidate-dir "artifacts/phase3/sahi_detector_trials/W04_cpu_probe1280_5frames01" `
  --output "artifacts/phase3/sahi_detector_trials/W04_cached640_vs_probe1280_5frames01.json"
```

The comparator rescales the **previously cached 21-frame reference
detections** to the **same five scored frames** as the candidate,
verifies checkpoint and ground-truth hashes and exact image size,
and reports all four same-frame scores (reference/candidate
standard and sliced). No re-inference for the 640-tile baseline.
Original reference 21-frame median and new 5-frame median come
from different sample counts, so do NOT claim their ratio is
a rigorous paired runtime benchmark; use as rough engineering
cost evidence only.

If candidate MOTORCYCLE recall falls dramatically on these same
five frames or latency remains prohibitive, do not tune blindly.
The optimization candidates then include accelerated CPU runtimes,
reduced spatial coverage via **independently specified** region
masks, or deployment on faster hardware, each requiring validation
that it does not exclude real motorcycles. Do not select masks
using the FLUID labels and claim held-out performance.
This May-26 footage remains tuned, not a true generalization test.

## W04 1280-tile CPU probe — rejected on same five detector frames

The user ran two configurations on the **same five May-26 W04 source frames**
10750, 10900, 11050, 11200 and 11350 (93 FLUID MOTORCYCLE observations):

| Detector-only score | Existing SAHI 640x640 tiles, 0.20 overlap | New 1280x1280 tiles, 0.10 overlap |
|---|---:|---:|
| Motorcycle matches / 93 | 80/93 = 86.02% | 46/93 = 49.46% |
| Motorcycle precision | 75.47% | 93.88% |
| Overall recall | 91.01% | 71.91% |
| Overall precision | 73.97% | 87.07% |
| Unmatched detector predictions | 57 | 19 |

**Reject 1280x1280 tiles at model size 640.** They lose 34 correct
motorcycle observations, a **36.56 percentage point** motorcycle-recall
regression, on this small paired sample. At median CPU time 3.535 s/frame
across five test frames they are cheaper than the prior 640-tile
21-frame median of 12.055 s, but those latency medians are unpaired
and do not establish a stable speedup ratio. The loss of motorcycle
recall is sufficient to reject the candidate regardless.

### Next low-risk CPU acceleration investigation: OpenVINO with identical 640 tiles

Keep the **640x640 image tiles, 0.20 overlap and model image size 640**.
Rather than make motorcycles smaller, try a separate **non-INT8** CPU
OpenVINO export of the same frozen YOLO checkpoint through Ultralytics.
SAHI's Ultralytics wrapper supports OpenVINO model directories according
to upstream documentation; this project's version-specific compatibility
and any gains must be established by a real local parity test.

Export requires optional OpenVINO packages. Install them into only the
experiment venv, not the project production environment. Pin the existing\nUltralytics version (8.4.174) to minimize software-version confounding.\nCheck the installed SAHI and Ultralytics versions in the new run manifest. The exporter
copies the original checkpoint into an isolated temporary folder,
converts once, records model bytes and checkpoint hashes and atomically
publishes a new immutable export. It never edits the original checkpoint.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$weights = "C:\Users\Admin\.cache\huggingface\hub\models--rfonod--geo-trax\snapshots\f512e0d1445e65fc2cf505d7474deccf33f11bf9\geotrax_hbb_yolov8s_1920_v1.pt"

# Optional CPU backend, kept in the isolated test virtual environment:
& $python -m pip install "ultralytics[export-openvino]==8.4.174"

# Convert original .pt to a separate non-quantized candidate model:
& $python scripts/phase3_export_openvino.py `
  --weights $weights --image-size 640 `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_openvino_export640_01"

$ov = "artifacts/phase3/sahi_detector_trials/W04_openvino_export640_01/checkpoint_openvino_model"

# A fresh 5-frame detector-only parity test, not tracker validation:
& $python scripts/phase3_sahi_audit.py `
  --video $video --fluid-tracks $truth --weights $weights `
  --runtime-model $ov `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_openvino640_probe5_01" `
  --start-frame 10750 --end-frame 11350 --sample-step 150 `
  --confidence 0.15 --image-size 640 --slice-height 640 `
  --slice-width 640 --overlap 0.20 --device cpu

# Detector gate may exit code 2; if report.json exists, it is evidence.
# Compare against cached PyTorch 640 tiles on identical five frames:
& $python scripts/phase3_compare_cached_detector_subsets.py `
  --reference-dir "artifacts/phase3/sahi_detector_trials/W04_spaced21_control640_01" `
  --candidate-dir "artifacts/phase3/sahi_detector_trials/W04_openvino640_probe5_01" `
  --output "artifacts/phase3/sahi_detector_trials/W04_torch_vs_openvino640_5frames01.json"
```

Actual outcomes are **unknown**; never promise the reported upstream
speedup on this model, video or CPU. Use the newly persisted runtime
model hash and reference .pt SHA to verify checkpoint lineage. The
comparator re-scores the previously cached .pt predictions on exactly
the same five frames and identifies the OpenVINO backend separately.
OpenVINO conversion may change detection scores numerically; observed
motorcycle parity is a required check. Do not equate original 21-frame
latency median to the new 5-frame median as a strictly paired benchmark.
Do not relax the detector gate and do not promote to production.

## May-26 W04 OpenVINO CPU five-frame experiment — completed locally

The user exported the frozen detector SHA-256
`7d462ae523b15f3679a83f74ab08aa79d2978f319ca2f0cf2892fbbb60df79da`
to an isolated, **non-quantized OpenVINO** 640-input model.
The successfully verified exported directory SHA-256 is
`feff81d24b5f7790d8b5ca34ec9d4641b3513ba87962ced52c7060491b0212ea`.
OpenVINO version **2026.4.1**, Ultralytics version **8.4.174**.
On Intel 12th Gen Core i5-1235U CPU, 640x640 slices with overlap
0.20 were evaluated on exactly video frames
10750, 10900, 11050, 11200, 11350. This **detector-only** test
used the unchanged checkpoint, 50px class-aware detector matching
and FLUID offset +1; no ByteTrack IDs.

| SAME five frames, sliced detector | PyTorch 640 | OpenVINO 640 |
|---|---:|---:|
| MOTORCYCLE correct-class matches | 80/93 | 80/93 |
| MOTORCYCLE recall | 86.02% | 86.02% |
| MOTORCYCLE precision | 75.47% | 76.19% |
| Overall recall | 162/178 = 91.01% | 162/178 = 91.01% |
| Overall precision | 162/219 = 73.97% | 162/217 = 74.65% |
| Unmatched predictions | 57 | 55 |
| Observed sliced median CPU time | 12.055s (21 different frames) | 1.611s (5 frames) |

The indicative `~7.5x` median ratio is NOT strictly paired in
timing cohort: a future same-five-frame PyTorch timing pass is needed
before asserting measured acceleration. The OpenVINO five-frame
detector gate still **failed both precision regression and max 5x
latency relative to its OWN OpenVINO full-frame-640 baseline**
(0.143s full-frame vs 1.611s sliced median). Do NOT change the
frozen detector gate or claim production readiness.

Although the same number of motorcycle truth observations was
matched, the model conversion may recover *different annotated vehicles*.
The new exact-FLUID-observation parity audit therefore compares
individual `(video_frame, FLUID track ID)` outcomes with the same
class-aware 50px detection-only association, using existing cached
prediction CSVs, and makes no ByteTrack identity claims.

### Next steps: zero-inference observation parity, then same-frame timing

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
$ref = "artifacts/phase3/sahi_detector_trials/W04_spaced21_control640_01"
$ov = "artifacts/phase3/sahi_detector_trials/W04_openvino640_probe5_01"

# STEP A: exact paired motorcycle observation identity, no model execution.
& $python scripts/phase3_paired_detector_truth_parity.py `
  --reference-dir $ref --candidate-dir $ov `
  --output "artifacts/phase3/sahi_detector_trials/W04_openvino_truth_parity5_01.json"

# STEP B: fresh PyTorch 640x640 tile timing on SAME 5 source frames.
# This reruns ONLY 5 images, not the video or tracker continuously.
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$weights = "C:\Users\Admin\.cache\huggingface\hub\models--rfonod--geo-trax\snapshots\f512e0d1445e65fc2cf505d7474deccf33f11bf9\geotrax_hbb_yolov8s_1920_v1.pt"
& $python scripts/phase3_sahi_audit.py `
  --video $video --fluid-tracks $truth --weights $weights `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_torch640_timing5_01" `
  --start-frame 10750 --end-frame 11350 --sample-step 150 `
  --confidence 0.15 --image-size 640 --slice-height 640 `
  --slice-width 640 --overlap 0.20 --device cpu

# Exit code 2 may mean detector gate rejected; report.json still saved.
$torch = Get-Content "artifacts/phase3/sahi_detector_trials/W04_torch640_timing5_01/report.json" -Raw | ConvertFrom-Json
$openvino = Get-Content "$ov/report.json" -Raw | ConvertFrom-Json
[pscustomobject]@{
  Torch5MedianS = $torch.sliced.latency_median_s
  OpenVINO5MedianS = $openvino.sliced.latency_median_s
  Torch5MotorcycleRecall = $torch.sliced.per_class.MOTORCYCLE.recall
  OpenVINO5MotorcycleRecall = $openvino.sliced.per_class.MOTORCYCLE.recall
} | Format-List
```

The timing probe compares the same five source-frame positions but
is still sequential session-level timing without repeated randomized
run order or warm-start controls. A longer paired throughput/quality
benchmark, inspection of prediction-level precision errors, and
independent unseen footage remain prerequisites for tracker promotion.
All outputs remain separate, immutable and non-production.

## OpenVINO paired five-frame observation parity and latency — completed

On October 8 the user executed the exact cached FLUID truth-observation parity
audit for 10750, 10900, 11050, 11200 and 11350, plus a fresh PyTorch
five-frame detector-only timing trial with the same image size and tile
geometry as the existing OpenVINO pilot. Source video and FLUID ground
truth remain local; original Geo-trax and T000 remain unchanged.

- **MOTORCYCLE FLUID observations:** 80 matched by both, zero matched
  only by PyTorch, zero matched only by OpenVINO, 13 matched by neither
  (93 truth observations). These are annotated *frame observations*, not
  ByteTrack or independent physical vehicle IDs.
- **CAR:** 81 matched by both, 0 backend-only, 3 by neither (84 annotated
  observations); **HEAVY_VEHICLE:** one matched by both. No BUS annotations
  were present. No class-level truth-observation disagreement was found.
- PyTorch sliced inference median **7.6989638 seconds/frame** across
  the same five sample locations; OpenVINO sliced median **1.6108624
  seconds/frame**. This is an observed **~4.78× median-time ratio**.
  Inference runs were conducted in separate sessions, without
  interleaved randomized repeat timings, so it is not a guaranteed
  benchmark speedup.
- **Same five-frame class-aware detector recall:** 162/178=91.01%
  overall and 80/93=86.02% MOTORCYCLE for both backends. Unmatched
  detector predictions were **57 PyTorch** versus **55 OpenVINO**;
  overall precision **73.97%** versus **74.65%**. Equal recovered
  annotated observations do not imply identical box confidence, geometry
  or false-positive behavior.
- The frozen detector-trial gate **still fails**: sliced precision is
  below allowed regression versus standard full-frame-640 and sliced
  CPU time exceeds the allowed relative latency limit. Do not promote
  any engine or IoU suppression threshold.

### Next: 21-frame OpenVINO parity across the same spaced W04 cohort

Rather than jump to a continuous SAHI ByteTrack run, reuse exactly
the original 21 predeclared source frames, 10750–11350 inclusive with
step 30. This is **still the May-26 tuning recording**, a wider
detector-only replication, not a holdout or an identity test.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$weights = "C:\Users\Admin\.cache\huggingface\hub\models--rfonod--geo-trax\snapshots\f512e0d1445e65fc2cf505d7474deccf33f11bf9\geotrax_hbb_yolov8s_1920_v1.pt"
$model = "artifacts/phase3/sahi_detector_trials/W04_openvino_export640_01/checkpoint_openvino_model"

# Run OpenVINO on the identical 21 cached PyTorch reference frames.
& $python scripts/phase3_sahi_audit.py `
  --video $video --fluid-tracks $truth --weights $weights `
  --runtime-model $model `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_openvino_spaced21_01" `
  --start-frame 10750 --end-frame 11350 --sample-step 30 `
  --confidence 0.15 --image-size 640 `
  --slice-height 640 --slice-width 640 --overlap 0.20 --device cpu

# The strict gate is expected to return code 2 while saving report.json.
# The comparator scores both modes on the EXACT same 21 samples.
& $python scripts/phase3_compare_cached_detector_subsets.py `
  --reference-dir "artifacts/phase3/sahi_detector_trials/W04_spaced21_control640_01" `
  --candidate-dir "artifacts/phase3/sahi_detector_trials/W04_openvino_spaced21_01" `
  --output "artifacts/phase3/sahi_detector_trials/W04_torch_ov_same21_01.json"

# Verify the *same annotated observations* were recovered.
& $python scripts/phase3_paired_detector_truth_parity.py `
  --reference-dir "artifacts/phase3/sahi_detector_trials/W04_spaced21_control640_01" `
  --candidate-dir "artifacts/phase3/sahi_detector_trials/W04_openvino_spaced21_01" `
  --output "artifacts/phase3/sahi_detector_trials/W04_torch_ov_truth_parity21_01.json"
```

Files are new/immutable: if the exact output path already exists,
inspect its `report.json` rather than repeating into that directory.
All three existing scripts have already been unit-tested for strict
provenance and fixed-frame denominators. Results will determine whether
OpenVINO's five-frame quality and runtime behavior persist in the
21-frame tuned cohort. Detector trial gate, tracking and true
held-out validation remain separate.

## W04 21-frame OpenVINO parity — completed; precision review next

The user executed the OpenVINO non-INT8 backend on the exact **21 May-26
W04 sampled frames** (10750 to 11350, every 30) previously evaluated
with PyTorch. All runs used the frozen original detector weights and
640×640 tiles (20% overlap); evaluation is **class-aware detector-only
50px**, **not the frozen class-agnostic P3B tracking score**.

| Same 21-frame detection-only sample | PyTorch SAHI | OpenVINO SAHI |
|---|---:|---:|
| Annotated observations | 739 | 739 |
| Overall matches | 671 | 670 |
| Overall recall | 90.80% | 90.66% |
| Overall precision | 78.02% | 78.18% |
| MOTORCYCLE matched / 380 | 321 | 320 |
| MOTORCYCLE recall | 84.47% | 84.21% |
| Unmatched predictions | 189 | 187 |
| CPU sliced inference median | 12.055s | 1.710s |

OpenVINO's *observed* median CPU inference is ~7.05x faster on the
identical frame positions, but these were separately timed sessions,
not interleaved, repeated random-order paired measurements.
**The strict detector gate still fails** because of sliced precision
relative to full-frame 640 and excessive sliced/full-frame latency.
OpenVINO remains **experimental and not eligible for production**.

Exact **FLUID observation ID** parity:
**MOTORCYCLE** = 319 matched by both, **2 PyTorch-only**,
**1 OpenVINO-only**, **58 neither** (380 annotated frame observations).
CAR = 345 matched by both (9 neither); HEAVY_VEHICLE = 5 both.
Thus three annotated motorcycle *observations* disagree.
No claimed ByteTrack identity consistency, no claim of three unique
motorcycles. Matching may differ due to converted detector numerics.

OpenVINO's 187 detector-unmatched predictions break down as
**89 MOTORCYCLE, 66 CAR, 20 HEAVY_VEHICLE and 12 BUS**.
FLUID includes **zero BUS annotations** in these 21 samples, so
BUS predictions are unmatched, not automatically physically absent.
The test cannot by itself distinguish detector false alarms,
incomplete FLUID annotation, class confusion, or clustered
real vehicles.

### Next: cache-only precision audit, no further inference

The script `phase3_sahi_unmatched_detection_review.py` reads the
immutable 21-frame OpenVINO CSV and report, recomputes all class-aware
50px matches using the established evaluator, rejects score drift
and mismatched checkpoints/exports, then reports class-specific
confidence bands and *nearest-center* indicators for the
187 unmatched observations. It also prioritizes the three
PyTorch/OpenVINO motorcycle observation disagreements for human
image review. Proximity to another prediction is **not proof of
duplicated box IoU**; no suppression is executed or proposed by
the code.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
& $python scripts/phase3_sahi_unmatched_detection_review.py `
  --audit-dir "artifacts/phase3/sahi_detector_trials/W04_openvino_spaced21_01" `
  --parity "artifacts/phase3/sahi_detector_trials/W04_torch_ov_truth_parity21_01.json" `
  --output "artifacts/phase3/sahi_detector_trials/W04_openvino_unmatched_review21_01.json"
```

Before any precision optimization, visually adjudicate a few
real motorcycles and unmatched high-confidence cars/large vehicles
using the original raw video/FLUID frames. Do not use evaluator
unmatched flags as a deployment-time filter; labels may be incomplete.
No model threshold fitting or tracker promotion is allowed on
this tuned W04 source. A separate held-out recording is still
mandatory.

## Evidence produced

- `*.txt`: Geo-trax 14-column track file with confirmed real IDs.
- `*.manifest.json`: exact model hash, versions, bounds, runtime and export counts.
- `*.pixel.json` and `*.identity.json`: fixed-window P3B report.
- `*.baseline_pixel.json`, `*.baseline_identity.json`, `*.gate.json`:
  optional same-window baseline and promotion decision.

Important: `x_stabilized_px` and `y_stabilized_px` duplicate raw coordinates
for **parser compatibility only**. The manifest explicitly marks
`stabilized_coordinates=false`. Never feed trial tracks directly into
georeferenced simulation or world-coordinates without separate calibration.

## Evidence gates

1. **Real API integration:** public SAHI sample with real YOLO checkpoint
   produces at least one confirmed ByteTrack ID and 14-column record.
2. **Development windows:** W01/W04/W05 fixed intervals with actual FLUID
   labels; track recall, precision, class agreement, pixel MAE, fragmentation
   and elapsed processing seconds.
3. **Unseen validation:** May-26 was already used for tuning and is **not**
   an untouched holdout. Require another recording before claiming
   generalization.
4. **Promotion:** only a model that satisfies same-cohort quality and latency
   guardrails and an untouched test can be considered for production.

Any failed or regressive experimental candidate should remain isolated.
There is no silent fallback that would mislabel Geo-trax outputs as SAHI.
