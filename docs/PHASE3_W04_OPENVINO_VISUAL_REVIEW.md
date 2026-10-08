# W04 OpenVINO visual case review — 15 source-video crops

**Status:** provisional human visual interpretation of user-local gallery, **NOT corrected FLUID ground truth**. Source: `W04_openvino_visual_review21_01.zip` supplied during the October 8, 2026 review; `index.json` lists 15 side-by-side raw/overlay JPEGs. The ZIP stays local to the user's conversation; no CCTV frames or image bytes committed to GitHub.

The gallery renders the **same original raw 3840×2160 frame crop** at left, and identical crop with **gold FLUID annotated centers, blue PyTorch detection centers, green OpenVINO detection centers, magenta review target** at right. The CSV audit stores centers only, **not original raw SAHI box extents, slice IDs, NMS groups or tracking identities**. Every review here is about individual *frame observations*, not unique physical vehicles or 21-frame-level FP prevalence. Video source is the already-tuned May-26 W04, not held-out evidence.

## Provisional image-by-image interpretation

| Image | Frame / review focus | What is actually visible in the raw image | Working hypothesis / limits |
|---|---|---|---|
| 01 | 10990 MOTORCYCLE, FLUID 2390, OpenVINO-only match | Motorcycle/two-wheeler near crosswalk under selected marker | Vehicle is present; slight prediction/backend differences require more than a single center to explain match disagreement |
| 02 | 11020 MOTORCYCLE, FLUID 2418, PyTorch-only match | Dark two-wheeler / rider-like figure on paved island; other two-wheeler-like objects also nearby | PyTorch center covers selected truth location; OpenVINO no match at the selected point. No tracker inference |
| 03 | 11230 MOTORCYCLE, FLUID 2386, PyTorch-only match | Small light two-wheeler at intersection/crosswalk beneath marker | PyTorch-only detection/truth association; requires inspecting exact exported detection coordinates before attributing reason |
| 04 | 10990 MOTORCYCLE close to a matched prediction | **Multiple distinct two-wheelers** near the curb/sidewalk, with several green markers | Dangerous to automatically collapse nearby centers: separate physical motorcycles coexist; exact selected same-bike duplicate remains uncertain |
| 05 | 10930 MOTORCYCLE near another matched prediction | One apparent two-wheeler/rider at median; multiple OpenVINO green markers closely cluster around it | **Stronger visual duplicate candidate**, but no SAHI box IoU/NMS group evidence; review the exact boxes before proposing suppression |
| 06 | 11200 MOTORCYCLE close to matched prediction | Several separate two-wheelers around an island; target green center appears displaced to the right of a visible two-wheeler | Could be redundant/poorly localized detection, but still an intersection with distinct nearby objects; no blanket center-distance suppression |
| 07 | 10780 high-confidence unmatched MOTORCYCLE | A visually plausible motorcyclist/vehicle is plainly present near the magenta circle on an open lane | Apparent **real object without nearby motorcycle FLUID label** (nearest class label ~665px), not safe to count as confirmed detector hallucination |
| 08 | 11080 high-confidence unmatched MOTORCYCLE | Dark small figure near sidewalk and crossing; identity as motorcycle vs person is **visually ambiguous** at this crop resolution | Candidate person/two-wheeler class confusion, needs nearby frames or original resolution, not a confirmed false alarm |
| 09 | 11260 high-confidence unmatched CAR | Real small **green commercial/utility vehicle** at magenta circle | Class ontology disagreement is plausible: nearest differently labeled FLUID center is ~2.51px away; not an absent object |
| 10 | 11050 high-confidence unmatched CAR | Real gray/white car within a queue of multiple distinct cars; more than one prediction marker over the selected car | Probable repeated center for an existing car or matching/annotation association; neighboring cars are real and must be preserved |
| 11 | 11200 high-confidence unmatched CAR | Similar queue of real closely spaced cars, likely recurring vehicles from frame 11050 | Possible repeat detector association/centering behavior, not proof from two separate snapshots of two unique FP objects |
| 12 | 10960 unmatched HEAVY_VEHICLE | **Long white truck/box commercial vehicle** carrying equipment/cargo on roof | Genuine large vehicle near another-class FLUID annotation (~6.83px). Class ontology must be checked |
| 13 | 10840 unmatched BUS | Clearly visible white full-size bus directly beneath OpenVINO BUS marker | **Not a hallucinated bus**; BUS label missing or assigned to a different canonical class in evaluation |
| 14 | 10750 unmatched BUS | Clearly visible white bus with red/dark side edge under BUS marker | **Not a hallucinated bus**; compare raw FLUID `type` for nearby track (11/12 BUS predictions had other-class FLUID centers within 50px) |
| 15 | 10900 unmatched HEAVY_VEHICLE | Large commercial vehicle (appears truck/large box-type vehicle) beneath target marker | Physical vehicle present; class agreement, body shape and source FLUID type need inspection |

### Most important constraints discovered

1. **Evaluation label/ontology problem is real.** BUS predictions under a genuine physical bus, and CAR/HEAVY_VEHICLE predictions over actual commercial vehicles, mean that an evaluator-unmatched prediction is **not equivalent** to a physical false-positive detection. In the 21-frame detector audit there were zero canonical BUS FLUID annotations and 12 unmatched OpenVINO BUS observations; 11 of those were within 50px of *other-class* annotated centers. Do **not** reassign missing ground-truth BUS labels without first inspecting the unnormalized FLUID `type` value and canonical class-mapping policy.
2. **Real duplicate candidates exist but require original box evidence.** In gallery 05 a single rider/vehicle looks multiply marked; other examples (04, 06) have multiple *distinct* motorcycles very close together. Nearest-center distance ≤25px alone is **unsafe** as a deployment-time suppression criterion.
3. **High-confidence unmatched detections often correspond to real objects** (gallery 07, 09, 12–15). Confidence cutoffs or matched/unmatched filtering would destroy genuine traffic observations.
4. Three backend-observation disagreements (gallery 01–03) involve visible two-wheeler-like objects; the source pixel center and model runtime conversion can affect assignment. These do **not** demonstrate long-run tracking identity loss.
5. May-26 W04 is tuning footage. This qualitative review does not change, correct or replace the frozen detector-only 50px matcher or original P3B, T000, identity benchmark or historic PR metrics.

## Frozen next-step experiment (do not choose rules from this review)

**First: annotation/ontology adjudication before suppression.** Build a *separate*, manually adjudicated evidence sidecar for a predeclared limited set including the 15 visual cases, preserving the original FLUID annotations exactly as provided. For each case record physical object presence, visual vehicle type, raw FLUID `type` (or no label), mapped class, whether a single vehicle has multiple markers, visibility/occlusion ambiguity and confidence. Preserve `UNKNOWN` rather than assuming a class. Independent second-reviewer labels are preferable.

**Second: inspect raw SAHI boxes or regenerate only fixed, bounded raw inference evidence**, if necessary, with original slice origins, xyxy coordinates, NMS/merge outcomes and source frame ID. Do **not** infer bounding-box IoU or NMS behavior from the current center-only CSV. Any candidate geometry-only rule must apply blindly to **all predictions**, never use FLUID matched/unmatched flags to choose removals. Compare correct-class recall, overall recall, precision, duplicate boxes, cars, motorcycles, object/track continuity and detection CPU latency against unchanged sources.

**Third: untouched validation and promotion gates.** Fixing evaluation label coverage/ontology does not authorize editing FLUID ground truth silently or rewriting frozen historical scorecards. Shadow adjudication and any new matcher must get separate filenames/scorecards. Require continuous tracking with proper warm-up and independent unseen CCTV before any production adoption. PR #10 stays DRAFT.

## Next local check: exact original FLUID `type` near each reviewed vehicle

The gallery suggests label gaps and type-ontology differences, but we have
**not yet inspected the raw FLUID CSV for the 15 selected positions**.
Before declaring any bus missing from ground truth, inspect the nearest
raw `type` string, its canonical mapping, and center distance. The
new audit preserves *all* raw types, even classes omitted by the
four-class detector scorer (e.g., `van` → `LIGHT_COMMERCIAL`).

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
& $python scripts/phase3_raw_fluid_label_review.py `
  --index "artifacts/phase3/sahi_detector_trials/W04_openvino_visual_review21_01/index.json" `
  --precision-review "artifacts/phase3/sahi_detector_trials/W04_openvino_unmatched_review21_01.json" `
  --fluid-tracks $truth `
  --output "artifacts/phase3/sahi_detector_trials/W04_raw_fluid_label_nearest15_01.json"
```

The command is **zero inference** and **zero video decoding**. It checks
the gallery's source references, reads only the original CSV, and lists
up to three nearest *raw* labels and Euclidean center distances for each
of 15 cases. Nearest-neighbor labels are hypotheses, never proof that
the annotation describes the same physical vehicle. The original FLUID
CSV remains byte-for-byte unchanged, and no score is rewritten. Preserve
the JSON output and report any raw bus/heavy/light-commercial disparities
rather than quietly relabeling the video.
