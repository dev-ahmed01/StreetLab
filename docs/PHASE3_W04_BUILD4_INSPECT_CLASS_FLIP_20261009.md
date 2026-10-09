# StreetLab Phase 3 — Build 4 original-source verification and cached class-flip attribution (9 October 2026)

**Experimental development evidence, not production approval or a T000 superiority claim.** This is a continuation of the real W04 201-frame Build 3 trial, not a new OpenVINO inference run.

## Original-source verification: passed

The user ran the repository's Build 4 `inspect` on the **original Windows workspace** and supplied `W04_continuous_build4_integrity_01.json`. Verified values:

| Check | Result |
|---|---|
| Status | `BUILD4_VERIFIED_BATCH_NOT_PRODUCTION` |
| Original video / FLUID source SHA verification | `true` |
| Exact `batch_report.json` SHA-256 | `dbb3f0539a788926c630c198f383421040f57938712c0de93531e9f694c00120` |
| Policy count | **25** |
| Consecutive evaluated frame count | **201** |
| Same-window T000 track baseline | **not provided** |
| Eligible for production | **false** |

This newly uploaded JSON has SHA-256 `31302c9a95c2ab8e5e9bd997ab14295c50c12b980dad0cce0a5908222f8f5fb4`. This closes the pending source-integrity check but **does not verify T000-relative accuracy, physical object classification, unseen-video performance or CPU p95**.

## Offline same-evidence detector/association attribution

Using the previously submitted unchanged `W04_CONTINUOUS_201_RESULTS.zip` (25 verified 14-column track files and **21,104** original per-tile boxes over 291 decoded frames), the new `scripts/phase3_offline_class_flip_attribution.py` matches each emitted tracker class-change event to **source pre-global-merge detection boxes from its current and previous observed frames**. It compares absolute bbox IoU ≥0.50, class IDs and source-frame consistency **without any inference, FLUID relabeling or tracker modification**.

The newly submitted Build 4 `batch_report_sha256` is validated against the *bytes of the batch report inside the ZIP*. All 25 policy track-file hashes and the raw premerge SHA are checked. Output is a JSON summary and event CSV, created in a new immutable directory. A five-test local suite passed (real fixture, synthetic class flip and overlap cases, corrupted provenance, duplicate frame-ID rejection).

### W04 raw-hypothesis overlap for the leading policy

| Quantity | `hard_nms_ios_0.30` |
|---|---:|
| Distinct tracker IDs | 119 |
| IDs changing emitted class | **27** |
| Observed class changes | **100** |
| Changes across adjacent observed frames | **80** |
| Current frame has overlapping raw detection of both former and new class | **50** |
| Current frame has overlapping raw detection of new class only | **50** |
| Current frame lacks new-class raw overlap | **0** |
| Prior track frame includes preceding emitted class in raw evidence | **100** |

Alternative policies:

| Policy | Class-changing IDs | Total class changes | Both-class raw hypothesis at flip | New-class-only raw hypothesis |
|---|---:|---:|---:|---:|
| `hard_nms_ios_0.30` | 27/119 | 100 | 50 | 50 |
| `hard_nms_ios_0.50` | 27/120 | 100 | 50 | 50 |
| `hard_nms_iou_0.30` | 37/189 | 114 | 51 | 63 |
| `weighted_fusion_ios_0.30` | 26/118 | 92 | 53 | 39 |
| `raw_unmerged` | 60/308 | 183 | 101 | 82 |

**Interpretation:** For the lead's 100 class flips, an overlapping same-class raw detector hypothesis exists for the current emitted class, and one also exists for the former class on 50 events. The raw evidence therefore documents both frame-to-frame detector class changes and simultaneous alternative-class hypotheses. It **does not causally identify** model uncertainty versus ByteTrack's cross-class matching; both may contribute. Inference that any event is a true physical class change is prohibited.

Most frequent directed transitions for the lead: BUS→HEAVY 20, HEAVY→BUS 18, CAR→HEAVY 18, HEAVY→CAR 16, MOTORCYCLE→CAR 13, CAR→MOTORCYCLE 12. Given that the 201-frame frozen FLUID cohort has **zero BUS truth points**, independent visual confirmation is needed; numeric FLUID class corrections are not justified.

## Reproducible command

In the experimental repo root, after pulling the latest branch, with the original `W04_box_tracking_batch201_01` directory and the SHA-verified Build 4 integrity JSON:

```powershell
$lab = ".\artifacts\phase3\sahi_detector_trials"
& ".\.venv-sahi-audit\Scripts\python.exe" scripts\phase3_offline_class_flip_attribution.py `
  --batch-dir "$lab\W04_box_tracking_batch201_01" `
  --integrity-report "$lab\W04_continuous_build4_integrity_01.json" `
  --output-dir "$lab\W04_class_flip_attribution_01"
```

The `--archive` option accepts the immutable batch ZIP instead of `--batch-dir`. The run refuses overwriting its destination and requires no detector or ByteTrack dependencies.

## Next engineering/research gate

1. Locate the original, **unaltered same-window T000** Geo-trax track evidence for source-video frames **10750–10950** (FLUID **10751–10951**, +1 offset). If unavailable, create a separate reproducible baseline run using the original T000 configuration and source provenance; do not substitute a truncated or mismatched benchmark.
2. Use cached detector-box replay for explicit *class-consistent versus class-agnostic tracker association* ablations, retaining the original 25 policies and reporting motorcycle/heavy class recall plus ID switches. Do **not** hide class flicker by relabeling benchmark truth or post-hoc changing existing tracks.
3. Once actual same-window T000 and class-preservation evidence are reviewed, preregister a single policy and validate on a genuinely untouched distinct source-video dataset. Require independent two-human physical review and measured p95 latency before any production decision.

**Draft PR #10 remains unmerged and research-only.**
