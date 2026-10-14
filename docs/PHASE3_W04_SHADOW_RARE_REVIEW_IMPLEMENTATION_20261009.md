# StreetLab Phase 3 W04 — Two-lane shadow rare-vehicle review evidence (2026-10-09)

**Development only. The primary IoS.30/ByteTrack output remains authoritative, unchanged and not promoted.** There is no new production model, FLUID relabeling, vehicle-count augmentation or T000 comparison.

## Architecture

- **Primary lane:** preserve original `hard_nms_ios_0.30` → ByteTrack `control_ios030.txt` as the trusted 14-column track export, exact original SHA-256 `6c695e384e05421ec25dbe450f00a83fb6a72a659a2d162a1122d67c7d0ce418`. This lead had 86.70% point precision, 91.37% recall, 71 contiguous FLUID identity switches and 15 fragmented FLUID truth tracks in W04.
- **Shadow evidence lane:** compare saved `rare_iou05.txt` rare-class tracks with primary same-frame, same-class tracks via **maximum-cardinality Hungarian pairing at bbox IoU >= 0.50**. Consider unmatched hybrid rows only as hypotheses. Cross-check against *all classes* of original primary tracks and source 21,104 raw tile detector boxes. Flag competing-class hypotheses separately.
- **Review queue:** group hypotheses by hybrid tracklet ID/class; require >=3 **consecutive unmatched** source frames and >=2 unmatched rows at confidence >=0.50. Neither this filter nor raw-box support certifies physical novelty. Reviewers must inspect original source frames; FLUID label classes are known to be incomplete/inconsistent for BUS and AUTO_RICKSHAW.

## Actual inspected data

The user uploaded both ZIPs and we independently checked original control byte parity, raw detector evidence hash and all four unified track+class-evidence pairs. A separate local audit generated a stronger reviewed CSV bundle: `W04_SHADOW_REVIEW_V2.zip`, containing an **all-191-row shadow table**, **all 40 shadow tracklets**, **7 human-review candidates**, and input SHA manifests. No source video inference or tracking rerun was performed.

| Metric | Actual |
|---|---:|
| Additional rare-class tracked observations spatially unmatched at same-class IoU >= 0.50 | **191** |
| Unmatched MOTORCYCLE observations | **113** |
| Unmatched HEAVY_VEHICLE observations | **78** |
| Distinct hybrid IDs with unmatched observations | **40** |
| IDs passing persistence and confidence filter | **7 (5 HEAVY, 2 MOTORCYCLE)** |
| Unmatched observations within the seven review-tracklets | **115** |
| Those 115 observations with supporting same-class detector box | **115** |
| Queue candidates containing at least one competing-class raw detection | **5 of 7** |

## Proposed review order, not credibility ranking

| Class | Hybrid ID | Unmatched rows | High-conf rows | Cross-class original control overlaps | Competing raw-class frames |
|---|---:|---:|---:|---:|---:|
| HEAVY_VEHICLE | 122 | 13 | 13 | 0 | 0 |
| MOTORCYCLE | 22 | 6 | 2 | 0 | 0 |
| HEAVY_VEHICLE | 137 | 20 | 18 | 2 | 2 |
| HEAVY_VEHICLE | 97 | 13 | 11 | 1 | 2 |
| HEAVY_VEHICLE | 96 | 12 | 9 | 8 | 8 |
| MOTORCYCLE | 102 | 44 | 4 | 0 | 26 |
| HEAVY_VEHICLE | 2 | 7 | 4 | 5 | 6 |

Every candidate is **PENDING HUMAN REVIEW**, not 'missed-vehicle verified'. Alternative-class overlap may indicate a class-label disagreement rather than new physical object. A long tracklet may be a repeated false positive, and a high detector confidence is not a ground-truth probability.

## Implemented

- `scripts/phase3_w04_shadow_review_queue.py` accepts the original 25-policy continuous benchmark ZIP and unified hybrid result ZIP. It verifies all 25 saved native track file hashes, original raw source boxes, original/unified manifests, all 4 candidate track+class sidecar hashes and exact frozen primary track byte identity.
- Uses bounded source frames 10660–10950, scores hypothesis pairing only on frames 10750–10950, and writes a new immutable folder with `shadow_observations.csv`, `shadow_tracklet_review.csv`, `human_review_queue.csv`, `shadow_lane_report.json` and SHA-256 manifest. No original source, FLUID annotations or primary track rows are changed.
- `tests/test_phase3_w04_shadow_review_queue.py` validates frame/class grouping, deterministic matching, 3-frame+confidence filters, cross-class ambiguity, malformed/duplicate original tracking rows, immutable output refusal, and uploaded fixture parity where available. Added to main Phase 3 CI sparse checkout.

## How to reproduce if needed

```powershell
$python = '.\.venv-sahi-audit\Scripts\python.exe'
& $python scripts\phase3_w04_shadow_review_queue.py --original-batch-zip 'W04_CONTINUOUS_201_RESULTS.zip' --unified-replay-zip 'W04_UNIFIED_HYBRID_REPLAY_01.zip' --output-dir 'artifacts\phase3\sahi_detector_trials\W04_shadow_review_queue_01'
```

The archive paths must reference exact immutable source packages; use absolute paths if those ZIPs are stored elsewhere. This is an optional reproducibility command, **not** a new required W04 benchmark run.

## Next quality gate

Generate visual review crops/contact sheets from original source video frame numbers, request at least two independent human reviewers, separately adjudicate whether proposals represent (a) truly missed physical vehicles, (b) alternate class of an already-tracked vehicle, (c) a duplicate, or (d) unsupported/unclear image evidence. Only an external ground-truth revision process may update labels; this script cannot change FLUID or promote any candidate automatically. Later validate on different, untouched footage and measure CPU p95.

**Warning:** zero true-vehicle additions have been established. The shadow review layer is implemented and testable, but production streaming deployment and independent physical validation are still pending.