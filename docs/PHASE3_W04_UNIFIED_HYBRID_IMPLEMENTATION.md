# W04 — integrated engineering stage: one-shot hybrid ByteTrack replay

**Research only.** The user requested to move beyond repeated single-policy iterations. This stage bundles all three previously supported hybrid suppression approaches into **one SHA-verified, real ByteTrack experiment** using the 291 consecutive cached detector frames. It **does not decode the original video, call OpenVINO, modify original FLUID, overwrite historical Geo-trax, or promote production**.

## Implemented in the experimental branch

Components: `streetlab_phase3/video/w04_hybrid_rare_suppression.py`, `streetlab_phase3/video/w04_class_evidence_stream.py`, `scripts/phase3_w04_unified_hybrid_replay.py`, `scripts/RUN_W04_UNIFIED_HYBRID_REPLAY.ps1` and corresponding synthetic regression tests.

The unified runner rejects the batch unless Build 4 `verify_batch(...,verify_originals=True)` checks all 25 original track hashes, original video and FLUID source hashes, the original 21,104 raw boxes, 291 decoded-source frame indexes (10660–10950), 90 warmup frames, and 201 frozen scoring frames. It then runs four separate real ByteTrack sequences using the **same original tracker parameters and FPS**:

| Replay name | Policy | Purpose |
|---|---|---|
| `control_ios030` | Original Hard NMS IoS .30 | Mandatory exact original-score reproduction |
| `rare_iou05` | CAR/BUS IoS .30 + MOTORCYCLE/HEAVY IoU .50 | Primary, moderate rare-class preservation |
| `rare_center_gate` | CAR/BUS IoS .30 + 0.35×minimum-side center guard for rare classes | Secondary recall-biased hybrid |
| `rare_cross_tile_center_gate` | Same center guard, requiring separate input tile IDs before suppression | Stress-control for possible duplicate proliferation |

The control **must exactly reproduce** the original saved point counts, FLUID match counts, identity switches and fragmentation, and per-class correct match counts. If any equality fails, the runner aborts before publishing the candidate scorecards. This prevents silent runtime/version drift or incorrect cached detection ordering from creating false hybrid accuracy claims.

Every candidate produces the frozen 14-column raw track export, SHA-256, original +1 frame/50-pixel precision/recall, MOTORCYCLE/CAR/HEAVY correct-class diagnostics, separate FLUID identity fragmentation/switch metrics and CPU stage medians. A separate CSV includes original instantaneous class and optional **detector-supported causal stabilized class**, keeping the original track classes and IDs immutable.

### Development safety filter (not a release decision)

Each candidate is flagged only when it meets *all* of the following against the original IoS .30 control on exactly the same 201 frames: point precision decline no worse than 2pp, point recall decline no worse than 1pp, no MOTORCYCLE or HEAVY correct-class recall loss, CAR correct-class recall loss no worse than 1pp, **no added FLUID identity switches**, and truth-track fragmentation no worse than 2pp. Failing candidates stay fully recorded (no result cherry-picking); passing candidates are **not** automatically adopted. No BUS recall conclusion is allowed because the frozen cohort has zero BUS truth points.

## Run exactly once on the original Windows workstation

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git switch codex/phase3-engine-shootout
git pull --ff-only origin codex/phase3-engine-shootout
& ".\scripts\RUN_W04_UNIFIED_HYBRID_REPLAY.ps1"
```

The PowerShell script fails on missing original evidence or existing output, executes all four trackers, validates exact control parity, prints an overview of all candidates and automatically packages a single ZIP: `W04_UNIFIED_HYBRID_REPLAY_01.zip`. This is the only next experiment artifact required, **not** a sequence of manual mini-tests. It uses the already-installed `.venv-sahi-audit` environment containing real ByteTrack.

## What is and is not established

The previous 25-policy continuous scores and 291-frame consolidated offline box/class counterfactuals are established **development evidence**. Hybrid candidates retained more plausible rare-class detections in the cached geometry but were not rescored through real tracker + FLUID when first proposed. The implementation here closes that design gap **once its real Windows replay is completed and validated**. The code and synthetic CI tests do not themselves justify asserting improved identity or physical recall.

Historical original T000 is unrecovered in the scanned workspace; any future Geo-trax run must be labeled a new replacement baseline. Separate held-out video, independent human BUS/rare-class review and end-to-end CPU p95 remain mandatory before production promotion. Draft PR #10 stays open and unmerged.