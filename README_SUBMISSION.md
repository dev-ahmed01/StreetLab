# StreetLab — submission and demo handoff (October 2026)

## What to submit

**StreetLab is an evidence-first, local-first traffic observation and decision-support research prototype.** It combines a real-video-derived traffic observation pipeline and source-locked FLUID evaluation with evidence sufficiency checks and **executed native SUMO synthetic scenario demonstrations**. It is **not** a field-calibrated digital twin, a live traffic-signal controller, or production-certified traffic infrastructure.

**Software:** the repository's `main` branch after the research-release PR merges: https://github.com/dev-ahmed01/StreetLab

**Research evidence:** download the `streetlab-research-demo-evidence` artifact from the successful [research-submission GitHub Actions workflow](https://github.com/dev-ahmed01/StreetLab/actions/workflows/research-submission.yml). The artifact contains a source-hashed `streetlab_research_bundle.zip` generated from an **actual native SUMO run of synthetic network scenarios** in CI. GitHub CI does **not** have the user's private May25 video or W04 OpenVINO model and cannot substantiate independent May25 re-verification.

**Historical May25 analysis:** `research/MAY25_FULL_COHORT_REPLAY_2026-10-23.md` records the independently recomputed saved-coordinates/tracking replay of the frozen 120-frame cohort. It is **not** the canonical original-video-SHA-verified `automated_object_audit.json`. Do not set `STREETLAB_MAY25_AUDIT_PATH` to a derivative report or claim the read-only gate succeeded until the original-source runner has created the verified canonical file.

**Six-phase acceptance:** `README_W04_COMPLETION.md` and `streetlab_integration/w04_completion_gate.py` implement the exactly six-phase fail-closed acceptance receipt. A missing canonical source-verified May25 object audit means the full gate remains **BLOCKED** despite the saved-cohort replay; the honest research prototype can still be submitted with its limitations stated clearly.

## Demonstrate the product

The product runs on Windows with Python 3.12, dependencies in `requirements.txt`, and locally installed SUMO for native simulation. For **an already set up** StreetLab checkout, the existing `STREETLAB_M8.ps1` launcher runs the local web service on `127.0.0.1:8000` (or another supplied port). Open:

- **/** — workspace, projects and evidence-first navigation
- **/research** — frozen FLUID development scores, historical May25 audit gate, Chennai failure, simulation/research evidence and limitations
- **/reports** — provenance export
- **/release** — operator release-readiness and evidence blockers
- **/fieldwork** — blank fieldwork data-collection templates, not fabricated measurements

A project with original video-based processing requires the separate frozen W04 model and locally retained research media. The model and original 505 MB video are deliberately **not committed** to the public repository. The web app is a **local** interface, not a publicly hosted service.

For a submission-day demonstration that does not depend on recomputing large videos, use the already produced GitHub Actions evidence archive to show the saved results, manifest, five synthetic comparison figures and scientific caveats. Then show the local `/research` and `/release` routes, preserving a visible *not verified* state if the private full-cohort canonical audit is not present. Do **not** manufacture extra proof just to fill the UI.

## Defensible research summary

- **Frozen May25 historical regression:** 150 original decoded frames; 120 evaluated (video 60–179; FLUID 61–180), original +1 alignment and 50px class-agnostic Hungarian matching. **7,892 FLUID points, 8,076 saved model points, 7,299 matches; F1 = 91.42034068136272%; precision = 90.37890044576523%; recall = 92.48606183476938%.**
- **Repeated class disagreement:** 236 HEAVY_VEHICLE annotation-relative wrong-class matches arise from **two original FLUID track IDs (7 and 36)**, 120 and 116 matched frames, respectively, each consistently predicted CAR by model track IDs 26 and 29. The model's agreement is not independent proof of physical vehicle type.
- **BUS limitation:** 185 predicted BUS-frame records across 12 predicted BUS IDs; zero annotated BUS truth points. Physical BUS precision/recall **cannot** be certified. Do not equate 184 unmatched BUS points with 184 proven false physical detections.
- **Predicted track stability:** 132 model tracks across the frozen saved cohort; 16 changed model class during their sampled frames. This is model-label consistency, not adjudicated physical vehicle identity.
- **Frozen May26 development:** W04 tuned-development precision/recall previously reported 86.70% / 91.37% on its 201-frame evaluation; any cross-date May25 comparison is **descriptive and unpaired**, not an untouched holdout.
- **Chennai historical field-transfer:** **FAILED** its recorded held-out generalization/capacity gate. Keep the failed outcome and no-retuning provenance intact.
- **Native SUMO:** real simulator binary execution on **assumed network geometry, routing/demand and response assignments**. Paired branch results from assumptions are a conditional synthetic study, not a measured reduction in Bangalore traffic.
- **CPU performance:** ~1.4 seconds/frame is a detector-stage median, **not** measured full pipeline end-to-end p95.
- **No production approval:** original same-window T000 is absent; independent physical taxonomy labels, field geometry calibration, genuinely unseen heldout transfer and production authorization remain unavailable.

## Evidence-integrity and reproducibility

The GitHub evidence ZIP includes `manifest.json` with SHA-256 of each included file (including its generated README; self-referential manifest omitted by design). The release code rejects altered native SUMO log and human-readable summary bytes relative to the original results manifest, altered generated figures/tables, production claims, or missing status fields. It does not contain the private source video.

Use the saved original W04 policy `hard_nms_ios_0.30` without retuning. Never use greedy nearest-neighbor in place of the original Hungarian assignment or change the May25 truth/prediction cohort. The completion gate checks policy-lock/protocol fingerprints from the specific May25 preflight, refusing unrelated, contradictory or malformed evidence.

## Submission acceptance boundary

**Research-prototype deliverable:** code repository, 120-frame frozen cohort numerical replay report, native SUMO synthetic-results ZIP with source hashes, reproducible local web app, CI tests, fixed six-phase completion gate and clear limits.

**Not delivered or claimed:** complete original-video-rescanned May25 autonomous audit with freshly verified MP4 SHA, independent holdout pass, calibrated real-junction prediction, measured field intervention effects, or production release.

This is the intentionally finite outcome permitted by the six-phase roadmap. There is no new reviewer, crop, or manual re-upload stage.
