# StreetLab — Research submission without new physical inspection

**Scope:** Academically defensible **research prototype** using pre-existing real-traffic datasets and actual native SUMO counterfactuals on a **synthetic demonstration network**. No new site visit is required. This does not certify a real-road digital twin or authorize operational traffic decisions.

## One-command Windows research run

Prerequisites: Python 3.12 and dependencies from requirements.txt; SUMO and netconvert on PATH; TraCI available via SUMO_HOME/tools or installed with python -m pip install traci sumolib.

From a clean checkout, run: **.\scripts\STREETLAB_RESEARCH.ps1**

This runs the existing Phase 2 M6 study under real native SUMO (three paired assumed scenario ensemble members), includes a deliberate NEEDS_DATA refusal for unsupported input, and creates a fresh timestamped folder under artifacts/.

- **submission_evidence.json** — source hashes, frozen status, including recorded failure gates
- **submission_summary.md** — scientific summary, citations and documented limits
- **native_sumo_study.json** — real simulation study output, only if engine succeeds
- **native_sumo.log** — actual execution log
- **native_sumo_runtime/** — native simulation scratch files, NOT field evidence
- **research_results.md** — actual synthetic branch medians, per-member min–max and paired deltas; notes lower simulated arrivals under detours
- **research_metrics.csv** — 15 source-linked scenario/metric rows for plotting and audit
- **figure_01_queue.svg** through **figure_05_reroutes.svg** — auto-generated numerical figures, deliberately marked synthetic
- **research_results_manifest.json** — SHA-256 hashes binding the generated outputs to source evidence and native SUMO JSON

The exporter refuses unexecuted/altered evidence or overclaims of real-world impact. The three assumed assignment members do not imply statistical confidence intervals. A successful native run now produces these additional files automatically; an evidence-only run does not. To export results from an earlier successful native SUMO folder, run `python -m streetlab_integration.research_results --run-dir <existing-successful-output-directory>` in the StreetLab virtualenv.

For evidence-only mode without SUMO: **.\scripts\STREETLAB_RESEARCH.ps1 -EvidenceOnly**.

For an occupied web port 8000: **.\scripts\STREETLAB_M8.ps1 -Mode web -Port 8001** and open http://127.0.0.1:8001/research.

## Exact evidence boundaries

**A. Chennai real traffic trajectories.** Two checked-in historical Chennai trajectory spreadsheets and the immutable Phase 1 held-out report. Frozen flags: **capacity_gate_pass=false; holdout_generalization_pass=false; retuning_allowed=false**. The study did NOT pass its final predictive generalization gate. Previously computed vehicle counts and speed/spacing metrics are reported as previously evaluated, not independently recomputed here.

**B. FLUID real footage and benchmark.** The paper and publicly released FLUID dataset provide genuine intersection videos, trajectories, maps and traffic-light observations. The user-local W04 frozen footage is associated with this dataset. StreetLab's checked-in W04 development QA states point precision 86.70% and recall 91.37% for the original hard_nms_ios_0.30 policy on the audited 201-frame evaluation cohort. **These are not independent held-out results**, and that cohort had no BUS ground-truth points. The research tool accurately attributes these previously recorded scores and does not re-score source truth that is absent from GitHub.

**C. SUMO counterfactuals.** Running the actual SUMO engine on the Phase 2 M6 network provides baseline, guided-detour, and mixed-detour results. Chennai-derived behavior priors may be used, but junction topology, demand and intervention response are ASSUMED. Numerical differences are simulated consequences, not improvements measured at a real road.

**The Chennai and FLUID sites are different.** The tool must not fabricate road geometry, physical censuses, travel-time holdouts, coordinate transforms, data signatures, or M8 approval by combining unrelated sources.

## Suggested submission sections

1. Research problem and scope: observation and uncertainty-aware traffic scenario analysis.
2. Related work: original FLUID dataset article; Chennai heterogeneous trajectory analysis; FHWA simulation calibration; SUMO.
3. Platform methods: evidence registry, frozen video inference, track association, native SUMO, evidence sufficiency gates and integrity checks.
4. Experiment A: documented FLUID W04 development-label metrics and measurement limits. Run independently on real video later if possible.
5. Experiment B: original Chennai recorded traffic and frozen held-out generalization failure, no post-hoc retuning.
6. Experiment C: actual synthetic-network SUMO paired scenario outcomes and uncertainty, explicit assumed topology.
7. Threats to validity: site transfer, unknown geometry, source annotation noise, assumptions, capacity gap, no measured causal effects.
8. Reproducibility: input SHAs, code ref, exact commands, outputs, environment version, diagrams, screenshots, CI.

## Primary sources

- Chen, Y., et al. **FLUID: A Fine-Grained Lightweight Urban Signalized-Intersection Dataset of Dense Conflict Trajectories** (arXiv 2025, published descriptor 2026): https://arxiv.org/abs/2509.00497
- **FLUID released dataset v2 (2026), CC BY 4.0** DOI 10.6084/m9.figshare.29974954: https://figshare.com/articles/dataset/FLUID_A_Fine-Grained_Lightweight_Urban_Signalized-Intersection_Dataset_of_Dense_Conflict_Trajectories/29974954
- **FHWA Traffic Analysis Toolbox Volume III — Calibration (2019)**: https://ops.fhwa.dot.gov/publications/fhwahop18036/chapter5.htm
- **FHWA Guidelines on Microsimulation Calibration (2014)**: https://www.fhwa.dot.gov/publications/research/operations/13026/007.cfm
- **Eclipse SUMO framework, Alvarez Lopez et al. (ITSC 2018)**: https://eclipse.dev/sumo/docs/

## Interpretation and final gates

A software-native SUMO PASS is sufficient for a prototype demonstration, NOT calibrated prediction. Dataset labels are not independently verified physical census. Keep the independent M8 field-specific gate untouched and **production_release_approved=false**. For a future externally validated digital twin, use independent site-matched data or a genuine field survey. Do not turn the simulation into a claim that measured Bangalore congestion improved.

The research approach may be acceptable for a software/academic prototype submission, subject to your actual grading rubric and explicit discussion of the frozen failed holdout.

## Downloadable research-prototype evidence bundle

A successful original/native SUMO research run can now be packaged as a self-checking, non-overwriting ZIP:

```powershell
python -m streetlab_integration.research_release --run-dir artifacts/research_submission_YYYYMMDD_HHMMSS --output artifacts/streetlab_research_bundle.zip
```

This reads existing successful outputs only. It verifies the source-artifact SHA values and derived figure/table SHA values against the original `research_results_manifest.json`, enforces the failed Chennai holdout and synthetic-only SUMO statuses, and adds `manifest.json` with the ZIP's complete per-file SHA inventory. The workflow's [GitHub Actions research job](https://github.com/dev-ahmed01/StreetLab/actions/workflows/research-submission.yml) also creates an archive download automatically from its own verified **synthetic fixture/native SUMO run**. This Action cannot validate your private May25 Drive footage.

The original May25 autonomous object audit is **not silently bundled or generated**. A previously completed and verified `automated_object_audit.json` can be included explicitly with `--may25-audit PATH`, after the original source SHA-verified runner succeeds. If omitted, the archive manifest honestly states `NOT_INCLUDED_OR_NOT_VERIFIED`. No archive changes the frozen W04 policy, the Chennai failure, or the lack of real-site production approval.
