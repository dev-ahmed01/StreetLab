"""Reproducible, no-new-field-visit *research prototype* submission evidence.

This path combines TWO DIFFERENT study sites: checked-in, previously reported
Chennai trajectory/SUMO heldout evidence and local FLUID W04 development benchmark
documentation; optionally a live native SUMO decision-assurance demo on a
SYNTHETIC intersection. Never passes an M8 real-site gate, never combines their
counts, and never fabricates observed movements, geometry or travel times.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

CHENNAI = "data/processed/phase1_final_heldout_validation.json"
CHENNAI_RAW = (
    "data/raw/ChennaiTrajectoryData2.45-3.00PM.xlsx",
    "data/raw/ChennaiTrajectoryData3.00-3.15PM.xlsx",
)
FLUID_MANIFEST = "data/benchmarks/fluid_fidrt/manifest.json"
W04_DEVELOPMENT = "docs/PHASE3_W04_CONTINUOUS_201_FINAL_QA_20261009.md"
PERSONA = "data/models/sumo_persona_priors.csv"

REFERENCES = (
    {
        "title": "Chen et al. FLUID: Fine-Grained Lightweight Urban Signalized-Intersection Dataset (Scientific Data / preprint)",
        "url": "https://arxiv.org/abs/2509.00497",
        "purpose": "Primary published description of FLUID imagery, trajectories, signal data and benchmark constraints",
    },
    {
        "title": "FLUID published dataset archive (Figshare, CC BY 4.0)",
        "url": "https://figshare.com/articles/dataset/FLUID_A_Fine-Grained_Lightweight_Urban_Signalized-Intersection_Dataset_of_Dense_Conflict_Trajectories/29974954",
        "purpose": "Source dataset citation and license; local raw video/truth are not included in this Git checkout",
    },
    {
        "title": "FHWA Traffic Analysis Toolbox Vol III — Model Calibration (2019)",
        "url": "https://ops.fhwa.dot.gov/publications/fhwahop18036/chapter5.htm",
        "purpose": "Observation-specific calibration and the need for data-supported prediction",
    },
    {
        "title": "FHWA Guidance on Level of Effort for Microsimulation, Ch. 6 (2014)",
        "url": "https://www.fhwa.dot.gov/publications/research/operations/13026/007.cfm",
        "purpose": "Statistical calibration and multiple measured periods",
    },
    {
        "title": "Eclipse SUMO documentation",
        "url": "https://eclipse.dev/sumo/docs/",
        "purpose": "Native SUMO simulation and reproducible route/network experimentation",
    },
)

class EvidenceError(RuntimeError):
    pass


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def source(root: Path, rel: str) -> dict[str, Any]:
    p=root/rel
    if not p.is_file() or p.is_symlink():
        raise EvidenceError("Missing/untrusted evidence: "+rel)
    return {"path":rel,"sha256":digest(p),"size_bytes":p.stat().st_size}


def checked_json(root: Path, rel: str) -> dict[str, Any]:
    source(root,rel)
    try:
        result=json.loads((root/rel).read_text(encoding="utf-8"))
    except (ValueError,UnicodeError) as err:
        raise EvidenceError("Invalid immutable JSON receipt: "+rel) from err
    if not isinstance(result,dict):
        raise EvidenceError("Invalid root JSON object: "+rel)
    return result


def chennai_evidence(report: dict[str,Any]) -> dict[str,Any]:
    """Preserve frozen failures and do not promote calibration/holdout claims."""
    if report.get("phase1_closed") is not True or report.get("retuning_allowed") is not False:
        raise EvidenceError("Chennai Phase 1 is not frozen/closed")
    if report.get("calibration_locked") is not True:
        raise EvidenceError("Chennai calibration is not locked")
    simulation=report["simulation"]
    if simulation.get("validation_total_vehicles",0)<=0 or simulation.get("departed_vehicles",0)<=0:
        raise EvidenceError("No nonzero recorded validation simulation")
    status=str(report.get("phase1_status",""))
    capacity=report.get("capacity_gate_pass")
    generalization=report.get("holdout_generalization_pass")
    if type(capacity) is not bool or type(generalization) is not bool:
        raise EvidenceError("Missing authoritative frozen validation gate")
    return {
        "study":"Chennai trajectory / Phase 1",
        "role":"previously evaluated frozen held-out evidence (not rerun in this workflow)",
        "phase1_status":status,
        "capacity_gate_pass":capacity,
        "holdout_generalization_pass":generalization,
        "calibration_locked":True,
        "retuning_allowed":False,
        "validation_total_vehicles":simulation["validation_total_vehicles"],
        "boundary_crossing_coverage":simulation["boundary_crossing_coverage"],
        "teleport_events":simulation["teleport_events"],
        "collision_events":simulation["collision_events"],
        "speed_objective":report.get("speed_objective"),
        "spacing_objective":report.get("spacing_objective"),
        "conclusion":(
            "Frozen held-out acceptance failed. This cannot support a claim of "
            "calibrated predictive accuracy for a new real site."
            if not (capacity and generalization)
            else "Frozen recorded gates passed; transfer to a new site remains unvalidated."
        ),
        "limitations":report.get("known_limitations",[]),
    }


def fluid_evidence(manifest: dict[str,Any], text: str) -> dict[str,Any]:
    studies=manifest.get("studies",[])
    heldout=[s for s in studies if s.get("role")=="held_out_validation"]
    if len(heldout)!=1 or heldout[0].get("study_id")!="fluid_fidrt_20250526":
        raise EvidenceError("FLUID held-out split does not match frozen manifest")
    # These scores are extracted from the checked-in development QA report,
    # NOT recalculated from source video/truth CSV. The distinction matters.
    match=re.search(
        r"\|\s*\`hard_nms_ios_0\.30\`\s*\|\s*\*\*(\d+(?:\.\d+)?)%\*\*\s*"
        r"\|\s*(\d+(?:\.\d+)?)%\s*\|",
        text,
    )
    if not match:
        raise EvidenceError("Frozen W04 source QA metric table cannot be parsed")
    p,r=(float(x) for x in match.groups())
    if not (0<=p<=100 and 0<=r<=100):
        raise EvidenceError("Out of bounds W04 source metrics")
    return {
        "study":"FLUID FIDRT, May 26 W04 window",
        "role":"previously documented development-only FLUID-label comparison",
        "held_out_manifest_study":heldout[0]["study_id"],
        "policy":"hard_nms_ios_0.30",
        "reported_point_precision_pct":p,
        "reported_point_recall_pct":r,
        "scores_recomputed_in_this_run":False,
        "data_split":"W04 tuned/development window; NOT independent final held-out validation",
        "claim":"Reported performance against FLUID labels only, not verified physical precision or generalization.",
        "limitations":[
            "Raw independent FLUID truth and complete 20250526 video are not committed to GitHub.",
            "Development-policy metrics are not a new blind or entire-video validation.",
            "The QA cohort has zero BUS FLUID truth points; BUS recall is not measurable there.",
            "The observed source-video study is distinct from the Chennai behavior study.",
        ],
    }


def validate_simulation(result: dict[str,Any]) -> dict[str,Any]:
    """Fail closed; never claim a simulation happened from a copied placeholder."""
    if result.get("incomplete_study",{}).get("status")!="NEEDS_DATA":
        raise EvidenceError("Native research demo did not refuse incomplete evidence")
    if result.get("supported_study",{}).get("status")!="SUPPORTED":
        raise EvidenceError("Assumed demonstration study is not supported")
    g=result.get("guardrails",{})
    if (g.get("confidence_fabricated") is not False or
        g.get("authority_remains_final_decision_maker") is not True or
        g.get("live_runtime_mutated_by_evaluated_branch") is not False):
        raise EvidenceError("Simulation guardrail failed")
    meta=result.get("ensemble",{}).get("metadata",{})
    if (meta.get("provenance")!="ASSUMED" or
        meta.get("same_snapshot_for_all_members") is not True or
        meta.get("members",0)<2):
        raise EvidenceError("Not a verified paired synthetic ensemble")
    rows=result.get("ensemble",{}).get("branch_summaries",{})
    if not {"BASELINE","DETOUR_GUIDED","DETOUR_MIXED"}<=set(rows):
        raise EvidenceError("Native scenario branches missing")
    # Do not silently collapse the stochastic run into a 'proven impact' number.
    return {
        "status":"SUMO_EXECUTED_SYNTHETIC_ONLY",
        "engine":"actual native SUMO through TraCI and existing M6 Decision Lab",
        "ensemble_members":meta["members"],
        "same_snapshot":True,
        "baseline_and_interventions":list(rows),
        "scorecard":rows,
        "study_evidence":{
            "speeds":"CALIBRATED from prior Chennai research",
            "geometry":"ASSUMED synthetic demonstration junction",
            "demand":"ASSUMED synthetic demonstration traffic",
            "turn_movements":"ASSUMED",
            "alternate_route":"ASSUMED",
        },
        "simulation_not_real_world_effect":True,
        "site_specific_field_validation":False,
        "production_release_approved":False,
    }


def _sumo_environment() -> dict[str, str]:
    """Inherit the normal environment and discover SUMO_HOME from a real installation.

    Windows package managers often expose sumo.exe on PATH without setting
    SUMO_HOME. The bundled sumolib/traci Python modules then remain invisible.
    Never silently download or change the project's frozen Python environment.
    """
    sumo=shutil.which("sumo")
    netconvert=shutil.which("netconvert")
    if not sumo or not netconvert:
        raise EvidenceError(
            "Native SUMO prerequisites missing: run 'sumo --version' and "
            "'netconvert --version' in this same PowerShell session."
        )
    env=dict(os.environ)
    supplied=env.get("SUMO_HOME", "").strip()
    supplied_tools=Path(supplied)/"tools" if supplied else None
    if not (supplied_tools and supplied_tools.is_dir()):
        # Common layout on Windows/Linux: <SUMO_HOME>/bin/sumo(.exe).
        home=Path(sumo).resolve().parent.parent
        if (home/"tools").is_dir():
            env["SUMO_HOME"]=str(home)
    return env


def _native_failure(output: Path, returncode: int) -> EvidenceError:
    """Keep the real failure visible, not merely 'exit 1'."""
    path=output/"native_sumo.log"
    try:
        lines=path.read_text(encoding="utf-8", errors="replace").splitlines()
        detail="\n".join(lines[-24:])[-3200:] or "(native process emitted no output)"
    except OSError:
        detail="(native engine log unavailable)"
    remedy=""
    if "No module named 'traci'" in detail or 'No module named "traci"' in detail:
        remedy=(
            "\nSUMO's Python TraCI module is missing. In the StreetLab venv "
            "run: python -m pip install traci sumolib. Alternatively point "
            "SUMO_HOME at the real SUMO installation containing tools/traci."
        )
    elif "No module named 'sumolib'" in detail or 'No module named "sumolib"' in detail:
        remedy=(
            "\nSUMO's Python sumolib module is missing. In the StreetLab venv "
            "run: python -m pip install traci sumolib."
        )
    return EvidenceError(
        f"Native SUMO failed (exit {returncode}); log: {path}\n"
        f"Last native output:\n{detail}{remedy}"
    )


def native_demo(root:Path, output:Path, *, timeout:int=720) -> dict[str,Any]:
    env=_sumo_environment()
    cmd=[
        sys.executable,str(root/"scripts"/"phase2_m6_study_demo.py"),
        "--project-root",str(root),
        "--workdir",str(output/"native_sumo_runtime"),
        "--output",str(output/"native_sumo_study.json"),
        "--members","3",
    ]
    try:
        p=subprocess.run(cmd,cwd=root,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=timeout,
            check=False,encoding="utf-8",errors="replace",env=env)
    except subprocess.TimeoutExpired as err:
        content=err.stdout or b""
        if isinstance(content,bytes):
            content=content.decode("utf-8","replace")
        (output/"native_sumo.log").write_text(
            str(content)[-60000:],encoding="utf-8"
        )
        raise EvidenceError(
            f"Native SUMO timed out; partial log: {output/'native_sumo.log'}"
        ) from err
    (output/"native_sumo.log").write_text(p.stdout[-60000:],encoding="utf-8")
    if p.returncode!=0:
        raise _native_failure(output,p.returncode)
    result=checked_json(output,"native_sumo_study.json")
    return validate_simulation(result)


def study(root:Path,output:Path,run_sumo:bool) -> dict[str,Any]:
    files=[
        source(root,p)
        for p in (CHENNAI,FLUID_MANIFEST,W04_DEVELOPMENT,PERSONA,*CHENNAI_RAW)
    ]
    chennai=chennai_evidence(checked_json(root,CHENNAI))
    fluid=fluid_evidence(
        checked_json(root,FLUID_MANIFEST),
        (root/W04_DEVELOPMENT).read_text(encoding="utf-8")
    )
    result={
        "schema_version":1,
        "track":"RESEARCH_SUBMISSION_NO_NEW_FIELD_SURVEY",
        "method":"Two separated empirical dataset studies plus a distinct native synthetic SUMO decision experiment",
        "dataset_summaries":{"chennai":chennai,"fluid_w04":fluid},
        "source_manifest":files,
        "no_physical_visit_needed_for_this_research_workflow":True,
        "datasets_are_from_different_sites":True,
        "physical_site_geometry_inferred_from_video":False,
        "field_census_claimed":False,
        "external_site_transfer_validated":False,
        "real_world_intervention_effect_proven":False,
        "production_release_approved":False,
        "m8_real_site_acceptance": "NOT_RUN_NOT_SUBSTITUTED",
        "native_sumo": {"status":"NOT_EXECUTED"} if not run_sumo else None,
        "references":list(REFERENCES),
    }
    if run_sumo:
        result["native_sumo"]=native_demo(root,output)
    result["research_workflow_status"]=(
        "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION"
        if run_sumo else "RECORDED_EVIDENCE_ONLY_NATIVE_SUMO_NOT_RUN"
    )
    return result


def markdown(d:dict[str,Any]) -> str:
    ch=d["dataset_summaries"]["chennai"];fl=d["dataset_summaries"]["fluid_w04"]
    native=d["native_sumo"]
    lines=[
        "# StreetLab — No-New-Field-Visit Research Submission Evidence",
        "",
        "**Track:** Research prototype / previously collected empirical datasets; not an operational digital twin.",
        "**Technical simulation:** "+native["status"]+".",
        "**Acceptance:** M8 physical-site acceptance NOT performed; production release NOT approved.",
        "",
        "## Research objective",
        "Use recorded real-world trajectory/video evidence to evaluate a traffic observation and simulation research platform, then demonstrate a separate reproducible SUMO intervention study. Do not assert both empirical datasets represent the same junction.",
        "",
        "## Dataset A — Chennai observed trajectories",
        f"- Original source: two checked-in trajectory spreadsheets, evidence SHA values in \`submission_evidence.json\`.",
        f"- Previously recorded validation vehicles: **{ch['validation_total_vehicles']}**.",
        f"- Boundary observation coverage: **{ch['boundary_crossing_coverage']:.2%}**.",
        f"- Capacity gate: **{'PASS' if ch['capacity_gate_pass'] else 'FAIL'}**.",
        f"- Frozen held-out generalization: **{'PASS' if ch['holdout_generalization_pass'] else 'FAIL'}**.",
        f"- Frozen decision: **{ch['phase1_status']}**. No retuning allowed.",
        "- This report faithfully reuses the existing immutable held-out evaluation; it is not an independent recomputation of the full trajectories.",
        "",
        "## Dataset B — FLUID W04 real recorded video research",
        f"- Frozen study: **{fl['held_out_manifest_study']}**.",
        f"- Development policy: \`{fl['policy']}\`.",
        f"- Previously documented FLUID-label precision: **{fl['reported_point_precision_pct']:.2f}%**.",
        f"- Previously documented FLUID-label recall: **{fl['reported_point_recall_pct']:.2f}%**.",
        "- These scores are copied from the versioned real-video development QA document with SHA provenance; **no unseen-video evaluation is implied**.",
        "- They are not physical ground truth or evidence of generalization to a new Indian road.",
        "",
        "## Actual SUMO counterfactual experiment",
        f"- Native run state: **{native['status']}**.",
    ]
    if native["status"]=="SUMO_EXECUTED_SYNTHETIC_ONLY":
        lines.extend([
            f"- Three SUMO ensemble members, paired baseline vs guided/mixed detours on a **synthetic demonstration network**.",
            "- Model demonstration uses Chennai-derived speed/persona priors, but geometry, demand and turn topology are **assumptions**.",
            "- Incomplete evidence is rejected as NEEDS_DATA; only explicitly declared assumed scenarios are simulated.",
            "- Actual per-branch numerical scorecards are in \`native_sumo_study.json\`; never report simulated changes as measured causal improvements.",
        ])
    else:
        lines.append("- Not executed. No simulated outcomes can be claimed until the native SUMO run succeeds.")
    lines.extend([
        "",
        "## Results and submission verdict",
        f"**{d['research_workflow_status']}**.",
        "Source provenance is recorded and a research workflow is reproducible without visiting a new road.",
        "**The existing Chennai frozen held-out study failed; the current system does not demonstrate validated citywide or site-specific predictive accuracy.**",
        "A video-to-specific-junction-to-independent-field-census-to-calibrated-SUMO pipeline cannot be certified without linked original observations.",
        "",
        "## Threats to validity",
        "1. Prior W04 evaluation was used during development, so its reported performance is not an independent holdout score.",
        "2. Existing FLUID labels have documented class mismatches; no automated ground-truth repair is claimed.",
        "3. Chennai held-out generalization failed. Low departure delay or no collision alone does not establish a calibrated micro-model.",
        "4. The synthetic SUMO network is not reconstructed from the FLUID video or Chennai original site geometry.",
        "5. Dataset permissions and the academic submission rubric should be confirmed prior to public redistribution.",
        "",
        "## Reproduction",
        "\`python -m streetlab_integration.research_submission --output-dir artifacts/research_01 --run-sumo\`",
        "Reports use only checked-in source references, real native SUMO when enabled, and non-overwriting output folders.",
        "The W04 local video worker can be independently tested in the usual local UI; its output is not silently inserted into this evidence report.",
        "",
        "## Sources and methodological references",
    ])
    lines.extend(f"- [{ref['title']}]({ref['url']}) — {ref['purpose']}" for ref in d["references"])
    return "\n".join(lines)+"\n"


def main() -> None:
    parser=argparse.ArgumentParser(description="Reproducible research submission without a NEW field inspection")
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir",type=Path,required=True,
                        help="A NEW output directory, never an existing one")
    parser.add_argument("--run-sumo",action="store_true",
                        help="Execute actual native SUMO on a separate assumed synthetic research network")
    args=parser.parse_args()
    root=args.project_root.resolve()
    dest=args.output_dir
    if not dest.is_absolute():dest=root/dest
    if dest.exists():
        parser.error("Output already exists; choose a new folder to preserve evidence")
    dest.mkdir(parents=True,exist_ok=False)
    try:
        result=study(root,dest,args.run_sumo)
        result["created_at_utc"]=datetime.now(timezone.utc).isoformat()
        (dest/"submission_evidence.json").write_text(
            json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
        )
        (dest/"submission_summary.md").write_text(markdown(result),encoding="utf-8")
    except (EvidenceError,ValueError,KeyError,OSError) as error:
        (dest/"FAILED_RESEARCH_RUN.txt").write_text(
            str(error)+"\nNo acceptance claims made.\n",encoding="utf-8"
        )
        print("FAILED: "+str(error),file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({
        "status":result["research_workflow_status"],
        "output_dir":str(dest),"sources":len(result["source_manifest"]),
        "chennai_holdout_pass":result["dataset_summaries"]["chennai"]["holdout_generalization_pass"],
        "native_sumo":result["native_sumo"]["status"],
        "production_release_approved":False,
    },indent=2))


if __name__=="__main__":
    main()
