"""M5 immutable project-specific scenario proposal and actual paired SUMO CLI.

Nine matched seed/load conditions, baseline + intervention per condition.
No runtime from HTTP; no unverified baseline can create a scenario proposal.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from streetlab_integration.video_jobs import VideoStore, _uuid
from streetlab_integration.reconstruction import _json_bytes
from streetlab_integration.site_baseline import verified_baseline, _folder as baseline_folder
from streetlab_integration.primary_runner import sha
from streetlab_integration.site_scenarios import (
    ScenarioError, validate_scenario, experiment_model, completed_trip_times,
    paired_scorecard,
)
from streetlab_integration.site_sumo import render_sumo

ALLOWED_STATIC={"proposal.json","quality.json"}
PER_RUN_LIMIT_BYTES=120_000_000
MAX_WALL_SECONDS=960
SUBPROCESS_SECONDS=50
SCENARIO_RE=re.compile(r"[a-f0-9]{64}")


def _folder(store: VideoStore, project_id: str, digest: str) -> Path:
    return store.root/"projects"/project_id/"scenarios"/digest


def _runtime(directory: Path) -> dict:
    folder=directory/"runtime"
    receipt=folder/"experiment_receipt.json"
    if not receipt.exists():
        return {"status":"NOT_EXECUTED","real_world_outcome_verified":False}
    if folder.is_symlink():
        raise ScenarioError("Untrusted experiment output directory")
    result=json.loads(receipt.read_text(encoding="utf-8"))
    hashes=result.get("files")
    if (not isinstance(hashes,dict) or not hashes or len(hashes)>75
        or any(not re.fullmatch(
            r"(?:baseline|scenario)\.net\.xml|(?:low|mid|high)\.rou\.xml|"
            r"(?:low|mid|high)_(?:baseline|scenario)_s(?:42|43|44)\."
            r"(?:tripinfo|summary)\.xml|netconvert\.log", name)
            for name in hashes)):
        raise ScenarioError("Unexpected experiment output manifest")
    required={"baseline.net.xml","scenario.net.xml","netconvert.log"}
    required.update(f"{label}.rou.xml" for label in ("low","mid","high"))
    for label in ("low","mid","high"):
        for seed in (42,43,44):
            for name in ("baseline","scenario"):
                required.add(f"{label}_{name}_s{seed}.tripinfo.xml")
                required.add(f"{label}_{name}_s{seed}.summary.xml")
    if set(hashes)!=required:
        raise ScenarioError("Partial paired simulation evidence manifest is not publishable")
    for name,digest in hashes.items():
        item=folder/name
        if item.is_symlink() or sha(item)!=digest:
            raise ScenarioError("Experiment output SHA integrity failed")
    if result.get("status") not in {"SIMULATED_COMPARISON_ELIGIBLE",
                                   "INDETERMINATE_INCOMPLETE_TRIPS"}:
        raise ScenarioError("Unexpected experiment result state")
    if result.get("real_world_outcome_verified") is not False:
        raise ScenarioError("Simulation result cannot assert real-world verification")
    if result.get("paired_runs")!=9:
        raise ScenarioError("Partial experimental grid is not publishable")
    return result


def verified_scenario(store: VideoStore, project_id: str, revision: str) -> dict:
    store.project(_uuid(project_id))
    if not isinstance(revision,str) or SCENARIO_RE.fullmatch(revision) is None:
        raise ScenarioError("Invalid counterfactual revision")
    folder=_folder(store,project_id,revision)
    if folder.is_symlink() or not folder.is_dir():
        raise ScenarioError("Scenario proposal not found")
    hashes=json.loads((folder/"SHA256SUMS.json").read_text(encoding="utf-8"))
    if not isinstance(hashes,dict) or set(hashes)!=ALLOWED_STATIC:
        raise ScenarioError("Scenario manifest is missing original proposal/quality")
    for name,digest in hashes.items():
        member=folder/name
        if member.is_symlink() or sha(member)!=digest:
            raise ScenarioError("Scenario proposal evidence integrity failed")
    proposal=json.loads((folder/"proposal.json").read_text(encoding="utf-8"))
    if (hashlib.sha256(_json_bytes(proposal)).hexdigest()!=revision
        or proposal.get("project_id")!=project_id):
        raise ScenarioError("Scenario proposal project/content binding mismatch")
    baseline=verified_baseline(store,project_id,proposal["baseline_revision"])
    receipt=baseline_folder(store,project_id,baseline["revision"])/"runtime"/"run_receipt.json"
    if (baseline["runtime"].get("status")!="BASELINE_FIDELITY_CHECKED"
        or baseline["runtime"].get("real_site_sumo_allowed") is not True
        or sha(receipt)!=proposal["m4_runtime_receipt_sha256"]
        or baseline["model"]["source_video_sha256"]!=proposal["source_video_sha256"]
        or baseline["model"]["source_track_sha256"]!=proposal["source_track_sha256"]
        or baseline["model"]["spatial_revision"]!=proposal["spatial_revision"]):
        raise ScenarioError("Underlying M4 field baseline or original evidence is untrusted")
    quality=json.loads((folder/"quality.json").read_text(encoding="utf-8"))
    runtime=_runtime(folder)
    if runtime["status"]!="NOT_EXECUTED":
        if (runtime.get("scenario_revision")!=revision or
            runtime.get("baseline_revision")!=proposal["baseline_revision"] or
            runtime.get("run_engine")!="ACTUAL_SUMO_PAIRED_BASELINE_INTERVENTION"):
            raise ScenarioError("Experiment execution receipt does not match proposal/source revisions")
        planned={
            (label,seed) for label in proposal["sensitivity"]["demand_multipliers"]
            for seed in proposal["sensitivity"]["seeds"]
        }
        recorded={
            (p.get("demand_multiplier"),p.get("seed"))
            for p in runtime.get("records",[])
        }
        if len(runtime.get("records",[]))!=9 or planned!=recorded:
            raise ScenarioError("Experiment paired seed/demand conditions differ from approved design")
    return {"revision":revision,"proposal":proposal,"quality":quality,"runtime":runtime}


def create_scenario(store: VideoStore, project_id: str, baseline_revision: str,
                    raw: dict) -> dict:
    baseline=verified_baseline(store,project_id,baseline_revision)
    scenario=validate_scenario(raw,baseline)
    scenario["project_id"]=project_id
    receipt=baseline_folder(store,project_id,baseline_revision)/"runtime"/"run_receipt.json"
    scenario["m4_runtime_receipt_sha256"]=sha(receipt)
    digest=hashlib.sha256(_json_bytes(scenario)).hexdigest()
    folder=_folder(store,project_id,digest)
    if not folder.exists():
        folder.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix=".proposal-",dir=folder.parent))
        try:
            (stage/"proposal.json").write_bytes(_json_bytes(scenario))
            quality={
                "status":"EXPERIMENT_PROPOSAL_ONLY",
                "scenario_execution_completed":False,
                "real_world_outcome_verified":False,
                "scenario_eligibility":"M4_BASELINE_QA_PASS_OPERATOR_ATTESTED",
                "missing_for_real_world_claims":[
                    "externally verified site survey and reviewed field counts",
                    "genuine independent M4 holdout validity",
                    "physical intervention feasibility and local compliance review",
                    "out-of-sample observed intervention effects",
                ],
            }
            (stage/"quality.json").write_bytes(_json_bytes(quality))
            (stage/"SHA256SUMS.json").write_bytes(_json_bytes({
                name:sha(stage/name) for name in sorted(ALLOWED_STATIC)}))
            os.replace(stage,folder)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    result=verified_scenario(store,project_id,digest)
    with store.connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS scenario_latest ("
                     "project_id TEXT PRIMARY KEY,revision TEXT NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO scenario_latest VALUES(?,?) "
                     "ON CONFLICT(project_id) DO UPDATE SET revision=excluded.revision",
                     (project_id,digest))
    return result


def latest_scenario(store: VideoStore, project_id: str) -> dict:
    store.project(project_id)
    with store.connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS scenario_latest("
                     "project_id TEXT PRIMARY KEY,revision TEXT NOT NULL)")
        row=conn.execute("SELECT revision FROM scenario_latest WHERE project_id=?",
                         (project_id,)).fetchone()
    if not row:
        return {"status":"NOT_CONFIGURED",
                "quality":{"status":"NEEDS_DATA",
                           "missing_evidence":["M4 fully executed independent baseline QA",
                                               "explicitly reviewed operational intervention"]},
                "runtime":{"status":"NOT_EXECUTED","real_world_outcome_verified":False}}
    return verified_scenario(store,project_id,row["revision"])


def _binary(name: str, root: str | None) -> str:
    if root:
        path=Path(root)/(name+".exe" if os.name=="nt" else name)
        if not path.is_file():
            raise ScenarioError(f"Trusted local {name} binary not found")
        return str(path)
    found=shutil.which(name)
    if not found:
        raise ScenarioError(f"Install SUMO/netconvert or supply --sumo-bin-dir; {name} absent")
    return found


def _call(args: list[str], cwd: Path, *, deadline: float) -> subprocess.CompletedProcess:
    remaining=deadline-time.monotonic()
    if remaining<=0:
        raise ScenarioError("Paired simulation exceeded total 16-minute runtime budget")
    try:
        result=subprocess.run(args,cwd=cwd,check=False,capture_output=True,text=True,
                              timeout=min(SUBPROCESS_SECONDS,remaining))
    except (subprocess.TimeoutExpired,OSError) as exc:
        raise ScenarioError("Bounded SUMO experimental subprocess failed or timed out") from exc
    if result.returncode!=0:
        raise ScenarioError("SUMO experiment rejected the scenario assumptions: "+result.stderr[-1000:])
    return result


def run_scenario(store: VideoStore, project_id: str, revision: str,
                 *, executable_dir: str | None = None) -> dict:
    item=verified_scenario(store,project_id,revision)
    if item["runtime"]["status"]!="NOT_EXECUTED":
        raise ScenarioError("Experiment already has immutable results; create a new proposal for changes")
    folder=_folder(store,project_id,revision)
    if (folder/"runtime").exists():
        raise ScenarioError("Scenario runtime exists or is incomplete; do not overwrite evidence")
    proposal=item["proposal"]
    base=verified_baseline(store,project_id,proposal["baseline_revision"])
    root=baseline_folder(store,project_id,base["revision"])
    netconvert=_binary("netconvert",executable_dir)
    sumo=_binary("sumo",executable_dir)
    deadline=time.monotonic()+MAX_WALL_SECONDS
    stage=Path(tempfile.mkdtemp(prefix=".scenario-run-",dir=folder))
    try:
        # Bind baseline network to the M4 real executed and verified net.
        shutil.copyfile(root/"runtime"/"site.net.xml",stage/"baseline.net.xml")
        change=experiment_model(base["model"],proposal,1.0,changed=True)
        compiled=render_sumo(change)
        for name,body in compiled.items():
            (stage/name).write_bytes(body)
        args=[netconvert,"--node-files","nodes.nod.xml",
              "--edge-files","edges.edg.xml","--connection-files","connections.con.xml"]
        if "signals.tll.xml" in compiled:
            args+=["--tllogic-files","signals.tll.xml"]
        args+=["--output-file","scenario.net.xml"]
        built=_call(args,stage,deadline=deadline)
        if not (stage/"scenario.net.xml").is_file():
            raise ScenarioError("netconvert returned success without the changed network")
        (stage/"netconvert.log").write_text(
            (built.stdout+ "\n"+built.stderr)[-4000:],encoding="utf-8")
        pairs=[]
        retained={"baseline.net.xml","scenario.net.xml","netconvert.log"}
        for label,load in zip(("low","mid","high"),proposal["sensitivity"]["demand_multipliers"]):
            demand_model=experiment_model(base["model"],proposal,load,changed=False)
            route_file=label+".rou.xml"
            (stage/route_file).write_bytes(render_sumo(demand_model)["routes.rou.xml"])
            retained.add(route_file)
            for seed in proposal["sensitivity"]["seeds"]:
                observations={}
                for condition in ("baseline","scenario"):
                    prefix=f"{label}_{condition}_s{seed}"
                    tripfile=prefix+".tripinfo.xml"
                    summaryfile=prefix+".summary.xml"
                    args=[sumo,"--net-file",condition+".net.xml",
                          "--route-files",route_file,"--begin","0",
                          "--end",str(base["model"]["window"]["duration_s"]+3600),
                          "--summary-output",summaryfile,
                          "--tripinfo-output",tripfile,"--no-step-log","true",
                          "--time-to-teleport","-1","--seed",str(seed)]
                    _call(args,stage,deadline=deadline)
                    for output in (tripfile,summaryfile):
                        file=stage/output
                        if not file.is_file() or file.stat().st_size>PER_RUN_LIMIT_BYTES:
                            raise ScenarioError("SUMO output missing or exceeds experiment size budget")
                        retained.add(output)
                    observations["baseline" if condition=="baseline" else "intervention"]=(
                        completed_trip_times(demand_model,(stage/tripfile).read_bytes()))
                pairs.append({"seed":seed,"demand_multiplier":load,**observations})
        score=paired_scorecard(pairs)
        hashes={name:sha(stage/name) for name in sorted(retained)}
        score.update(scenario_revision=revision,baseline_revision=base["revision"],
                     executed_at_unix=time.time(),files=hashes,
                     run_engine="ACTUAL_SUMO_PAIRED_BASELINE_INTERVENTION",
                     source="HYPOTHETICAL_MODEL_EXPERIMENT_NOT_FIELD_EVIDENCE")
        (stage/"experiment_receipt.json").write_bytes(_json_bytes(score))
        # No corrupt or partially written publication allowed.
        os.replace(stage,folder/"runtime")
        return _runtime(folder)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser=argparse.ArgumentParser(description="Run a fully reviewed M5 site counterfactual")
    parser.add_argument("--workdir",default=".streetlab-m5")
    parser.add_argument("--project",required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--sumo-bin-dir",default=None)
    args=parser.parse_args()
    store=VideoStore(args.workdir)
    print(json.dumps(run_scenario(store,args.project,args.revision,
                                  executable_dir=args.sumo_bin_dir),indent=2))


if __name__=="__main__":
    main()
