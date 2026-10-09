"""M4 immutable observed-site SUMO package and bounded, standalone baseline CLI.

No SUMO subprocess runs in HTTP threads, and no model, video or executable
paths are taken from client requests. M4 never executes counterfactual scenarios.
"""
from __future__ import annotations

import argparse
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
from streetlab_integration.reconstruction import _json_bytes, verified_reconstruction
from streetlab_integration.primary_runner import sha
from streetlab_integration.site_inputs import SiteInputError, validate_site
from streetlab_integration.site_sumo import render_sumo, evaluate_tripinfo

BLUEPRINT={"site_input.json","quality.json","nodes.nod.xml","edges.edg.xml",
           "connections.con.xml","signals.tll.xml","routes.rou.xml"}
EXEC_FILES={"site.net.xml","summary.xml","tripinfo.xml","run.log"}

def _folder(store,project_id,revision):
    return store.root/"projects"/project_id/"baselines"/revision

def save_baseline(store: VideoStore, project_id: str, spatial_revision: str,
                  raw: dict) -> dict:
    store.project(_uuid(project_id))
    spatial=verified_reconstruction(store,project_id,spatial_revision)
    model,quality=validate_site(raw,spatial)
    model.update(project_id=project_id,
                 source_video_sha256=spatial["model"]["source_video_sha256"],
                 source_track_sha256=spatial["model"]["source_track_sha256"])
    revision=hashlib.sha256(_json_bytes(model)).hexdigest()
    target=_folder(store,project_id,revision)
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix=".blueprint-",dir=target.parent))
        try:
            data={"site_input.json":_json_bytes(model),
                  "quality.json":_json_bytes(quality),**render_sumo(model)}
            for name,payload in data.items():
                (staging/name).write_bytes(payload)
            (staging/"SHA256SUMS.json").write_bytes(
                _json_bytes({name:sha(staging/name) for name in sorted(data)}))
            os.replace(staging,target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    result=verified_baseline(store,project_id,revision)
    with store.connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS baseline_latest ("
                     "project_id TEXT PRIMARY KEY,revision TEXT NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO baseline_latest VALUES(?,?) "
                     "ON CONFLICT(project_id) DO UPDATE SET revision=excluded.revision",
                     (project_id,revision))
    return result


def _runtime(directory: Path) -> dict:
    folder=directory/"runtime"
    receipt=folder/"run_receipt.json"
    if not receipt.exists():
        return {"status":"NOT_EXECUTED","real_site_sumo_allowed":False}
    if folder.is_symlink():
        raise SiteInputError("Runtime path changed to a symlink")
    result=json.loads(receipt.read_text(encoding="utf-8"))
    hashes=result.get("files")
    if not isinstance(hashes,dict) or not {"site.net.xml","tripinfo.xml","summary.xml","run.log"}<=set(hashes) or not set(hashes)<=EXEC_FILES:
        raise SiteInputError("Missing or unexpected baseline runtime outputs")
    for name,digest in hashes.items():
        file=folder/name
        if file.is_symlink() or sha(file)!=digest:
            raise SiteInputError("Baseline runtime output integrity failed")
    if result.get("status") not in {"BASELINE_FIDELITY_CHECKED","BASELINE_NEEDS_REVIEW"}:
        raise SiteInputError("Unexpected baseline QA state")
    if result.get("real_site_sumo_allowed")!=(result["status"]=="BASELINE_FIDELITY_CHECKED"):
        raise SiteInputError("Invalid baseline QA gate")
    return result


def verified_baseline(store: VideoStore, project_id: str, revision: str) -> dict:
    store.project(_uuid(project_id))
    if not isinstance(revision,str) or not re.fullmatch("[a-f0-9]{64}",revision):
        raise SiteInputError("Baseline revision must be a SHA-256 hex digest")
    path=_folder(store,project_id,revision)
    if path.is_symlink() or not path.is_dir():
        raise SiteInputError("Baseline not found or is untrusted")
    checks=json.loads((path/"SHA256SUMS.json").read_text(encoding="utf-8"))
    required={"site_input.json","quality.json","nodes.nod.xml",
              "edges.edg.xml","connections.con.xml","routes.rou.xml"}
    if not isinstance(checks,dict) or not required<=set(checks) or not set(checks)<=BLUEPRINT:
        raise SiteInputError("Site package manifest has unknown or missing members")
    for name,expected in checks.items():
        member=path/name
        if member.is_symlink() or sha(member)!=expected:
            raise SiteInputError("Site baseline package integrity failed")
    model=json.loads((path/"site_input.json").read_text(encoding="utf-8"))
    if hashlib.sha256(_json_bytes(model)).hexdigest()!=revision or model["project_id"]!=project_id:
        raise SiteInputError("Site baseline manifest revision/project mismatch")
    upstream=verified_reconstruction(store,project_id,model["spatial_revision"])
    if (model["source_video_sha256"]!=upstream["model"]["source_video_sha256"] or
        model["source_track_sha256"]!=upstream["model"]["source_track_sha256"]):
        raise SiteInputError("Upstream observed site evidence changed")
    quality=json.loads((path/"quality.json").read_text(encoding="utf-8"))
    return {"revision":revision,"model":model,"quality":quality,"runtime":_runtime(path)}


def latest_baseline(store: VideoStore, project_id: str) -> dict:
    store.project(project_id)
    with store.connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS baseline_latest ("
                     "project_id TEXT PRIMARY KEY,revision TEXT NOT NULL)")
        row=conn.execute("SELECT revision FROM baseline_latest WHERE project_id=?",
                         (project_id,)).fetchone()
    if row is None:
        return {"status":"NOT_CONFIGURED",
                "quality":{"status":"NEEDS_DATA","real_site_sumo_allowed":False,
                           "missing_evidence":[
                               "M3 metric geometry and manually reviewed turns",
                               "field measured road arms and lane connections",
                               "separate physical vehicle census by movement/class",
                               "independent observed baseline travel-time holdout"]},
                "runtime":{"status":"NOT_EXECUTED","real_site_sumo_allowed":False}}
    return verified_baseline(store,project_id,row["revision"])


def run_baseline(store: VideoStore, project_id: str, revision: str,
                 executable_dir: str | None = None) -> dict:
    """On-demand command-line operation with bounded CPU/runtime and immutable result."""
    snapshot=verified_baseline(store,project_id,revision)
    directory=_folder(store,project_id,revision)
    if (directory/"runtime").exists():
        raise SiteInputError("Immutable baseline already executed; create a new revision")
    if executable_dir:
        root=Path(executable_dir)
        binaries=[root/("netconvert.exe" if os.name=="nt" else "netconvert"),
                  root/("sumo.exe" if os.name=="nt" else "sumo")]
        if not all(p.is_file() for p in binaries):
            raise SiteInputError("SUMO executables not present at operator CLI directory")
        binaries=[str(p) for p in binaries]
    else:
        binaries=[shutil.which("netconvert"),shutil.which("sumo")]
    if not all(binaries):
        raise SiteInputError("Install SUMO + netconvert or supply --sumo-bin-dir on the CLI")
    stage=Path(tempfile.mkdtemp(prefix=".sumo-exec-",dir=directory))
    try:
        for name in ("nodes.nod.xml","edges.edg.xml","connections.con.xml",
                     "signals.tll.xml","routes.rou.xml"):
            if (directory/name).is_file():
                shutil.copyfile(directory/name,stage/name)
        args=[binaries[0],"--node-files","nodes.nod.xml",
              "--edge-files","edges.edg.xml","--connection-files","connections.con.xml"]
        if (stage/"signals.tll.xml").is_file():
            args+=["--tllogic-files","signals.tll.xml"]
        args+=["--output-file","site.net.xml"]
        try:
            built=subprocess.run(args,cwd=stage,text=True,capture_output=True,
                                 timeout=240,check=False)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise SiteInputError("netconvert could not run within the bounded limit") from exc
        if built.returncode!=0 or not (stage/"site.net.xml").is_file():
            raise SiteInputError("netconvert failed site topology checks: "+built.stderr[-1000:])
        horizon=snapshot["model"]["window"]["duration_s"]+3600
        args=[binaries[1],"--net-file","site.net.xml","--route-files","routes.rou.xml",
              "--begin","0","--end",str(horizon),
              "--summary-output","summary.xml",
              "--tripinfo-output","tripinfo.xml",
              "--no-step-log","true","--seed","42",
              "--time-to-teleport","-1"]
        try:
            simulated=subprocess.run(args,cwd=stage,text=True,capture_output=True,
                                     timeout=240,check=False)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise SiteInputError("SUMO did not complete within bounded runtime") from exc
        if simulated.returncode!=0 or not (stage/"tripinfo.xml").is_file() or not (stage/"summary.xml").is_file():
            raise SiteInputError("SUMO could not execute reviewed baseline: "+simulated.stderr[-1000:])
        assessment=evaluate_tripinfo(snapshot["model"],(stage/"tripinfo.xml").read_bytes())
        (stage/"run.log").write_text(
            "netconvert:\n"+built.stdout[-2500:]+built.stderr[-2500:]+
            "\nsumo:\n"+simulated.stdout[-2500:]+simulated.stderr[-2500:],
            encoding="utf-8")
        assessment.update(
            executed_at_unix=time.time(),
            files={name:sha(stage/name) for name in sorted(EXEC_FILES)},
            model_revision=revision)
        (stage/"run_receipt.json").write_bytes(_json_bytes(assessment))
        os.replace(stage,directory/"runtime")
        return _runtime(directory)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser=argparse.ArgumentParser(description="Run one fully reviewed real-site SUMO baseline")
    parser.add_argument("--workdir",default=".streetlab-m5")
    parser.add_argument("--project",required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--sumo-bin-dir",help="Operator-controlled directory for SUMO binaries")
    args=parser.parse_args()
    print(json.dumps(run_baseline(VideoStore(args.workdir),args.project,args.revision,
                                 executable_dir=args.sumo_bin_dir),indent=2))


if __name__=="__main__":
    main()
