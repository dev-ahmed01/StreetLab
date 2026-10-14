"""Operator M8 release gate CLI; read-only audit with no simulation side effects."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.release_acceptance import evaluate_release
from streetlab_integration.reconstruction import _json_bytes


def environment() -> dict:
    """Operational preflight hints; cannot establish real source/field validity."""
    modules={name:importlib.util.find_spec(name) is not None
             for name in ("fastapi","uvicorn","cv2","sahi","supervision")}
    binaries={name:bool(shutil.which(name)) for name in ("sumo","netconvert")}
    return {
        "platform":platform.system(),"python_version":platform.python_version(),
        "runtime_modules_present":modules,"sumo_binaries_on_path":binaries,
        "security_mode":os.getenv("STREETLAB_SECURITY_MODE","local"),
        "network_deployment_approved":False,
        "real_site_science_validated":False,
        "note":"Presence checks do not exercise video inference or field survey",
    }


def human_markdown(report: dict) -> str:
    status=report["acceptance_status"]
    lines=[
        "# StreetLab M8 release acceptance",
        "",
        "Project: "+report["project"]["name"],
        "Project UUID: "+report["project"]["id"],
        "Technical status: **"+status+"**",
        "Production release approved: **NO**",
        "Observed real-world impact proven: **NO**",
        "",
        "## Acceptance gates",
        "",
        "| Gate | State | Reason |",
        "|---|---|---|",
    ]
    for name,gate in report["checks"].items():
        reason=gate["reason"].replace("|","/").replace("\n"," ")
        lines.append(f"| {name} | {gate['status']} | {reason} |")
    lines+=["","## Release blockers", ""]
    if report["blocked_by"]:
        lines.extend("- "+item for item in report["blocked_by"])
    else:
        lines.append("- External observer verification, legal permissions, and release authority remain required")
    lines+=["","## Scientific and operational limitations",""]
    lines.extend("- "+item for item in report["limits"])
    return "\n".join(lines)+"\n"


def main() -> None:
    parser=argparse.ArgumentParser(description="Run real-site readiness checks. Never certifies actual road outcomes.")
    parser.add_argument("--workdir",default=".streetlab-m5")
    parser.add_argument("--project",help="Existing observed-site project UUID")
    parser.add_argument("--model-dir",type=Path,help="Frozen W04 OpenVINO model location")
    parser.add_argument("--frozen-provenance",type=Path,help="Frozen SHA/provenance JSON")
    parser.add_argument("--field-dir",type=Path,help="Original field survey/census/holdout records")
    parser.add_argument("--verify-source-sha",action="store_true",help="Rehash entire original video (can be multi-GiB)")
    parser.add_argument("--output-dir",type=Path,help="Save the acceptance JSON and Markdown to this empty directory")
    parser.add_argument("--environment",action="store_true",help="Display installation preflight without examining projects")
    args=parser.parse_args()
    if args.environment:
        print(json.dumps(environment(),indent=2))
        return
    if not args.project:
        parser.error("--project is required unless --environment was given")
    report=evaluate_release(
        VideoStore(args.workdir),args.project,
        model_dir=args.model_dir,frozen_provenance=args.frozen_provenance,
        field_dir=args.field_dir,full_video_sha=args.verify_source_sha)
    if args.output_dir:
        # Explicit operator-selected new output directory. Never overwrite a
        # previous acceptance record; no internet I/O.
        if args.output_dir.exists():
            parser.error("--output-dir already exists: choose a fresh folder to preserve old acceptance records")
        args.output_dir.mkdir(parents=True,exist_ok=False)
        (args.output_dir/"m8_acceptance.json").write_bytes(_json_bytes(report))
        (args.output_dir/"m8_acceptance.md").write_text(human_markdown(report),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["technical_acceptance_checks_passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
