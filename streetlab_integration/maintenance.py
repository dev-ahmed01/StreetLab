"""M7 local recovery tools: SQLite online-safe backups and evidence audit.

Backups contain database METADATA only, never camera footage or model weights.
Source media stays immutable in original user-controlled project directories.
Use a real filesystem snapshot for full disaster recovery, including media.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sqlite3
import tempfile

from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.primary_runner import sha
from streetlab_integration.reconstruction import _json_bytes
from streetlab_integration.workspace_read import project_workspace


def check_db(db: Path) -> None:
    if db.is_symlink() or not db.is_file():
        raise ValueError("Missing or unsafe project metadata database")
    with sqlite3.connect(str(db),timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        if conn.execute("PRAGMA quick_check").fetchone()[0]!="ok":
            raise ValueError("SQLite metadata consistency check failed")


def audit(store: VideoStore) -> dict:
    check_db(store.db)
    records=[]
    for proj in store.projects():
        try:
            snapshot=project_workspace(store,proj["id"])
            records.append({
                "project_id":proj["id"],"status":"PASS",
                "observation":snapshot["stages"]["observation"]["status"],
                "geometry":snapshot["stages"]["geometry"]["status"],
                "baseline":snapshot["stages"]["baseline"]["status"],
                "scenarios":snapshot["stages"]["scenarios"]["status"]})
        except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError,sqlite3.Error):
            # Do not expose private media paths and survey records in audit logs.
            records.append({"project_id":proj["id"],"status":"EVIDENCE_INTEGRITY_FAILURE"})
    return {"status":"PASS" if all(p["status"]=="PASS" for p in records) else "FAIL",
            "project_count":len(records),
            "projects":records,
            "no_simulation_executed":True,
            "note":"Receipt validation is not a full-field survey or causal validation"}


def backup_metadata(store: VideoStore) -> dict:
    check_db(store.db)
    folder=store.root/"backups"
    if folder.is_symlink():
        raise ValueError("Untrusted metadata backup destination")
    folder.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename=f"streetlab-db-{stamp}-{secrets.token_hex(4)}.sqlite3"
    target=folder/filename
    stage=folder/("."+filename+".tmp")
    if target.exists() or stage.exists():
        raise ValueError("Refusing to overwrite an existing backup")
    try:
        # SQLite online backup is consistent in WAL mode with local writers.
        with sqlite3.connect(str(store.db),timeout=30) as source:
            with sqlite3.connect(str(stage),timeout=30) as destination:
                source.backup(destination,pages=256,sleep=.02)
        check_db(stage)
        digest=sha(stage)
        result={"schema_version":1,"backup_file":filename,"sha256":digest,
                "size_bytes":stage.stat().st_size,
                "scope":"SQLITE_METADATA_ONLY_NO_MEDIA_OR_MODEL_WEIGHTS",
                "created_utc":stamp,
                "restore_requires_original_source_media":True}
        receipt=folder/(filename+".json")
        if receipt.exists():
            raise ValueError("Backup receipt already exists")
        os.replace(stage,target)
        try:
            receipt.write_bytes(_json_bytes(result))
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return result
    finally:
        stage.unlink(missing_ok=True)


def verify_backup(store: VideoStore, name: str) -> dict:
    if (not name.startswith("streetlab-db-") or
        not name.endswith(".sqlite3") or "/" in name or "\\" in name
        or len(name)>100):
        raise ValueError("Invalid metadata backup name")
    folder=store.root/"backups"
    item=folder/name
    receipt=folder/(name+".json")
    if item.is_symlink() or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("Missing or untrusted metadata backup")
    record=json.loads(receipt.read_text(encoding="utf-8"))
    if (record.get("backup_file")!=name
        or record.get("scope")!="SQLITE_METADATA_ONLY_NO_MEDIA_OR_MODEL_WEIGHTS"
        or sha(item)!=record.get("sha256")):
        raise ValueError("Metadata backup checksum or binding mismatch")
    check_db(item)
    return {"verified":True,"sha256":record["sha256"],
            "metadata_only":True,"backup_file":name}


def main() -> None:
    parser=argparse.ArgumentParser(description="Audit StreetLab receipts and back up project metadata")
    parser.add_argument("--workdir",default=".streetlab-m5")
    operations=parser.add_mutually_exclusive_group(required=True)
    operations.add_argument("--audit",action="store_true")
    operations.add_argument("--backup",action="store_true")
    operations.add_argument("--verify-backup",metavar="BACKUP_FILENAME")
    args=parser.parse_args()
    store=VideoStore(args.workdir)
    if args.audit:
        report=audit(store)
        print(json.dumps(report,indent=2))
        if report["status"]!="PASS":
            raise SystemExit(2)
    elif args.backup:
        print(json.dumps(backup_metadata(store),indent=2))
    else:
        print(json.dumps(verify_backup(store,args.verify_backup),indent=2))


if __name__=="__main__":
    main()
