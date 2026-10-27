"""Repository hygiene audit for tracked files. Does not rewrite Git history.

The original raw Chennai datasets and phase-1 scientific simulation artifacts are
tracked research evidence, NOT auto-deleted garbage. This protects future commits
against unreviewed binaries, local environments, credentials and generated data.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

FORBIDDEN_TOP={".venv","venv","env","artifacts",".pytest_cache",".mypy_cache",
               ".ruff_cache","htmlcov","build","dist",".streetlab-m5"}
FORBIDDEN_EXT={".pyc",".pyo",".mp4",".avi",".mov",".mkv",".db",
               ".sqlite",".sqlite3",".pem",".key",".p12",".pfx",".exe",".dll",".pyd"}
SENSITIVE_PATTERNS=[
    (re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),"private key"),
    (re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),"GitHub credential"),
    (re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),"GitHub fine-grained token"),
    (re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),"AWS access key"),
]
SOURCE_ROOTS=("streetlab_integration/","streetlab_phase1/",
              "streetlab_phase2/","streetlab_phase3/","scripts/","tests/")
SCANNABLE_EXT={".py",".js",".ps1",".yml",".yaml",".toml",".ini",".sh"}


def check(root: Path, strict: bool=True) -> dict:
    cmd=subprocess.run(["git","-C",str(root),"ls-files","-z"],
                       capture_output=True,check=True)
    files=[x.decode("utf-8","surrogateescape") for x in cmd.stdout.split(b"\0") if x]
    issues=[]
    total=0
    largest=[]
    for name in files:
        p=Path(name)
        components=p.parts
        suffix=p.suffix.lower()
        is_data=name.startswith("data/")
        if (any(c in FORBIDDEN_TOP or c.startswith(".venv") or c=="__pycache__"
                for c in components) or suffix in FORBIDDEN_EXT
            or p.name in {".env",".DS_Store"} or (p.name.startswith(".env.") and p.name != ".env.example")):
            issues.append("Forbidden tracked local/generated/secret file: "+name)
        if not is_data and (name.endswith(".parquet") or name.endswith(".xlsx")
                            or name.endswith(".csv") and not name.startswith("tests/")):
            issues.append("Generated data outside research dataset area: "+name)
        local=root/p
        if not local.is_file():
            # Sparse checkout can omit file content while preserving its index.
            continue
        n=local.stat().st_size
        total+=n
        if n>5_000_000 and not is_data:
            issues.append("Oversized nonresearch tracked asset (>5MB): "+name)
        if n>1_000_000: largest.append((name,n))
        if ((name.startswith(SOURCE_ROOTS) or name.startswith(".github/")) and suffix in SCANNABLE_EXT) or p.name == ".env.example":
            if n>5_000_000:
                continue
            buf=local.read_bytes()
            for pattern,label in SENSITIVE_PATTERNS:
                if pattern.search(buf):
                    issues.append("Possible "+label+" in "+name+"; review and rotate if real")
    result={"tracked_files":len(files),"checked_local_bytes":total,
            "largest_files":sorted(largest,key=lambda x:-x[1])[:8],
            "blocked_issue_count":len(issues),"issues":issues[:80],
            "historic_blobs_scrubbed":False,
            "scans_all_secret_formats":False}
    if strict and issues:
        raise RuntimeError("\n".join(issues[:40]))
    return result


def main() -> None:
    import json
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("."))
    args=parser.parse_args()
    result=check(args.root.resolve(),strict=False)
    print(json.dumps(result,indent=2))
    if result["blocked_issue_count"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
