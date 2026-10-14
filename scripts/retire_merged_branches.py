"""Retire ONLY the 26 audited StreetLab branch refs once absorbed into main.

Preserves Git commit objects and history; never rewrites, force-pushes, or
deletes main. Uses immutable expected SHA snapshots, protected-branch checks,
and exact GitHub ahead_by=0 ancestry checks. Dry-run unless --apply is explicit.
"""

from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORY = "dev-ahmed01/StreetLab"
EXPECTED_BRANCHES = {
    "codex/phase3-engine-shootout": "ed8fd83a883b2d86323fcbe7677560e31cd08d79",
    "codex/streetlab-integration-m1": "2e268ecfecb2fdae1c779d6e78fd8d7a7008e5f0",
    "codex/streetlab-integration-m2": "6af17b9204813f5be075466dcdd173de911fdc75",
    "codex/streetlab-integration-m3": "dd7fdb6785e60e6682898655bf881b5a7509d480",
    "codex/streetlab-integration-m4": "ebb16219435839b337c54122339597cc8a81c08a",
    "codex/streetlab-integration-m5": "73b36cdb92db0b4ce59bc1e4d0fa277be7a40a26",
    "codex/streetlab-integration-m6": "76138084a9658a415f3e64234847c4935b298d0a",
    "codex/streetlab-integration-m7": "5aec03ee1ac1c47c8231328fe80c1553c9943b3d",
    "codex/streetlab-integration-m8": "13b5913caa019122a0a9822f0221014a38da9582",
    "codex/streetlab-integration-m9": "a3d0575a6d530ee62a20cda2ad9c96d5560c9c08",
    "codex/streetlab-repo-consolidation": "d1d289a3f2ae9b385075fb400285b110e7070ea3",
    "fix/p3b-compare-script-import": "0d33bf8e09a5bd63e9069c46b9cbb604b1cbe2cb",
    "fix/p3b-dataset-pairing-guard": "fa64faa6ff24f9ccf2ae2f918677bf700b708ed0",
    "phase2/decision-lab-m1": "a4367f94337cb5924b6d5da8d7fcb222958dbbbd",
    "phase2/m1-closure-proof": "9c6d9aae344ee72ef97b080ffa8ceab9f6337885",
    "phase2/m1-decision-loop": "5c3b2b33017fe228c265ff4d42003aee89edb742",
    "phase2/m2-runtime-decisions": "9c6d9aae344ee72ef97b080ffa8ceab9f6337885",
    "phase2/m2-runtime-state-manager": "68ef1963d0360067ec3f1783c9ae0a1d291acec9",
    "phase2/m3-heterogeneous-response": "62abd1a2b68c24e4dd450bdcb4a905e570d945e2",
    "phase2/m4-scenario-ensembles": "7d528feaeda44c5bede9f1119a6d29c7ad75005a",
    "phase2/m5-decision-lab-api-ui": "4a0dba2ea8f40ab052888c8b2cc1398645c5e734",
    "phase2/m6-study-evidence-decisions": "45001605388f8e4689f1cf46ef5b6a2b60b020e7",
    "phase3/p3a-observation-platform": "49b89fc2e0953b5940a8cbe48636372e063bf592",
    "phase3/p3b-fluid-benchmark": "b073e2476e7cda55b2ae472e25fe22ee6fcd559d",
    "phase3/p3b-identity-benchmark": "90e6c1a77e81fb670781e399f540b8895281d125",
    "phase3/p3b-identity-diagnostics": "90e6c1a77e81fb670781e399f540b8895281d125"
}


def eligible(*, branch: str, current_sha: str, protected: bool,
             ahead_by: int, expected_sha: str | None = None) -> tuple[bool, str]:
    """A testable deletion gate that fails closed on any mismatch."""
    expected=EXPECTED_BRANCHES.get(branch) if expected_sha is None else expected_sha
    if branch == "main" or branch not in EXPECTED_BRANCHES:
        return False,"not in audited retirement allowlist"
    if protected:
        return False,"GitHub branch protection is enabled"
    if not isinstance(current_sha,str) or current_sha != expected:
        return False,"branch head moved since audit"
    if ahead_by != 0:
        return False,"branch contains commits outside main"
    return True,"original branch head is an ancestor of main"


def api(method: str, endpoint: str, token: str) -> object:
    url="https://api.github.com/repos/"+REPOSITORY+"/"+endpoint
    headers={
        "Accept":"application/vnd.github+json",
        "Authorization":"Bearer "+token,
        "X-GitHub-Api-Version":"2022-11-28",
        "User-Agent":"StreetLab-safe-branch-retirement",
    }
    req=Request(url,method=method,headers=headers)
    try:
        with urlopen(req,timeout=30) as response:
            payload=response.read()
            return json.loads(payload) if payload else None
    except HTTPError as err:
        message=err.read(500).decode("utf-8","replace")
        raise RuntimeError(
            "GitHub REST "+method+" "+endpoint+" returned "+str(err.code)+
            " (details redacted; check repo permissions and branch rules)"
        ) from err


def main() -> None:
    parser=argparse.ArgumentParser(description="Delete only fully absorbed audited legacy branches")
    parser.add_argument("--apply",action="store_true",
                        help="Actually delete; defaults to read-only dry run")
    args=parser.parse_args()
    if os.getenv("GITHUB_REPOSITORY") != REPOSITORY:
        raise SystemExit("Repository identity mismatch; refusing branch operations")
    if os.getenv("GITHUB_REF") != "refs/heads/main":
        raise SystemExit("Only run from the main branch")
    token=os.getenv("GH_TOKEN","")
    if not token:
        raise SystemExit("GH_TOKEN is required for read and deletion safety checks")

    main_ref=api("GET","git/ref/heads/main",token)
    main_sha=main_ref["object"]["sha"]
    workflow_sha=os.getenv("GITHUB_SHA")
    if workflow_sha and workflow_sha != main_sha:
        raise SystemExit("Main moved after workflow start; rerun on the new main head")

    branches={}
    page=1
    while True:
        rows=api("GET","branches?per_page=100&page="+str(page),token)
        for row in rows:
            branches[row["name"]]=row
        if len(rows)<100:break
        page+=1

    deleted=[]
    safe=[]
    excluded=[]
    errors=[]
    for branch,expected in EXPECTED_BRANCHES.items():
        meta=branches.get(branch)
        if meta is None:
            excluded.append({"branch":branch,"reason":"already absent"})
            continue
        # Avoid quoting ambiguity from slash-containing branch names: compare
        # immutable HEAD commit SHA to the exact CURRENT main commit SHA.
        comparison=api("GET","compare/"+main_sha+"..."+expected,token)
        permitted,why=eligible(branch=branch,
            current_sha=meta["commit"]["sha"],
            protected=bool(meta.get("protected",True)),
            ahead_by=comparison["ahead_by"])
        if not permitted:
            errors.append({"branch":branch,"reason":why})
            continue
        safe.append(branch)
        if args.apply:
            # Re-read immediately before deleting, and abort if anything moved.
            remote=api("GET","git/ref/heads/"+quote(branch,safe="/"),token)
            if remote["object"]["sha"]!=expected:
                errors.append({"branch":branch,"reason":"head moved immediately before deletion"})
                continue
            try:
                api("DELETE","git/refs/heads/"+quote(branch,safe="/"),token)
            except RuntimeError:
                errors.append({"branch":branch,"reason":"GitHub rejected deletion; inspect branch settings"})
                continue
            deleted.append(branch)

    result={
        "repository":REPOSITORY,"main_sha":main_sha,
        "mode":"APPLY" if args.apply else "DRY_RUN",
        "audited_branch_count":len(EXPECTED_BRANCHES),
        "eligible_before_deletion":safe,
        "deleted":deleted,
        "skipped":excluded,
        "blocked":errors,
        "main_deleted":False,
        "git_history_rewritten":False,
    }
    print(json.dumps(result,indent=2))
    if errors:
        raise SystemExit("One or more branches could not safely be retired")
    if args.apply:
        final=api("GET","git/refs/heads",token)
        remaining=[x["ref"].removeprefix("refs/heads/") for x in final]
        print("Remote branches after retirement:",json.dumps(remaining))
        if remaining!=["main"]:
            raise SystemExit("Cleanup incomplete: see remaining branches above")


if __name__=="__main__":
    main()
