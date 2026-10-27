"""Black-box local HTTP smoke for a clean installed StreetLab checkout.

Does not pretend to process a real W04 model or run a physical traffic survey.
Starts an actual isolated Uvicorn instance, exercises project persistence/UI
and checks local-mode Host rejection without contacting external services.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen


def request(base: str,path: str,*,method: str="GET",body=None,host=None) -> tuple[int,bytes]:
    headers={"Accept":"application/json"}
    if body is not None: headers["Content-Type"]="application/json"
    if host is not None: headers["Host"]=host
    req=Request(base+path,
                data=json.dumps(body).encode() if body is not None else None,
                headers=headers,method=method)
    try:
        with urlopen(req,timeout=3) as response:
            return response.status,response.read()
    except HTTPError as exc:
        return exc.code,exc.read()


def run() -> None:
    root=Path(__file__).resolve().parent.parent
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port=sock.getsockname()[1]
    base=f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="streetlab-http-smoke-") as td:
        log=Path(td)/"uvicorn.log"
        with log.open("wb") as stdout:
            env=os.environ.copy()
            env["PYTHONPATH"]=str(root)
            env["STREETLAB_SECURITY_MODE"]="local"
            env.pop("STREETLAB_ACCESS_PASSWORD",None)
            # The clean-checkout smoke must not inherit private research
            # paths or accidentally turn a developer audit into CI proof.
            env["STREETLAB_MAY25_AUDIT_PATH"]=str(Path(td)/"missing_original_audit.json")
            env["STREETLAB_RESEARCH_BUNDLE_PATH"]=str(Path(td)/"missing_native_sumo.zip")
            proc=subprocess.Popen(
                [sys.executable,"-m","uvicorn","streetlab_phase2.api:app",
                 "--host","127.0.0.1","--port",str(port),"--no-access-log"],
                cwd=td,env=env,stdout=stdout,stderr=subprocess.STDOUT)
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise RuntimeError("Uvicorn failed to start: "+log.read_text(errors="replace")[-3500:])
                    try:
                        if request(base,"/api/health")[0]==200:break
                    except (URLError,TimeoutError,ConnectionError):
                        time.sleep(.1)
                else: raise RuntimeError("Timed out waiting for local Uvicorn")
                for page in ("/","/reports","/release","/fieldwork","/calibration",
                             "/baseline","/scenarios","/research"):
                    status,_=request(base,page)
                    assert status==200,(page,status)
                # A clean checkout must expose real frozen research source
                # summaries, while refusing absent private May25 audit and
                # absent saved native SUMO ZIP (never fabricated demo data).
                status,body=request(base,"/api/research/summary")
                assert status==200,("research summary",status,body[:500])
                summary=json.loads(body)
                assert summary["production_release_approved"] is False
                assert summary["chennai"]["holdout_generalization_pass"] is False
                for path in ("/api/research/may25-audit",
                             "/api/research/simulation"):
                    status,body=request(base,path)
                    assert status==409,(path,status,body[:500])
                    detail=json.loads(body)["detail"]
                    assert isinstance(detail,str),(path,detail)
                    expected=("absent or inconsistent" if path.endswith("may25-audit")
                              else "unavailable or inconsistent")
                    assert expected in detail.lower(),(path,detail)
                status,body=request(base,"/api/projects",method="POST",body={"name":"HTTP integration smoke"})
                assert status==201,(status,body)
                project=json.loads(body)
                id_=project["id"]
                for path in ("/api/projects",
                             f"/api/projects/{id_}/workspace",
                             f"/api/projects/{id_}/fieldwork-plan",
                             f"/api/projects/{id_}/report"):
                    status,body=request(base,path)
                    assert status==200,(path,status,body[:500])
                status,body=request(base,f"/api/projects/{id_}/fieldwork-plan")
                assert json.loads(body)["status"]=="NEEDS_REVIEWED_GEOMETRY"
                status,_=request(base,"/api/projects",host="nonloopback.example")
                assert status==403,("Host bypass",status)
                status,_=request(base,"/api/projects",method="POST",
                                 body={"name":"bad"},host="nonloopback.example")
                assert status==403,("Host bypass for mutations",status)
                status,_=request(base,"/api/projects/not-a-uuid/fieldwork-plan")
                assert status in {404,409},("Malformed project should not expose internals",status)
                assert (Path(td)/".streetlab-m5"/"video_jobs.sqlite3").is_file()
                print("Black-box real HTTP: PASS (pages, research source/fail-closed gates, create/query project, localhost guard, SQLite durability)")
            finally:
                proc.terminate()
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=8)


if __name__=="__main__":
    run()
