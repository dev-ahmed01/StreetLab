"""Read-only local research-submission page; does not execute untrusted commands."""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from streetlab_integration.research_audit_display import AuditDisplayError, read_audit
from streetlab_integration.research_submission import EvidenceError, study


def mount_research_routes(app:FastAPI) -> None:
    root=Path(__file__).resolve().parents[1]

    @app.get("/api/research/summary")
    def research_summary() -> dict:
        try:
            report=study(root,root,False)
        except (EvidenceError,KeyError,ValueError,OSError) as exc:
            # Never return an incorrectly completed summary.
            raise HTTPException(status_code=409,
                detail="Research source unavailable or inconsistent; check repository evidence") from exc
        return {
            "schema_version":1,"track":report["track"],
            "chennai":report["dataset_summaries"]["chennai"],
            "fluid_w04":report["dataset_summaries"]["fluid_w04"],
            "research_workflow_status":report["research_workflow_status"],
            "production_release_approved":False,
            "sources":report["source_manifest"],
        }

    @app.get("/api/research/may25-audit")
    def may25_audit() -> dict:
        # Only an explicit original-source completed artifact can be displayed.
        # No source files or inferred results are silently loaded from Drive.
        import os
        configured = os.environ.get("STREETLAB_MAY25_AUDIT_PATH", "").strip()
        if not configured:
            raise HTTPException(status_code=409,
                detail="Original-source autonomous audit has not been configured")
        try:
            return read_audit(Path(configured))
        except (AuditDisplayError, OSError, ValueError) as exc:
            raise HTTPException(status_code=409,
                detail="Original-source autonomous audit is absent or inconsistent") from exc

    @app.get("/research",response_class=HTMLResponse)
    def research_page() -> str:
        return r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetLab — Research submission</title>
<style>
:root{font:14px Inter,ui-sans-serif,system-ui,sans-serif;color:#233942;background:#f6f8f8}
*{box-sizing:border-box}body{margin:0}
header{background:#fff;border-bottom:1px solid #dbe5e5;padding:23px max(24px,calc((100% - 1060px)/2))}
main{max-width:1060px;margin:auto;padding:24px;display:grid;gap:18px}
section{border:1px solid #dde5e5;border-radius:13px;background:#fff;padding:22px}
h1{font-size:26px;margin:10px 0}h2{font-size:18px;margin:0 0 12px}
p{line-height:1.65;color:#506872}a{color:#1c6960;text-decoration:none}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:15px}
.stat{font-size:26px;font-weight:700;letter-spacing:-.04em}
small{color:#536872}pre{white-space:pre-wrap;overflow-wrap:anywhere;
font:12px Consolas,monospace;line-height:1.7;background:#f1f5f5;
border-radius:9px;padding:15px}
.warning{background:#fffbf4;border-color:#e3d9bf}
.note{font-size:12px;color:#667a83}
@media(max-width:720px){.cards{grid-template-columns:1fr}}
</style></head><body>
<header><a href="/">← Back to workspace</a>
<h1>Research submission</h1>
<p>Reproducible evidence from existing datasets and a distinct native SUMO study. No new physical junction survey is required for this limited research track.</p>
</header><main>
<section class="warning"><h2>Research prototype · not real-site certification</h2>
<p>Existing frozen Chennai held-out results are reported exactly, including failures. W04 development precision/recall are previously documented FLUID-label estimates, not a new independent holdout. The SUMO demonstration assumes its junction layout and demand; it does not measure effects on a real road.</p></section>
<div class="cards">
<section><h2>Chennai behavior evidence</h2>
<div class="stat" id="chennai">Loading verified source…</div>
<p id="chennaiDetail">Checking immutable Phase 1 held-out record</p></section>
<section><h2>FLUID W04 observation</h2>
<div class="stat" id="fluid">Loading verified source…</div>
<p id="fluidDetail">Checking reported development-only metrics</p></section>
</div>
<section><h2>Run your research package locally</h2>
<p>This is the submission route when independent new fieldwork is unavailable. Run it in a <strong>new</strong> Windows PowerShell terminal from the repository folder. It executes actual SUMO through StreetLab's existing M6 simulation and writes a results report and SHA-based evidence inventory. It does not change source models or original data.</p>
<pre>.\scripts\STREETLAB_RESEARCH.ps1</pre>
<p class="note">Outputs: submission_evidence.json, submission_summary.md, native_sumo_study.json, native_sumo.log and internal simulation files. Each run uses a new timestamped directory. For report-only mode, pass -EvidenceOnly.</p>
<p>Separately, use the original M8 worker to inspect real footage and report video-run evidence. Never use its inferred tracks as an independent physical census without validation.</p>
</section>
<section><h2>May 25 · full-cohort object attribution</h2>
<div class="cards">
<div><small>Historical spatial F1</small><div class="stat" id="auditF1">Not verified</div></div>
<div><small>Distinct HEAVY label-disagreement tracks</small><div class="stat" id="auditHeavy">—</div></div>
</div>
<p id="auditStatus" class="note">Original-source full-cohort audit is required. Previous 236 events are frame-level, not unique vehicles.</p>
</section>
<section><h2>What the result can support</h2>
<p><strong>Supported:</strong> reproducible source provenance, previously recorded FLUID model comparison, explicit failure reporting for Chennai heldout transfer, and genuine SUMO computation for a clearly assumed traffic scenario.</p>
<p><strong>Not supported:</strong> a Bengaluru-specific measured geometry, physically observed turn counts, independently verified original W04 full-video accuracy, traffic signal recommendations for actual deployment, or M8 production release approval.</p>
<a href="/reports">Real-site project reports →</a> · <a href="/release">Real-site acceptance →</a>
</section>
</main>
<script>
(async()=>{
 try{
   const response=await fetch('/api/research/may25-audit');
   if(!response.ok)throw Error('Original-source object audit not yet verified');
   const audit=await response.json();
   document.getElementById('auditF1').textContent=audit.spatial_f1_pct.toFixed(2)+'%';
   document.getElementById('auditHeavy').textContent=String(audit.heavy_truth_track_ids_with_disagreement);
   document.getElementById('auditStatus').textContent=audit.heavy_wrong_class_frame_points+
     ' matched HEAVY class disagreements · '+audit.bus_predicted_unique_track_ids+
     ' predicted BUS track IDs · FLUID-label relative, not physical truth';
 }catch(e){
   document.getElementById('auditStatus').textContent='Awaiting original-source verified autonomous audit; no inferred counts displayed';
 }
})();
(async()=>{
 try{
  const r=await fetch('/api/research/summary');
  if(!r.ok)throw Error('Evidence not available ('+r.status+')');
  const x=await r.json();
  const ch=x.chennai,fw=x.fluid_w04;
  document.getElementById('chennai').textContent=ch.holdout_generalization_pass?'PASS':'FAIL — held-out gate';
  document.getElementById('chennaiDetail').textContent=
    ch.validation_total_vehicles+' vehicles · capacity gate '+(ch.capacity_gate_pass?'PASS':'FAIL')+' · no retuning';
  document.getElementById('fluid').textContent=fw.reported_point_precision_pct.toFixed(2)+'% precision';
  document.getElementById('fluidDetail').textContent=
    fw.reported_point_recall_pct.toFixed(2)+'% recall · '+fw.policy+' · development-only FLUID labels';
 }catch(e){
  document.getElementById('chennai').textContent='Source unavailable';
  document.getElementById('fluid').textContent='Not verified';
  document.getElementById('chennaiDetail').textContent='Do not assume any result until the checked-in evidence is available';
  document.getElementById('fluidDetail').textContent=e.message;
 }
})();
</script></body></html>"""
