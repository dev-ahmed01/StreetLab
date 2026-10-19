"""Compact accessible research workspace HTML for the existing local StreetLab app.

No new frontend framework or route tree; the screen consumes only local read-only
evidence APIs. Missing audits remain visibly unverified.
"""
from __future__ import annotations


def research_page_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetLab | Research</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Poppins:wght@400;500;600&display=swap");
:root{font-family:Poppins,system-ui,sans-serif;color:#233a38;background:#f6f8f7;
  font-size:13px;-webkit-font-smoothing:antialiased}
*{box-sizing:border-box}
body{margin:0}h1,h2,h3,.number,.brand{font-family:Manrope,Poppins,system-ui,sans-serif}
a{color:inherit;text-decoration:none}
a:focus-visible{outline:3px solid #4c9385;outline-offset:4px}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:20px 38px;
  background:#fff;border-bottom:1px solid #e3eae7;gap:18px}
.brand{font-weight:800;font-size:20px;letter-spacing:-.06em;color:#215c51}
.brand span{display:inline-grid;place-items:center;background:#215c51;color:#fff;
  width:30px;height:30px;border-radius:9px;letter-spacing:-.04em;margin-right:10px;font-size:15px}
.toplinks{display:flex;align-items:center;gap:23px;flex-wrap:wrap;font-weight:500;color:#536b66}
.toplinks a:hover,.nav a:hover{color:#226b60}
.shell{max-width:1430px;display:grid;grid-template-columns:225px minmax(0,1fr);margin:0 auto;min-height:calc(100vh - 75px)}
.nav{border-right:1px solid #e3eae7;padding:31px 18px;display:flex;flex-direction:column;gap:5px}
.nav-label,.eyebrow{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#81938e;font-weight:600}
.nav-label{margin:4px 15px 14px}
.nav a{border-radius:9px;padding:12px 14px;font-weight:500;color:#617570}
.nav a.active{background:#e6f1ec;color:#215e51;font-weight:600}
.nav .tail{margin-top:auto;border-top:1px solid #e1eae7;padding-top:16px;font-size:11px;color:#6d807b;line-height:1.8}
main{padding:33px 38px 65px;min-width:0}
.page-title{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:27px}
h1{font-weight:800;letter-spacing:-.055em;font-size:30px;line-height:1.18;margin:6px 0 0}
h2{font-size:17px;line-height:1.4;letter-spacing:-.035em;font-weight:800;margin:0}
h3{font-size:14px;margin:0;font-weight:750;letter-spacing:-.02em}
.muted{color:#71827d}.small{font-size:11px}.meta{font-size:11px;color:#78918a}
.grids{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px}
.metric,.pane{border:1px solid #e1e9e5;border-radius:14px;background:#fff}
.metric{padding:20px;min-height:123px}
.metric .label{font-size:11px;font-weight:500;color:#72867e}
.number{font-size:29px;font-weight:800;letter-spacing:-.06em;margin:10px 0 1px;line-height:1.2}
.detail{font-size:11px;color:#788780;line-height:1.5}
.pane{padding:23px;margin-top:18px}
.rowhead{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:21px}
.chip{border:1px solid #dfe9e4;padding:5px 9px;border-radius:999px;color:#62776e;font-size:10px;font-weight:600}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.fact{border:1px solid #edf0ee;border-radius:10px;padding:16px;min-width:0}
.fact strong{display:block;font:800 26px Manrope,Poppins,sans-serif;letter-spacing:-.06em;margin-top:6px}
.fact p,.context{font-size:11px;color:#71837d;line-height:1.65;margin:8px 0 0}
.warn{border-left:3px solid #c89a54;padding-left:14px;color:#6c6150;line-height:1.65;font-size:12px}
.statusline{font-size:11px;color:#698078;line-height:1.7}
pre{white-space:pre-wrap;word-break:break-word;background:#f5f8f6;
  border:1px solid #e5ece7;padding:16px 18px;border-radius:10px;color:#2b5a4e;font:12px Consolas,monospace}
.buttonrow{display:flex;flex-wrap:wrap;gap:10px}
.buttonrow a{padding:10px 14px;background:#205e51;color:white;border-radius:9px;font-size:11px;font-weight:600}
.buttonrow a.secondary{background:#f0f6f2;color:#276658;border:1px solid #dae9e0}
.line{border-top:1px solid #ebf0ed;padding-top:16px;margin-top:16px}
.result{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}
.result>div{padding:14px;border:1px solid #ebf0ed;border-radius:10px;font-size:11px;color:#6a7c77}
.result strong{display:block;font:700 14px Manrope,sans-serif;color:#2d4f48;margin:8px 0}
footer{margin-top:26px;font-size:11px;color:#83938e}
@media(max-width:1050px){.grids{grid-template-columns:repeat(2,1fr)}.shell{grid-template-columns:175px 1fr}main{padding:26px}}
@media(max-width:690px){.topbar{padding:15px 19px}.toplinks{gap:12px}.shell{display:block}.nav{border-right:0;border-bottom:1px solid #e3eae7;padding:10px 13px;flex-direction:row;overflow-x:auto}.nav-label,.nav .tail{display:none}.nav a{white-space:nowrap;padding:9px 12px}main{padding:20px 15px 35px}.grids,.two,.result{grid-template-columns:1fr 1fr}.page-title{display:block}h1{font-size:25px}}
@media(max-width:450px){.grids,.two,.result{grid-template-columns:1fr}.metric{min-height:95px}}
</style>
</head>
<body>
<header class="topbar">
  <a href="/" class="brand" aria-label="StreetLab workspace"><span>S</span>StreetLab</a>
  <nav class="toplinks" aria-label="Project navigation">
    <a href="/">Workspace</a><a href="/reports">Reports</a><a href="/release">Release</a>
  </nav>
</header>
<div class="shell">
<nav class="nav" aria-label="Research sections">
  <span class="nav-label">RESEARCH</span>
  <a href="#overview" class="active">Overview</a>
  <a href="#diagnostics">Diagnostics</a>
  <a href="#simulation">Simulation</a>
  <a href="#boundaries">Release gates</a>
  <div class="tail">Historical evidence and synthetic simulation remain separate.</div>
</nav>
<main>
<div class="page-title" id="overview">
  <div><span class="eyebrow">Evidence workspace / Research</span><h1>Traffic research</h1></div>
  <span class="chip">Research prototype</span>
</div>
<div class="grids" aria-label="Research summary metrics">
  <div class="metric"><div class="label">May26 · W04 precision</div>
    <div class="number" id="fluid" aria-live="polite">—</div>
    <div class="detail">Tuned development cohort</div></div>
  <div class="metric"><div class="label">May26 · W04 recall</div>
    <div class="number" id="fluidRecall" aria-live="polite">—</div>
    <div class="detail" id="fluidDetail">FLUID-label comparison</div></div>
  <div class="metric"><div class="label">May25 · spatial F1</div>
    <div class="number" id="auditF1" aria-live="polite">—</div>
    <div class="detail">Historical regression · audit required</div></div>
  <div class="metric"><div class="label">Chennai · held-out transfer</div>
    <div class="number" id="chennai" aria-live="polite">—</div>
    <div class="detail" id="chennaiDetail">Loading frozen evidence</div></div>
</div>
<section class="pane" id="diagnostics">
  <div class="rowhead"><h2>May25 · object diagnostics</h2><span class="chip">Historical regression</span></div>
  <div class="two">
    <div class="fact"><span class="meta">FLUID HEAVY tracks with class disagreement</span>
      <strong id="auditHeavy">—</strong><p>Distinct annotated tracks, not physical-object census</p></div>
    <div class="fact"><span class="meta">Predicted BUS track IDs</span>
      <strong id="auditBus">—</strong><p>BUS correctness cannot be measured with zero BUS FLUID truth points</p></div>
  </div>
  <p class="statusline" id="auditStatus" role="status" aria-live="polite">Source-verified full-cohort audit not loaded. No unique-object counts inferred.</p>
</section>
<section class="pane" id="simulation">
  <div class="rowhead"><h2>Native SUMO experiment</h2><span class="chip">Synthetic only</span></div>
  <div class="result">
    <div>01 <strong>Baseline</strong> Assumed junction and demand</div>
    <div>02 <strong>Guided detour</strong> Controlled modeled alternative</div>
    <div>03 <strong>Mixed detour</strong> Paired synthetic comparison</div>
  </div>
  <p class="context">Execute the existing reproducible local research runner. The results describe a modeled scenario, not measured road improvement.</p>
  <pre> .\scripts\STREETLAB_RESEARCH.ps1</pre>
  <div class="buttonrow"><a href="/reports">Open evidence reports</a><a class="secondary" href="/">Observe a project</a></div>
</section>
<section class="pane" id="boundaries">
  <div class="rowhead"><h2>Scientific release gates</h2><span class="chip">Not production approved</span></div>
  <div class="warn">The Chennai independent transfer failed. May25 is historical regression, May26 is tuned development, and the native SUMO experiment uses an assumed network. Independent field calibration and physical vehicle labels remain unavailable.</div>
  <div class="line buttonrow"><a href="/release">View release acceptance</a><a href="/fieldwork" class="secondary">Fieldwork tools</a></div>
</section>
<footer>Source pixels are not geospatial measurements · Tracker IDs are not independently verified unique vehicles · Modeled gains are not observed causal effects.</footer>
</main></div>
<script>
(async function(){
  try{
    const response=await fetch("/api/research/summary",{cache:"no-store"});
    if(!response.ok)throw Error("Unavailable");
    const result=await response.json();
    const w=result.fluid_w04, c=result.chennai;
    document.getElementById("fluid").textContent=w.reported_point_precision_pct.toFixed(2)+"%";
    document.getElementById("fluidRecall").textContent=w.reported_point_recall_pct.toFixed(2)+"%";
    document.getElementById("fluidDetail").textContent=w.policy+" · development only";
    document.getElementById("chennai").textContent=c.holdout_generalization_pass?"PASS":"FAIL";
    document.getElementById("chennaiDetail").textContent=c.holdout_generalization_pass?"Review held-out evidence":"Frozen transfer did not pass";
  }catch(e){
    document.getElementById("fluidDetail").textContent="Source evidence unavailable";
    document.getElementById("chennaiDetail").textContent="Source evidence unavailable";
  }
})();
(async function(){
  try{
    const response=await fetch("/api/research/may25-audit",{cache:"no-store"});
    if(!response.ok)throw Error("Audit unavailable");
    const result=await response.json();
    document.getElementById("auditF1").textContent=result.spatial_f1_pct.toFixed(2)+"%";
    document.getElementById("auditHeavy").textContent=String(result.heavy_truth_track_ids_with_disagreement);
    document.getElementById("auditBus").textContent=String(result.bus_predicted_unique_track_ids);
    document.getElementById("auditStatus").textContent=result.heavy_wrong_class_frame_points+
      " HEAVY frame-level class disagreements across the frozen 120 frames · FLUID label-relative";
  }catch(e){
    document.getElementById("auditStatus").textContent=
      "Original-source audit unavailable or inconsistent. Distinct-track counts remain unverified.";
  }
})();
</script>
</body></html>"""
