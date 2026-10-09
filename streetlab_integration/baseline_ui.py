"""Guided M4 evidence review and packaged baseline UI.

Avoid example counts, fabricated coordinates, or default vType measurements:
operator must explicitly provide a complete, independently evidenced model.
"""
from __future__ import annotations


def baseline_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetLab | Observed-site SUMO baseline</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#f6f7f9;color:#1d2935}
*{box-sizing:border-box}body{margin:0}
header{background:#fff;border-bottom:1px solid #dfe4ea;padding:18px max(24px,calc((100% - 1250px)/2))}
main{max-width:1250px;margin:auto;padding:24px;display:grid;gap:16px}
section{background:#fff;border:1px solid #e0e5ea;border-radius:12px;padding:22px}
h1{font-size:25px;margin:8px 0}h2{font-size:18px;margin:0 0 10px}
p,li{font-size:14px;line-height:1.55;color:#5e6977}
a{color:#225c62;text-decoration:none}
.columns{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}
label{font-size:13px;font-weight:600;display:grid;gap:7px}
input,select,textarea{font:inherit;border:1px solid #cbd3dd;border-radius:8px;padding:11px;width:100%;background:white}
textarea{min-height:390px;resize:vertical;font-family:ui-monospace,Consolas,monospace;font-size:12px;line-height:1.5}
button{border:0;background:#253c42;color:#fff;padding:10px 14px;border-radius:8px;cursor:pointer}
button.secondary{border:1px solid #cbd2da;background:#f5f7f9;color:#243a41}
.row{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap}
.row>*{flex:1;min-width:170px}
.meta{background:#f7fafb;border:1px solid #dce4e8;padding:12px;border-radius:8px;font-size:13px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
strong{color:#273542}
@media(max-width:860px){.columns{grid-template-columns:1fr} main{padding:14px}}
</style></head>
<body>
<header><a href="/">← StreetLab dashboard</a><h1>Observed-site simulation baseline</h1>
<p>Connect a reviewed local-meter junction model to field-surveyed traffic and an independent baseline. The separate synthetic Decision Lab is not used here.</p></header>
<main>
<section>
<h2>1. Select validated geometry</h2>
<div class="row">
<label>Project<select id="project"><option value="">Choose site project</option></select></label>
<label>Spatial reconstruction revision<input id="spatialRevision" readonly placeholder="No calibrated junction revision"></label>
</div>
<p id="geometry">Choose a project. Real-site simulation is blocked until M3 has checked metric geometry, approaches and permitted turns.</p>
</section>
<div class="columns">
<div>
<section>
<h2>2. Enter field-reviewed baseline inputs</h2>
<p>Paste a JSON site model using the required schema. This form does not populate traffic volumes, vehicle parameters, dimensions or signal phases with guesses.</p>
<textarea id="siteJson" spellcheck="false" aria-label="Field reviewed site network, vehicle census and independent holdout JSON"
 placeholder='{
  "center_world_m": null,
  "arms": [],
  "connections": [],
  "control": {},
  "window": {},
  "vehicle_types": [],
  "demand": [],
  "holdout": [],
  "review": {}
}'></textarea>
<div class="row" style="margin-top:12px">
<button id="save">Validate and package baseline</button>
<button class="secondary" id="showSaved">Restore most recently packaged inputs</button>
</div>
<p><a href="https://github.com/dev-ahmed01/StreetLab/blob/codex/streetlab-integration-m4/README_INTEGRATION_M4.md" target="_blank" rel="noopener noreferrer">Open complete field-input schema and CLI instructions ↗</a></p>
</section>
</div>
<div>
<section>
<h2>Evidence checklist</h2>
<p>Only measured or explicitly field-reviewed information is eligible:</p>
<ul>
<li>Center and all road-arm endpoints in M3's local-meter plane, plus posted speeds and observed lane counts.</li>
<li>Lane-to-lane links for every manually approved M3 movement.</li>
<li>Confirmed unsignalized control or the actual fixed-time signal phases and reviewed link-order mapping.</li>
<li>Vehicle-type parameters with source references, not silently substituted generic defaults.</li>
<li>Manual counts of <strong>distinct physical vehicles</strong> per permitted movement and type. Tracker IDs do not count.</li>
<li>Separately obtained travel-time holdouts with at least three field samples per movement.</li>
</ul>
</section>
<section>
<h2>3. Baseline execution</h2>
<div id="status" class="meta" role="status" aria-live="polite">Not yet configured</div>
<div style="margin-top:12px" class="row"><button class="secondary" id="refresh">Refresh validation</button></div>
<p>Site SUMO runs in a separate operating-system process, not in the HTTP server. Only an input package with verified evidence may be compiled and run.</p>
<label>Windows PowerShell command (after packaging)<textarea id="command" readonly style="min-height:145px"></textarea></label>
<div class="row"><button class="secondary" id="copy">Copy CLI command</button></div>
</section>
<section>
<h2>4. Review simulation evidence</h2>
<div id="fidelity" class="meta">No independent baseline comparison has been run.</div>
<div id="files" style="display:grid;gap:8px;margin-top:12px;font-size:13px"></div>
<p>The M4 95% completion / 20% movement travel-time checks are QA heuristics, not certified transport model accuracy.</p>
</section>
</div>
</div></main>
<script>
(()=>{
const $=id=>document.getElementById(id);
let project='',stored=null;
async function api(path,options={}){
 const res=await fetch(path,options);
 const body=await res.json().catch(()=>({detail:'Server sent an invalid JSON response'}));
 if(!res.ok)throw Error(body.detail||'Request failed');
 return body;
}
const note=text=>$('status').textContent=text;
function checkResult(v){
 stored=v.status==='NOT_CONFIGURED'?null:v;
 if(!stored){
   $('fidelity').textContent='No independently verified baseline package exists.';
   $('files').replaceChildren();$('command').value='';
   return;
 }
 const q=stored.quality||{},r=stored.runtime||{};
 const lines=['Input evidence: '+(q.status||'UNKNOWN'),
  'Baseline execution: '+(r.status||'NOT_EXECUTED'),
  'Eligible for M5 scenario consideration: '+(r.real_site_sumo_allowed?'YES (subject to site review)':'NO'),
  'Count provenance: field census, never tracker IDs',
  'Input hash: '+stored.revision];
 note(lines.join('\n'));
 const report=['Actual SUMO baseline: '+(r.status||'NOT_EXECUTED')];
 if(r.completion_ratio!==undefined)report.push('Simulated trip completion ratio: '+r.completion_ratio);
 for(const row of r.per_movement||[])report.push(row.from_zone+' → '+row.to_zone+
   ': reference '+row.holdout_mean_s+' s; simulation '+(row.simulated_mean_s??'not completed')+
   ' s; relative error '+(row.relative_error??'unavailable'));
 $('fidelity').textContent=report.join('\n');
 const rev=stored.revision;
 const base='/api/projects/'+project+'/baseline/'+rev+'/files/';
 $('files').replaceChildren();
 for(const name of ['nodes.nod.xml','edges.edg.xml','connections.con.xml',
   'signals.tll.xml','routes.rou.xml','site.net.xml','summary.xml','tripinfo.xml']){
  if(name==='signals.tll.xml'&&stored.model.control.kind!=='FIXED_TIME_SIGNAL')continue;
  if(['site.net.xml','summary.xml','tripinfo.xml'].includes(name)&&r.status==='NOT_EXECUTED')continue;
  const link=document.createElement('a');link.href=base+name;link.textContent='View verified '+name;
  link.target='_blank';link.rel='noopener noreferrer';$('files').appendChild(link);
 }
 const cmd=['cd C:\\Users\\Admin\\Desktop\\StreetLab-engine-trial',
  '& ".\\.venv-sahi-audit\\Scripts\\python.exe" -m streetlab_integration.site_baseline',
  '  --workdir ".streetlab-m5"',
  '  --project "'+project+'"',
  '  --revision "'+rev+'"'].join(' ');
 $('command').value=cmd;
}
async function load(){
 if(!project)return;
 const geo=await api('/api/projects/'+project+'/reconstruction');
 $('spatialRevision').value=geo.revision||'';
 $('geometry').textContent=geo.status==='NOT_CONFIGURED'
  ?'No reconstruction saved. Open /calibration first.'
  :'M3: '+geo.quality.status+' · independent metric check: '+
   (geo.quality.metric_transform_checked?'passed':'unverified')+
   ' · permitted turns: '+geo.model.movements.length+
   ' · note: geometry readiness does not prove traffic demand.';
 const res=await api('/api/projects/'+project+'/baseline');
 checkResult(res);
 if(res.status==='NOT_CONFIGURED')note('NEEDS_DATA: submit independent site survey, vehicle counts, controls and holdout.');
}
$('project').onchange=()=>{project=$('project').value;load().catch(e=>note(e.message))};
$('refresh').onclick=()=>load().catch(e=>note(e.message));
$('save').onclick=async()=>{
 try{
  if(!project)throw Error('Choose an existing video project');
  const revision=$('spatialRevision').value;
  if(!revision)throw Error('Complete geometry calibration before M4');
  const model=JSON.parse($('siteJson').value);
  const payload=await api('/api/projects/'+project+'/baseline',{
   method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({spatial_revision:revision,site:model})});
  checkResult(payload);
  note('Immutable SUMO baseline input package created: '+payload.revision+
   '\nRun the copied CLI command on the local machine with SUMO installed.');
 }catch(e){note('NOT READY: '+e.message)}
};
$('showSaved').onclick=()=>{
 if(!stored){note('No saved site package to restore.');return}
 const copy=structuredClone(stored.model);
 for(const key of ['schema_version','spatial_revision','project_id',
    'source_video_sha256','source_track_sha256','coordinate_system'])delete copy[key];
 $('siteJson').value=JSON.stringify(copy,null,2);
 note('Rehydrated last saved, immutable input for revision. Changes create a new revision.');
};
$('copy').onclick=async()=>{
 try{await navigator.clipboard.writeText($('command').value);
     note('Copied local SUMO baseline runner command.')}catch{note('Select and copy the command manually.')}
};
(async()=>{
 try{
  const list=(await api('/api/projects')).projects;
  for(const p of list)$('project').add(new Option(p.name,p.id));
  const selected=new URLSearchParams(location.search).get('project_id');
  if(selected&&list.some(p=>p.id===selected))$('project').value=selected;
  else if(list.length)$('project').value=list[0].id;
  project=$('project').value;
  if(project)await load();
  else note('No project yet. Analyze a real video in the main dashboard.');
 }catch(e){note(e.message)}
})();
})();
</script></body></html>"""
