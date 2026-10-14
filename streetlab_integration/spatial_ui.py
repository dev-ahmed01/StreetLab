"""Deliberately guided StreetLab M3 editor, separate from observation and demo UI.

A user clicks real source pixels and supplies measured, independent LOCAL-meter
coordinates. The application never silently picks scale, GPS, lanes or turns.
"""
from __future__ import annotations


def calibration_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>StreetLab | Guided junction calibration</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#202936;background:#f6f7f9}
*{box-sizing:border-box} body{margin:0}
header{padding:18px max(24px,calc((100% - 1320px)/2));background:#fff;border-bottom:1px solid #e0e4e9}
header a{color:#245d60;text-decoration:none;font-size:14px}
main{max-width:1320px;margin:auto;padding:24px;display:grid;gap:18px}
section{background:#fff;border:1px solid #dde2e8;border-radius:12px;padding:22px}
h1{font-size:25px;margin:8px 0}h2{font-size:19px;margin:0 0 12px}
p{color:#566273;line-height:1.52;font-size:14px}
label{display:grid;gap:6px;font-size:13px;font-weight:600;min-width:0}
input,select{background:#fff;border:1px solid #c8cfd9;border-radius:8px;padding:10px;width:100%;font:inherit;color:inherit}
button{background:#243b43;color:white;border:0;border-radius:8px;padding:11px 14px;cursor:pointer;font-size:14px}
button.secondary{background:#f1f5f7;color:#263744;border:1px solid #cdd6dd}
button:disabled{opacity:.45;cursor:not-allowed}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:end}.row>*{flex:1;min-width:145px}
.workspace{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(320px,1fr);gap:18px;align-items:start}
.stack{display:grid;gap:16px}.note{font-size:13px;color:#606b79}
#frameCanvas{width:100%;max-height:620px;background:#0e1923;cursor:crosshair;border-radius:8px}
#frameCanvas image{pointer-events:none}#frameCanvas .mark{pointer-events:none}
#notice{border:1px solid #ccd9e0;background:#f7fafb;padding:14px;border-radius:8px;white-space:pre-wrap}
#quality{white-space:pre-wrap;font-size:13px;line-height:1.6;color:#405363}
#zoneList,#movementList,#controlList{font-size:13px;line-height:1.75;max-height:150px;overflow:auto}
#draftMode{font-weight:700;color:#205a5c;font-size:14px}
@media(max-width:880px){.workspace{grid-template-columns:1fr}main{padding:14px}header{padding:16px}}
</style>
</head>
<body>
<header><a href="/">← StreetLab projects</a><h1>Junction calibration</h1>
<p>Guide the reconstruction from genuine source video. Coordinates below are a local measured ground plane, never GPS.</p></header>
<main>
<section>
<h2>1. Select observation</h2>
<div class="row">
<label>Project<select id="project"><option value="">Choose a project</option></select></label>
<label>Completed video analysis<select id="job"><option value="">Choose a successful job</option></select></label>
<label>Source frame index<input id="frameIndex" type="number" min="0" step="1"></label>
<button id="loadFrame">Show real frame</button>
</div>
<p id="sourceNote">Only successfully processed projects can be calibrated. No simulation outputs are generated on this page.</p>
</section>
<div class="workspace">
<div class="stack">
<section>
<h2>2. Mark actual road geometry</h2>
<p>Choose a mode, then click the real source image. Source-image pixels are preserved even when the browser scales the preview.</p>
<div class="row">
<button class="secondary" id="modeAnchor">Mark anchor</button>
<button class="secondary" id="modeCheck">Mark independent check</button>
<button class="secondary" id="modeZone">Draw zone polygon</button>
<button class="secondary" id="stopMode">Stop drawing</button>
</div>
<p>Current tool: <span id="draftMode">None</span></p>
<svg id="frameCanvas" viewBox="0 0 640 360" role="img"
 aria-label="Source traffic video frame. Click to place selected road geometry points.">
<text x="20" y="40" fill="white">Choose a successful source job to display real footage</text>
</svg>
<p class="note">Blue circles: calibration anchors. Orange squares: independent scale checks. Outlined shapes: manually drawn approach and exit zones.</p>
</section>
<section>
<h2>3. Define approaches, exits and movements</h2>
<div class="row">
<label>Zone ID (lowercase)<input id="zoneId" placeholder="e.g. north_entry" maxlength="30"></label>
<label>Zone role<select id="zoneRole"><option value="APPROACH">Approach</option><option value="EXIT">Exit</option></select></label>
<label>Approach lanes (verified)<input id="zoneLanes" type="number" min="1" max="12" placeholder="Unknown"></label>
<button id="finishZone">Save drawn polygon</button>
</div>
<div id="zoneList">No road polygons saved</div>
<div class="row">
<label>From approach<select id="fromZone"></select></label>
<label>To exit<select id="toZone"></select></label>
<button id="addMove">Add permitted movement</button>
</div>
<div id="movementList">No movement links supplied</div>
</section>
</div>
<div class="stack">
<section>
<h2>Measured calibration</h2>
<p>For every anchor or independent check, enter coordinates that were measured or surveyed on the same flat road plane. Click after entering them.</p>
<div class="row">
<label>World X (meters)<input id="worldX" type="number" step="any" placeholder="Measured X"></label>
<label>World Y (meters)<input id="worldY" type="number" step="any" placeholder="Measured Y"></label>
</div>
<label style="margin-top:12px">Scale measurement / survey provenance
<input id="scaleBasis" maxlength="500" placeholder="e.g. field-surveyed distance with reference drawing"></label>
<p>Exactly four non-collinear anchors determine a homography. At least one separately measured check point is required to publish local-meter trajectories.</p>
<div class="row">
<button class="secondary" id="clearAnchors">Clear anchors</button>
<button class="secondary" id="clearChecks">Clear checks</button>
<button class="secondary" id="clearDraft">Discard unsaved zone</button>
</div>
<div id="controlList">No calibration controls added</div>
</section>
<section>
<h2>Review and save</h2>
<p>Measured geometry is versioned and SHA-verified, linked to the original video, frozen tracker and M1 observation receipt. Revisions never overwrite earlier evidence.</p>
<button id="saveModel">Save reconstruction revision</button>
<div id="notice" role="status" aria-live="polite" style="margin-top:12px">Start by selecting a project and completed job.</div>
<h2 style="margin-top:20px">Evidence readiness</h2>
<div id="quality">Road geometry, independent calibration, verified demand and site SUMO inputs are not yet available.</div>
<p><a id="worldDownload" href="#" hidden>Export checked local-meter trajectories (CSV)</a></p>
<p><strong>Simulation gate:</strong> real-site SUMO remains blocked until verified demand, route connectivity, signal inputs and baseline fidelity are completed in M4.</p>
</section>
</div>
</div>
</main>
<script>
(()=>{
const $=id=>document.getElementById(id);
const NS='http://www.w3.org/2000/svg';
const state={project:'',job:'',width:640,height:360,mode:'',model:null,draft:[]};
const empty=()=>({anchors:[],checks:[],zones:[],movements:[],scale_basis:''});
state.model=empty();
async function api(url,options={}){
 const r=await fetch(url,options);const data=await r.json().catch(()=>({detail:'Invalid server response'}));
 if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'Request failed');
 return data;
}
const notice=msg=>$('notice').textContent=msg;
const currentProject=()=>{if(!state.project)throw Error('Select a project first');return state.project};
const currentJob=()=>{if(!state.job)throw Error('Select a completed job first');return state.job};
const number=id=>{const n=Number($(id).value);if($(id).value.trim()===''||!Number.isFinite(n))throw Error('Enter measured world X and Y in meters');return n};
function mode(name){state.mode=name;$('draftMode').textContent=name||'None'}
function svg(name,attrs){
 const element=document.createElementNS(NS,name);
 for(const [key,value] of Object.entries(attrs))element.setAttribute(key,String(value));
 return element;
}
function draw(){
 const area=$('frameCanvas');area.setAttribute('viewBox','0 0 '+state.width+' '+state.height);
 area.replaceChildren();
 if(state.job){
  const img=svg('image',{x:0,y:0,width:state.width,height:state.height,
   href:'/api/jobs/'+state.job+'/source-frame?frame='+encodeURIComponent($('frameIndex').value)});
  area.appendChild(img);
 }
 for(const zone of state.model.zones){
  const vertices=zone.polygon_pixels.map(p=>p.join(',')).join(' ');
  area.appendChild(svg('polygon',{points:vertices,fill:'rgba(45,170,140,.15)',
   stroke:zone.role==='APPROACH'?'#2ec19c':'#ffb351','stroke-width':3,'class':'mark'}));
 }
 if(state.draft.length){
  area.appendChild(svg('polyline',{points:state.draft.map(p=>p.join(',')).join(' '),
    fill:'none',stroke:'#ffb351','stroke-width':3,'class':'mark'}));
 }
 for(const [items,color] of [[state.model.anchors,'#38bdf8'],[state.model.checks,'#f59e0b']]){
  for(const item of items){
   area.appendChild(svg('circle',{cx:item.pixel[0],cy:item.pixel[1],r:6,
    stroke:'#12202c','stroke-width':2,fill:color,'class':'mark'}));
  }
 }
}
function refresh(){
 $('scaleBasis').value=state.model.scale_basis||'';
 $('controlList').textContent=state.model.anchors.length+' / 4 anchors · '+
    state.model.checks.length+' independent check point(s)';
 $('zoneList').textContent=state.model.zones.length
   ?state.model.zones.map(z=>z.id+' ('+z.role+', '+z.polygon_pixels.length+
     ' vertices, lanes '+(z.lane_count??'unverified')+')').join(' · ')
   :'No approach/exit polygons saved';
 $('movementList').textContent=state.model.movements.length
   ?state.model.movements.map(m=>m.from_zone+' → '+m.to_zone).join(' · ')
   :'No movement links supplied';
 for(const [select,role] of [['fromZone','APPROACH'],['toZone','EXIT']]){
  const sel=$(select);sel.replaceChildren();
  for(const z of state.model.zones.filter(z=>z.role===role))sel.add(new Option(z.id,z.id));
 }
 draw();
}
async function loadProject(){
 state.job='';state.model=empty();state.draft=[];mode('');
 const project=await api('/api/projects/'+currentProject());
 const source=project.source;
 $('sourceNote').textContent=source
  ?source.original_filename+' · '+source.metadata.width+' × '+source.metadata.height+
   ' · '+source.metadata.fps+' source fps · local image pixels'
  :'Upload video from the project dashboard before starting calibration';
 state.width=source?.metadata.width||640;state.height=source?.metadata.height||360;
 const jobs=(await api('/api/projects/'+state.project+'/jobs')).jobs.filter(j=>j.status==='SUCCEEDED');
 $('job').replaceChildren(new Option('Select completed analysis',''));
 for(const item of jobs)$('job').add(new Option(item.id.slice(0,10)+' · '+item.total_frames+' frames',item.id));
 const existing=await api('/api/projects/'+state.project+'/reconstruction');
 if(existing.status!=='NOT_CONFIGURED'){
  state.model=existing.model;
  $('job').value=existing.model.source_job_id;state.job=$('job').value;
  renderQuality(existing);
 } else renderQuality(existing);
 refresh();
 if(state.job)await selectJob();
 else notice('Select a completed job, then draw your actual site geometry.');
}
async function selectJob(){
 state.job=$('job').value;
 if(!state.job)return;
 const job=await api('/api/jobs/'+state.job);
 $('frameIndex').min=job.config.first_frame;
 $('frameIndex').max=job.config.last_frame;
 $('frameIndex').value=job.config.first_frame;
 draw();
 notice('Genuine source frame loaded. Four anchors plus a separate check point are required for metric mapping.');
}
function renderQuality(data){
 const q=data.quality||{};
 const parts=[q.status||'NEEDS_DATA'];
 if(q.metric_transform_checked)parts.push('Independent planar check: passed (operator-supplied measurements).');
 if(q.independent_checkpoint_error_m?.length)
   parts.push('Measured check residuals (m): '+q.independent_checkpoint_error_m.join(', '));
 if(q.missing_evidence?.length)parts.push('Missing: '+q.missing_evidence.join('; '));
 if(q.mapped_observations!==undefined)parts.push('Calibrated source observations: '+q.mapped_observations);
 if(q.tracked_id_candidate_movements)
   parts.push('Tracker-ID movement candidates (not physical counts): '+
     JSON.stringify(q.tracked_id_candidate_movements));
 $('quality').textContent=parts.join('\n');
 $('worldDownload').hidden=!q.metric_transform_checked;
 if(! $('worldDownload').hidden)
  $('worldDownload').href='/api/projects/'+state.project+'/reconstruction/export';
}
$('project').onchange=()=>{state.project=$('project').value;loadProject().catch(e=>notice(e.message))};
$('job').onchange=()=>selectJob().catch(e=>notice(e.message));
$('loadFrame').onclick=()=>{
 const v=Number($('frameIndex').value);
 if(!Number.isInteger(v)||v<Number($('frameIndex').min)||v>Number($('frameIndex').max)){
  notice('Choose a valid frame inside the completed job interval.');return;
 }
 draw();notice('Displaying source-frame '+v+' from the verified video.');
};
for(const [id,value] of [['modeAnchor','anchor'],['modeCheck','check'],['modeZone','zone'],['stopMode','']]){
 $(id).onclick=()=>mode(value);
}
$('frameCanvas').addEventListener('click',event=>{
 try{
  currentJob();
  if(!state.mode)throw Error('Choose a calibration or zone drawing mode');
  const svgEl=$('frameCanvas');
  const p=svgEl.createSVGPoint();p.x=event.clientX;p.y=event.clientY;
  const matrix=svgEl.getScreenCTM();if(!matrix)throw Error('Source display is not ready');
  const at=p.matrixTransform(matrix.inverse());
  const pixel=[Number(at.x.toFixed(3)),Number(at.y.toFixed(3))];
  if(pixel[0]<0||pixel[1]<0||pixel[0]>state.width||pixel[1]>state.height)
    throw Error('Click inside the video image');
  if(state.mode==='zone'){
   state.draft.push(pixel);notice('Zone vertex '+state.draft.length+' added. Finish polygon with at least three.');
  }else{
   const pair={pixel,world_m:[number('worldX'),number('worldY')],provenance:'OBSERVED_MANUAL'};
   if(state.mode==='anchor'){
    if(state.model.anchors.length>=4)throw Error('Four anchors already recorded; clear anchors to revise');
    state.model.anchors.push(pair);
   }else {
    if(state.model.checks.length>=8)throw Error('Maximum eight independent check points');
    state.model.checks.push(pair);
   }
   notice('Measured '+state.mode+' correspondence added; entered meter coordinates were supplied manually.');
  }
  refresh();
 }catch(e){notice(e.message)}
});
$('finishZone').onclick=()=>{
 try{
  currentJob();
  const id=$('zoneId').value.trim();
  if(!/^[a-z0-9_]{1,30}$/.test(id))throw Error('Use a lowercase ID (a-z, 0-9, underscore)');
  if(state.model.zones.some(z=>z.id===id))throw Error('That zone ID already exists');
  if(state.draft.length<3)throw Error('Click at least three polygon corners first');
  const lanes=$('zoneLanes').value.trim();
  const lc=lanes===''?null:Number(lanes);
  state.model.zones.push({id,role:$('zoneRole').value,label:id,
   polygon_pixels:state.draft.slice(),lane_count:lc,provenance:'OBSERVED_MANUAL'});
  state.draft=[];mode('');refresh();notice('Manual polygon '+id+' added to draft.');
 }catch(e){notice(e.message)}
};
$('addMove').onclick=()=>{
 try{
  const from=$('fromZone').value,to=$('toZone').value;
  if(!from||!to)throw Error('Define approach and exit polygons first');
  if(state.model.movements.some(m=>m.from_zone===from&&m.to_zone===to))
    throw Error('Movement already registered');
  state.model.movements.push({from_zone:from,to_zone:to,provenance:'OBSERVED_MANUAL'});
  refresh();notice('Movement manually declared; traffic demand is not inferred from the link.');
 }catch(e){notice(e.message)}
};
$('clearAnchors').onclick=()=>{state.model.anchors=[];refresh()};
$('clearChecks').onclick=()=>{state.model.checks=[];refresh()};
$('clearDraft').onclick=()=>{state.draft=[];refresh()};
$('saveModel').onclick=async()=>{
 try{
  currentProject();currentJob();
  state.model.scale_basis=$('scaleBasis').value.trim();
  const data=await api('/api/projects/'+state.project+'/reconstruction',{
   method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({job_id:state.job,model:state.model})});
  renderQuality(data);
  notice('Immutable reconstruction revision saved: '+data.revision.slice(0,16)+
    '. Real-site SUMO remains blocked pending M4 demand and network validation.');
 }catch(e){notice(e.message)}
};
(async()=>{
 try{
  const projects=(await api('/api/projects')).projects;
  const selector=$('project');
  for(const p of projects)selector.add(new Option(p.name,p.id));
  const chosen=new URLSearchParams(location.search).get('project_id');
  if(chosen&&projects.some(p=>p.id===chosen))selector.value=chosen;
  else if(projects.length)selector.value=projects[0].id;
  state.project=selector.value;
  if(state.project)await loadProject();
  else notice('Create a video project and process an observation from the main dashboard first.');
 }catch(e){notice(e.message)}
})();
})();
</script></body></html>"""
