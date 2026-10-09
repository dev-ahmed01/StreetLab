"""M5 evidence-aware experiment and comparison UI (no invented site statistics)."""
from __future__ import annotations

def scenarios_html() -> str:
    return r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetLab | Scenario Decision Comparison</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#f6f7f9;color:#1d2a37}
*{box-sizing:border-box}body{margin:0}
header{background:white;border-bottom:1px solid #dfe3e9;padding:18px max(24px,calc((100% - 1180px)/2))}
main{max-width:1180px;margin:auto;padding:22px;display:grid;gap:16px}
section{background:white;border:1px solid #dfe4ea;border-radius:13px;padding:22px}
h1{font-size:25px;margin:8px 0}h2{font-size:18px;margin:0 0 12px}
p{font-size:14px;color:#586573;line-height:1.55}
a{color:#246468;text-decoration:none}
.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.split{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}
label{display:grid;gap:7px;font-size:13px;font-weight:600}
input,select,textarea{font:inherit;width:100%;padding:10px;border-radius:8px;border:1px solid #ccd5df;background:white;color:#1d2a37}
textarea{min-height:78px}
button{border:0;border-radius:8px;background:#243b43;color:#fff;padding:11px 14px;cursor:pointer}
button:disabled{opacity:.45;cursor:not-allowed}
button.secondary{background:#f3f6f8;border:1px solid #ccd5dc;color:#233c46}
.stack{display:grid;gap:14px}
.mono{white-space:pre-wrap;word-break:break-word;background:#f7f9fb;border:1px solid #dee4e9;border-radius:8px;padding:13px;font-size:13px;line-height:1.65}
.caption{font-size:12px;color:#687581}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px;border-bottom:1px solid #e1e6eb;text-align:left}
th{color:#556270}
@media(max-width:820px){.split,.row{grid-template-columns:1fr}main{padding:12px}header{padding:16px}}
</style></head><body>
<header><a href="/">← StreetLab workspace</a><h1>Scenario comparison</h1>
<p>Paired simulations against a checked site baseline. This page compares model hypotheses; it does not predict a proven real-world impact.</p></header>
<main>
<section>
<h2>1. Verified baseline</h2>
<div class="row"><label>Project<select id="project"><option value="">Choose project</option></select></label>
<label>M4 baseline revision<input id="revision" readonly placeholder="No completed site baseline"></label>
<label>Independent M4 baseline QA<input id="gate" readonly value="NEEDS_DATA"></label></div>
<p id="source">Select a project to check real-source provenance and baseline suitability.</p>
</section>
<div class="split">
<div class="stack">
<section>
<h2>2. Define a single operational change</h2>
<label>Intervention type
<select id="kind"><option value="APPROACH_SPEED_LIMIT">Reduced approach speed limit</option>
<option value="FIXED_SIGNAL_PLAN">Reviewed fixed-time signal phase durations</option></select></label>
<div id="speedForm" class="row" style="grid-template-columns:1fr 1fr;margin-top:14px">
<label>Measured road approach<select id="arm"></select></label>
<label>Proposed speed limit (meters / second)<input id="speed" type="number" min="1" max="45" step="any" placeholder="Enter proposed speed"></label>
</div>
<div id="signalForm" style="display:none;margin-top:14px">
<label>New ordered phase durations (seconds), comma-separated<input id="phases" placeholder="For each existing phase; do not change phase state order"></label>
<div class="caption" id="signalInfo"></div>
</div>
<p>Speed changes are restricted to the reviewed approach, at most 50% lower. Signal changes keep phase state order and require fixed-time baseline evidence. Road closures and rerouting are intentionally not guessed.</p>
</section>
<section>
<h2>3. Document operational assumptions</h2>
<div class="stack">
<label>Decision question<textarea id="question" placeholder="What site-level operational decision is this model experiment exploring?"></textarea></label>
<label>Implementation/feasibility review reference<textarea id="feasibility" placeholder="Describe field feasibility, lane and sign implications, permissions, and the review record"></textarea></label>
<label>Reviewer / sign-off record<input id="reviewer" placeholder="Reviewer and dated decision record"></label>
</div>
<p class="caption">Sensitivity grid: 90%, 100%, 110% of the independently counted flow; paired random seeds 42, 43, 44. These are experimental stress assumptions, not observed demand ranges.</p>
<button id="create">Validate and save experiment proposal</button>
<p id="message" class="mono" role="status" aria-live="polite">A passing M4 baseline is required.</p>
</section>
</div>
<div class="stack">
<section>
<h2>4. Run controlled simulation</h2>
<div id="status" class="mono">No scenario proposal executed</div>
<p>SUMO runs in the separate local CLI. Nine matched seed-and-demand combinations are executed for both the baseline and the change (18 SUMO runs), preserving the same demand and seed within each pair.</p>
<label>Windows runner command<textarea id="command" readonly style="min-height:140px;font:12px ui-monospace,Consolas,monospace"></textarea></label>
<button class="secondary" id="copy">Copy command</button>
<button class="secondary" id="refresh">Refresh result</button>
</section>
<section>
<h2>5. Comparison evidence</h2>
<div id="result" class="mono">No comparison is available. The model will not synthesize an improvement percentage.</div>
<div id="comparisons"></div>
<p class="caption">Difference is the intervention’s simulated trip duration minus the paired baseline (seconds). Negative values mean faster in the simulator only. Missing or censored trips make comparisons indeterminate. A min/max across scenarios is not a confidence interval.</p>
</section>
</div>
</div>
</main><script>
(()=>{
const $=id=>document.getElementById(id);
let project='',baseline=null,scenario=null;
async function api(url,options={}){
 const resp=await fetch(url,options);
 const data=await resp.json().catch(()=>({detail:'Unusable server response'}));
 if(!resp.ok)throw Error(data.detail||'Request failed');
 return data;
}
const message=t=>$('message').textContent=t;
$('kind').onchange=()=>{
 const fixed=$('kind').value==='FIXED_SIGNAL_PLAN';
 $('speedForm').style.display=fixed?'none':'grid';
 $('signalForm').style.display=fixed?'block':'none';
};
function showScenario(value){
 scenario=value.status==='NOT_CONFIGURED'?null:value;
 const state=scenario?.runtime?.status||'NOT_EXECUTED';
 $('status').textContent=scenario
  ?'Revision '+scenario.revision+'\nExecution: '+state+
    '\nReal-world outcome verified: NO'
  :'No immutable scenario proposal is configured for this project.';
 if(!scenario){$('command').value='';$('result').textContent='No simulated comparison exists';$('comparisons').replaceChildren();return}
 $('command').value=[
  'cd C:\\Users\\Admin\\Desktop\\StreetLab-engine-trial;',
  '& ".\\.venv-sahi-audit\\Scripts\\python.exe" -m streetlab_integration.scenario_experiments',
  '--workdir ".streetlab-m5"','--project "'+project+'"',
  '--revision "'+scenario.revision+'"'].join(' ');
 const runtime=scenario.runtime||{};
 if(state==='NOT_EXECUTED'){
  $('result').textContent='Proposal saved. No SUMO experiment has run. Results cannot be claimed.';
  $('comparisons').replaceChildren();return
 }
 let summary='Experiment state: '+runtime.status+'\n';
 if(runtime.all_conditions_comparable)
   summary+='Mean paired simulated difference: '+runtime.mean_paired_simulation_difference_s+' s\n'+
   'Range across tested settings: '+runtime.min_paired_simulation_difference_s+
   ' to '+runtime.max_paired_simulation_difference_s+' s\n'+
   'Faster in every simulated condition: '+runtime.simulated_improvement_in_every_condition;
 else summary+='Some simulation conditions are incomplete. No aggregate advantage is published.';
 summary+='\nPhysical-site improvement proven: NO';
 $('result').textContent=summary;
 const host=$('comparisons');host.replaceChildren();
 const table=document.createElement('table');
 const tr=document.createElement('tr');
 for(const label of ['Demand','Seed','Baseline mean (s)','Changed mean (s)','Delta (s)','Valid?']){
  const th=document.createElement('th');th.textContent=label;tr.append(th)
 }
 table.append(tr);
 for(const x of runtime.records||[]){
  const row=document.createElement('tr');
  for(const value of [x.demand_multiplier,x.seed,
    x.baseline.mean_trip_duration_s?.toFixed(2)??'—',
    x.intervention.mean_trip_duration_s?.toFixed(2)??'—',
    x.paired_difference_s??'—',x.comparison_eligible?'YES':'NO']){
    const td=document.createElement('td');td.textContent=String(value);row.append(td)
  }table.append(row);
 }
 host.append(table);
}
async function load(){
 if(!project)return;
 baseline=await api('/api/projects/'+project+'/baseline');
 const runtime=baseline.runtime||{};
 const passed=runtime.status==='BASELINE_FIDELITY_CHECKED'
   &&runtime.real_site_sumo_allowed===true;
 $('gate').value=passed?'BASELINE FIDELITY CHECKED':'NEEDS_DATA';
 $('revision').value=baseline.revision||'';
 $('source').textContent=passed
   ?'M4 qualified by software fidelity checks. Field-grounded data validity and causal transport behavior are still not independently certified.'
   :'A genuine executed and passing M4 baseline is required. Go to /baseline for missing evidence.';
 $('create').disabled=!passed;
 $('arm').replaceChildren();
 if(baseline.model){
  for(const a of baseline.model.arms.filter(a=>a.role==='APPROACH')){
   $('arm').add(new Option(a.zone_id+' · reviewed '+a.speed_mps+' m/s',a.zone_id))
  }
  if(baseline.model.control.kind==='FIXED_TIME_SIGNAL'){
   $('signalInfo').textContent='Reviewed original durations: '+
      baseline.model.control.phases.map(x=>x.duration_s).join(', ')+
      ' seconds. Keep the same state ordering.';
  }else{
   $('signalInfo').textContent='Not a fixed-time signal; this intervention will be rejected.';
  }
 }
 const existing=await api('/api/projects/'+project+'/scenario');
 showScenario(existing);
}
$('project').onchange=()=>{project=$('project').value;load().catch(e=>message(e.message))};
$('create').onclick=async()=>{
 try{
  if(!baseline?.revision)throw Error('A passing real-site baseline must be selected');
  const kind=$('kind').value;
  const intervention=kind==='APPROACH_SPEED_LIMIT'
   ?{kind,zone_id:$('arm').value,speed_mps:Number($('speed').value)}
   :{kind,phase_durations_s:$('phases').value.split(',').map(x=>Number(x.trim()))};
  const proposal={
   intervention,demand_multipliers:[.9,1.0,1.1],
   assumptions:{decision_question:$('question').value,
    feasibility_evidence_ref:$('feasibility').value,
    reviewer:$('reviewer').value,provenance:'HYPOTHETICAL_REVIEWED'}};
  const result=await api('/api/projects/'+project+'/scenario',{
   method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({baseline_revision:baseline.revision,scenario:proposal})});
  showScenario(result);
  message('Immutable reviewed experiment prepared: '+result.revision+
   '\nCopy and run CLI on a local SUMO installation. No effects have been claimed.');
 }catch(e){message('NOT READY: '+e.message)}
};
$('refresh').onclick=async()=>{
 try{const v=await api('/api/projects/'+project+'/scenario');showScenario(v);message('Verified original evidence and checked latest result.')}
 catch(e){message(e.message)}
};
$('copy').onclick=async()=>{
 try{await navigator.clipboard.writeText($('command').value);message('Command copied')}
 catch(e){message('Copy the local runner command manually')}
};
(async()=>{
 try{
  const records=(await api('/api/projects')).projects;
  for(const p of records)$('project').add(new Option(p.name,p.id));
  const chosen=new URLSearchParams(location.search).get('project_id');
  if(chosen&&records.some(p=>p.id===chosen))$('project').value=chosen;
  else if(records.length)$('project').value=records[0].id;
  project=$('project').value;
  if(project)await load();
  else message('Create and analyze a real project first.');
 }catch(e){message(e.message)}
})();
})();
</script></body></html>"""
