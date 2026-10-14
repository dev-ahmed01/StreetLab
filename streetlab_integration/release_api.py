"""M8 human release-readiness navigator.

The browser only displays project readiness hints. Full source rehash and
native tripinfo reconciliation run via local CLI, never HTTP.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


def mount_release_page(app: FastAPI) -> None:
    @app.get("/release",response_class=HTMLResponse)
    def release_page():
        return r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetLab | Release acceptance</title>
<style>
:root{font:14px Inter,ui-sans-serif,system-ui,sans-serif;color:#22343e;background:#f5f7f8}
*{box-sizing:border-box}body{margin:0}
header{background:#fff;border-bottom:1px solid #dce4e8;padding:20px max(24px,calc((100% - 1080px)/2))}
main{max-width:1080px;margin:auto;padding:22px;display:grid;gap:16px}
section{background:#fff;border:1px solid #dae3e8;padding:22px;border-radius:12px}
h1{font-size:26px;margin:10px 0}h2{font-size:18px;margin:0 0 12px}
p{color:#596a75;font-size:14px;line-height:1.6}
a{color:#1e6169;text-decoration:none}
label{display:grid;gap:7px;font-size:13px;font-weight:650}
select{padding:10px;border:1px solid #c9d4da;border-radius:8px;font:inherit;max-width:480px}
.columns{display:grid;grid-template-columns:1.4fr .9fr;gap:15px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.check{border:1px solid #dee5e9;border-radius:9px;padding:14px}
.check strong{display:block;font-size:14px}
.check span{display:block;color:#526875;font-size:12px;margin-top:8px}
.warn{font-size:13px;padding:14px;border:1px solid #eadcc5;background:#fbf9f3;border-radius:9px}
pre{white-space:pre-wrap;word-break:break-word;font:12px Consolas,monospace;line-height:1.7;background:#f3f6f7;padding:15px;border-radius:9px}
button{padding:10px 13px;background:#215e66;color:white;border:0;border-radius:8px;cursor:pointer}
.status{font-size:12px;color:#596b77}
@media(max-width:790px){.columns,.grid{grid-template-columns:1fr}}
</style></head><body>
<header><a href="/">← StreetLab workspace</a><h1>Release acceptance</h1>
<p>Seven integration stages are software-built. Final acceptance must reproduce the entire real-site chain before any operational release decision.</p>
</header><main>
<section><label>Project<select id="project"><option value="">Choose a project</option></select></label>
<p id="status" class="status" role="status" aria-live="polite">Select your project to inspect source-stage readiness.</p>
</section>
<div class="columns"><section><h2>Preliminary project evidence</h2>
<div class="grid" id="checks"></div>
<p class="warn"><strong>Production release: NOT APPROVED.</strong> Browser readiness is not an acceptance pass. The offline M8 CLI must rehash the original video, verify the actual frozen model, re-score all SUMO outputs and reconcile original field records. Human review and external approval still remain.</p>
</section>
<section><h2>Next action</h2>
<p id="nextAction">No source selected.</p>
<p>Use separate local terminals for the web app and frozen video worker. Original survey, census and holdout files must come from independent physical observations, not a demo fixture.</p>
<a href="/reports">View project evidence report →</a></section>
</div>
<section><h2>Generate a formal acceptance report locally</h2>
<p>Run the following command in PowerShell from the repository directory. Replace the placeholder paths with your actual frozen model and physically collected field evidence folder.</p>
<pre id="command">Select a real project first</pre><button type="button" id="copy">Copy command</button>
<p>The command produces explicit PASS, NEEDS_DATA, NEEDS_VERIFICATION, BLOCKED or FAIL states. A green software test cannot establish real-junction traffic data or intervention outcomes.</p>
</section></main>
<script>
(()=>{
const $=id=>document.getElementById(id);
async function fetchJson(url){
 const response=await fetch(url);
 const data=await response.json().catch(()=>({detail:'Cannot decode server response'}));
 if(!response.ok)throw Error(data.detail||'Source evidence unavailable');
 return data;
}
function show(s){
 const checks=$('checks');checks.replaceChildren();
 const sections=[
  ['Original footage',s.stages.observation.status],
  ['Measured geometry',s.stages.geometry.status],
  ['Actual SUMO baseline',s.stages.baseline.status],
  ['Paired simulation',s.stages.scenarios.status],
  ['Frozen Windows W04 model','LOCAL CLI VERIFICATION REQUIRED'],
  ['Independent raw field records','LOCAL CLI REVIEW REQUIRED'],
 ];
 for(const [name,state] of sections){
  const card=document.createElement('div');card.className='check';
  const title=document.createElement('strong');title.textContent=name;
  const detail=document.createElement('span');detail.textContent=state.replaceAll('_',' ');
  card.append(title,detail);checks.append(card);
 }
 $('nextAction').textContent=s.next_action.reason;
 $('status').textContent='Current project: '+s.project.name+' · Preview only. No final acceptance claim.';
 $('command').textContent=[
   '.\\scripts\\STREETLAB_M8.ps1 -Mode acceptance',
   '-Project "'+s.project.id+'"',
   '-ModelDir "C:\\PATH\\TO\\FROZEN_MODEL"',
   '-FrozenProvenance "C:\\PATH\\TO\\FROZEN_PROVENANCE.json"',
   '-FieldDir "C:\\PATH\\TO\\REAL_FIELD_RECORDS"',
   '-VerifySourceSha',
   '-OutputDir ".\\artifacts\\m8_operator_acceptance"'
 ].join(' ');
}
$('project').onchange=async()=>{
 const id=$('project').value;
 if(!id)return;
 try{show(await fetchJson('/api/projects/'+encodeURIComponent(id)+'/workspace'))}
 catch(e){$('status').textContent='Release preview blocked: '+e.message;$('checks').replaceChildren()}
};
$('copy').onclick=()=>{
 const text=$('command').textContent;
 if(navigator.clipboard)navigator.clipboard.writeText(text).catch(()=>{$('status').textContent='Select and copy the command manually'});
};
(async()=>{try{
 const records=(await fetchJson('/api/projects')).projects;
 for(const p of records)$('project').add(new Option(p.name,p.id));
 const requested=new URLSearchParams(location.search).get('project_id');
 if(requested&&records.some(p=>p.id===requested))$('project').value=requested;
 else if(records.length)$('project').value=records[0].id;
 if($('project').value)show(await fetchJson('/api/projects/'+$('project').value+'/workspace'));
 }catch(e){$('status').textContent=e.message}
})();
})();
</script></body></html>"""
