"""M9 site-based collection preview, deliberately read-only and metadata-only.

Only a locally invoked CLI can create blank paper forms or inspect originals.
No field files, observer details or media can be uploaded through HTTP here.
"""
from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from streetlab_integration.video_jobs import VideoStore,JobError
from streetlab_integration.fieldwork import field_plan,FieldworkError


def mount_fieldwork_routes(app: FastAPI,workdir: Path) -> None:
    store=VideoStore(workdir)

    @app.get("/api/projects/{project_id}/fieldwork-plan")
    def plan(project_id: str):
        try:
            return field_plan(store,project_id)
        except JobError as exc:
            raise HTTPException(status_code=404,detail="Project unavailable") from exc
        except (FieldworkError,ValueError,OSError,KeyError,TypeError) as exc:
            raise HTTPException(status_code=409,detail="Project evidence must pass integrity review") from exc

    @app.get("/fieldwork",response_class=HTMLResponse)
    def page():
        return r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetLab | Physical fieldwork</title><style>
:root{font:14px Inter,ui-sans-serif,system-ui,sans-serif;color:#253440;background:#f5f7f8}
*{box-sizing:border-box}body{margin:0}
header{padding:24px max(20px,calc((100% - 1000px)/2));background:white;border-bottom:1px solid #dbe3e8}
main{max-width:1000px;margin:0 auto;padding:20px;display:grid;gap:16px}
section{background:white;border:1px solid #dbe3e8;border-radius:12px;padding:22px}
h1{font-size:26px;margin:12px 0}h2{font-size:18px;margin:0 0 10px}
p{line-height:1.6;color:#566775}a{color:#1d5964;text-decoration:none}
select,button{font:inherit;border-radius:8px;padding:10px;border:1px solid #cbd4da}
select{width:100%;max-width:490px;background:#fff}
button{background:#245e68;color:white;cursor:pointer}button:disabled{opacity:.5}
ol{padding-left:23px;line-height:1.8}li{padding-bottom:6px}
code,pre{font:12px Consolas,ui-monospace,monospace}pre{background:#f3f6f7;padding:15px;white-space:pre-wrap;word-break:break-word;border-radius:8px}
.alert{background:#fcfaf3;border:1px solid #ebe0c6;padding:14px;border-radius:8px}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:15px}
@media(max-width:750px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><a href="/">← StreetLab</a><h1>Physical fieldwork</h1>
<p>Collect genuine measured road and traffic evidence. The software never generates physical counts or observed travel-time samples.</p>
</header><main><section><label for="project"><strong>Source project</strong></label>
<div style="margin-top:8px"><select id="project"><option value="">Choose project</option></select></div>
<p id="status" role="status" aria-live="polite">Select an existing project.</p></section>
<div class="grid"><section><h2>Physical observation plan</h2>
<p id="source">A reviewed M3 site geometry is required before making a project-specific field sheet.</p>
<ol id="moves"></ol></section><section><h2>Collection protocol</h2>
<ol>
<li>Verify actual surveyed road widths/lanes and control on site.</li>
<li>Count distinct observed vehicles by movement and class.</li>
<li>Time genuine trips in a <strong>separate</strong> holdout session.</li>
<li>Have a different field reviewer inspect the original evidence.</li>
<li>Bind original records to the M4 baseline without rewriting them to match SUMO.</li>
</ol>
<p class="alert"><strong>Not certification.</strong> A passed field-file check cannot prove the observations are real, the reviewer independent, or the intervention effective.</p></section></div>
<section><h2>Create blank field observation sheets</h2>
<p>Use this command locally. It creates only labeled, empty templates; original measured data stays on your machine and is never auto-uploaded.</p>
<pre id="command">Select a reviewed geometry project first</pre>
<button type="button" id="copy" disabled>Copy Windows command</button>
<p>After an independently reviewed M4 baseline exists, use <code>--hashes</code> and <code>--check</code> with the four authentic original files. Read <a href="/release">the release acceptance workflow</a> for the remaining tests.</p>
</section></main><script>
(()=>{
const $=x=>document.getElementById(x);
async function get(url){const res=await fetch(url);const data=await res.json().catch(()=>({detail:'No JSON response'}));if(!res.ok)throw Error(data.detail||'Request failed');return data}
function clear(){ $('moves').replaceChildren();$('copy').disabled=true;$('command').textContent='A reviewed M3 geometry is required'}
async function view(){
 clear();const id=$('project').value;if(!id)return;
 try {
  const plan=await get('/api/projects/'+encodeURIComponent(id)+'/fieldwork-plan');
  $('status').textContent=plan.status.replaceAll('_',' ');
  $('source').textContent=plan.next_action;
  if(plan.status!=='PLAN_READY_FOR_INDEPENDENT_FIELD_MEASUREMENT')return;
  for(const m of plan.movements){const li=document.createElement('li');li.textContent=m.from_zone+' → '+m.to_zone;$('moves').append(li)}
  $('command').textContent='.\\scripts\\STREETLAB_M9.ps1 -Mode kit -Project "'+id+'" -OutputDir ".\\artifacts\\real_fieldkit_'+id.slice(0,8)+'"';
  $('copy').disabled=false;
 }catch(e){$('status').textContent='Evidence unavailable: '+e.message}
}
$('project').onchange=view;
$('copy').onclick=()=>navigator.clipboard?.writeText($('command').textContent).catch(()=>{$('status').textContent='Copy the command manually'});
(async()=>{try{
 const response=await get('/api/projects');
 const projects=response.projects;
 for(const p of projects)$('project').add(new Option(p.name,p.id));
 const requested=new URLSearchParams(location.search).get('project_id');
 if(requested&&projects.some(p=>p.id===requested))$('project').value=requested;
 else if(projects.length)$('project').value=projects[0].id;
 await view();
 }catch(e){$('status').textContent=e.message}
})();
})();
</script></body></html>"""
