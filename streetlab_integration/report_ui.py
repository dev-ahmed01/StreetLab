"""M7 plain-language project report preview with controlled evidence export."""
from __future__ import annotations

def report_html() -> str:
    return r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetLab | Project evidence report</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#f5f7f8;color:#24333e}
*{box-sizing:border-box}body{margin:0}
header{background:#fff;border-bottom:1px solid #dde4e9;padding:19px max(20px,calc((100% - 1050px)/2))}
main{max-width:1050px;margin:0 auto;padding:24px;display:grid;gap:17px}
section{background:#fff;padding:23px;border:1px solid #dfe6e9;border-radius:12px}
h1{font-size:27px;margin:7px 0}h2{font-size:19px;margin:0 0 12px}
p{font-size:14px;color:#5a6b77;line-height:1.6}
label{font-size:13px;font-weight:650;display:grid;gap:8px;max-width:500px}
select{font-size:14px;padding:10px;border:1px solid #c6d2dc;border-radius:8px}
a{color:#226169;text-decoration:none}
.button{display:inline-block;background:#235b63;color:white;padding:11px 16px;border-radius:8px;font-size:13px;text-decoration:none}
.button.ghost{background:#fff;border:1px solid #bdced3;color:#235b63}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.note{background:#f4f8f9;border:1px solid #dfe6e9;border-radius:8px;padding:14px;line-height:1.5;font-size:13px;color:#435867}
pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7;font:13px ui-monospace,Consolas,monospace}
.state{font-size:12px;color:#6b7e87}
</style></head><body>
<header><a href="/">← Project workspace</a>
<h1>Evidence and study report</h1>
<p>Reproducible project records, explicit scientific boundaries, and verified provenance—without exporting raw source videos or full tracking logs.</p>
</header><main>
<section><h2>Choose project</h2>
<label>Site study<select id="slReportProject"><option value="">Select a project</option></select></label>
<div class="actions"><a class="button ghost" id="slBack" href="/">Return to project</a>
<a class="button" id="slZip" aria-disabled="true">Export verification package (.zip)</a>
<a class="button ghost" id="slMarkdown" aria-disabled="true">Download readable report (.md)</a></div>
<p class="state" role="status" aria-live="polite" id="slState">Choose a study to load its verified report.</p>
</section>
<section><h2>What the report says</h2>
<div class="note">Evidence files in the ZIP are checksummed and bound to one project. A checksum does not prove who took the measurements. Video frames, full track rows, original video and large SUMO traces are intentionally excluded.</div>
<pre id="slReport">No report loaded.</pre></section>
</main>
<script>
(()=>{
const $=id=>document.getElementById(id);
const escapeDownload=id=>'/api/projects/'+encodeURIComponent(id);
async function request(url){const resp=await fetch(url);const j=await resp.json().catch(()=>({detail:'Unusable response'}));if(!resp.ok)throw Error(j.detail||'Request failed');return j}
async function show(){
 const id=$('slReportProject').value;
 for(const key of ['slZip','slMarkdown']){$(key).removeAttribute('href');$(key).setAttribute('aria-disabled','true')}
 if(!id){$('slReport').textContent='No report loaded.';return}
 try{
  const url=escapeDownload(id);
  const report=await request(url+'/report');
  $('slState').textContent='Verified project evidence loaded. No physical causal outcome claimed.';
  const blocks=['STREETLAB / PROJECT STUDY REPORT','',
    'Project: '+report.project.name,
    'Original video SHA: '+(report.source_video?.sha256||'Not yet uploaded'),
    'Source observations: '+report.observation.status,
    'Tracker identities: '+(report.observation.source_tracker_identity_count??'Not observed'),
    'Physical vehicle census from tracking: NOT established','',
    'Spatial reconstruction: '+report.survey_geometry.status,
    'Calibration revision: '+(report.survey_geometry.revision||'Not available'),'',
    'SUMO baseline: '+report.site_baseline.status,
    'Field distinct-vehicle count: '+(report.site_baseline.reviewed_field_vehicle_count??'Not measured'),
    'Baseline validity claim: SOFTWARE QA ONLY','',
    'Paired scenario status: '+report.scenarios.status,
    'Simulated mean paired delta (seconds): '+
     (report.scenarios.paired_simulation_comparison?.mean_paired_simulation_difference_s??'Not available'),
    'Observed real-world intervention effect: NOT VERIFIED','',
    'Next step: '+report.next_action.reason,'',
    'LIMITATIONS',...report.limitations.map(s=>'• '+s)];
  $('slReport').textContent=blocks.join('\n');
  $('slBack').href='/?project_id='+encodeURIComponent(id);
  $('slZip').href=url+'/evidence.zip';$('slMarkdown').href=url+'/report.md';
  for(const key of ['slZip','slMarkdown'])$(key).setAttribute('aria-disabled','false');
 }catch(error){$('slState').textContent=error.message;$('slReport').textContent='Unable to verify study evidence.'}
}
$('slReportProject').onchange=show;
(async()=>{try{
 const list=(await request('/api/projects')).projects;
 for(const p of list)$('slReportProject').add(new Option(p.name,p.id));
 const selected=new URLSearchParams(location.search).get('project_id');
 if(selected&&list.some(p=>p.id===selected))$('slReportProject').value=selected;
 else if(list.length)$('slReportProject').value=list[0].id;
 await show();
 }catch(e){$('slState').textContent=e.message}
})();
})();
</script></body></html>"""
