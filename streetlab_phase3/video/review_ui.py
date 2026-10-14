"""Generate a standalone browser-side manual annotator (no hosted service)."""
from __future__ import annotations
import json
from typing import Any


def offline_review_html(manifest: dict[str, Any]) -> str:
    cases=[{'case_id':r['case_id'],'image_file':r['image_file'],
            'crop_size':r['crop_size']} for r in manifest['cases']]
    encoded=json.dumps(cases,separators=(',',':')).replace('<','\\u003c')
    return r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>StreetLab — Blinded Independent Review</title>
<style>body{font:16px system-ui,Arial,sans-serif;color:#172a37;background:#f5f7f9;margin:0}
main{max-width:1090px;margin:auto;padding:25px}h1{font-size:24px;margin-bottom:4px}
.lede{color:#586b77;line-height:1.5}.panel{background:white;border:1px solid #dce3e8;border-radius:12px;padding:20px;margin-top:20px}
button,input,select{font:inherit;padding:10px;border:1px solid #b6c4cf;border-radius:8px;background:white;color:#172a37}
button{cursor:pointer}button:focus-visible{outline:3px solid #3f86b8}
button.primary{background:#155d83;color:white;border-color:#155d83}button:disabled{opacity:.5;cursor:not-allowed}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar label{display:flex;gap:6px;align-items:center}
.frames{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}
figure{margin:0;max-width:480px}figcaption{font-size:13px;color:#536573;margin-top:5px}
canvas{width:min(100%,480px);height:auto;border:1px solid #cdd7df;touch-action:none;cursor:crosshair}
.choices{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:16px}
#help{font-size:14px;color:#536573}.counter{font-weight:650}.error{color:#a42c35;margin-top:8px;min-height:20px}
</style></head><body><main><h1>StreetLab — Independent object review</h1>
<p class="lede">This is a <strong>blinded, local-only</strong> review. No FLUID labels or detector classes are shown. Left: original pixels; right: target ring. Draw a bounding box on the <strong>left</strong> by dragging. Confirm the physical object at the ring, or select absent/uncertain. Each of two reviewers must independently export their own CSV; never collaborate on labels.</p>
<section class="panel"><div class="toolbar"><label>Reviewer ID <input id="reviewer" placeholder="Your unique reviewer ID" autocomplete="off"></label>
<button id="prev">← Previous</button><span class="counter" id="progress"></span><button id="next">Next →</button>
<button id="download" class="primary">Export completed CSV</button></div>
<div class="frames"><figure><canvas id="raw" width="480" height="480" aria-label="Original crop; drag to box the selected object"></canvas><figcaption>Original crop — draw a bounding box here</figcaption></figure>
<figure><canvas id="marked" width="480" height="480" aria-label="Target point on the original crop"></canvas><figcaption>Reference ring — identify only the object at this position</figcaption></figure></div>
<div class="choices"><label>Presence <select id="presence"><option value="">Choose…</option><option value="PRESENT">PRESENT</option><option value="ABSENT">ABSENT</option><option value="UNCERTAIN">UNCERTAIN</option></select></label>
<label>Visible class <select id="klass"><option value="">Choose…</option><option>CAR</option><option>BUS</option><option>HEAVY_VEHICLE</option><option>MOTORCYCLE</option><option>PEDESTRIAN</option><option>BICYCLE</option><option>AUTO_RICKSHAW</option><option>OTHER</option><option>UNKNOWN</option></select></label>
<label>Notes <input id="note" placeholder="Optional uncertainty details" size="28"></label></div>
<p id="help">PRESENT requires class and crop-relative box. ABSENT/UNCERTAIN require no box.</p><p class="error" id="error" role="alert"></p></section>
<p class="lede">Scope: deliberately selected W04 cases, not exhaustive all-object annotation. The exported CSV is validated later against a second independent review. Unresolved cases stay unresolved. Do not modify the original FLUID labels.</p></main>
<script>const cases=__CASES__;
const state=cases.map(()=>({presence:'',physical_class:'',bbox:null,note:''}));let pos=0, drag=null, image=null;
const get=id=>document.getElementById(id), raw=get('raw'),marked=get('marked'), ctx=raw.getContext('2d'),mctx=marked.getContext('2d');
const vals=['presence','klass','note'];
function clamp(v,n){return Math.max(0,Math.min(v,n));}
function render(){const c=cases[pos],a=state[pos];get('progress').textContent=`Case ${pos+1}/${cases.length} · ${state.filter(x=>x.presence).length} selected`;
get('prev').disabled=pos===0;get('next').disabled=pos===cases.length-1;
get('presence').value=a.presence;get('klass').value=a.physical_class;get('note').value=a.note;
get('klass').disabled=a.presence!=='PRESENT';get('error').textContent='';
image=new Image();const idx=pos;image.onload=()=>{if(idx!==pos)return;
for(const [target,x] of [[ctx,0],[mctx,c.crop_size]]){target.canvas.width=c.crop_size;target.canvas.height=c.crop_size;
target.drawImage(image,x,0,c.crop_size,c.crop_size,0,0,c.crop_size,c.crop_size);}
showBox();};image.onerror=()=>get('error').textContent=`Cannot load ${c.image_file}. Keep this HTML next to the images folder.`;image.src=c.image_file;}
function showBox(){if(!image?.complete || !image.naturalWidth)return;ctx.drawImage(image,0,0,cases[pos].crop_size,cases[pos].crop_size,0,0,cases[pos].crop_size,cases[pos].crop_size);
let b=drag?.bbox||state[pos].bbox;if(b){ctx.strokeStyle='#ff3aa5';ctx.lineWidth=3;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}}
function point(evt){const r=raw.getBoundingClientRect();return [clamp((evt.clientX-r.left)*raw.width/r.width,raw.width),clamp((evt.clientY-r.top)*raw.height/r.height,raw.height)];}
raw.addEventListener('pointerdown',e=>{if(state[pos].presence!=='PRESENT')return;raw.setPointerCapture(e.pointerId);drag={start:point(e),bbox:null};});
raw.addEventListener('pointermove',e=>{if(!drag)return;const a=drag.start,b=point(e);drag.bbox=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];showBox();});
raw.addEventListener('pointerup',e=>{if(!drag)return;state[pos].bbox=drag.bbox;drag=null;showBox();});
get('presence').addEventListener('change',e=>{state[pos].presence=e.target.value; if(e.target.value!=='PRESENT'){state[pos].physical_class='';state[pos].bbox=null;}render();});
get('klass').addEventListener('change',e=>{state[pos].physical_class=e.target.value;});
get('note').addEventListener('input',e=>{state[pos].note=e.target.value;});
get('prev').addEventListener('click',()=>{pos--;render();});get('next').addEventListener('click',()=>{pos++;render();});
function quote(s){return '"'+String(s??'').replaceAll('"','""')+'"';}
get('download').addEventListener('click',()=>{const reviewer=get('reviewer').value.trim();
if(!reviewer || reviewer.startsWith('REPLACE_')){get('error').textContent='Please enter your own reviewer ID.';return;}
const lines=[['case_id','reviewer_id','presence','physical_class','x1','y1','x2','y2','note']];
for(let i=0;i<cases.length;i++){const c=cases[i],r=state[i];if(!['PRESENT','ABSENT','UNCERTAIN'].includes(r.presence)){pos=i;render();get('error').textContent='Complete every case before exporting.';return;}
if(r.presence==='PRESENT'&&(!r.physical_class||!r.bbox||(r.bbox[2]-r.bbox[0]<1)||(r.bbox[3]-r.bbox[1]<1))){pos=i;render();get('error').textContent='PRESENT requires a physical class and a drawn box on the left.';return;}
let box=r.presence==='PRESENT'?r.bbox.map(x=>x.toFixed(2)):['','','',''];lines.push([c.case_id,reviewer,r.presence,r.presence==='PRESENT'?r.physical_class:'',...box,r.note]);}
const csv=lines.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';const blob=new Blob([csv],{type:'text/csv;charset=utf-8'});
const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='w04_independent_review_'+reviewer.replace(/[^a-z0-9_-]/gi,'_')+'.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});render();</script></body></html>'''.replace('__CASES__',encoded)