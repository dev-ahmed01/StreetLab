(function(){
"use strict";
const root=document.getElementById("slw");if(!root)return;
const get=id=>document.getElementById(id);
const names=["observation","geometry","baseline","scenarios"];
const nice={NEEDS_SOURCE:"Needs video",NEEDS_ANALYSIS:"Ready to analyze",
QUEUED:"Queued",RUNNING:"Running",FAILED:"Failed",CANCELLED:"Cancelled",
OBSERVATION_VERIFIED:"Source verified",WAITING_FOR_OBSERVATION:"Waiting for video",
NEEDS_GEOMETRY:"Needs measured geometry",GEOMETRY_REVIEWED:"Reviewed",
STALE_OBSERVATION:"Old observation",NEEDS_DATA:"Needs evidence",
WAITING_FOR_GEOMETRY:"Waiting for geometry",NEEDS_BASELINE:"Needs census",
BASELINE_NOT_EXECUTED:"Needs SUMO run",BASELINE_QA_PASSED:"Baseline QA passed",
BASELINE_NEEDS_REVIEW:"Needs review",STALE_GEOMETRY:"Outdated geometry",
WAITING_FOR_BASELINE:"Waiting for baseline",NEEDS_SCENARIO:"Ready to explore",
PROPOSAL_NOT_EXECUTED:"Awaiting SUMO",SIMULATED_COMPARISON_ELIGIBLE:"Simulations available",
INDETERMINATE:"Results indeterminate",STALE_BASELINE:"Outdated baseline"};
const paths={observation:"/?advanced=1#m2Panel",geometry:"/calibration",
baseline:"/baseline",scenarios:"/scenarios"};
let project="",snapshot=null,selected="observation",busy=false;
function url(stage){return stage==="observation"?paths[stage]:
 paths[stage]+"?project_id="+encodeURIComponent(project)}
async function api(path,opts={}){
 const response=await fetch(path,opts);
 const data=await response.json().catch(()=>({detail:"Unexpected server response"}));
 if(!response.ok)throw Error(data.detail||"Request failed");
 return data;
}
function make(tag,txt,cls){
 const el=document.createElement(tag);
 if(txt!==undefined&&txt!==null)el.textContent=String(txt);
 if(cls)el.className=cls;
 return el;
}
function number(v,unit=""){
 return v===null||v===undefined||!Number.isFinite(Number(v))?"—":
 Number(v).toLocaleString(undefined,{maximumFractionDigits:2})+unit
}
function fact(label,value){
 const el=make("div",null,"fact");
 el.append(make("strong",value),make("span",label));return el
}
function facts(list){
 const box=make("div",null,"facts");
 for(const [key,val] of list)box.append(fact(key,val));return box
}
function paragraph(text,cls="sub"){return make("p",text,cls)}
function action(text,stage){
 const a=make("a",text,"cta");a.href=url(stage);return a
}
function soft(text){const box=make("div",null,"soft");box.append(paragraph(text));return box}
function addTable(host,head,rows){
 const t=make("table",null,"mini-table"),tr=make("tr");
 for(const h of head)tr.append(make("th",h));t.append(tr);
 for(const cells of rows){
  const r=make("tr");
  for(const cell of cells)r.append(make("td",cell));t.append(r);
 }
 host.append(t);
}
function diagram(graph){
 const ns="http://www.w3.org/2000/svg";
 const svg=document.createElementNS(ns,"svg");svg.setAttribute("viewBox","0 0 500 240");
 svg.setAttribute("class","localmap");
 svg.setAttribute("aria-label","Schematic from surveyed local-meter road arms; not GPS");
 svg.setAttribute("role","img");
 const all=[graph.center_world_m,...graph.arms.map(a=>a.outer_world_m)];
 const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
 const midX=(Math.min(...xs)+Math.max(...xs))/2;
 const midY=(Math.min(...ys)+Math.max(...ys))/2;
 const scale=Math.min(400/Math.max(1,Math.max(...xs)-Math.min(...xs)),
                      170/Math.max(1,Math.max(...ys)-Math.min(...ys)));
 const coord=p=>[250+(p[0]-midX)*scale,120-(p[1]-midY)*scale];
 const svgEl=(tag,attrs)=>{
  const e=document.createElementNS(ns,tag);
  for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));
  return e;
 };
 for(const arm of graph.arms){
  const a=coord(graph.center_world_m),b=coord(arm.outer_world_m);
  const color=arm.role==="APPROACH"?"#276e74":"#d19a4b";
  svg.append(svgEl("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:color,
   "stroke-width":Math.min(15,4+arm.lane_count*2),"stroke-linecap":"round"}));
  const lbl=svgEl("text",{x:b[0]+(b[0]>=250?8:-8),y:b[1]-9,
   "text-anchor":b[0]>=250?"start":"end","font-size":12,fill:"#324b56"});
  lbl.textContent=arm.zone_id;svg.append(lbl);
 }
 const c=coord(graph.center_world_m);
 svg.append(svgEl("circle",{cx:c[0],cy:c[1],r:8,fill:"#182e36",stroke:"white","stroke-width":2}));
 return svg;
}
function renderObservation(host){
 const stage=snapshot.stages.observation;
 const data=stage.last_verified_job,job=stage.latest_job;
 host.append(facts([
 ["Observed tracking points",data?number(data.observed_points):"—"],
 ["Source tracker IDs",data?number(data.tracking_identity_count):"—"],
 ["Frames with tracks",data?number(data.observed_frames):"—"],
 ["Source video",stage.source?.original_filename||"Not uploaded"]
 ]));
 host.append(paragraph(stage.note));
 if(job){
  const box=soft("Most recent analysis: "+job.status+" · "+job.completed_frames+
    " of "+job.total_frames+" source frames.");
  const bar=make("progress");bar.value=job.progress_percent;bar.max=100;
  box.append(bar);
  if(job.error)box.append(paragraph(job.error));host.append(box);
 }
 if(stage.overlay_url){
  const img=make("img");
  img.src=stage.overlay_url;img.alt="Actual source-video vehicle tracking overlay";
  img.className="source-image";img.loading="lazy";host.append(img);
 }
 const controls=make("div",null,"soft");
 controls.append(paragraph("Analysis uses the frozen primary in a separate local worker. This interface does not execute inference."));
 if(!stage.source){
  const upload=make("input");upload.type="file";upload.accept=".mp4,.mov,.avi,.mkv,video/*";
  upload.setAttribute("aria-label","Choose real video footage");
  const button=make("button","Upload source video");
  button.onclick=()=>mutate(async()=>{
   const video=upload.files?.[0];
   if(!video)throw Error("Choose a video");
   if(video.size>2147483648)throw Error("Video exceeds 2 GiB");
   const filename=video.name.replace(/[^\x20-\x7E]/g,"_");
   await api("/api/projects/"+project+"/video",{method:"POST",
    headers:{"Content-Type":"application/octet-stream","X-StreetLab-Filename":filename},body:video});
  },"Video uploaded and verified");
  controls.append(upload,button);
 }else if(!job||["SUCCEEDED","FAILED","CANCELLED"].includes(job.status)){
  const start=make("button","Start analysis");
  start.onclick=()=>mutate(()=>api("/api/projects/"+project+"/jobs",{method:"POST",
   headers:{"Content-Type":"application/json"},body:JSON.stringify({first_frame:0,last_frame:null})}),
   "Processing job queued; local worker required");
  controls.append(start);
 }
 if(job&&["QUEUED","RUNNING"].includes(job.status)){
  const cancel=make("button","Cancel job","secondary");
  cancel.onclick=()=>mutate(()=>api("/api/jobs/"+job.id+"/cancel",{method:"POST"}),"Cancellation requested");
  controls.append(cancel);
 }
 if(job&&["FAILED","CANCELLED"].includes(job.status)){
  const retry=make("button","Retry job","secondary");
  retry.onclick=()=>mutate(()=>api("/api/jobs/"+job.id+"/retry",{method:"POST"}),"Analysis requeued");
  controls.append(retry);
 }
 host.append(controls);
 const row=make("div",null,"actions");row.append(action("Open detailed video analysis →","observation"));
 host.append(row);
}
function renderGeometry(host){
 const x=snapshot.stages.geometry;
 host.append(facts([["Measured-scale check",x.surveyed_local_meters?"QA checked":"Missing"],
 ["Manual anchors",x.anchors],["Independent checks",x.checks],
 ["Declared movements",x.movement_count??"—"]]));
 host.append(paragraph(x.note));
 if(x.site){
  const box=soft("Manual road zones in original source pixels; no GPS or route demand inferred.");
  const list=make("ul",null,"sub");
  for(const z of x.site.zones)list.append(make("li",z.id+" · "+z.role.toLowerCase()+
     " · "+(z.lane_count??"unverified")+" lanes"));
  box.append(list);host.append(box);
 }
 if(x.missing_evidence?.length)host.append(soft("Still needed: "+x.missing_evidence.join("; ")));
 host.append(action("Open guided calibration →","geometry"));
}
function renderBaseline(host){
 const x=snapshot.stages.baseline;
 host.append(facts([["Baseline runtime",nice[x.status]||x.status],
 ["Physical vehicles in reviewed census",number(x.manual_distinct_vehicle_count)],
 ["Movement QA comparisons",x.movement_fidelity.length],
 ["Scenario gate",x.scenario_eligible?"Provisional pass":"Blocked"]]));
 host.append(paragraph(x.note));
 if(x.road_graph){
  host.append(diagram(x.road_graph));
  host.append(paragraph("Plan from operator-measured local road arms, not a verified geographic map."));
 }
 if(x.movement_fidelity.length){
  addTable(host,["Movement","Field mean","SUMO mean","Relative error"],x.movement_fidelity.map(r=>
   [r.from_zone+" → "+r.to_zone,number(r.holdout_mean_s," s"),
    number(r.simulated_mean_s," s"),
    r.relative_error===null?"—":number(r.relative_error*100,"%")]));
 }
 if(x.missing_evidence?.length)host.append(soft("Review: "+x.missing_evidence.join("; ")));
 host.append(action("Open field-evidence baseline →","baseline"));
}
function renderScenarios(host){
 const x=snapshot.stages.scenarios,cmp=x.comparison;
 host.append(facts([["Paired model conditions",cmp?number(cmp.paired_runs):"—"],
 ["Comparable conditions",cmp?number(cmp.comparable_runs):"—"],
 ["Mean simulated difference",cmp?number(cmp.mean_paired_simulation_difference_s," s"):"—"],
 ["Physical-world outcome verified","No"]]));
 host.append(paragraph(x.note));
 if(x.intervention)host.append(soft("Reviewed hypothetical intervention: "+x.intervention.kind.replaceAll("_"," ").toLowerCase()));
 if(cmp?.records?.length){
  addTable(host,["Demand","Seed","Delta (simulation)","Comparable"],cmp.records.map(row=>[
   number(row.demand_multiplier*100,"%"),row.seed,number(row.paired_difference_s," s"),
   row.comparison_eligible?"Yes":"No"]));
 }else host.append(soft("No eligible comparison currently available. Missing, incomplete or stale runs are not promoted."));
 host.append(action("Open paired simulation comparison →","scenarios"));
}
function render(){
 if(!snapshot)return;
 get("slwEmpty").classList.add("hidden");get("slwContent").classList.remove("hidden");
 for(const name of names){
  const button=get("slwTab"+name);
  button.setAttribute("aria-selected",selected===name?"true":"false");
  get("slwStatus"+name).textContent=nice[snapshot.stages[name].status]||
    snapshot.stages[name].status.replaceAll("_"," ");
 }
 get("slwHeading").textContent={
  observation:"Source observations",geometry:"Junction reconstruction",
  baseline:"Observed-site SUMO",scenarios:"Paired scenario evidence"}[selected];
 get("slwDescription").textContent={
  observation:"Source evidence is measured in pixels; tracker identity is not a physical count.",
  geometry:"Source-frame correspondences and human-reviewed road geometry.",
  baseline:"Surveyed road arms, manual physical census and independent baseline holdout.",
  scenarios:"Tested model hypotheses under paired seed and demand stress assumptions."}[selected];
 const dest=get("slwDetails");dest.replaceChildren();
 ({observation:renderObservation,geometry:renderGeometry,
   baseline:renderBaseline,scenarios:renderScenarios})[selected](dest);
 const next=snapshot.next_action;get("slwNext").textContent=next.reason;
 get("slwNextLink").href=next.url;
 get("slwNextLink").textContent=next.stage==="review"?
    "Review scenario evidence →":"Continue workflow →";
 get("slwIntegrity").textContent="M1–M5 evidence verified against this project's original video, tracking and published revisions. Missing receipts fail closed.";
 const links=get("slwLinks");links.replaceChildren();
 for(const stage of names)links.append(make("div",null,"stack"));
 for(let i=0;i<names.length;i++){
  const a=make("a",["Video analysis","Spatial editor","SUMO baseline","Scenario comparison"][i]+" →","textlink");
  a.href=url(names[i]);links.children[i].append(a);
 }
}
async function refresh(chooseStage=false){
 if(!project)return;
 snapshot=await api("/api/projects/"+project+"/workspace");
 if(chooseStage){
  selected=snapshot.next_action.stage==="review"?"scenarios":snapshot.next_action.stage;
 }
 render();get("slwLatest").textContent="Project evidence verified";
}
async function mutate(fn,message){
 if(busy)return;busy=true;
 try{await fn();get("slwLatest").textContent=message;await refresh(false)}
 catch(err){get("slwLatest").textContent=err.message}finally{busy=false}
}
async function projects(){
 const items=(await api("/api/projects")).projects;
 const picker=get("slwProject");picker.replaceChildren(new Option("Select a project",""));
 for(const item of items)picker.add(new Option(item.name,item.id));
 const query=new URLSearchParams(location.search).get("project_id");
 project=[project,query,items[0]?.id].find(x=>x&&items.some(i=>i.id===x))||"";
 picker.value=project;
 if(project)await refresh(true);
 else{get("slwContent").classList.add("hidden");get("slwEmpty").classList.remove("hidden")}
}
get("slwProject").onchange=()=>{
 project=get("slwProject").value;
 if(!project){get("slwContent").classList.add("hidden");get("slwEmpty").classList.remove("hidden");return}
 refresh(true).catch(e=>get("slwLatest").textContent=e.message);
};
get("slwCreate").onclick=()=>mutate(async()=>{
 const item=await api("/api/projects",{method:"POST",
  headers:{"Content-Type":"application/json"},
  body:JSON.stringify({name:get("slwNewName").value})});
 project=item.id;await projects();
},"Project created");
get("slwRefresh").onclick=()=>refresh(false).catch(e=>get("slwLatest").textContent=e.message);
for(const stage of names)get("slwTab"+stage).onclick=()=>{selected=stage;render()};
projects().catch(e=>get("slwLatest").textContent="Workspace evidence unavailable: "+e.message);
setInterval(()=>{
 if(snapshot&&["QUEUED","RUNNING"].includes(snapshot.stages.observation.latest_job?.status)){
  refresh(false).catch(e=>get("slwLatest").textContent=e.message);
 }
},5000);
})();
