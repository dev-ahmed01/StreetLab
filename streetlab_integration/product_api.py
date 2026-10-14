"""M2 HTTP endpoints and progressive product UI, mounted beside existing M1 demo.

Video uploads are streamed through a bounded temporary spool; never accept
arbitrary client-provided disk paths or run inference inside HTTP handlers.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from streetlab_integration.video_jobs import JobError, MAX_VIDEO_BYTES, VideoStore
from streetlab_integration.primary_runner import verified_run
from streetlab_integration.worker import job_report


class CreateProject(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class RunConfig(BaseModel):
    first_frame: int = Field(default=0, ge=0)
    last_frame: int | None = Field(default=None, ge=0)


def mount_product_routes(app: FastAPI, workdir: Path) -> None:
    store = VideoStore(workdir)

    def detail(exc: Exception) -> HTTPException:
        msg = str(exc)
        return HTTPException(status_code=404 if "not found" in msg.lower() else 409,
                             detail=msg)

    @app.get("/api/projects")
    def projects():
        items = store.projects()
        for item in items:
            if item["source_metadata"] is not None:
                item["source_metadata"] = json.loads(item["source_metadata"])
        return {"projects": items}

    @app.post("/api/projects", status_code=201)
    def create_project(payload: CreateProject):
        try:
            return store.create_project(payload.name)
        except JobError as exc:
            raise detail(exc) from exc

    @app.get("/api/projects/{project_id}")
    def project(project_id: str):
        try:
            result = store.project(project_id)
            with store.connection() as conn:
                row = conn.execute("SELECT * FROM sources WHERE project_id=?",
                                   (project_id,)).fetchone()
            result["source"] = store._dict(row)
            return result
        except JobError as exc:
            raise detail(exc) from exc

    @app.post("/api/projects/{project_id}/video", status_code=201)
    async def upload(project_id: str, request: Request):
        filename = request.headers.get("x-streetlab-filename", "")
        if not filename:
            raise HTTPException(status_code=400, detail="Select a named source video")
        if request.headers.get("content-length", "").isdigit():
            if int(request.headers["content-length"]) > MAX_VIDEO_BYTES:
                raise HTTPException(status_code=413, detail="2 GiB upload limit exceeded")
        try:
            # TemporaryFile spills to disk; never buffer large footage in RAM.
            with tempfile.TemporaryFile(mode="w+b") as spool:
                total = 0
                async for block in request.stream():
                    total += len(block)
                    if total > MAX_VIDEO_BYTES:
                        raise HTTPException(status_code=413, detail="2 GiB upload limit exceeded")
                    spool.write(block)
                spool.seek(0)
                return store.save_source(project_id, filename, spool)
        except JobError as exc:
            raise detail(exc) from exc

    @app.post("/api/projects/{project_id}/jobs", status_code=202)
    def start_job(project_id: str, payload: RunConfig):
        try:
            return store.queue(project_id, first_frame=payload.first_frame,
                               last_frame=payload.last_frame)
        except JobError as exc:
            raise detail(exc) from exc

    @app.get("/api/projects/{project_id}/jobs")
    def list_jobs(project_id: str):
        try:
            return {"jobs": store.jobs(project_id)}
        except JobError as exc:
            raise detail(exc) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return store.job(job_id)
        except JobError as exc:
            raise detail(exc) from exc

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return store.cancel(job_id)
        except JobError as exc:
            raise detail(exc) from exc

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry(job_id: str):
        try:
            return store.retry(job_id)
        except JobError as exc:
            raise detail(exc) from exc

    @app.get("/api/jobs/{job_id}/report")
    def observation_report(job_id: str):
        try:
            return job_report(store, job_id)
        except JobError as exc:
            raise detail(exc) from exc
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409,
                                detail="Completed observation receipt is missing or invalid") from exc

    @app.get("/api/jobs/{job_id}/overlays")
    def overlay_list(job_id: str):
        try:
            job = store.job(job_id)
            if job["status"] != "SUCCEEDED":
                return {"overlay_files": []}
            manifest = verified_run(store.root / "runs" / job_id,
                                    store.source(job["source_id"])["sha256"])
            return {"overlay_files": manifest["overlay_files"]}
        except JobError as exc:
            raise detail(exc) from exc
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409,
                                detail="Run overlay integrity check failed") from exc

    @app.get("/api/jobs/{job_id}/overlays/{name}")
    def overlay_image(job_id: str, name: str):
        try:
            job = store.job(job_id)
            if job["status"] != "SUCCEEDED":
                raise JobError("Only completed jobs have verified overlays")
            folder = store.root / "runs" / job_id
            manifest = verified_run(folder, store.source(job["source_id"])["sha256"])
            if name not in manifest["overlay_files"]:
                raise JobError("Overlay not found")
            return FileResponse(folder / name, media_type="image/jpeg",
                                headers={"Cache-Control": "no-store",
                                         "X-Content-Type-Options": "nosniff"})
        except JobError as exc:
            raise detail(exc) from exc
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409,
                                detail="Run overlay integrity check failed") from exc


def m2_ui() -> str:
    # Included on M1's existing / homepage: no extra frontend framework/runtime.
    return r"""
<section class="panel" aria-label="Traffic video projects" id="m2Panel">
<style>
#m2Panel h2{margin:0 0 6px;font-size:24px}
#m2Panel .m2row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}
#m2Panel .m2actions{display:flex;flex-wrap:wrap;gap:9px;margin-top:12px;align-items:center}
#m2Panel .m2actions button{font-size:14px}
#m2Panel .m2quiet{font-size:14px;color:#4b5563;line-height:1.55}
#m2Panel .m2message{padding:12px;background:#f7f8fa;border:1px solid #d6dbe1;border-radius:8px}
#m2Panel .m2progress{height:11px;width:100%;accent-color:#1f8069}
#m2Panel .m2previews{display:flex;gap:12px;overflow:auto;max-width:100%}
#m2Panel .m2previews img{max-width:260px;max-height:190px;object-fit:contain;background:#111;border-radius:6px}
#m2Panel .m2secondary{background:white;color:#262c35;border:1px solid #9ca3af}
@media(max-width:700px){#m2Panel .m2row{grid-template-columns:1fr}}
</style>
<h2>Analyze a traffic video</h2>
<p class="m2quiet">Create a project, upload a real source, then queue primary OpenVINO/ByteTrack analysis. The worker runs independently; uploads and progress are persistent locally. This is observation, not yet a calibrated junction reconstruction.</p>
<div class="m2row">
<label>New project<input id="m2ProjectName" maxlength="120" placeholder="e.g. BTM junction morning survey"></label>
<label>Active project<select id="m2ProjectSelect" aria-label="Active traffic video project"><option value="">Choose a project</option></select></label>
</div>
<div class="m2actions"><button id="m2Create">Create project</button><span class="m2quiet" id="m2SourceInfo">No source uploaded</span></div>
<div class="m2row">
<label>Source video (MP4, MOV, AVI, MKV; max 2 GiB)<input type="file" id="m2File" accept=".mp4,.mov,.avi,.mkv,video/*"></label>
<label>Last source frame (optional, blank means full footage)<input id="m2LastFrame" type="number" min="0" placeholder="All available frames"></label>
</div>
<div class="m2actions">
<button id="m2Upload">Upload source</button><button id="m2Start">Start analysis</button>
<button class="m2secondary" id="m2Cancel">Cancel</button>
<button class="m2secondary" id="m2Retry">Retry failed analysis</button>
</div>
<p class="m2quiet">The standalone CPU worker must be running with the frozen model and provenance path. A queued job remains saved until a worker claims it.</p>
<div class="m2message" role="status" aria-live="polite" id="m2Status">Select a project to begin.</div>
<progress id="m2Progress" class="m2progress" value="0" max="100"></progress>
<div id="m2Report" class="m2quiet"></div>
<div id="m2Images" class="m2previews" aria-label="Real video tracking overlays"></div>
</section>
<script>
window.addEventListener('DOMContentLoaded',()=>{
 const $=id=>document.getElementById(id);
 const state={project:null,job:null};
 async function api(path,opt={}){
   const r=await fetch(path,opt);
   let v;try{v=await r.json()}catch{v={detail:'Server returned an invalid response'}}
   if(!r.ok)throw new Error(v.detail||'Request failed');
   return v;
 }
 function msg(text){$('m2Status').textContent=text}
 async function projects(){
   const items=(await api('/api/projects')).projects;
   const s=$('m2ProjectSelect');const previous=state.project;
   s.replaceChildren(new Option('Choose a project',''));
   for(const p of items)s.add(new Option(p.name,p.id));
   if(previous&&items.some(p=>p.id===previous))s.value=previous;
   if(!state.project&&items.length){state.project=items[0].id;s.value=state.project}
   await refreshProject();
 }
 async function refreshProject(){
   if(!state.project)return;
   const project=await api('/api/projects/'+state.project);
   const source=project.source;
   $('m2SourceInfo').textContent=source
     ?source.original_filename+' · '+source.metadata.width+'×'+source.metadata.height+
      ' · '+source.metadata.duration_seconds+' s · SHA '+source.sha256.slice(0,12)+'…'
     :'No video uploaded for this project';
   const jobs=(await api('/api/projects/'+state.project+'/jobs')).jobs;
   state.job=jobs.length?jobs[0].id:null;
   await refreshJob();
 }
 async function refreshJob(){
   if(!state.job){msg('Upload a source video to start.');$('m2Progress').value=0;return}
   const job=await api('/api/jobs/'+state.job);
   const pct=job.total_frames?Math.round(100*job.completed_frames/job.total_frames):0;
   $('m2Progress').value=pct;
   msg('Analysis '+job.status+' · '+pct+'% ('+job.completed_frames+' / '+job.total_frames+
      ' source frames)'+(job.error?' · '+job.error:''));
   if(job.status==='SUCCEEDED'){
     const r=await api('/api/jobs/'+state.job+'/report');
     $('m2Report').textContent='Observed tracking rows: '+r.observed_points+
       ' · Tracker IDs: '+r.tracking_identity_count+
       ' · Frames with tracks: '+r.observed_frames+
       ' · Real-site study: '+r.study_gate.status+
       ' · Missing: '+r.study_gate.missing_evidence.join(', ')+
       '. IDs are not a physical vehicle census.';
     const view=await api('/api/jobs/'+state.job+'/overlays');
     const box=$('m2Images');box.replaceChildren();
     for(const name of view.overlay_files){
       const im=document.createElement('img');
       im.src='/api/jobs/'+state.job+'/overlays/'+encodeURIComponent(name);
       im.alt='Tracked vehicles on actual source frame '+name;box.appendChild(im)
     }
   }else{$('m2Report').textContent='';$('m2Images').replaceChildren()}
 }
 $('m2ProjectSelect').onchange=async()=>{state.project=$('m2ProjectSelect').value;state.job=null;
   try{await refreshProject()}catch(e){msg(e.message)}};
 $('m2Create').onclick=async()=>{try{
   const p=await api('/api/projects',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:$('m2ProjectName').value})});
   state.project=p.id;await projects();msg('Project created. Add its source video.');
 }catch(e){msg(e.message)}};
 $('m2Upload').onclick=async()=>{try{
   if(!state.project)throw Error('Create or select a project first');
   const f=$('m2File').files[0];if(!f)throw Error('Choose a local video');
   if(f.size>2147483648)throw Error('Video exceeds 2 GiB');
   msg('Uploading and validating video; large files may take time…');
   const filename=f.name.replace(/[^\x20-\x7E]/g,'_');
   await api('/api/projects/'+state.project+'/video',{method:'POST',
     headers:{'Content-Type':'application/octet-stream','X-StreetLab-Filename':filename},body:f});
   await refreshProject();msg('Video validated and saved; ready to queue analysis.');
 }catch(e){msg(e.message)}};
 $('m2Start').onclick=async()=>{try{
   if(!state.project)throw Error('Select a project first');
   const last=$('m2LastFrame').value;
   const j=await api('/api/projects/'+state.project+'/jobs',
      {method:'POST',headers:{'Content-Type':'application/json'},
       body:JSON.stringify({first_frame:0,last_frame:last===''?null:Number(last)})});
   state.job=j.id;await refreshJob();
 }catch(e){msg(e.message)}};
 $('m2Cancel').onclick=async()=>{try{
   if(!state.job)throw Error('No job selected');
   await api('/api/jobs/'+state.job+'/cancel',{method:'POST'});await refreshJob()
 }catch(e){msg(e.message)}};
 $('m2Retry').onclick=async()=>{try{
   if(!state.job)throw Error('No job selected');
   await api('/api/jobs/'+state.job+'/retry',{method:'POST'});await refreshJob()
 }catch(e){msg(e.message)}};
 projects().catch(e=>msg('M2 project storage unavailable: '+e.message));
 setInterval(()=>{if(state.job)refreshJob().catch(e=>msg(e.message))},3000);
});
</script>
"""
