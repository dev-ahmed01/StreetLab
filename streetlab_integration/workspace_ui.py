"""M6 compact home workspace fragment, with existing detailed routes preserved."""
from __future__ import annotations

def workspace_ui() -> str:
    return """
<section class="panel" id="slw" aria-label="StreetLab site project workspace">
<link rel="stylesheet" href="/assets/streetlab-workspace.css">
<div class="heading">
<div><span class="eyebrow">StreetLab / Project workspace</span>
<h1>One junction. One evidence trail.</h1>
<p>Start with real footage, then build an evidence-supported site comparison. Next actions appear only when inputs are ready.</p></div>
<label>Active project<select id="slwProject"><option value="">Choose a project</option></select></label>
</div>
<div class="eyelinks"><a class="cta" id="slwReport" href="/reports">Evidence report ↗</a><button type="button" class="secondary" id="slwAddProject">New project</button><button type="button" class="secondary" id="slwRefresh">Refresh evidence</button>
<span id="slwLatest" role="status" aria-live="polite"></span></div>
<div class="surface" id="slwEmpty" style="margin-top:18px">
<h2>Create a site project</h2><p>A project preserves its own video, calibration, baseline and experiments.</p>
<div class="workline"><label>Project name<input id="slwNewName" maxlength="120" placeholder="e.g. Junction morning survey"></label>
<button type="button" id="slwCreate">Create project</button></div></div>
<div id="slwContent" class="hidden">
<div role="tablist" aria-label="StreetLab analysis milestones" class="tabrow">
<button class="stage" id="slwTabobservation" type="button" role="tab" aria-selected="true" aria-controls="slwPanel"><span class="index">01 / Video</span><strong>Observe</strong><span class="state" id="slwStatusobservation">Needs video</span></button>
<button class="stage" id="slwTabgeometry" type="button" role="tab" aria-selected="false" aria-controls="slwPanel"><span class="index">02 / Survey</span><strong>Reconstruct</strong><span class="state" id="slwStatusgeometry">Needs geometry</span></button>
<button class="stage" id="slwTabbaseline" type="button" role="tab" aria-selected="false" aria-controls="slwPanel"><span class="index">03 / Field counts</span><strong>Validate SUMO</strong><span class="state" id="slwStatusbaseline">Needs evidence</span></button>
<button class="stage" id="slwTabscenarios" type="button" role="tab" aria-selected="false" aria-controls="slwPanel"><span class="index">04 / Experiments</span><strong>Compare</strong><span class="state" id="slwStatusscenarios">No results</span></button>
</div>
<div class="columns"><section class="surface" role="tabpanel" id="slwPanel" tabindex="0">
<h2 id="slwHeading">Source observations</h2><p id="slwDescription"></p><div id="slwDetails"></div>
</section><aside aria-label="Next steps" class="sidebar">
<section class="surface"><h2>Needs your attention</h2><p id="slwNext">Project evidence is loading.</p>
<a id="slwNextLink" class="cta" href="#">Continue workflow →</a></section>
<section class="surface"><div class="sideheading">Evidence integrity</div>
<div class="status" id="slwIntegrity">Project source checks are required before simulation.</div>
<p class="sub">Tracker IDs are not physical vehicle counts. Simulation is not a proven real-world impact.</p></section>
<section class="surface"><div class="sideheading">Detailed tools</div><p class="sub">Open only the step you need.</p>
<div class="stack" id="slwLinks"></div></section>
</aside></div></div>
<p class="foot">The synthetic Phase 2 Decision Lab remains separate from this observed-site workflow.</p>
<script defer src="/assets/streetlab-workspace.js"></script>
</section>"""