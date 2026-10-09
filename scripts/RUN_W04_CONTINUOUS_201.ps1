# Run from the root of StreetLab-engine-trial after pulling the experimental GitHub branch.
# W04 RESEARCH DEVELOPMENT ONLY. Original FLUID, Geo-trax, T000 and video files stay unchanged.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo '.venv-sahi-audit\Scripts\python.exe'
$runner = Join-Path $repo 'scripts\phase3_continuous_box_tracking_lab.py'
$lab = Join-Path $repo 'artifacts\phase3\sahi_detector_trials'
$bundle = Join-Path $lab 'W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip'
$model = Join-Path $lab 'W04_openvino_export640_01\checkpoint_openvino_model'
$video = 'C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4'
$truth = 'C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv'
$output = Join-Path $lab 'W04_box_tracking_batch201_01'

# If an unchanged, source-aligned 14-column T000 track file exists, set its absolute path here.
# Without it, the batch runs but same-window T000 and Build 4 promotion gates remain pending.
$baseline = $null
foreach ($item in @($python, $runner, $bundle, $model, $video, $truth)) {
  if (-not (Test-Path -LiteralPath $item)) { throw "Missing required input: $item" }
}
if (Test-Path -LiteralPath $output) {
  throw "Immutable output already exists: $output. Use a new unique output folder; never overwrite evidence."
}
if ($baseline -and -not (Test-Path -LiteralPath $baseline)) {
  throw "T000 baseline path does not exist: $baseline"
}

# Frame 10660..10749: 90 consecutive warmup frames.
# Frame 10750..10950: 201 consecutive evaluation frames.
# One shared per-frame inference stream with 25 independent ByteTrack instances.
$argsForRunner = @(
  $runner, '--video', $video, '--bundle', $bundle,
  '--runtime-model', $model, '--fluid-tracks', $truth,
  '--start-frame', '10750', '--end-frame', '10950',
  '--warmup-frames', '90', '--fps', '30',
  '--output-dir', $output
)
if ($baseline) { $argsForRunner += @('--baseline-tracks', $baseline) }

Write-Host 'W04 Build 3: continuous original 4K video; 90 warmup + 201 evaluation frames; all 25 box policies'
& $python @argsForRunner
if ($LASTEXITCODE -ne 0) { throw "W04 continuous ByteTrack experiment failed: Python exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath (Join-Path $output 'batch_report.json'))) {
  throw 'Runner exited successfully but expected batch_report.json is missing'
}
Write-Host "Complete immutable continuous results: $output"
