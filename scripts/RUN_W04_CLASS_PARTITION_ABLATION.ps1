# W04 development-only cached detector replay, class-specific ByteTrack ablations.
# Exactly 291 existing raw-box frames; NO source MP4 decode / OpenVINO inference.
# Original 25-policy tracking evidence, T000, FLUID and Geo-trax stay untouched.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo ".venv-sahi-audit\Scripts\python.exe"
$runner = Join-Path $repo "scripts\phase3_cached_class_partition_ablation.py"
$lab = Join-Path $repo "artifacts\phase3\sahi_detector_trials"
$batch = Join-Path $lab "W04_box_tracking_batch201_01"
$output = Join-Path $lab "W04_cached_class_partition_ablation_01"

foreach ($path in @($python, $runner, $batch, (Join-Path $batch "batch_report.json"))) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing original, required input: $path"
    }
}
if (Test-Path -LiteralPath $output) {
    throw "Immutable output already exists: $output. Choose a new unique destination."
}

Write-Host "W04 offline: four class-specific trackers per policy, 4 predeclared policies, 291 cached frames."
Write-Host "Validating ALL original 25-policy SHA evidence and original video/FLUID hashes first."
& $python $runner --batch-dir $batch --output-dir $output
if ($LASTEXITCODE -ne 0) {
    throw "Cached ByteTrack ablation failed: Python exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath (Join-Path $output "class_partition_ablation_summary.json"))) {
    throw "Runner exited but the expected complete immutable summary was not created."
}

Write-Host "Immutable ablation evidence: $output"
Write-Host "No model inference was executed. Class stability is enforced by design; judge true class recall and identity."
