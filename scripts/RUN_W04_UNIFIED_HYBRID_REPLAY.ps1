# W04 ONE-SHOT hybrid research benchmark: 4 trackers on cached 291 frames.
# 0 OpenVINO inference, 0 MP4 decode. No original evidence overwritten.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo '.venv-sahi-audit\Scripts\python.exe'

# Ensure child Python finds StreetLab package regardless of the caller's session.
$priorPythonPath = $env:PYTHONPATH
if ($priorPythonPath) {
    $env:PYTHONPATH = $repo + [System.IO.Path]::PathSeparator + $priorPythonPath
} else {
    $env:PYTHONPATH = $repo
}
$runner = Join-Path $repo 'scripts\phase3_w04_unified_hybrid_replay.py'
$lab = Join-Path $repo 'artifacts\phase3\sahi_detector_trials'
$original = Join-Path $lab 'W04_box_tracking_batch201_01'
$output = Join-Path $lab 'W04_unified_hybrid_replay_01'
$zip = Join-Path $repo 'W04_UNIFIED_HYBRID_REPLAY_01.zip'

foreach ($path in @($python, $runner,
                   (Join-Path $original 'batch_report.json'),
                   (Join-Path $original 'source_provenance.json'),
                   (Join-Path $original 'original_pre_global_merge_boxes.jsonl'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing original W04 evidence: $path"
    }
}
if ((Test-Path -LiteralPath $output) -or (Test-Path -LiteralPath $zip)) {
    throw 'Existing experiment output / ZIP cannot be overwritten'
}
Write-Host 'W04 one-shot: verify all 25 original trackers and SHA sources, then run four candidates.'
Write-Host 'Original IoS0.30 must exactly reproduce original pixel, class and identity scores.'
Write-Host 'No OpenVINO inference or 4K video decoding will run.'
& $python $runner --batch-dir $original --output-dir $output
if ($LASTEXITCODE -ne 0) {
    throw "Unified hybrid replay failed with Python exit code $LASTEXITCODE"
}
$summary = Join-Path $output 'unified_hybrid_report.json'
if (-not (Test-Path -LiteralPath $summary -PathType Leaf)) {
    throw "Missing required immutable result: $summary"
}
$report = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
if (-not $report.original_control_reproduction.successful -or
    $report.experimental_results.Count -ne 4 -or
    $report.eligible_for_production) {
    throw 'Missing control parity, four candidate runs, or development-only safety flag'
}
Write-Host ''
Write-Host "Original tracking control reproduced: $($report.original_control_reproduction.successful)"
foreach ($item in $report.experimental_results) {
    $m = $item.class_diagnostics.by_class.MOTORCYCLE.correct_class_recall
    $h = $item.class_diagnostics.by_class.HEAVY_VEHICLE.correct_class_recall
    $line = ('{0}: P={1:P2} R={2:P2} Motorcycle={3:P2} Heavy={4:P2} Switches={5} Gate={6}' -f
        $item.name, $item.pixel.point_precision, $item.pixel.point_recall, $m, $h,
        $item.identity.total_contiguous_id_switches,
        $item.development_safety_gate.development_gate_passed)
    Write-Host $line
}
Compress-Archive -Path (Join-Path $output '*') -DestinationPath $zip -CompressionLevel Optimal
if (-not (Test-Path -LiteralPath $zip -PathType Leaf)) {
    throw "Completed result ZIP was not created: $zip"
}
Write-Host "Full W04 hybrid comparison ZIP: $zip"
Write-Host 'No additional per-policy diagnostic runs are needed.'
