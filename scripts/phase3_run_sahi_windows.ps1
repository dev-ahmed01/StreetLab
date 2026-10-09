[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$VideoPath,
    [Parameter(Mandatory=$true)][string]$FluidTracks,
    [Parameter(Mandatory=$true)][string]$WeightsPath,
    [string]$Python = ".venv-sahi-audit\Scripts\python.exe",
    [ValidateSet("W01","W04","W05")][string[]]$Windows = @("W01","W04","W05"),
    [int]$SampleStep = 10,
    [int]$SliceSize = 640,
    [double]$Confidence = 0.15,
    [string]$Device = "cpu",
    [switch]$GeoTraxNumericClasses,
    [string]$OutputRoot = "artifacts\phase3\sahi_audit"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$env:PYTHONPATH = $root

if ($SampleStep -lt 1 -or $SliceSize -lt 128) {
    throw "SampleStep must be positive and SliceSize >= 128"
}

$py = (Resolve-Path $Python -ErrorAction Stop).Path
$video = (Resolve-Path $VideoPath -ErrorAction Stop).Path
$truth = (Resolve-Path $FluidTracks -ErrorAction Stop).Path
$weights = (Resolve-Path $WeightsPath -ErrorAction Stop).Path

$intervals = @{
    W01 = @(4850,5250)
    W04 = @(10750,11350)
    W05 = @(13250,13850)
}

$reportRows = @()
foreach ($window in $Windows) {
    $span = $intervals[$window]
    $folder = Join-Path $OutputRoot ("{0}_{1}_step{2}" -f $window,$SliceSize,$SampleStep)
    if (Test-Path $folder) {
        throw "Refusing to overwrite existing result: $folder. Choose a new OutputRoot."
    }
    $argsList = @(
        "scripts/phase3_sahi_audit.py",
        "--video",$video, "--fluid-tracks",$truth,
        "--weights",$weights, "--output-dir",$folder,
        "--start-frame",[string]$span[0], "--end-frame",[string]$span[1],
        "--sample-step",[string]$SampleStep,
        "--slice-height",[string]$SliceSize, "--slice-width",[string]$SliceSize,
        "--confidence",[string]$Confidence, "--device",$Device
    )
    if ($GeoTraxNumericClasses) {
        $argsList += @("--class-map","config/phase3/geotrax_class_ids.json")
    }
    Write-Host "Running detector-only SAHI A/B experiment for $window..."
    & $py @argsList
    $code = $LASTEXITCODE
    if ($code -ne 0 -and $code -ne 2) {
        throw "SAHI audit failed on $window (exit $code): no automatic promotion"
    }
    $report = Get-Content (Join-Path $folder "report.json") -Raw | ConvertFrom-Json
    $reportRows += [pscustomobject]@{
        Window = $window
        StandardMotorcycleRecall = $report.standard.per_class.MOTORCYCLE.recall
        SlicedMotorcycleRecall = $report.sliced.per_class.MOTORCYCLE.recall
        StandardPrecision = $report.standard.precision
        SlicedPrecision = $report.sliced.precision
        StandardMedianSeconds = $report.standard.latency_median_s
        SlicedMedianSeconds = $report.sliced.latency_median_s
        EligibleForTrackingTrial = $report.gate.eligible_for_tracking_trial
        ReportPath = (Join-Path $folder "report.json")
    }
}
$reportRows | Format-Table -AutoSize
Write-Host "This is detector-only evidence, not a tracked vehicle accuracy claim."
