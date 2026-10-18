# Create category-blinded reviewer-only May25 frames, or audit two completed independent forms.
[CmdletBinding()]
param(
    [ValidateSet('Prepare','Summarize')][string]$Mode = 'Prepare',
    [string]$RunDir = '',
    [string]$ForensicsDir = '',
    [string]$OriginalPacketDir = '',
    [string]$BlindPacketDir = '',
    [string]$Reviewer1 = '',
    [string]$Reviewer2 = '',
    [string]$Output = '',
    [string]$Python = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
$artifacts = Join-Path $Repo 'artifacts'
function Find-VerifiedDir([string]$filter,[string]$file,[string]$missing) {
    $latest = Get-ChildItem -Path $artifacts -Directory -Filter $filter -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName $file) -PathType Leaf } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw $missing }
    return $latest.FullName
}
if ($Mode -eq 'Prepare') {
    if ([string]::IsNullOrWhiteSpace($RunDir)) {
        $RunDir = Find-VerifiedDir 'w04_may25_crossdate_*' 'cross_date_summary.json' 'Original completed May25 regression batch not found.'
    }
    if ([string]::IsNullOrWhiteSpace($ForensicsDir)) {
        $ForensicsDir = Find-VerifiedDir 'w04_may25_forensics_*' 'regression_forensics.json' 'May25 source-backed forensics not found.'
    }
    if ([string]::IsNullOrWhiteSpace($OriginalPacketDir)) {
        $OriginalPacketDir = Find-VerifiedDir 'w04_may25_review_packet_*' 'review_packet_manifest.json' 'Create original May25 image review packet first.'
    }
    if ([string]::IsNullOrWhiteSpace($Output)) {
        $Output = Join-Path $artifacts ('w04_may25_blind_review_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    }
    if (Test-Path -LiteralPath $Output) { throw "Refusing evidence overwrite: $Output" }
    Write-Host 'Preparing blinded original-frame crop packets for two NEW independent reviewers.'
    & $Python -m streetlab_integration.w04_blind_physical_review prepare --run-dir $RunDir --forensics-dir $ForensicsDir --old-packet-dir $OriginalPacketDir --output-dir $Output
    if ($LASTEXITCODE -ne 0) { throw 'Blinded review package preparation failed; do not edit frozen evidence.' }
    Write-Host "Reviewer packages: $(Join-Path $Output 'reviewer_1'), $(Join-Path $Output 'reviewer_2')"
    Write-Host 'Do NOT give either reviewer sealed_reference.json or the earlier labeled contact sheet.'
} else {
    if ([string]::IsNullOrWhiteSpace($BlindPacketDir)) {
        $BlindPacketDir = Find-VerifiedDir 'w04_may25_blind_review_*' 'blind_review_receipt.json' 'Blind review packet not found.'
    }
    if ([string]::IsNullOrWhiteSpace($Reviewer1) -or [string]::IsNullOrWhiteSpace($Reviewer2)) {
        throw 'Summarize requires -Reviewer1 and -Reviewer2 paths to separately completed independent CSV forms.'
    }
    if ([string]::IsNullOrWhiteSpace($Output)) {
        $Output = Join-Path $artifacts ('w04_may25_two_reviewer_consensus_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.json')
    }
    if (Test-Path -LiteralPath $Output) { throw "Refusing evidence overwrite: $Output" }
    & $Python -m streetlab_integration.w04_blind_physical_review summarize --packet-dir $BlindPacketDir --reviewer-1 $Reviewer1 --reviewer-2 $Reviewer2 --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Review consensus not accepted; check independent forms and sealed source.' }
    Write-Host "Non-production selected-sample consensus: $Output"
}
