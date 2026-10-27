# Secure one-command, localhost-only StreetLab Docker Desktop launcher.
[CmdletBinding()]
param(
    [ValidateSet("Initialize","Start","Stop","Status","Logs")]
    [string]$Mode = "Start",
    [ValidateRange(1024,65535)]
    [int]$Port = 8000,
    [switch]$WithWorker
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo
$EnvFile = Join-Path $Repo ".env"
if ($Mode -in @("Initialize","Start")) {
    if (-not (Test-Path -LiteralPath $EnvFile)) {
        $Bytes = [byte[]]::new(32)
        [System.Security.Cryptography.RandomNumberGenerator]::Fill($Bytes)
        $Secret = [Convert]::ToBase64String($Bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
        @("STREETLAB_ACCESS_PASSWORD=$Secret","STREETLAB_PORT=$Port") |
            Set-Content -LiteralPath $EnvFile -Encoding utf8
        Write-Host "Created private access credentials in .env."
    }
    $Config = Get-Content -LiteralPath $EnvFile -Raw
    $Match = [regex]::Match($Config,'(?m)^STREETLAB_ACCESS_PASSWORD=([^\r\n]*)')
    if (-not $Match.Success -or $Match.Groups[1].Value.Length -lt 20 -or
        $Match.Groups[1].Value.StartsWith("REPLACE_WITH_")) {
        throw "Local .env must contain a unique 20+ character STREETLAB_ACCESS_PASSWORD."
    }
    New-Item -ItemType Directory -Force (Join-Path $Repo "artifacts") | Out-Null
    if ($Mode -eq "Initialize") {
        Write-Host "Initialized. Login: streetlab; password is in the local .env file."
        return
    }
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is unavailable. Install and start Docker Desktop."
}
$ArgsList = @("compose","-f","compose.yaml")
if ($WithWorker) {
    $Model = Join-Path $Repo "private/w04-model"
    $Provenance = Join-Path $Repo "private/w04-provenance.json"
    if ($Mode -eq "Start" -and
        (-not (Test-Path (Join-Path $Model "checkpoint.xml") -PathType Leaf) -or
         -not (Test-Path (Join-Path $Model "checkpoint.bin") -PathType Leaf) -or
         -not (Test-Path $Provenance -PathType Leaf))) {
        throw "Original W04 model/provenance absent; refusing placeholder inference."
    }
    $ArgsList += @("-f","compose.worker.yaml","--profile","inference")
}
switch ($Mode) {
    "Start"  { & docker @ArgsList up --build -d }
    "Stop"   { & docker @ArgsList down }
    "Status" { & docker @ArgsList ps }
    "Logs"   { & docker @ArgsList logs --tail 100 }
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Mode -eq "Start") {
    Write-Host "StreetLab URL: http://127.0.0.1:$Port"
    Write-Host "HTTP Basic user: streetlab; password saved locally in .env."
}
