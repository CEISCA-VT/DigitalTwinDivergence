param(
    [Parameter(Mandatory=$true)][string]$VideoDirectory,
    [string]$OutputDirectory = "results\ugv01_smooth_video_audit",
    [double]$SampleHz = 4.0,
    [int]$ExpectedRuns = 5,
    [int]$Workers = 0
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $ScriptDirectory
$Python = Join-Path $RepositoryRoot ".venv-video-audit\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    py -m venv (Join-Path $RepositoryRoot ".venv-video-audit")
    & $Python -m pip install -r (Join-Path $ScriptDirectory "requirements.txt")
}

$Arguments = @(
    (Join-Path $ScriptDirectory "audit_videos.py"),
    $VideoDirectory,
    "--output", $OutputDirectory,
    "--moving-tag-ids", "0",
    "--expected-runs", "$ExpectedRuns",
    "--sample-hz", "$SampleHz"
)
if ($Workers -gt 0) {
    $Arguments += @("--workers", "$Workers")
}

& $Python @Arguments
exit $LASTEXITCODE

