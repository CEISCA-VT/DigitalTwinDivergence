param(
    [string]$InputRoot = "",
    [string]$Summary = "",
    [string]$Config = "service_relative_fidelity\service_relative_config.json"
)
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "============================================================"
Write-Host "Service-relative operational fidelity audit"
Write-Host "============================================================"
Write-Host "This reads frozen V2 results only; it does not train or tune."

Write-Host "[1/3] Deterministic self-test"
python -m DigitalTwin.analysis.service_relative_fidelity_selftest
if ($LASTEXITCODE -ne 0) { throw "service-relative self-test failed" }

$argsList = @("-m", "DigitalTwin.analysis.i2nav_service_relative_fidelity", "--config", $Config)
if ($InputRoot -ne "") { $argsList += @("--input-root", $InputRoot) }
if ($Summary -ne "") { $argsList += @("--summary", $Summary) }

Write-Host "[2/3] Frozen signature + full service analysis"
& python @argsList
if ($LASTEXITCODE -ne 0) { throw "service-relative fidelity analysis failed" }

Write-Host "[3/3] Key outputs"
$Out = "results\service_relative_fidelity"
Write-Host "  Verification: $Out\parking00_vs_parking02_verification.md"
Write-Host "  Macro table : $Out\loso_monitor_macro.csv"
Write-Host "  Report      : $Out\service_relative_fidelity_report.md"
Write-Host "  Manifest    : $Out\analysis_manifest.json"
Write-Host ""
Write-Host "DONE"
