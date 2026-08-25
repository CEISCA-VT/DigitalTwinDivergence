param(
  [string]$Config = "service_risk_estimator\service_risk_config.json",
  [string]$WindowTable = "",
  [string]$RawRoot = ""
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
Write-Host "============================================================"
Write-Host "Nested service-risk estimator for frozen i2Nav Twin V2"
Write-Host "============================================================"
python -c "import numpy,pandas,sklearn"
if ($LASTEXITCODE -ne 0) {
  Write-Host "[deps] Installing required Python packages..."
  python -m pip install -r requirements-service-risk-estimator.txt
  if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }
}
Write-Host "[1/2] Deterministic self-test"
python -m DigitalTwin.analysis.service_risk_estimator_selftest
if ($LASTEXITCODE -ne 0) { throw "self-test failed" }
$argsList=@("-m","DigitalTwin.analysis.i2nav_nested_service_risk","--config",$Config)
if ($WindowTable -ne "") { $argsList += @("--window-table",$WindowTable) }
if ($RawRoot -ne "") { $argsList += @("--raw-root",$RawRoot) }
Write-Host "[2/2] Nested sequence-level evaluation"
& python @argsList
if ($LASTEXITCODE -ne 0) { throw "nested service-risk evaluation failed" }
Write-Host "DONE: results\service_risk_estimator_nested_loso\service_risk_report.md"
