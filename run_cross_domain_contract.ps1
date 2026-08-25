param(
    [switch]$SkipInstall,
    [switch]$NoDownload
)
$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch {}
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolRoot = Join-Path $RepoRoot "cross_domain_generalization"
$VenvRoot = Join-Path $ToolRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$Req = Join-Path $RepoRoot "requirements-cross-domain-contract.txt"
$Script = Join-Path $RepoRoot "DigitalTwin\analysis\cross_domain_contract_generalization.py"
$SelfTest = Join-Path $RepoRoot "DigitalTwin\analysis\cross_domain_contract_selftest.py"
$Out = Join-Path $RepoRoot "results\cross_domain_contract_generalization"

$BasePython = $null
$BaseArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) { $BasePython = "py"; $BaseArgs = @("-3") }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $BasePython = "python" }
else { throw "Python 3 was not found." }

if (-not (Test-Path $VenvPython)) {
    Write-Host "[env] Creating isolated environment..."
    & $BasePython @BaseArgs -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv." }
}
if (-not $SkipInstall) {
    Write-Host "[env] Installing/updating dependencies..."
    & $VenvPython -m pip install --disable-pip-version-check -q -r $Req
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
Write-Host "[test] Running contract self-test..."
& $VenvPython $SelfTest
if ($LASTEXITCODE -ne 0) { throw "Self-test failed." }

$argsList = @("--repo-root", $RepoRoot, "--output", $Out)
if ($NoDownload) { $argsList += "--no-download" }
Write-Host "[run] MAGNET + FreeTwinEV + TU Wien SNG cross-domain contract audit..."
& $VenvPython $Script @argsList
$RunCode = $LASTEXITCODE
$Report = Join-Path $Out "CROSS_DOMAIN_GENERALIZATION_REPORT.md"
Write-Host ""
Write-Host "============================================================"
if ($RunCode -eq 0) { Write-Host "Cross-domain audit finished: PASS" }
else { Write-Host "Cross-domain audit finished: INCOMPLETE / FAILED" }
Write-Host "Results: $Out"
Write-Host "============================================================"
if (Test-Path $Report) { Get-Content -Encoding UTF8 $Report }
if ($RunCode -ne 0) { throw "Cross-domain audit did not pass all three required dataset gates. See dataset_errors.csv and diagnostics in $Out" }
