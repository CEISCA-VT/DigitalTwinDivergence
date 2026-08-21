param(
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Classical baseline diagnostic v4"
Write-Host "============================================================"

$smokeArg = ""
if ($Smoke) { $smokeArg = "--smoke" }

Write-Host ""
Write-Host "[1/2] Diagnose yaw fusion candidates (strict LOSO)"
if ($Smoke) {
    python -m DigitalTwin.analysis.diagnose_classical_yaw_fusion --smoke
} else {
    python -m DigitalTwin.analysis.diagnose_classical_yaw_fusion
}
if ($LASTEXITCODE -ne 0) { throw "Yaw-fusion diagnostic failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "[2/2] Run official i2Nav-style evo alignment"
python -m DigitalTwin.analysis.validate_i2nav_evo_alignment
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "If evo is missing, install it with:"
    Write-Host "  python -m pip install evo"
    throw "evo validation failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Diagnostic complete."
Write-Host "Send these files back:"
Write-Host "  results\i2nav_fidelity_baselines\validation\classical_yaw_fusion_summary.csv"
Write-Host "  results\i2nav_fidelity_baselines\validation\classical_yaw_fusion_selected_by_training.csv"
Write-Host "  results\i2nav_fidelity_baselines\validation\i2nav_evo_ape_per_sequence.csv"
Write-Host "  results\i2nav_fidelity_baselines\validation\i2nav_evo_ape_summary.csv"
Write-Host "============================================================"
