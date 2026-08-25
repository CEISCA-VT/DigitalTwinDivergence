$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m DigitalTwin.analysis.service_contract_e1_e2_selftest
if (-not (Test-Path "results/service_relative_fidelity/service_pass_rates_per_sequence.csv")) { throw "Missing E1 service-relative outputs." }
if (-not (Test-Path "results/aifarms_terrasentia_full_study/sequence_quality_summary.csv")) {
  python -m DigitalTwin.analysis.aifarms_terrasentia_full_study
}
python -m DigitalTwin.analysis.service_contract_e1_e2 --config service_contract_e1_e2/experiment_config.json
Write-Host "Results: results/e1_e2_service_contract_publication"
