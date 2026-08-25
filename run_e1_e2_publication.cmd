@echo off
setlocal
cd /d "%~dp0"

echo [E1/E2] Running self-test...
python -m DigitalTwin.analysis.service_contract_e1_e2_selftest
if errorlevel 1 goto :fail

if not exist "results\service_relative_fidelity\service_pass_rates_per_sequence.csv" (
  echo [ERROR] E1 frozen service-relative outputs are missing.
  echo Run run_service_relative_fidelity.ps1/cmd first.
  goto :fail
)

if not exist "results\aifarms_terrasentia_full_study\sequence_quality_summary.csv" (
  echo [E2] Frozen TerraSentia full-study outputs not found; generating them with the existing repository study...
  python -m DigitalTwin.analysis.aifarms_terrasentia_full_study
  if errorlevel 1 goto :fail
)

echo [E1/E2] Running publication-grade service-contract analysis...
python -m DigitalTwin.analysis.service_contract_e1_e2 --config service_contract_e1_e2\experiment_config.json
if errorlevel 1 goto :fail

echo.
echo [DONE] Results: results\e1_e2_service_contract_publication
exit /b 0

:fail
echo.
echo [FAILED] E1/E2 analysis did not complete.
exit /b 1
