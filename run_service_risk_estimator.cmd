@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Nested service-risk estimator for frozen i2Nav Twin V2
echo ============================================================
python -c "import numpy,pandas,sklearn" >nul 2>&1
if errorlevel 1 (
  echo [deps] Installing required Python packages...
  python -m pip install -r requirements-service-risk-estimator.txt
  if errorlevel 1 exit /b 1
)
echo [1/2] Deterministic self-test
python -m DigitalTwin.analysis.service_risk_estimator_selftest
if errorlevel 1 exit /b 1
echo [2/2] Nested sequence-level evaluation
python -m DigitalTwin.analysis.i2nav_nested_service_risk --config service_risk_estimator\service_risk_config.json %*
if errorlevel 1 exit /b 1
echo.
echo DONE. Key report:
echo results\service_risk_estimator_nested_loso\service_risk_report.md
endlocal
