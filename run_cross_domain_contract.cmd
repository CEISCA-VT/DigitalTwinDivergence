@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_cross_domain_contract.ps1" %*
exit /b %ERRORLEVEL%
