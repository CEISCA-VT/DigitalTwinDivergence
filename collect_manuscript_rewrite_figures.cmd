@echo off
setlocal
cd /d "%~dp0"

echo [preflight] Validating PowerShell syntax...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile('%~dp0collect_manuscript_rewrite_figures.ps1',[ref]$tokens,[ref]$errors) ^| Out-Null; if ($errors.Count -gt 0) { $errors ^| ForEach-Object { Write-Host ('SYNTAX ERROR: ' + $_.Message) }; exit 97 }"
if errorlevel 1 (
    echo [error] PowerShell syntax preflight failed.
    endlocal & exit /b 97
)

echo [run] Collecting manuscript rewrite figures...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collect_manuscript_rewrite_figures.ps1" %*
set RC=%ERRORLEVEL%
endlocal & exit /b %RC%
