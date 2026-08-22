param(
    [switch]$ForceDownload,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch {}
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolRoot = Join-Path $RepoRoot "magnet_tfp"
$DataRoot = Join-Path $ToolRoot "data"
$ResultsRoot = Join-Path $ToolRoot "results"
$VenvRoot = Join-Path $ToolRoot ".venv"
$Requirements = Join-Path $ToolRoot "requirements.txt"
$Script = Join-Path $ToolRoot "run_analysis.py"
$HardeningScript = Join-Path $ToolRoot "publication_hardening.py"
$Config = Join-Path $ToolRoot "config.json"

New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null

$Physical = Join-Path $DataRoot "MAGNET_Heat_Pipe_2022-03-30.csv"
$Forecast = Join-Path $DataRoot "ML_MAGNET_2022-03-30.csv"

$PhysicalUrl = "https://raw.githubusercontent.com/IdahoLabResearch/MAGNET-Heat-Pipe-Data/main/Experiment/Single_File/MAGNET_Heat_Pipe_2022-03-30.csv"
$ForecastUrl = "https://raw.githubusercontent.com/IdahoLabResearch/MAGNET-Heat-Pipe-Data/main/Machine_Learning/Single_File/ML_MAGNET_2022-03-30.csv"

function Download-IfNeeded([string]$Url, [string]$Destination, [string]$Label) {
    if ((Test-Path $Destination) -and (-not $ForceDownload)) {
        Write-Host "[data] $Label already present: $Destination"
        return
    }
    Write-Host "[data] Downloading $Label from the official INL GitHub archive..."
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
    }
    catch {
        if (Test-Path $Destination) { Remove-Item $Destination -Force -ErrorAction SilentlyContinue }
        throw "Failed to download $Label. Download it manually into magnet_tfp/data and rerun. URL: $Url`n$($_.Exception.Message)"
    }
}

Download-IfNeeded $PhysicalUrl $Physical "physical experiment CSV"
Download-IfNeeded $ForecastUrl $Forecast "digital-twin forecast CSV"

# Find a Python launcher.
$BasePython = $null
$BaseArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $BasePython = "py"
    $BaseArgs = @("-3")
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $BasePython = "python"
}
else {
    throw "Python 3 was not found. Install Python 3.10+ and rerun .\run_magnet_tfp.ps1"
}

$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[env] Creating isolated virtual environment..."
    & $BasePython @BaseArgs -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python virtual environment." }
}

if (-not $SkipInstall) {
    Write-Host "[env] Installing/updating analysis dependencies..."
    & $VenvPython -m pip install --disable-pip-version-check -q -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

if (Test-Path $ResultsRoot) {
    Remove-Item $ResultsRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

Write-Host "[run] Running MAGNET cross-domain fidelity audit..."
& $VenvPython $Script --physical $Physical --forecast $Forecast --output $ResultsRoot --config $Config
if ($LASTEXITCODE -ne 0) { throw "MAGNET analysis failed. See the error above." }

Write-Host "[run] Running publication-hardening checks..."
& $VenvPython $HardeningScript --results $ResultsRoot
if ($LASTEXITCODE -ne 0) { throw "MAGNET publication-hardening analysis failed. See the error above." }

Write-Host ""
Write-Host "============================================================"
Write-Host "MAGNET analysis finished"
Write-Host "Results: $ResultsRoot"
Write-Host "============================================================"
Write-Host ""
Get-Content -Encoding UTF8 (Join-Path $ResultsRoot "SUMMARY.md")
Write-Host ""
Write-Host "---------------- Publication hardening ----------------"
Get-Content -Encoding UTF8 (Join-Path $ResultsRoot "PUBLICATION_HARDENING_SUMMARY.md")

$ResultsZip = Join-Path $ToolRoot "MAGNET_publication_results.zip"
if (Test-Path $ResultsZip) { Remove-Item $ResultsZip -Force }
Compress-Archive -Path (Join-Path $ResultsRoot "*") -DestinationPath $ResultsZip -Force
Write-Host ""
Write-Host "Upload this file for review: $ResultsZip"
