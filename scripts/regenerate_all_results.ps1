[CmdletBinding()]
param(
    [string]$PythonExecutable = "python",
    [int]$BootstrapIterations = 2000,
    [int]$ThresholdGridPoints = 80,
    [int]$RuntimeBenchmarkRuns = 4,
    [switch]$RunTests,
    [switch]$IncludeBufferedAttackCampaign
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$analysisRoot = Join-Path $repoRoot "DigitalTwin\datasets\analysis"
$paperResults = Join-Path $repoRoot "results"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory)]
        [string]$Label,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Label"
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Push-Location $repoRoot
try {
    if ($RunTests) {
        Invoke-PythonStep -Label "Unit tests" -Arguments @(
            "-m", "pytest", "-q"
        )
    }

    Invoke-PythonStep -Label "Stationary GPS analysis" -Arguments @(
        "-m", "DigitalTwin.analysis.analyze_stationary"
    )
    Invoke-PythonStep -Label "Bench telemetry analysis" -Arguments @(
        "-m", "DigitalTwin.analysis.analyze_bench_telemetry"
    )
    Invoke-PythonStep -Label "Hardware replay consistency review" -Arguments @(
        "-m", "DigitalTwin.analysis.review_hardware_replay"
    )
    Invoke-PythonStep -Label "Digital-twin accuracy and consistency" -Arguments @(
        "-m", "DigitalTwin.analysis.digital_twin_accuracy"
    )

    Invoke-PythonStep -Label "Primary frozen attack campaign" -Arguments @(
        "-m", "DigitalTwin.analysis.real_data_study",
        "--bootstrap-iterations", "$BootstrapIterations"
    )
    Invoke-PythonStep -Label "Revised mathematical diagnostics" -Arguments @(
        "-m", "DigitalTwin.analysis.math_revision_analysis"
    )
    Invoke-PythonStep -Label "Post-campaign statistical analysis" -Arguments @(
        "-m", "DigitalTwin.analysis.post_campaign_analysis",
        "--bootstrap-iterations", "$BootstrapIterations"
    )
    Invoke-PythonStep -Label "Targeted covariance-poisoning analysis" -Arguments @(
        "-m", "DigitalTwin.analysis.covariance_poisoning",
        "--bootstrap-iterations", "$BootstrapIterations"
    )
    Invoke-PythonStep -Label "False-alarm versus detection threshold sweep" -Arguments @(
        "-m", "DigitalTwin.analysis.threshold_sweep",
        "--expanded-attacks",
        "--grid-points", "$ThresholdGridPoints"
    )

    Invoke-PythonStep -Label "Expanded replay-only attack grid" -Arguments @(
        "-m", "DigitalTwin.analysis.real_data_study",
        "--expanded-attack-grid",
        "--bootstrap-iterations", "$BootstrapIterations"
    )
    Invoke-PythonStep -Label "Expanded-grid mathematical diagnostics" -Arguments @(
        "-m", "DigitalTwin.analysis.math_revision_analysis",
        "--out-dir", "DigitalTwin/datasets/analysis/real_data_study/expanded_grid",
        "--campaign-file", "campaign_summary_expanded_grid.csv"
    )

    Invoke-PythonStep -Label "Offline runtime benchmark" -Arguments @(
        "-m", "DigitalTwin.analysis.runtime_benchmark",
        "--max-runs", "$RuntimeBenchmarkRuns"
    )

    if ($IncludeBufferedAttackCampaign) {
        Invoke-PythonStep -Label "Optional combined GPS and buffered-transport campaign" -Arguments @(
            "-m", "DigitalTwin.analysis.real_data_study",
            "--include-buffered-attacks",
            "--bootstrap-iterations", "$BootstrapIterations",
            "--out-dir", "DigitalTwin/datasets/analysis/real_data_study_buffered_attacks"
        )
    }

    New-Item -ItemType Directory -Force -Path $paperResults | Out-Null
    Copy-Item -Path (Join-Path $analysisRoot "real_data_study\*") `
        -Destination $paperResults -Recurse -Force

    $covarianceResults = Join-Path $paperResults "covariance_poisoning"
    New-Item -ItemType Directory -Force -Path $covarianceResults | Out-Null
    Copy-Item -Path (Join-Path $analysisRoot "covariance_poisoning\*") `
        -Destination $covarianceResults -Recurse -Force

    Write-Host ""
    Write-Host "All analyses completed."
    Write-Host "Paper-facing results: $paperResults"
    Write-Host "Primary report: $(Join-Path $paperResults 'real_data_study_report.md')"
    Write-Host "Revised-math report: $(Join-Path $paperResults 'math_revision_report.md')"
}
finally {
    Pop-Location
}
