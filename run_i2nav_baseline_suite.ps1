param(
    [switch]$Smoke,
    [switch]$SkipTraining,
    [switch]$SkipMunoz,
    [switch]$SkipYNet,
    [string]$Config = "baseline_suite_config.json"
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Location).Path
$ConfigPath = Join-Path $Repo $Config
if (-not (Test-Path $ConfigPath)) {
    throw "Config not found: $ConfigPath"
}
$cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json

$models = [string]$cfg.generated_models
if ($SkipYNet) {
    $parts = $models.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -and $_ -ne 'ynet_reduced' }
    $models = ($parts -join ',')
}

Write-Host "============================================================"
Write-Host "i2Nav external-baseline + fidelity suite"
Write-Host "Raw source : $($cfg.raw_source_root)"
Write-Host "Models     : $models"
Write-Host "Seeds      : $($cfg.seeds)"
Write-Host "============================================================"

if (-not $SkipTraining) {
    $trainArgs = @(
        "-m", "DigitalTwin.baselines.run_i2nav_baselines",
        "--input-root", [string]$cfg.raw_source_root,
        "--glob", [string]$cfg.raw_source_glob,
        "--output-root", [string]$cfg.generated_baseline_root,
        "--models", $models,
        "--seeds", [string]$cfg.seeds
    )
    if ($Smoke) { $trainArgs += "--smoke" }
    & python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Baseline generation failed with exit code $LASTEXITCODE" }
}

# Always run the cheap synthetic alignment checks.
& python -m DigitalTwin.analysis.validate_munoz_alignment
if ($LASTEXITCODE -ne 0) { throw "Muñoz synthetic validation failed" }

if ($Smoke) {
    Write-Host "Smoke test passed. Full fidelity evaluation was intentionally not started."
    Write-Host "Run: .\run_i2nav_baseline_suite.ps1"
    exit 0
}

$evalRoot = [string]$cfg.evaluation_root
$manifest = Join-Path $evalRoot "trajectory_manifest.csv"

& python -m DigitalTwin.analysis.build_fidelity_manifest --config $Config --output $manifest
if ($LASTEXITCODE -ne 0) { throw "Trajectory manifest build failed" }

& python -m DigitalTwin.analysis.tfp_multimethod `
    --manifest $manifest `
    --output (Join-Path $evalRoot "tfp")
if ($LASTEXITCODE -ne 0) { throw "TFP evaluation failed" }

& python -m DigitalTwin.analysis.bergs_dt_fidelity `
    --manifest $manifest `
    --output (Join-Path $evalRoot "bergs")
if ($LASTEXITCODE -ne 0) { throw "Bergs-style evaluation failed" }

if (-not $SkipMunoz) {
    $m = $cfg.munoz
    $munozArgs = @(
        "-m", "DigitalTwin.analysis.munoz_trace_alignment_multimethod",
        "--manifest", $manifest,
        "--output", (Join-Path $evalRoot "munoz"),
        "--alignment-hz", [string]$m.alignment_hz,
        "--max-snapshots", [string]$m.max_snapshots,
        "--position-mads-m", [string]$m.position_mads_m,
        "--heading-mads-deg", [string]$m.heading_mads_deg,
        "--gap-open", [string]$m.gap_open,
        "--gap-extend", [string]$m.gap_extend,
        "--lcaw", [string]$m.lcaw,
        "--resume"
    )
    & python @munozArgs
    if ($LASTEXITCODE -ne 0) { throw "Muñoz-style evaluation failed" }

    & python -m DigitalTwin.analysis.compare_fidelity_frameworks `
        --tfp (Join-Path $evalRoot "tfp\tfp_per_sequence.csv") `
        --munoz (Join-Path $evalRoot "munoz\munoz_per_sequence.csv") `
        --bergs (Join-Path $evalRoot "bergs\bergs_per_sequence.csv") `
        --output (Join-Path $evalRoot "comparison")
    if ($LASTEXITCODE -ne 0) { throw "Framework comparison failed" }
} else {
    Write-Host "Muñoz and framework-comparison stages skipped by request."
}

& python -m DigitalTwin.analysis.validate_baseline_suite `
    --manifest $manifest `
    --generated-manifest ([string]$cfg.generated_baseline_manifest) `
    --tfp-summary (Join-Path $evalRoot "tfp\tfp_dataset_summary.csv") `
    --output (Join-Path $evalRoot "validation")
if ($LASTEXITCODE -ne 0) { throw "Final baseline-suite validation failed" }

Write-Host "============================================================"
Write-Host "DONE"
Write-Host "Main outputs: $evalRoot"
Write-Host "============================================================"
