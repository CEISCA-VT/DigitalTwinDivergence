param(
    [switch]$Smoke,
    [switch]$SkipMunoz,
    [string]$Config = "baseline_suite_config.json"
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Location).Path
$ConfigPath = Join-Path $Repo $Config
if (-not (Test-Path $ConfigPath)) { throw "Config not found: $ConfigPath" }
$cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json

Write-Host "============================================================"
Write-Host "Repairing Fixed Physics + EKF-IW only"
Write-Host "This preserves existing LWOI/YNet trained outputs."
Write-Host "============================================================"

$repairArgs = @(
    "-m", "DigitalTwin.baselines.repair_fixed_ekf",
    "--input-root", [string]$cfg.raw_source_root,
    "--glob", [string]$cfg.raw_source_glob,
    "--output-root", [string]$cfg.generated_baseline_root
)
if ($Smoke) { $repairArgs += "--smoke" }
& python @repairArgs
if ($LASTEXITCODE -ne 0) { throw "Fixed/EKF repair failed" }

& python -m DigitalTwin.analysis.diagnose_ekf_iw `
    --input-root ([string]$cfg.raw_source_root) `
    --glob ([string]$cfg.raw_source_glob) `
    --ekf-root (Join-Path ([string]$cfg.generated_baseline_root) "EKF_IW") `
    --output-root (Join-Path ([string]$cfg.evaluation_root) "validation")
if ($LASTEXITCODE -ne 0) { throw "EKF diagnostics failed" }

if ($Smoke) {
    Write-Host "Smoke repair complete. Inspect EKF diagnostic rows for parking00/parking02."
    exit 0
}

& python -m DigitalTwin.analysis.validate_fixed_physics_official `
    --fixed-root (Join-Path ([string]$cfg.generated_baseline_root) "Fixed_Physics_Recomputed") `
    --v2-root ([string]$cfg.raw_source_root) `
    --v2-glob ([string]$cfg.raw_source_glob) `
    --output-root (Join-Path ([string]$cfg.evaluation_root) "validation")
if ($LASTEXITCODE -ne 0) { throw "Fixed-Physics alignment validation failed" }

$evalRoot = [string]$cfg.evaluation_root
$manifest = Join-Path $evalRoot "trajectory_manifest.csv"

& python -m DigitalTwin.analysis.build_fidelity_manifest --config $Config --output $manifest
if ($LASTEXITCODE -ne 0) { throw "Trajectory manifest build failed" }

& python -m DigitalTwin.analysis.tfp_multimethod --manifest $manifest --output (Join-Path $evalRoot "tfp")
if ($LASTEXITCODE -ne 0) { throw "TFP reevaluation failed" }

& python -m DigitalTwin.analysis.bergs_dt_fidelity --manifest $manifest --output (Join-Path $evalRoot "bergs")
if ($LASTEXITCODE -ne 0) { throw "Bergs reevaluation failed" }

if (-not $SkipMunoz) {
    # Do not use --resume after changing trajectory contents; recompute to avoid stale alignments.
    $m = $cfg.munoz
    & python -m DigitalTwin.analysis.munoz_trace_alignment_multimethod `
        --manifest $manifest `
        --output (Join-Path $evalRoot "munoz") `
        --alignment-hz ([string]$m.alignment_hz) `
        --max-snapshots ([string]$m.max_snapshots) `
        --position-mads-m ([string]$m.position_mads_m) `
        --heading-mads-deg ([string]$m.heading_mads_deg) `
        --gap-open ([string]$m.gap_open) `
        --gap-extend ([string]$m.gap_extend) `
        --lcaw ([string]$m.lcaw)
    if ($LASTEXITCODE -ne 0) { throw "Muñoz reevaluation failed" }

    & python -m DigitalTwin.analysis.compare_fidelity_frameworks `
        --tfp (Join-Path $evalRoot "tfp\tfp_per_sequence.csv") `
        --munoz (Join-Path $evalRoot "munoz\munoz_per_sequence.csv") `
        --bergs (Join-Path $evalRoot "bergs\bergs_per_sequence.csv") `
        --output (Join-Path $evalRoot "comparison")
    if ($LASTEXITCODE -ne 0) { throw "Framework comparison failed" }
}

& python -m DigitalTwin.analysis.validate_baseline_suite `
    --manifest $manifest `
    --generated-manifest ([string]$cfg.generated_baseline_manifest) `
    --tfp-summary (Join-Path $evalRoot "tfp\tfp_dataset_summary.csv") `
    --output (Join-Path $evalRoot "validation")
if ($LASTEXITCODE -ne 0) { throw "Final validation failed" }

Write-Host "============================================================"
Write-Host "REPAIR + REEVALUATION COMPLETE"
Write-Host "Check:"
Write-Host "  $evalRoot\validation\ekf_iw_diagnostics.csv"
Write-Host "  $evalRoot\validation\official_alignment_validation_summary.csv"
Write-Host "  $evalRoot\tfp\tfp_dataset_summary.csv"
Write-Host "============================================================"
