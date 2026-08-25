param(
    [string]$RepoRoot = $PSScriptRoot,
    [switch]$Strict,
    [switch]$RefreshExisting
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Get-Location).Path
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$Dest = Join-Path $RepoRoot 'figures'
$Supp = Join-Path $Dest 'supplement'
$MagnetSupp = Join-Path $Supp 'magnet_source'

New-Item -ItemType Directory -Force -Path @($Dest, $Supp, $MagnetSupp) | Out-Null

Write-Host '============================================================'
Write-Host 'DigitalTwinDivergence manuscript-rewrite figure collector'
Write-Host ('Repository  : {0}' -f $RepoRoot)
Write-Host ('Main figures: {0}' -f $Dest)
Write-Host ('Supplement  : {0}' -f $Supp)
Write-Host '============================================================'

$manifest = New-Object 'System.Collections.Generic.List[object]'
$coreMissing = New-Object 'System.Collections.Generic.List[string]'

function Get-NormalizedPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    try { return [System.IO.Path]::GetFullPath($Path).TrimEnd('\\') } catch { return $Path }
}

function Resolve-FigureSource {
    param(
        [string[]]$Candidates,
        [string]$FileName,
        [string]$DestinationFile
    )

    $destinationNormalized = Get-NormalizedPath $DestinationFile

    foreach ($rel in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($rel)) { continue }
        $p = Join-Path $RepoRoot $rel
        if (Test-Path -LiteralPath $p -PathType Leaf) {
            $resolved = (Resolve-Path -LiteralPath $p).Path
            if ((Get-NormalizedPath $resolved) -ne $destinationNormalized) {
                return $resolved
            }
        }
    }

    $hits = @(
        Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -Filter $FileName -ErrorAction SilentlyContinue |
            Where-Object {
                (Get-NormalizedPath $_.FullName) -ne $destinationNormalized -and
                $_.FullName -notmatch '\\figures\\supplement\\'
            }
    )

    if ($hits.Count -eq 0) { return $null }

    $ranked = $hits | Sort-Object -Property @(
        @{ Expression = {
                if ($_.FullName -match '\\results\\e1_e2_service_contract_publication\\') { 0 }
                elseif ($_.FullName -match '\\results\\cross_domain_contract_generalization\\') { 1 }
                elseif ($_.FullName -match '\\results\\publication_hardening\\') { 2 }
                elseif ($_.FullName -match '\\results\\service_relative_fidelity\\') { 3 }
                elseif ($_.FullName -match '\\results\\') { 4 }
                elseif ($_.FullName -match '\\magnet_tfp\\results\\') { 5 }
                else { 6 }
            }
        },
        @{ Expression = 'LastWriteTime'; Descending = $true }
    )

    return ($ranked | Select-Object -First 1).FullName
}

function Collect-Figure {
    param(
        [string]$Name,
        [string]$Section,
        [string]$Role,
        [string[]]$Candidates,
        [bool]$Required = $false,
        [string]$Subdir = '',
        [string]$Note = ''
    )

    if ([string]::IsNullOrWhiteSpace($Subdir)) {
        $targetDir = $Dest
    } else {
        $targetDir = Join-Path $Dest $Subdir
    }
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    $target = Join-Path $targetDir $Name

    $existing = Test-Path -LiteralPath $target -PathType Leaf
    $source = Resolve-FigureSource -Candidates $Candidates -FileName $Name -DestinationFile $target

    if ($null -ne $source) {
        if ($RefreshExisting -or -not $existing) {
            Copy-Item -LiteralPath $source -Destination $target -Force
            $status = 'COPIED'
            Write-Host ('COPIED [{0}] {1}' -f $Section, $target)
            Write-Host ('   <- {0}' -f $source)
        } else {
            $status = 'READY'
            Write-Host ('READY  [{0}] {1}' -f $Section, $target)
        }
    } elseif ($existing) {
        $status = 'READY_EXISTING'
        $source = $target
        Write-Host ('READY  [{0}] {1}' -f $Section, $target)
    } else {
        if ($Required) {
            $status = 'MISSING_REQUIRED'
            $coreMissing.Add($Name) | Out-Null
        } else {
            $status = 'MISSING_OPTIONAL'
        }
        Write-Warning ('{0}: {1}' -f $status, $Name)
    }

    if ([string]::IsNullOrWhiteSpace($Subdir)) {
        $relativeTarget = $Name
    } else {
        $relativeTarget = Join-Path $Subdir $Name
    }

    $manifest.Add([pscustomobject]@{
        section = $Section
        role = $Role
        file = $relativeTarget
        status = $status
        source = $source
        note = $Note
    }) | Out-Null
}

# A. Frozen E1/E2/E3 core figures.
$coreSpecs = @(
    [pscustomobject]@{ Name='e1_parking_full_grid_inversion.png'; Section='E1'; Role='CORE'; Required=$true; Subdir=''; Candidates=@('results\e1_e2_service_contract_publication\E1_i2nav\e1_parking_full_grid_inversion.png'); Note='Full tolerance-grid parking00/parking02 local/global inversion; preferred over a single selected threshold.' },
    [pscustomobject]@{ Name='e1_metric_service_rank_alignment.png'; Section='E1'; Role='CORE'; Required=$true; Subdir=''; Candidates=@('results\e1_e2_service_contract_publication\E1_i2nav\e1_metric_service_rank_alignment.png'); Note='Shows ATE and finite-horizon RPE align with different service-validity questions.' },
    [pscustomobject]@{ Name='E1_E2_cross_platform_position_contracts.png'; Section='E2'; Role='CORE'; Required=$true; Subdir=''; Candidates=@('results\e1_e2_service_contract_publication\E1_E2_cross_platform_position_contracts.png'); Note='Unchanged service-contract structure across i2Nav and TerraSentia.' },
    [pscustomobject]@{ Name='e2_terrasentia_contract_transfer.png'; Section='E2'; Role='SUPPORT'; Required=$false; Subdir='supplement'; Candidates=@('results\e1_e2_service_contract_publication\E2_terrasentia\e2_terrasentia_contract_transfer.png'); Note='TerraSentia-only contract transfer; RTK position primary.' },
    [pscustomobject]@{ Name='cross_domain_horizon_profile.png'; Section='E3'; Role='CORE'; Required=$true; Subdir=''; Candidates=@('results\cross_domain_contract_generalization\figures\cross_domain_horizon_profile.png'); Note='MAGNET + FreeTwinEV + TU Wien SNG horizon-dependent contract validity.' },
    [pscustomobject]@{ Name='cross_domain_contract_heatmap.png'; Section='E3'; Role='CORE'; Required=$true; Subdir=''; Candidates=@('results\cross_domain_contract_generalization\figures\cross_domain_contract_heatmap.png'); Note='Cross-domain tolerance/horizon contract summary.' }
)

foreach ($spec in $coreSpecs) {
    Collect-Figure -Name $spec.Name -Section $spec.Section -Role $spec.Role -Candidates $spec.Candidates -Required $spec.Required -Subdir $spec.Subdir -Note $spec.Note
}

foreach ($name in @('MAGNET_contract_surface.png', 'FreeTwinEV_1S4P_contract_surface.png', 'TUWien_SNG_contract_surface.png')) {
    Collect-Figure -Name $name -Section 'E3' -Role 'SUPPORT' -Candidates @((Join-Path 'results\cross_domain_contract_generalization\figures' $name)) -Required $false -Subdir 'supplement' -Note 'Individual cross-domain contract surface.'
}

# B. Existing curated figures useful for the rewrite.
$existingMain = @(
    [pscustomobject]@{ Name='persistent_yaw_mechanism.png'; Section='Mechanism'; Note='Persistent yaw / accumulated yaw / global divergence mechanism diagnostic.' },
    [pscustomobject]@{ Name='condition_dependent_fidelity.png'; Section='Conditions'; Note='Condition dependence across fidelity dimensions.' },
    [pscustomobject]@{ Name='coverage_vs_sharpness.png'; Section='Conditions'; Note='Conditioned vs unconditional benign-envelope calibration/sharpness.' },
    [pscustomobject]@{ Name='parking02_tfp_global_tail.png'; Section='E1'; Note='Detailed parking02 local-good/global-bad diagnostic; use if a case-study panel is retained.' },
    [pscustomobject]@{ Name='magnet_horizon_component_counterexample.png'; Section='MAGNET'; Note='Hardened MAGNET composite: horizon, component, matched-RMSE counterexample.' },
    [pscustomobject]@{ Name='ugv01_asset_instantiation.png'; Section='UGV01'; Note='Existing asset-specific calibration/instantiation evidence; descriptive until prospective campaign is added.' },
    [pscustomobject]@{ Name='ugv01_condition_fidelity_profile.png'; Section='UGV01'; Note='Existing physical condition profile; descriptive/supplementary until new prospective runs.' },
    [pscustomobject]@{ Name='ugv01_local_vs_global_fidelity.png'; Section='UGV01'; Note='Existing UGV01 local-vs-global evidence; descriptive/supplementary until new prospective runs.' },
    [pscustomobject]@{ Name='representative_divergence_trace.png'; Section='Framework'; Note='Representative physical-virtual divergence trace.' }
)

foreach ($spec in $existingMain) {
    Collect-Figure -Name $spec.Name -Section $spec.Section -Role 'CURATED_EXISTING' -Candidates @((Join-Path 'figures' $spec.Name), (Join-Path 'results\publication_hardening\figures' $spec.Name)) -Required $false -Note $spec.Note
}

Collect-Figure -Name 'official_benchmark_results.png' -Section 'Benchmark' -Role 'CONTEXT_ONLY' -Candidates @('figures\official_benchmark_results.png') -Required $false -Subdir 'supplement' -Note 'Context only. Re-check exact official-evaluator provenance before citing headline values.'

Collect-Figure -Name 'v1_v2_local_vs_global_clean.png' -Section 'Legacy i2Nav' -Role 'SUPPORT' -Candidates @('figures\v1_v2_local_vs_global_clean.png', 'results\publication_hardening\figures\v1_v2_local_vs_global_clean.png') -Required $false -Subdir 'supplement' -Note 'Internal V1/V2 comparison; keep separate from later official Fixed-Physics-vs-V2 evaluation.'

foreach ($name in @('sensing_fidelity_tradeoff.png', 'benign_envelope_by_condition.png', 'loso_envelope_coverage.png', 'ugv01_rpe_vs_global_fidelity.png')) {
    Collect-Figure -Name $name -Section 'Existing supporting evidence' -Role 'SUPPORT' -Candidates @((Join-Path 'figures' $name)) -Required $false -Subdir 'supplement' -Note 'Retained for rewrite/supplement consideration; verify against the final claim set before main-paper use.'
}

foreach ($name in @('parking_local_contrast.png', 'parking_global_contrast.png', 'local_1s_tight_decision_summary.png', 'local_5s_moderate_decision_summary.png', 'local_10s_preview_decision_summary.png', 'global_state_tracking_decision_summary.png')) {
    Collect-Figure -Name $name -Section 'Service-relative audit' -Role 'SUPPORT' -Candidates @((Join-Path 'results\service_relative_fidelity\figures' $name)) -Required $false -Subdir 'supplement' -Note 'Audit/provenance figure; not preferred over frozen E1 full-grid figures.'
}

# Keep source MAGNET plots in a supplement provenance folder.
$magnetRoots = @(
    (Join-Path $RepoRoot 'magnet_tfp\results'),
    (Join-Path $RepoRoot 'results\magnet_tfp'),
    (Join-Path $RepoRoot 'MAGNET_publication_results')
) | Where-Object { Test-Path -LiteralPath $_ }

foreach ($root in $magnetRoots) {
    Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -match '^\.(png|pdf|jpg|jpeg)$' } |
        ForEach-Object {
            $target = Join-Path $MagnetSupp $_.Name
            if ($RefreshExisting -or -not (Test-Path -LiteralPath $target)) {
                Copy-Item -LiteralPath $_.FullName -Destination $target -Force
            }
        }
}

# C. Write machine-readable and human-readable manifests.
$csvPath = Join-Path $Dest 'FIGURE_REWRITE_MANIFEST.csv'
$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csvPath

$mdPath = Join-Path $Dest 'FIGURE_REWRITE_MANIFEST.md'
$lines = New-Object 'System.Collections.Generic.List[string]'
$lines.Add('# Manuscript rewrite figure manifest') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('Generated by collect_manuscript_rewrite_figures.ps1.') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## Preferred main-paper figures available now') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('1. e1_parking_full_grid_inversion.png - threshold-robust service-relative inversion.') | Out-Null
$lines.Add('2. e1_metric_service_rank_alignment.png - why one scalar metric is insufficient across services.') | Out-Null
$lines.Add('3. E1_E2_cross_platform_position_contracts.png - i2Nav/TerraSentia contract portability.') | Out-Null
$lines.Add('4. cross_domain_horizon_profile.png - MAGNET/FreeTwinEV/SNG cross-domain horizon profile.') | Out-Null
$lines.Add('5. cross_domain_contract_heatmap.png - cross-domain tolerance/horizon structural-transfer view.') | Out-Null
$lines.Add('6. condition_dependent_fidelity.png - operating-condition decomposition.') | Out-Null
$lines.Add('7. persistent_yaw_mechanism.png - mechanism evidence.') | Out-Null
$lines.Add('8. magnet_horizon_component_counterexample.png - hardened MAGNET case study if space allows.') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## Physical UGV01 figures already available') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('The existing UGV01 figures are retained, but should be treated as current/descriptive evidence until the planned prospective IoT resource campaign is completed.') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## Important claim boundary') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('v1_v2_local_vs_global_clean.png is placed in supplement because it is an internal V1/V2 comparison. Do not mix it with the later official Fixed-Physics-vs-V2 evaluator results.') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## Collection status') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('| Section | Role | Figure | Status |') | Out-Null
$lines.Add('|---|---|---|---|') | Out-Null
foreach ($row in $manifest) {
    $lines.Add(('| {0} | {1} | {2} | {3} |' -f $row.section, $row.role, $row.file, $row.status)) | Out-Null
}
$lines | Set-Content -Encoding UTF8 -Path $mdPath

Write-Host ''
Write-Host ('Manifest CSV: {0}' -f $csvPath)
Write-Host ('Manifest MD : {0}' -f $mdPath)
Write-Host ''

if ($coreMissing.Count -gt 0) {
    Write-Warning ('Missing {0} CORE figure(s): {1}' -f $coreMissing.Count, ($coreMissing -join ', '))
    Write-Warning 'Run the corresponding frozen E1/E2/E3 analysis before manuscript rewrite.'
    if ($Strict) { exit 2 }
} else {
    Write-Host 'All frozen E1/E2/E3 core figures are present in figures/.'
}

Write-Host 'Figure collection complete.'
exit 0
