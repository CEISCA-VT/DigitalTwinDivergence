param(
    [string]$TexPath = ".\\main_iotj_rewritten_with_magnet.tex",
    [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path
$TexPath = (Resolve-Path $TexPath).Path
$Dest = Join-Path $RepoRoot "figures"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$tex = Get-Content -Raw $TexPath
$matches = [regex]::Matches($tex, 'figures/([A-Za-z0-9_.-]+\.(?:png|pdf|jpg|jpeg|eps))')
$names = $matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

Write-Host "Repository : $RepoRoot"
Write-Host "Destination: $Dest"
Write-Host "Figures referenced by manuscript: $($names.Count)"

$missing = @()
foreach ($name in $names) {
    $destFile = Join-Path $Dest $name
    if (Test-Path $destFile) {
        Write-Host "READY : $name"
        continue
    }

    # Prefer publication-hardening outputs, then results/, then the newest remaining source.
    $hits = @(Get-ChildItem -Path $RepoRoot -Recurse -File -Filter $name -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -ne $destFile })

    if ($hits.Count -eq 0) {
        Write-Warning "NOT FOUND: $name"
        $missing += $name
        continue
    }

    $ranked = $hits | Sort-Object `
        @{Expression={ if ($_.FullName -match '\\results\\publication_hardening\\figures\\') {0} elseif ($_.FullName -match '\\results\\') {1} else {2} }}, `
        @{Expression='LastWriteTime';Descending=$true}

    $src = $ranked | Select-Object -First 1
    Copy-Item -Force $src.FullName $destFile
    Write-Host ("COPIED: {0}`n        <- {1}" -f $name,$src.FullName)
}

# Also collect the user's existing MAGNET analysis figures without renaming them.
# These are optional supplementary/source figures; the manuscript's stable MAGNET
# cross-domain schematic is already distributed as figures\magnet_horizon_component_counterexample.png.
$magnetRoots = @(
    (Join-Path $RepoRoot "magnet_tfp\\results"),
    (Join-Path $RepoRoot "results\\magnet_tfp"),
    (Join-Path $RepoRoot "MAGNET_publication_results")
) | Where-Object { Test-Path $_ }

if ($magnetRoots.Count -gt 0) {
    $magnetDest = Join-Path $Dest "magnet_source"
    New-Item -ItemType Directory -Force -Path $magnetDest | Out-Null
    foreach ($mr in $magnetRoots) {
        Get-ChildItem -Path $mr -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -match '^\.(png|pdf|jpg|jpeg)$' } |
            ForEach-Object {
                $target = Join-Path $magnetDest $_.Name
                if (-not (Test-Path $target)) {
                    Copy-Item $_.FullName $target
                    Write-Host ("MAGNET SOURCE: {0}`n             <- {1}" -f $_.Name,$_.FullName)
                }
            }
    }
}

Write-Host ""
if ($missing.Count -gt 0) {
    Write-Warning ("Missing {0} manuscript figure(s): {1}" -f $missing.Count, ($missing -join ', '))
    exit 2
}

Write-Host "All manuscript figures are now in $Dest"
