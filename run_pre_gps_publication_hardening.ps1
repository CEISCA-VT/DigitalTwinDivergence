param(
    [string]$TexPath = ""
)
$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Pre-GPS publication hardening"
Write-Host "============================================================"

Write-Host "[1/6] Sequence-level paired statistics"
python -m DigitalTwin.analysis.publication_sequence_stats
if ($LASTEXITCODE -ne 0) { throw "sequence statistics failed" }

Write-Host "[2/6] TFP-vs-Munoz bootstrap uncertainty"
python -m DigitalTwin.analysis.munoz_bootstrap_uncertainty
if ($LASTEXITCODE -ne 0) { throw "Munoz bootstrap analysis failed" }

Write-Host "[3/6] Controlled yaw-bias replay sensitivity"
python -m DigitalTwin.analysis.yaw_bias_replay_sensitivity
if ($LASTEXITCODE -ne 0) { throw "yaw-bias replay failed" }

Write-Host "[4/6] Claim audit"
if ($TexPath -ne "") {
    python -m DigitalTwin.analysis.claim_audit --tex "$TexPath"
} else {
    python -m DigitalTwin.analysis.claim_audit
}
if ($LASTEXITCODE -ne 0) { throw "claim audit failed" }

Write-Host "[5/6] Reproducibility snapshot + figures"
python -m DigitalTwin.analysis.reproducibility_snapshot
if ($LASTEXITCODE -ne 0) { throw "reproducibility snapshot failed" }
python -m DigitalTwin.analysis.publication_figures
if ($LASTEXITCODE -ne 0) { throw "publication figure generation failed" }

Write-Host "[6/6] Combined report"
python -m DigitalTwin.analysis.build_publication_hardening_report
if ($LASTEXITCODE -ne 0) { throw "report build failed" }

Write-Host ""
Write-Host "DONE"
Write-Host "Results: results\publication_hardening"
Write-Host "Paper additions: publication_hardening\paper_additions.tex"
Write-Host "Protocol freeze: publication_hardening\PRE_GPS_PROTOCOL.md"
