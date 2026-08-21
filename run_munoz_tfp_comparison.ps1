# Run from the DigitalTwinDivergence repository root.
$root = "results\i2nav_v2_full_loso\i2nav_v2_full_loso"

python -m DigitalTwin.analysis.munoz_trace_alignment_i2nav `
  --input-root $root `
  --alignment-hz 1.0 `
  --position-mads-m "0.25,0.5,1.0" `
  --heading-mads-deg "2,5,10" `
  --gap-open -1.0 `
  --gap-extend -0.1

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m DigitalTwin.analysis.compare_tfp_vs_munoz `
  --input-root $root `
  --munoz-csv "results\munoz_trace_alignment\munoz_per_sequence.csv"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Read:"
Write-Host "  results\munoz_trace_alignment\munoz_report.md"
Write-Host "  results\tfp_vs_munoz\tfp_vs_munoz_report.md"
