# E3 v2 corrective patch

This patch corrects two issues discovered by auditing the first E3 output.

1. **TU Wien SNG unit harmonization.** The measured product-gas channels are on an approximately percent scale (e.g. H2 around 38--55), while the corresponding soft-sensor channels are fractions (e.g. H2 around 0.41). The v1 script paired them without multiplying the soft-sensor fraction by 100, which artificially produced very large normalized errors. V2 harmonizes the soft-sensor channels to percent before computing discrepancy and records the scale factor in `sng_pairing_audit.csv`.

2. **FreeTwinEV CSV parsing/diagnostics.** The v1 flexible CSV reader could silently accept a semicolon-delimited file as a single comma-delimited column. V2 scores comma/semicolon/tab/sniffed parses and chooses a real multi-column table. It also records any remaining dataset loader failure in `dataset_errors.csv` and in the Markdown report.

The frozen contract itself is unchanged: horizons 60/300/600 s, normalized tolerance grid 0.01/0.02/0.05/0.10/0.20/0.50, and within-window q95 discrepancy.

Do not use the v1 SNG validity numbers in a manuscript. Rerun E3 with this patch and use only the v2 outputs.
