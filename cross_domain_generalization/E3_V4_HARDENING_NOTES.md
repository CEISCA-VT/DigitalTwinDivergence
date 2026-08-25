# E3 v4 hardening notes

This patch fixes the complete class of FreeTwinEV ingestion/alignment failures observed in the E3 audit without changing the frozen scientific contract (horizons, tolerances, q95 statistic, or transfer gates).

## Fixed

- Excel-style `sep=` directives may appear after blank/preamble lines.
- European decimal-comma and common grouped numeric strings are coerced robustly.
- NaN/Inf timestamps are removed before any alignment.
- Duplicate physical and solver timestamps are collapsed deterministically.
- FreeTwinEV alignment no longer uses `pandas.merge_asof`; it uses an explicit nearest-time matcher that cannot fail on null merge keys.
- Relative vs absolute simulation clocks are detected from declared segment timing only.
- Conventional time-unit conversions (s/ms/min/h) are handled and audited.
- Physical/simulation aggregate columns are explicitly prefixed; pairing no longer depends on pandas merge suffixes.
- Mean/max/spread temperature aggregates receive documented semantic fallback pairing when exact labels differ.
- Conservative temperature-channel fallback handles channel-ID style headers while excluding obvious electrical/flow variables.
- SNG as-of timestamps are sanitized as well.
- The output directory is regenerated from scratch on each run, preventing stale files from earlier partial audits.
- A required-dataset failure now returns a non-zero exit status after writing the full report.
- FreeTwinEV failures emit `freetwinev_schema_diagnostic.csv`, `freetwinev_time_diagnostic.csv`, and full traceback diagnostics.

## Scientific protocol unchanged

- Horizons: 60, 300, 600 s
- Normalized tolerances: 0.01, 0.02, 0.05, 0.10, 0.20, 0.50
- Within-window statistic: q95 normalized absolute discrepancy
- Structural transfer gates: unchanged
- SNG fraction-to-percent harmonization from v2 remains in force
