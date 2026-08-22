# v2 publication-hardening changes

This version keeps the original MAGNET audit and adds reviewer-facing safeguards:

- headline horizon statistics use only the non-overlapping subset;
- paired bootstrap 95% intervals, one-sided Wilcoxon signed-rank test, and paired rank-biserial effect size;
- similar-RMSE counterexamples must be temporally disjoint, with both 5% and 10% RMSE matching tolerances;
- outlier trimming checks whether extreme numerical forecast failures drive the conclusion;
- persistence is re-evaluated across fixed thresholds from 1 to 50 °C;
- robust-envelope results are tested across multiple MAD multipliers and floors;
- operating-regime results are checked across multiple slope thresholds;
- component-by-horizon localization is summarized explicitly;
- a dedicated publication-hardening report is generated;
- PowerShell output is forced to UTF-8 to avoid mojibake such as `Â°C` and `Ã—`.
