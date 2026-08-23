# MAGNET exact-results integration

The manuscript now uses the hardened MAGNET publication archive rather than only audit counts.

Main-paper evidence incorporated:
- 111 candidate forecast windows; 97 strict eligible windows.
- 23 greedy non-overlapping windows used for dependence-reduced inference.
- Median 0-60 s RMSE: 0.0876 C.
- Median 301-599 s RMSE: 6.6176 C.
- Median paired increase: 6.0425 C; bootstrap 95% CI [2.2828, 6.6952] C.
- Long > short in 95.7% of independent windows.
- One-sided Wilcoxon p=2.38e-07; paired rank-biserial=0.993.
- Median paired long/short ratio 43.04x; 95% CI [32.13,80.69]x.
- After trimming the worst 20% by max absolute error: long/short median ratio 60.80x; long>short fraction 0.987.
- All 10 thermowells show median long/short ratio above 7.99x and long>short fraction at least 0.918.
- TC-06 has the largest median long-horizon RMSE: 7.669 C.
- 7 independent window pairs have aggregate RMSE within 5%; 3 show >=2x p99 differences and 3 show >=0.25 persistence differences.
- Strongest pair: windows 57 and 99 have RMSE 13.325 vs 13.131 C (1.47% apart), but p99 21.10 vs 73.25 C (3.47x), persistence 0.680 vs 0.146, and short-horizon RMSE 13.353 vs 0.162 C.

Claim boundaries retained:
- Long-horizon RMSE is highly correlated with aggregate RMSE (Spearman rho=0.997); p99 rho=0.966; persistence rho=0.784. Do not claim every TFP dimension is statistically independent of RMSE.
- MAGNET operating-regime comparison is null on independent windows (Kruskal-Wallis p=0.426, epsilon-squared=0.000). Do not use MAGNET as evidence for condition-dependent fidelity.
- Persistence thresholds are sensitivity analyses, not physics-certified safety tolerances.
- Claim cross-domain structural transfer, not universal DT validity.
