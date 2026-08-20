# UGV01 Exact Fidelity-Protocol Comparison

Core pose and RPE metrics were computed by `DigitalTwin.analysis.i2nav_fidelity_evaluator.evaluate_fidelity_frames` with 1/5/10 s horizons.
Rate metrics in this report are derived from finite differences of the aligned physical and twin poses because these AprilTag artifacts do not contain the original prediction trace.

| Condition | ATE (m) | Heading MAE (deg) | RPE1 (m) | RPE5 (m) | RPE10 (m) | Dp p95 (m) | Dp max (m) | Dtheta p95 (deg) | Dtheta max (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carpet_low_speed | 0.114 | 6.7 | 0.029 | 0.086 | 0.109 | 0.207 | 0.222 | 19.3 | 37.8 |
| smooth_floor_trapezoid | 0.101 | 17.4 | 0.023 | 0.069 | 0.096 | 0.165 | 0.280 | 46.8 | 64.5 |
| smooth_floor_square_1p5 | 0.338 | 28.6 | 0.021 | 0.057 | 0.089 | 0.918 | 1.261 | 70.0 | 92.3 |

## Interpretation

- The trapezoid condition has lower ATE and RPE than the carpet headline, but substantially larger heading error and derived yaw disagreement.
- The 1.5 m square has low local RPE relative to its global ATE and Dp p95, providing physical evidence that local and global fidelity are distinct.
- These comparisons are descriptive across three recorded conditions, not independent repeated-condition statistics. The non-carpet runs use motion-correlated synchronization and remain supplemental evidence.
- The current result supports a cautious condition-dependent UGV01 discussion. It does not establish all-surface or higher-speed generalization.
