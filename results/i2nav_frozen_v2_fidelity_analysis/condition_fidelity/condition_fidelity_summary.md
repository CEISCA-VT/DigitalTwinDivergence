# Condition-Dependent Twin V2 Fidelity

Script version: `2026-08-20-condition-fidelity-v1`
Frozen full-LOSO commit expected by context: `6540c01f90f3c1074de0d8dae9964a5276fbbc91`

This analysis uses only saved frozen V2 LOSO artifacts and reconstructed canonical i2Nav sensor context. It does not retrain, tune, or alter V2.

## Frozen Condition Definitions

Condition-bin definitions are recorded in `condition_definitions.json`. Numeric variables use global tertiles of condition variables only; outcome metrics are not used to choose thresholds.

## Statistical Hierarchy

Condition metrics are computed within each seed run, then the three seeds are aggregated within each physical sequence. Dataset-level interpretation uses the 10 physical sequences as the unit. Timestamp-level samples are not treated as independent replicates.

## Which Conditions Consistently Degrade Local Fidelity?

| Metric | strongest condition contrast | median delta | degraded sequences |
|---|---|---:|---:|
| RPE1_m | wheel_imu_disagreement: high vs low | 0.005 | 10/10 |
| RPE5_m | turning: high vs low | 0.018 | 9/10 |
| RPE10_m | turning: high vs low | 0.026 | 9/10 |

Local fidelity changes are generally weaker and less monotonic than global synchronization changes. This matches the earlier local-vs-global finding: low short-horizon RPE can coexist with long-horizon drift.

## Which Conditions Consistently Degrade Global Synchronization?

| Metric | strongest condition contrast | median delta | degraded sequences |
|---|---|---:|---:|
| Dp_p95_m | acceleration: high vs low | 0.024 | 9/10 |
| Dtheta_p95_deg | curvature: high vs low | 0.278 | 10/10 |

## Are Global-Divergence Conditions Different From RPE Conditions?

Yes, in the current frozen V2 evidence they are not the same object. RPE degradation is finite-horizon and often modest, while Dp/Dtheta degradation is more sensitive to elapsed time and accumulated orientation mismatch. Therefore the paper should avoid using RPE alone as the definition of digital-twin fidelity.

## Is parking02 Explained By an Extreme Measurable Condition?

parking02 is not fully explained by a single extreme bin of speed, acceleration, turning, curvature, or wheel-IMU disagreement. It remains the largest global-divergence sequence even though some simple operating-condition variables are not uniquely extreme. This supports a sequence-specific behavior interpretation: the measurable benign conditions help characterize stress, but they do not by themselves collapse parking02 into an ordinary high-speed or high-turning case.

parking02 high-bin diagnostic:

| Variable | high-bin duration approx [s] | parking02 Dp-p95 rank within high bin |
|---|---:|---:|
| speed | 99.5 | 1 |
| acceleration | 486.0 | 1 |
| turning | 433.7 | 1 |
| curvature | 442.5 | 1 |
| wheel_imu_disagreement | 386.1 | 1 |
| persistent_yaw_mismatch | 1256.9 | 1 |

## Variables Supported For Later Benign Fidelity Envelope

Supported conditioning variables are those with enough sequence/bin coverage to summarize without leaning on one sequence.

`speed`, `acceleration`, `turning`, `curvature`, `wheel_imu_disagreement`, `elapsed_time`

The environment category is useful descriptively, but it is not an ordered stress variable and should not be treated like speed or turning.
The lateral/slip proxy is not supported by finite values in the current frozen canonical i2Nav context, so it should not be used for an envelope from this run.

## Files Produced

- `condition_definitions.json`
- `per_run_condition_fidelity.csv`
- `per_sequence_condition_fidelity.csv`
- `condition_degradation_summary.csv`
- `fidelity_by_speed.png`
- `fidelity_by_turning.png`
- `fidelity_by_wheel_imu_disagreement.png`
- `fidelity_by_time.png`

## Null / Weak Findings

Not every condition variable produces a strong or monotonic relationship. The analysis should preserve these weak findings because they are scientifically important: the twin's hard failures are not reducible to a single obvious stress scalar.
