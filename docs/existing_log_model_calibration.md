# Existing-Log Motion and Covariance Calibration

Status: completed from the existing 20-run benign corpus without new rover
collection.

## Data Discipline

- Development: trials 1-3, 12 complete runs.
- Validation: trial 4, 4 complete runs.
- Test: trial 5, 4 complete runs.
- Motion and covariance parameters were selected only from development runs.
- Waveshare drive radius, encoder count, sign, and nominal track width remain
  unchanged.

## Motion Model Changes

Each run estimates z-gyro bias from its stationary pre-motion prefix. The
bias-corrected signal is clipped, low-pass filtered, and assigned a 5% weight:

```text
omega_fused = 0.95 * omega_encoder + 0.05 * omega_imu
```

The low weight is deliberate. Development profiling showed correct IMU turn
sign but only moderate encoder/IMU rate correlation, so heavier fusion was not
supported. The frozen filter coefficient is `0.40`.

Initial local heading is estimated by aligning the first 16 fused
dead-reckoning updates to the clean pre-monitoring GPS displacement using one
2D rotation. GPS is not used to correct the security branch after this
initialization interval.

## Imbalance and Slip Diagnostics

Across the 20 runs:

- median stationary gyro bias: `-0.094 deg/s`
- gyro-bias range: `-0.433 to 0.151 deg/s`
- median run-level yaw-rate disagreement: `0.081 rad/s`
- median run-level p95 normalized slip indicator: `0.684`

These diagnose encoder/IMU inconsistency. They do not identify true physical
slip without independent trajectory ground truth.

| Speed / surface | Mean yaw disagreement | Mean slip p95 | Mean straight IMU yaw | Mean turn IMU/encoder ratio |
| --- | ---: | ---: | ---: | ---: |
| Low / rough permeable concrete | 0.081 rad/s | 0.686 | -3.225 deg/s | 0.490 |
| Low / smooth kitchen floor | 0.075 rad/s | 0.668 | -3.390 deg/s | 0.482 |
| Medium / rough permeable concrete | 0.083 rad/s | 0.700 | -4.643 deg/s | 0.481 |
| Medium / smooth kitchen floor | 0.085 rad/s | 0.685 | -3.061 deg/s | 0.468 |

The condition differences are modest except for the stronger straight-motion
yaw tendency on medium rough concrete. These values are indicators, not
terrain-specific physical slip coefficients.

## Covariance Calibration

The development unscaled NIS 95th percentile was divided by the
two-dimensional chi-square 95th percentile, producing:

```text
covariance_scale = 1.6481096867023477
```

The same scalar multiplies initial `P`, process covariance `Q`, and GPS
measurement covariance `R`. Therefore the Kalman gain and state trajectory are
unchanged; only uncertainty normalization and gate statistics change.

## Evaluation

| Split | Runs | Operational-GPS RMSE | Security-GPS RMSE | NIS 95% coverage | Mean NIS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 12 | 1.213 m | 2.474 m | 94.9% | 1.570 |
| Validation | 4 | 1.579 m | 3.737 m | 83.3% | 3.622 |
| Test | 4 | 1.227 m | 3.250 m | 89.8% | 2.912 |

Matched ablations use the same 16-update evaluation horizon:

- encoder-only versus 5% IMU-fused operational GPS agreement:
  `1.291 m` versus `1.297 m`
- encoder-only versus 5% IMU-fused security-predictor GPS agreement:
  `2.932 m` versus `2.925 m`
- five-fix displacement initialization versus 16-update shape-aligned
  initialization for the security predictor: `3.127 m` versus `2.925 m`

The initialization change is the material improvement, reducing pooled
security-predictor GPS disagreement by about `6.5%`. The conservative gyro
fusion is nearly neutral: it slightly worsens operational sensor agreement
and slightly improves protected-predictor agreement. It is retained primarily
for bias correction, turn-rate cross-checking, and slip diagnostics rather
than claimed as an accuracy gain.

The held-out coverage gap is retained. It indicates that a single global
covariance scale does not fully describe condition and run variability.
Further tuning against validation or test data would invalidate their role as
held-out evidence.

Generate the current report with:

```powershell
python -m DigitalTwin.analysis.digital_twin_accuracy
```

The generated report is
`DigitalTwin/datasets/analysis/digital_twin_accuracy/accuracy_report.md`.
These values quantify sensor agreement and statistical consistency, not
physical localization accuracy.
