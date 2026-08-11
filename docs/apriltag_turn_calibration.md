# AprilTag Tracked-Turn Calibration

## Status

The UGV01 physical/vendor track-width parameter remains `0.141 m`. Camera
ground truth shows that this value is too small for encoder-based skid-steer
turn prediction on the recorded kitchen-floor run. The provisional empirical
turn parameter is:

| Parameter | Value |
|---|---:|
| Nominal physical width | `0.141 m` |
| Effective tracked-turn width | `0.192 m` |
| Left encoder sign | `-1` |
| Right encoder sign | `-1` |
| Gyro scale | `1.0` |
| Direct gyro fusion weight | `0.00` |

## Evidence

- Video: `docs/footage/footage_trapezoid.mp4`
- Telemetry: `raw_logs/telemetry/ugv_t147_interactive_20260805_192736.csv`
- Camera/telemetry offset: `-10.10 +/- 0.28 s`
- Motion-signature alignment correlation: `0.856`
- Direct interval regression estimate: `0.18984 m`
- Yaw-focused trajectory-grid selection: `0.192 m`
- Earlier encoder-only estimate from independent motion logs: `0.1818 m`

The encoder signs are confirmed by a positive correlation of `0.838` between
signed encoder rotation and AprilTag heading change. The nominal-width model
systematically over-rotates because angular displacement is divided by a width
that is too small for a laterally slipping tracked vehicle.

## Fidelity Change

| Metric | Nominal `0.141 m` | Calibrated `0.190 m` |
|---|---:|---:|
| Position ATE RMSE | `0.561 m` | `0.101 m` |
| 1-second RPE RMSE | `0.072 m` | `0.025 m` |
| Heading MAE | `102.7 deg` | `17.4 deg` |

Total path length was already close under the nominal model. The improvement
comes primarily from assigning the encoder distance to more accurate turn
angles and therefore to more accurate trajectory directions.

## IMU Decision

The stationary z-gyro bias for this run was approximately `0.07375 deg/s`.
Direct gyro fusion degraded complete-run and late-run trajectory fidelity,
including at weights below and above the previous `0.05`. Gyro scale is
retained at `1.0`, but direct pose-mean weight is set to zero. The IMU remains
an independent bias, slip, uncertainty, and security-evidence source.

## Slip-Aware Uncertainty

An event-level audit paired encoder counter changes with synchronized AprilTag
heading changes in both camera runs. Nine valid short turns gave a median
absolute encoder-to-camera ratio error of `6.7%`, a median absolute angle error
of `10.43 deg`, and a 90th-percentile ratio error of `31.2%`. The median event
effective width was `0.194 m`, close to the frozen `0.192 m` mean, but individual
events ranged from `0.176 m` to `0.252 m`.

The mean turn update therefore remains unchanged. The EKF now adds independent
tracked-turn variance after the selected uncertainty estimator and covariance
scale are applied:

```text
sigma_turn = 0.10 * abs(omega_encoder * dt)
Q[theta, theta] += sigma_turn^2
```

The empirical `31.2%` value is retained as a robust tail bound for reporting;
it is not used to shift the state mean. Straight motion receives no additional
turn variance. This term is GPS-independent and is applied consistently in
simulation, hardware replay, and real-data campaign replay.

The sparse firmware yaw samples had substantially larger short-event error than
encoder yaw, so they are not used to adapt the pose mean. High-rate onboard IMU
integration would be required before an IMU-driven slip correction is justified.

## Limitation

The `0.192 m` parameter is calibrated on this camera run and surface. It must
be evaluated without retuning on a separate synchronized run before it is
reported as a held-out accuracy result. Surface-dependent effective width may
be required if rough-ground validation shows a systematic difference.

Adding turn-slip covariance changes NIS, alarm, and campaign outputs. Analysis
artifacts generated before this policy must be regenerated before publication.
