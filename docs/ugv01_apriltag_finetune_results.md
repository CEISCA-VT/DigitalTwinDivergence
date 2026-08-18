# UGV01 AprilTag Fine-Tuning Results

This is a development calibration on the current UGV01 AprilTag carpet pilot.
It is useful for improving the digital-twin motion model, but it is not a final
independent validation because the train and validation windows come from the
same recording session.

## Tuned Parameters

| Parameter | Old current value | Tuned value |
|---|---:|---:|
| Distance scale | 1.000 | 0.975 |
| Clockwise effective track width | 0.192 m | 0.200 m |
| Counterclockwise effective track width | 0.192 m | 0.190 m |
| Bias-corrected gyro weight | 0.20 | 0.20 |
| Gyro sign/scale | 1.0 | 1.0 |

These values are recorded as the carpet development candidate in
`DigitalTwin/kinematics.py`. The vendor/base geometry and firmware behavior are
unchanged.

## Full Usable AprilTag Windows

The full-window comparison uses the same selected AprilTag intervals as the
current pilot:

- `0.00-162.66 s`
- `188.66-239.30 s`

| Metric | Old current twin | Fine-tuned twin | Change |
|---|---:|---:|---:|
| Position ATE RMSE | 0.131 m | 0.099 m | 24.4% lower |
| 1-second RPE RMSE | 0.035 m | 0.028 m | 20.0% lower |
| Heading MAE | 13.3 deg | 5.6 deg | 57.9% lower |

## Temporal Holdout Check

Parameters were selected on the first 75% of the video and evaluated once on
the final 25%.

| Metric | Baseline holdout | Fine-tuned holdout |
|---|---:|---:|
| Position RMSE | 0.092 m | 0.092 m |
| 1-second RPE RMSE | 0.038 m | 0.036 m |
| Heading MAE | 9.6 deg | 7.5 deg |
| Samples within 10 cm | 83.0% | 56.4% |
| Samples within 25 cm | 100.0% | 100.0% |
| Path-length agreement | 99.2% | 96.7% |

## Interpretation

The fine-tuned model improves the full-window trajectory and especially heading
agreement. On the temporal holdout, heading and short-horizon trajectory error
improve, while the fraction within 10 cm and path-length agreement decrease.
That means the tuning is promising, but not a final proof of generalization.

For a paper, report this as development calibration. The final claim still
needs a separate synchronized run with telemetry, GPS, AprilTag video, and a
clear sync event.

## Generated Artifacts

- `DigitalTwin/datasets/analysis/ugv01_apriltag_finetune_142023/`
- `DigitalTwin/datasets/analysis/ugv01_apriltag_old_current_142023/`
- `DigitalTwin/datasets/analysis/ugv01_apriltag_finetuned_full_142023/`
- `DigitalTwin/datasets/analysis/ugv01_apriltag_finetuned_carpet_142023/`
