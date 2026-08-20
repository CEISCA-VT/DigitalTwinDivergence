# Official i2Nav Benchmark Evaluation: Frozen Twin V2

This report uses the verified public i2Nav-WHU `evaluate_odometry` protocol and the already-frozen Twin V2 outputs. No model was retrained, retuned, checkpoint-selected, or altered.

## Protocol

- Trajectory format: TUM rows `t tx ty tz qx qy qz qw`.
- Frame: i2Nav local NED reference; V2 internal ENU/FLU estimates exported to NED with yaw conversion.
- Association tolerance: `0.005 s`.
- Alignment: SE(3), no scale correction.
- Metrics: APE translation/rotation RMSE and all-pairs distance RPE at 50/100/150/200/250/300 m with relative delta tolerance `0.002`.

## What Are The Official Frozen V2 Benchmark Results?

- V2 official APE translation macro mean: **1.635 m**.
- V2 official APE rotation macro mean: **3.011 deg**.
- V2 official RPE 50 m translation macro mean: **1.310 m** (2.620%).
- V2 official RPE 100 m translation macro mean: **2.217 m** (2.217%).
- V2 official RPE 300 m translation macro mean: **3.635 m** (1.212%).

## Fixed Physics / V1 Availability

- Twin V2: 30 frozen runs found and evaluated (10 sequences x 3 base seeds).
- Fixed Physics: included as the deterministic `fixed_v5_replay` trajectory where one official-format run per sequence exists.
- V1: not included in official scoring because no exact matching frozen V1 trajectory files were found; only scalar internal V1 metrics exist in V2 summaries.
- Fixed Physics official APE translation macro mean: **3.299 m**.
- Fixed Physics official RPE 50 m translation macro mean: **46.154 m**.

## V2 Compared With Fixed Physics

- APE translation macro mean: V2 changes by **-1.664 m**, which is a **50.441% reduction in the macro mean** versus Fixed Physics. The mean sequence-wise relative change is **-34.894%**. V2 improved on 9/10 sequences; bootstrap CI [-3.492, -0.279], sign-flip p=0.0059.
- RPE 50 m translation macro mean: V2 changes by **-44.844 m**, which is a **97.162% reduction in the macro mean** versus Fixed Physics. The mean sequence-wise relative change is **-97.162%**. V2 improved on 10/10 sequences.

## Hard Sequences

Largest V2 official APE translation sequences:

| sequence | official APE trans. RMSE (m) | internal ATE RMSE (m) | internal Dp p95 (m) |
|---|---:|---:|---:|
| parking02 | 5.747 | 11.350 | 22.345 |
| parking01 | 1.920 | 4.071 | 7.763 |
| building02 | 1.659 | 1.663 | 4.735 |

parking01/parking02 remain important, but the official SE(3)-aligned APE layer compresses some of the long-horizon drift that is visible in the internal DT-fidelity layer.

## Official vs Internal DT-Fidelity Layer

- Official APE/RPE are benchmark metrics after SE(3) alignment; they are suitable for protocol-compatible odometry-style comparison.
- Internal DT-fidelity metrics remain the correct evidence for physical-virtual synchronization because they do not use post-hoc alignment to hide drift.
- The local-vs-global result remains visible by comparison: short-horizon internal RPE can stay small while internal Dp/Dtheta grows, even if official aligned APE is reduced.

## Carry-Forward Benchmark Numbers

Use the V2 macro means in `official_macro_summary.csv` for later sensing-fidelity comparison, especially:

- `official_ape_translation_rmse_m_macro_mean`
- `official_ape_rotation_rmse_deg_macro_mean`
- `official_rpe_50m_translation_rmse_m_macro_mean`
- `official_rpe_100m_translation_rmse_m_macro_mean`
- `official_rpe_300m_translation_rmse_m_macro_mean`

Do not claim state of the art unless published systems are evaluated under this same protocol.

## Files Produced

- `official_export_manifest.json`
- `official_per_run_results.csv`
- `official_per_sequence_results.csv`
- `official_macro_summary.csv`
- `official_method_comparison.csv`
- `official_internal_comparison.csv`
- `official_benchmark_results.png`
- `official_hard_sequence_comparison.png`
