# AprilTag Validation Split

This note freezes how the existing UGV01 AprilTag footage should be used after
continuity repair. The goal is to avoid wasting the recorded videos while still
being honest about which segments are strong enough for validation claims.

## Strict Validation Set

Use these runs for the main digital-twin fidelity discussion.

| Run | Output folder | Reason |
|---|---|---|
| Carpet 2 m x 1 m run | `DigitalTwin/datasets/analysis/carpet_2x1_rect_continuity_repaired/` | Best long AprilTag track. Rover tag is valid in `11591/11900` frames before repair and `11900/11900` after repair. Only `309` short-gap frames are interpolated; longest gap is `4.20 s`. |
| Fine-tuned 142023 validation run | `DigitalTwin/datasets/analysis/ugv01_apriltag_finetuned_full_142023_continuity_repaired/` | Best paired digital-twin fidelity result using the repaired AprilTag track and matched `T:147` telemetry. It is the current headline UGV01 fidelity result. |

Strict-set wording:

> The strict AprilTag validation set is limited to runs where the rover tag is
> continuously available after short-gap continuity repair, and where the
> measured/recovered tag track dominates the reconstructed portion.

## Supplemental Set

Use these for robustness discussion, qualitative plots, route-shape examples, or
engineering diagnostics. They should not be the main publication-grade accuracy
claim unless their limitations are explicitly disclosed.

| Run | Output folder | Use | Limitation |
|---|---|---|---|
| Trapezoid run | `DigitalTwin/datasets/analysis/trapezoid_continuity_repaired/` | Good supplemental route-shape evidence. | Position is useful after calibration, but heading agreement is weaker than the main carpet run. |
| Carpet 1428 run | `DigitalTwin/datasets/analysis/carpet_1428_2x1_continuity_repaired/` | Supplemental continuity/visualization. | Contains one `25.97 s` long interpolated gap. |
| Trial 1 square 1.5 m run | `DigitalTwin/datasets/analysis/trial1_square_1p5_continuity_repaired/` | Supplemental square-route visualization. | Contains `1418` long-gap interpolated frames; longest gap is `26.24 s`. |
| Carpet 1433 run | `DigitalTwin/datasets/analysis/carpet_1433_2x1_continuity_repaired/` | Partial diagnostic only. | `1015` frames remain missing after repair. |

## Excluded From Ground-Truth Claims

| Run | Reason |
|---|---|
| `still footage.mp4` | It contains the fixed reference tags but no rover tag `ID 0`, so it is a setup/reference video rather than a rover trajectory. |
| `carpet_low_speed_2m_continuity_repaired` | The rover tag is valid in only `15/13696` frames before repair and `505/13696` after repair, so it is not reliable ground truth. |

## Current Fine-Tuned Digital-Twin Fidelity

Current headline result:

| Metric | Value |
|---|---:|
| Evaluation samples | `2153` |
| Position ATE RMSE | `0.114 m` |
| Median position error | `0.090 m` |
| 95th percentile position error | `0.207 m` |
| 1 s RPE RMSE | `0.030 m` |
| Heading MAE | `6.7 deg` |
| Heading 95th percentile error | `19.3 deg` |
| Samples within 5 cm | `20.9%` |
| Samples within 10 cm | `58.2%` |
| Samples within 25 cm | `100.0%` |
| Truth path length | `8.94 m` |
| Estimated path length | `9.12 m` |
| Path-length ratio | `1.021` |
| Camera stationary jitter RMSE / p95 | `0.001 / 0.004 m` |

Compared with the old/current model on the same 142023 run, fine-tuning improved:

| Metric | Old/current | Fine-tuned strict windows | Fine-tuned repaired full window |
|---|---:|---:|---:|
| Position ATE RMSE | `0.131 m` | `0.099 m` | `0.114 m` |
| 1 s RPE RMSE | `0.035 m` | `0.028 m` | `0.030 m` |
| Heading MAE | `13.3 deg` | `5.6 deg` | `6.7 deg` |

## Required Caveat

The current UGV01 AprilTag fidelity result is a strong development validation,
but it is not yet the final publication-standard validation run because GPS was
disconnected and the camera/telemetry sync was estimated from motion rather than
from a deliberate hardware-visible synchronization event. The final paper should
separate this result from any later GPS-plus-AprilTag synchronized run.
