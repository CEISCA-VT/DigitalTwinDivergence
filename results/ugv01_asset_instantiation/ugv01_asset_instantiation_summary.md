# UGV01 Asset-Specific Digital Twin Instantiation Summary

## Scientific Answer

Existing artifacts support a qualified asset-specific instantiation result:
binding the generic/current rover representation to this physical UGV01 and
calibrating low-speed carpet motion improves physical-virtual agreement on the
recorded AprilTag pilot. The strongest same-window comparison is:

| Metric | Current UGV01 model | Asset-specific fitted | Change |
|---|---:|---:|---:|
| ATE RMSE | 0.131 m | 0.099 m | -24.4% |
| RPE1 RMSE | 0.035 m | 0.028 m | -19.9% |
| Heading MAE | 13.3 deg | 5.6 deg | -57.9% |

The current full repaired headline for the fitted asset-specific twin is:

| Metric | Value |
|---|---:|
| ATE RMSE | 0.114 m |
| RPE1 RMSE | 0.030 m |
| RPE5 RMSE | 0.087 m |
| RPE10 RMSE | 0.104 m |
| Heading MAE | 6.7 deg |
| Position p95 / max | 0.207 / 0.222 m |
| Heading p95 / max | 19.3 / 37.8 deg |

## What Changed Between Stages

- The current comparator already has a UGV01 frame/sensor adapter, encoder signs,
  AprilTag frame alignment, and motion-correlation timing.
- The asset-specific fitted stage adds a carpet-specific distance scale, asymmetric
  effective track widths, and bounded gyro contribution recorded in
  `DigitalTwin/datasets/analysis/ugv01_apriltag_finetune_142023/temporal_calibration_summary.json`.
- The full repaired headline applies that candidate to the complete repaired
  142023 window.

## Interpretation

UGV01-specific calibration measurably improves the same-window development result,
especially heading and short-horizon path shape. The remaining weak point is that
this is one low-speed carpet recording with motion-correlation synchronization and
partly recovered AprilTag samples. The model is defensibly a twin of this specific
UGV01 only for the tested low-speed carpet condition. It is not yet evidence for
all surfaces, higher speeds, or sustained slip-heavy motion.

GPS is not required for this particular fidelity audit because the independent
reference is AprilTag pose. GPS would become necessary for later GPS-trust or
security claims, not for proving physical-virtual agreement in this run.

## Is New Physical Collection Scientifically Necessary?

New data are not necessary to report a careful existing-data asset-instantiation
analysis. New data are necessary for stronger claims:

- An untouched synchronized repeat enables a publication-grade held-out UGV01
  fidelity claim.
- A second surface enables cross-surface fidelity claims.
- A higher-speed run enables speed-regime claims.
- A sustained-turn or figure-eight run enables turn/slip robustness claims.

The minimum next experiment is therefore one clean synchronized UGV01 AprilTag
run with a deliberate visible sync event; add only one extra targeted condition
if the paper needs a generalization claim.
