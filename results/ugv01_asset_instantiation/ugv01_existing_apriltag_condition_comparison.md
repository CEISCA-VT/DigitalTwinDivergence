# Existing UGV01 AprilTag Condition Comparison

The repo does contain non-carpet AprilTag fidelity artifacts. They were not part of the strict headline split, but they are useful as supplemental evidence for a condition-dependent UGV01 discussion.

| Condition | Role | ATE RMSE | RPE1 RMSE | Heading MAE | p95 position | Sync correlation | Main limitation |
|---|---|---:|---:|---:|---:|---:|---|
| carpet_strict_headline | strict headline | 0.114 m | 0.030 m | 6.7 deg | 0.207 m | 0.933 | explicit in filename/docs |
| smooth_floor_trapezoid | supplemental condition evidence | 0.101 m | 0.025 m | 17.4 deg | 0.165 m | 0.842 | docs/apriltag_turn_calibration.md calls this the recorded kitchen-floor run |
| smooth_floor_square_1p5 | supplemental condition evidence | 0.338 m | 0.046 m | 28.6 deg | 0.918 m | 0.835 | video/analysis artifact exists, but surface is not as explicitly named as the carpet run |

Interpretation:

- The trapezoid run is the best existing smooth/kitchen-floor-like supplemental condition: ATE and RPE1 are close to the carpet headline, but heading error is worse and sync uncertainty is larger.
- The 1.5 m square run shows much larger global/tail error despite reasonable short-horizon RPE, making it useful for the local-versus-global fidelity story.
- These runs can support a cautious UGV01 condition-dependent discussion now, but they should be labeled supplemental unless the manuscript explicitly accepts motion-correlation sync and repaired tracks as sufficient.