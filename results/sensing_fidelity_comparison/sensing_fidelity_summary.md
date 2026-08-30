# Sensing-Fidelity Tradeoff Summary

## Scope

This analysis positions the frozen Twin V2 official i2Nav benchmark result by sensing burden. It does not retrain, retune, re-export, or alter frozen benchmark trajectories.

Twin V2 runtime sensing:

- wheel/odometry: yes
- IMU: yes
- camera: no
- LiDAR: no
- radar: no
- GNSS: no
- ground truth: training supervision and held-out evaluation only, not runtime inference

## Frozen Twin V2 Numbers

- Official APE translation RMSE, arithmetic macro mean: 1.635 m
- Official APE rotation RMSE, arithmetic macro mean: 3.011 deg
- README-compatible ATE sequence-RMS: 2.187 m
- README-compatible ARE sequence-RMS: 3.706 deg
- RPE 50 m macro mean: 1.310 m
- RPE 100 m macro mean: 2.217 m
- RPE 300 m macro mean: 3.635 m

## Protocol-Compatible Positioning

Twin V2 ranks 5/9 among directly comparable 10-sequence ATE/ARE rows when using the README-compatible ATE sequence-RMS aggregate.

Methods with lower ATE than Twin V2 in this official table are: FAST-LIO2, FAST-LIVO2, LE-VINS, OpenVINS (Stereo).

No directly comparable proprioceptive-only external method in the audited i2Nav README table outperforms Twin V2, but this does not prove Pareto optimality because the table does not include every possible wheel-inertial method.

## Interpretation

Twin V2 is not as accurate as the strongest heavier exteroceptive systems such as LiDAR/IMU or LiDAR/visual/IMU methods, and it should not be described as odometry SOTA. Its value is that it achieves a usable official trajectory result while requiring only wheel/odometry and IMU at runtime, which supports the sensor-lightweight digital-twin framing.

The sensing-fidelity analysis should be used as supporting positioning, not as the main contribution. The main paper should still center on local/global fidelity, condition dependence, benign fidelity characterization, and asset-specific instantiation.

## Claim Tests

- Claim A: SUPPORTED_WITH_QUALIFICATION. Twin V2 provides usable and sometimes competitive trajectory fidelity using only wheel/odometry and IMU, but it does not beat the strongest LiDAR/visual systems.
- Claim B: SUPPORTED_WITH_QUALIFICATION. The results support a favorable sensing-fidelity tradeoff as contextual positioning, not a universal Pareto claim.
- Claim C: NOT_SUPPORTED. Pareto optimality is not established because sensing burden is componentwise and the audited benchmark table is not exhaustive.
- Claim D: NOT_SUPPORTED. The penalty for removing camera/LiDAR/radar/GNSS is not necessarily modest; several heavier methods have substantially lower ATE.
- Claim E: SUPPORTED. The runtime modality audit supports describing the proposed twin as sensor-lightweight.

## Recommendation

Include a compact sensing-fidelity table and a qualitative grouped plot only as context. Avoid a full leaderboard-style claim.
