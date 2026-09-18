# TerraSentia/AIFARMS Frozen External Validation

## Scope

- Adapter conventions are frozen from the Phase-2 audit.
- No TerraSentia tuning, target-domain normalization, V2 weight changes, or EKF/RTK sign selection was performed.
- RTK is the primary positional reference. Dataset EKF is used only for secondary heading/yaw diagnostics.
- Motor and IMU provenance remains unresolved and is treated as a dataset limitation.

## Accepted Sequences

- `ts_2022_06_09_13h16m39s_one_row`: 1395 samples, 139.4 s, RTK path 68.3 m, motor/RTK distance ratio 1.058.
- `ts_2022_06_15_11h48m34s_four_rows`: 3274 samples, 327.3 s, RTK path 262.7 m, motor/RTK distance ratio 1.052.
- `ts_2022_09_01_11h20m00s_two_random`: 2352 samples, 235.1 s, RTK path 187.0 m, motor/RTK distance ratio 1.031.
- `ts_2022_09_01_12h32m56s_double_loop_corridor`: 2558 samples, 255.7 s, RTK path 225.8 m, motor/RTK distance ratio 0.981.
- `ts_2022_09_06_12h37m11s_four_rows`: 3598 samples, 359.7 s, RTK path 265.1 m, motor/RTK distance ratio 0.987.

## Physics-Only Positional Fidelity

- `ts_2022_06_09_13h16m39s_one_row`: ATE 32.712 m; RPE1/5/10 0.294/1.555/3.423 m; Dp p95/max 39.628/39.783 m.
- `ts_2022_06_15_11h48m34s_four_rows`: ATE 43.491 m; RPE1/5/10 0.146/1.015/2.538 m; Dp p95/max 66.662/75.945 m.
- `ts_2022_09_01_11h20m00s_two_random`: ATE 48.836 m; RPE1/5/10 0.240/1.491/3.706 m; Dp p95/max 108.223/122.580 m.
- `ts_2022_09_01_12h32m56s_double_loop_corridor`: ATE 49.904 m; RPE1/5/10 0.785/4.027/8.260 m; Dp p95/max 98.251/104.712 m.
- `ts_2022_09_06_12h37m11s_four_rows`: ATE 60.688 m; RPE1/5/10 0.319/1.687/3.522 m; Dp p95/max 92.259/100.119 m.

## Frozen V2 Portability

- `ts_2022_06_09_13h16m39s_one_row`: V2 median ATE 32.525 m (mean 32.521, SD 0.230); 80.0% of checkpoints improve physics ATE.
- `ts_2022_06_15_11h48m34s_four_rows`: V2 median ATE 42.527 m (mean 42.503, SD 0.200); 100.0% of checkpoints improve physics ATE.
- `ts_2022_09_01_11h20m00s_two_random`: V2 median ATE 48.298 m (mean 48.274, SD 0.346); 93.3% of checkpoints improve physics ATE.
- `ts_2022_09_01_12h32m56s_double_loop_corridor`: V2 median ATE 50.149 m (mean 50.215, SD 0.469); 26.7% of checkpoints improve physics ATE.
- `ts_2022_09_06_12h37m11s_four_rows`: V2 median ATE 60.181 m (mean 60.129, SD 0.335); 96.7% of checkpoints improve physics ATE.

## Yaw-Dominated Failure Attribution

- `ts_2022_06_09_13h16m39s_one_row`: replacing translation improves ATE by 4.833 m; replacing yaw improves by -12.458 m; yaw-dominant=`False`.
- `ts_2022_06_15_11h48m34s_four_rows`: replacing translation improves ATE by 1.570 m; replacing yaw improves by 23.408 m; yaw-dominant=`True`.
- `ts_2022_09_01_11h20m00s_two_random`: replacing translation improves ATE by 1.796 m; replacing yaw improves by -1.021 m; yaw-dominant=`False`.
- `ts_2022_09_01_12h32m56s_double_loop_corridor`: replacing translation improves ATE by 5.842 m; replacing yaw improves by -14.952 m; yaw-dominant=`False`.
- `ts_2022_09_06_12h37m11s_four_rows`: replacing translation improves ATE by 5.904 m; replacing yaw improves by 17.090 m; yaw-dominant=`True`.

## Local-Versus-Global Divergence

- Low finite-horizon RPE coexists with high global ATE in 16 evaluated rows.
- This supports reporting local synchronization and long-horizon global drift as separate fidelity dimensions.

## Conclusions

- Portability of the fidelity framework: supported. The same RTK-based evaluator runs across all accepted TerraSentia sequences without changing definitions.
- Portability of the frozen V2 maintenance mechanism: limited. It must be judged sequence-by-sequence because feature shift and yaw provenance are substantial.
- Local-versus-global divergence: recurs. Short-horizon RPE can remain comparatively low while global ATE/Dp grows large.
- Yaw-dominated drift: recurs where replacing yaw with the fused-reference yaw rate improves the diagnostic oracle much more than replacing translation.
- Limitation: unresolved TerraSentia motor and IMU provenance prevents treating poor global drift as purely a model failure.

## Figures

- `results\aifarms_terrasentia_full_study\ts_2022_06_09_13h16m39s_one_row\representative_trajectory.png`
- `results\aifarms_terrasentia_full_study\ts_2022_06_15_11h48m34s_four_rows\representative_trajectory.png`
- `results\aifarms_terrasentia_full_study\ts_2022_09_01_11h20m00s_two_random\representative_trajectory.png`
- `results\aifarms_terrasentia_full_study\ts_2022_09_01_12h32m56s_double_loop_corridor\representative_trajectory.png`
- `results\aifarms_terrasentia_full_study\ts_2022_09_06_12h37m11s_four_rows\representative_trajectory.png`
- `results\aifarms_terrasentia_full_study\local_vs_global_comparison.png`
- `results\aifarms_terrasentia_full_study\sequence_physics_vs_v2.png`
