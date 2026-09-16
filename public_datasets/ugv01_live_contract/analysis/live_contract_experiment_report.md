# UGV01 Live Contract Experiment Summary

This report summarizes JSONL logs produced by the live UGV01 service-contract dashboard.
Unobservable service windows are retained as unobservable; they are not counted as successful contract satisfaction.

## Campaign Status

- Runs analyzed: 20
- Policies present: aoi-only, contract-aware, static-high, static-low
- Runs with any observable contract samples: 20
- Required final policy set: static-low, static-high, aoi-only, contract-aware

## Policy-Level Summary

| Physical condition | Wireless condition | Policy | Runs | Observable | Qualified | Withdrawn | p95 AoI (s) | Bytes/s | Requests/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smooth_floor_mixed_motion | wifi_baseline | aoi-only | 5 | 0.201 | 0.003 | 0.198 | 0.181 | 937.3 | 1.52 |
| smooth_floor_mixed_motion | wifi_baseline | contract-aware | 5 | 0.180 | 0.001 | 0.178 | 0.148 | 948.7 | 1.52 |
| smooth_floor_mixed_motion | wifi_baseline | static-high | 5 | 0.234 | 0.002 | 0.232 | 0.149 | 942.8 | 1.49 |
| smooth_floor_mixed_motion | wifi_baseline | static-low | 5 | 0.204 | 0.002 | 0.202 | 0.163 | 915.1 | 1.45 |

## Per-Service Summary

| Run | Policy | Service | Observable | Qualified | At risk | Withdrawn | Unobservable | p95 position (m) | p95 heading (deg) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial1 | aoi-only | Immediate motion | 0.007 | 0.000 | 0.000 | 0.007 | 0.993 | 0.013 | 7.810 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial1 | aoi-only | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial1 | aoi-only | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial1 | aoi-only | Global asset tracking | 0.681 | 0.014 | 0.000 | 0.667 | 0.319 | 6.492 | 138.535 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial2 | aoi-only | Immediate motion | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial2 | aoi-only | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial2 | aoi-only | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial2 | aoi-only | Global asset tracking | 0.919 | 0.000 | 0.000 | 0.919 | 0.081 | 1.742 |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial3 | aoi-only | Immediate motion | 0.014 | 0.000 | 0.000 | 0.014 | 0.986 | 0.948 | 44.209 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial3 | aoi-only | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial3 | aoi-only | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial3 | aoi-only | Global asset tracking | 0.830 | 0.000 | 0.000 | 0.830 | 0.170 | 7.094 | 44.209 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial4 | aoi-only | Immediate motion | 0.014 | 0.000 | 0.000 | 0.014 | 0.986 | 0.396 | 46.622 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial4 | aoi-only | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial4 | aoi-only | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial4 | aoi-only | Global asset tracking | 0.621 | 0.014 | 0.000 | 0.607 | 0.379 | 2.210 | 157.788 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial5 | aoi-only | Immediate motion | 0.064 | 0.007 | 0.000 | 0.057 | 0.936 | 0.467 | 9.022 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial5 | aoi-only | Short prediction | 0.007 | 0.000 | 0.000 | 0.007 | 0.993 | 0.578 | 86.099 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial5 | aoi-only | Planning support | 0.014 | 0.000 | 0.000 | 0.014 | 0.986 | 0.853 | 178.176 |
| smooth_floor_mixed_motion_wifi_baseline_aoi_only_trial5 | aoi-only | Global asset tracking | 0.850 | 0.021 | 0.000 | 0.829 | 0.150 | 2.197 | 173.546 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial1 | contract-aware | Immediate motion | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial1 | contract-aware | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial1 | contract-aware | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial1 | contract-aware | Global asset tracking | 0.843 | 0.000 | 0.000 | 0.843 | 0.157 | 1.899 | 0.000 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial2 | contract-aware | Immediate motion | 0.036 | 0.000 | 0.000 | 0.036 | 0.964 | 0.279 | 14.062 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial2 | contract-aware | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial2 | contract-aware | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial2 | contract-aware | Global asset tracking | 0.755 | 0.014 | 0.000 | 0.741 | 0.245 | 2.057 | 49.582 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial3 | contract-aware | Immediate motion | 0.050 | 0.000 | 0.000 | 0.050 | 0.950 | 0.391 | 124.989 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial3 | contract-aware | Short prediction | 0.029 | 0.000 | 0.000 | 0.029 | 0.971 | 0.729 | 166.984 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial3 | contract-aware | Planning support | 0.022 | 0.000 | 0.000 | 0.022 | 0.978 | 0.389 | 68.883 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial3 | contract-aware | Global asset tracking | 0.353 | 0.014 | 0.000 | 0.338 | 0.647 | 1.859 | 148.841 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial4 | contract-aware | Immediate motion | 0.014 | 0.000 | 0.000 | 0.014 | 0.986 | 0.148 | 0.026 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial4 | contract-aware | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial4 | contract-aware | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial4 | contract-aware | Global asset tracking | 0.669 | 0.000 | 0.000 | 0.669 | 0.331 | 2.956 | 0.025 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial5 | contract-aware | Immediate motion | 0.058 | 0.000 | 0.000 | 0.058 | 0.942 | 0.399 | 98.871 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial5 | contract-aware | Short prediction | 0.022 | 0.000 | 0.000 | 0.022 | 0.978 | 1.642 | 136.281 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial5 | contract-aware | Planning support | 0.014 | 0.000 | 0.000 | 0.014 | 0.986 | 0.654 | 105.883 |
| smooth_floor_mixed_motion_wifi_baseline_contract_aware_trial5 | contract-aware | Global asset tracking | 0.732 | 0.000 | 0.014 | 0.717 | 0.268 | 2.290 | 168.551 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial1 | static-high | Immediate motion | 0.044 | 0.007 | 0.000 | 0.036 | 0.956 | 0.371 | 25.660 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial1 | static-high | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial1 | static-high | Planning support | 0.022 | 0.000 | 0.000 | 0.022 | 0.978 | 1.752 | 117.419 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial1 | static-high | Global asset tracking | 0.577 | 0.000 | 0.000 | 0.577 | 0.423 | 2.245 | 143.946 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial2 | static-high | Immediate motion | 0.065 | 0.000 | 0.000 | 0.065 | 0.935 | 0.688 | 25.857 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial2 | static-high | Short prediction | 0.029 | 0.000 | 0.000 | 0.029 | 0.971 | 1.852 | 31.842 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial2 | static-high | Planning support | 0.007 | 0.000 | 0.000 | 0.007 | 0.993 | 2.682 | 90.996 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial2 | static-high | Global asset tracking | 0.768 | 0.000 | 0.000 | 0.768 | 0.232 | 4.013 | 94.983 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial3 | static-high | Immediate motion | 0.073 | 0.007 | 0.000 | 0.066 | 0.927 | 0.632 | 32.774 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial3 | static-high | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial3 | static-high | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial3 | static-high | Global asset tracking | 0.956 | 0.007 | 0.007 | 0.942 | 0.044 | 6.879 | 96.105 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial4 | static-high | Immediate motion | 0.080 | 0.000 | 0.000 | 0.080 | 0.920 | 0.698 | 25.369 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial4 | static-high | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial4 | static-high | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial4 | static-high | Global asset tracking | 0.927 | 0.007 | 0.000 | 0.920 | 0.073 | 3.212 | 159.126 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial5 | static-high | Immediate motion | 0.103 | 0.000 | 0.000 | 0.103 | 0.897 | 1.220 | 92.943 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial5 | static-high | Short prediction | 0.044 | 0.000 | 0.000 | 0.044 | 0.956 | 2.338 | 149.803 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial5 | static-high | Planning support | 0.007 | 0.000 | 0.000 | 0.007 | 0.993 | 3.308 | 81.536 |
| smooth_floor_mixed_motion_wifi_baseline_static_high_trial5 | static-high | Global asset tracking | 0.978 | 0.007 | 0.000 | 0.971 | 0.022 | 8.821 | 149.750 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial1 | static-low | Immediate motion | 0.082 | 0.000 | 0.000 | 0.082 | 0.918 | 0.576 | 20.746 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial1 | static-low | Short prediction | 0.033 | 0.000 | 0.000 | 0.033 | 0.967 | 1.103 | 3.503 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial1 | static-low | Planning support | 0.016 | 0.000 | 0.000 | 0.016 | 0.984 | 1.283 | 3.970 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial1 | static-low | Global asset tracking | 0.418 | 0.000 | 0.000 | 0.418 | 0.582 | 1.877 | 157.318 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial2 | static-low | Immediate motion | 0.070 | 0.000 | 0.000 | 0.070 | 0.930 | 0.727 | 145.914 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial2 | static-low | Short prediction | 0.031 | 0.000 | 0.000 | 0.031 | 0.969 | 0.864 | 112.982 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial2 | static-low | Planning support | 0.031 | 0.000 | 0.000 | 0.031 | 0.969 | 0.855 | 166.523 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial2 | static-low | Global asset tracking | 0.876 | 0.000 | 0.008 | 0.868 | 0.124 | 4.395 | 174.741 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial3 | static-low | Immediate motion | 0.017 | 0.000 | 0.000 | 0.017 | 0.983 | 0.518 | 15.592 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial3 | static-low | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial3 | static-low | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial3 | static-low | Global asset tracking | 0.650 | 0.000 | 0.000 | 0.650 | 0.350 | 2.275 | 120.464 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial4 | static-low | Immediate motion | 0.024 | 0.000 | 0.000 | 0.024 | 0.976 | 0.372 | 23.645 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial4 | static-low | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial4 | static-low | Planning support | 0.016 | 0.000 | 0.000 | 0.016 | 0.984 | 1.100 | 165.674 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial4 | static-low | Global asset tracking | 0.789 | 0.024 | 0.000 | 0.764 | 0.211 | 1.742 | 169.401 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial5 | static-low | Immediate motion | 0.051 | 0.000 | 0.000 | 0.051 | 0.949 | 0.494 | 38.692 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial5 | static-low | Short prediction | 0.037 | 0.000 | 0.000 | 0.037 | 0.963 | 1.760 | 119.864 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial5 | static-low | Planning support | 0.015 | 0.000 | 0.000 | 0.015 | 0.985 | 0.623 | 137.355 |
| smooth_floor_mixed_motion_wifi_baseline_static_low_trial5 | static-low | Global asset tracking | 0.926 | 0.007 | 0.000 | 0.919 | 0.074 | 4.347 | 167.923 |

## Interpretation Rules

- `qualified` and `at_risk` are observable service windows that remain within the declared contract.
- `withdrawn` means an observable contract exceeded position, heading, or AoI limits.
- `unobservable` usually means GPS position/course quality was unavailable for that service.
- The final IoT-J claim requires repeated matched runs across all four policies, not a single dashboard smoke test.
