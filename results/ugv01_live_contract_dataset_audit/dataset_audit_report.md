# UGV01 Live-Contract Dataset Audit

## Audit Verdict

The dataset is structurally complete and suitable for demonstrating a live UGV01 digital-twin contract prototype. It does not yet support the stronger causal claim that the contract-aware policy preserves service satisfaction while reducing delivered communication cost relative to the baselines.

## Structural Quality

- Balanced matrix: 4 policies x 5 trials = 20 runs.
- Complete video/telemetry/JSONL triplets: 20/20.
- JSONL parse errors: 0.
- JSONL records: 2702 across 30.1 minutes of live traces.
- Total unexplained sequence gaps: 7 (0.258% of expected sequence steps).
- Distinct telemetry schemas: 1; distinct JSON record schemas: 1.
- Duplicate file-content groups across trial assets: 0.
- Physical conditions: smooth_floor_mixed_motion.
- Wireless conditions: wifi_baseline.

## Policy-Level Evidence

| Policy | Runs | Requested rate (Hz) | Delivered rate (Hz) | Bytes/s | p95 AoI (s) | Observable | Qualified | Withdrawn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aoi-only | 5 | 2.77 | 1.52 | 935.8 | 0.181 | 0.201 | 0.003 | 0.198 |
| contract-aware | 5 | 8.61 | 1.52 | 948.7 | 0.148 | 0.180 | 0.001 | 0.178 |
| static-high | 5 | 10.00 | 1.49 | 942.8 | 0.149 | 0.234 | 0.002 | 0.232 |
| static-low | 5 | 2.00 | 1.41 | 886.4 | 0.163 | 0.204 | 0.002 | 0.202 |

Requested policy rates differ strongly, but the mean delivered-rate spread is only 0.111 Hz. The rover/HTTP stream therefore acted as the bottleneck and did not realize the requested 2/5/10 Hz policy actions.

## Service Observability

| Policy | Service | Observable | Satisfied given observable | Withdrawn | Main reason for missing evaluation |
|---|---|---:|---:|---:|---|
| aoi-only | global_state_tracking | 0.780 | 0.014 | 0.770 | GPS course unobservable at low speed |
| aoi-only | local_10s_preview | 0.003 | 0.000 | 0.003 | GPS course unobservable at low speed |
| aoi-only | local_1s_tight | 0.020 | 0.028 | 0.019 | GPS course unobservable at low speed |
| aoi-only | local_5s_moderate | 0.001 | 0.000 | 0.001 | GPS course unobservable at low speed |
| contract-aware | global_state_tracking | 0.670 | 0.016 | 0.662 | GPS course unobservable at low speed |
| contract-aware | local_10s_preview | 0.007 | 0.000 | 0.007 | GPS course unobservable at low speed |
| contract-aware | local_1s_tight | 0.032 | 0.000 | 0.032 | GPS course unobservable at low speed |
| contract-aware | local_5s_moderate | 0.010 | 0.000 | 0.010 | GPS course unobservable at low speed |
| static-high | global_state_tracking | 0.841 | 0.006 | 0.835 | GPS course unobservable at low speed |
| static-high | local_10s_preview | 0.007 | 0.000 | 0.007 | GPS course unobservable at low speed |
| static-high | local_1s_tight | 0.073 | 0.053 | 0.070 | GPS course unobservable at low speed |
| static-high | local_5s_moderate | 0.015 | 0.000 | 0.015 | GPS course unobservable at low speed |
| static-low | global_state_tracking | 0.732 | 0.010 | 0.724 | GPS course unobservable at low speed |
| static-low | local_10s_preview | 0.016 | 0.000 | 0.016 | GPS course unobservable at low speed |
| static-low | local_1s_tight | 0.049 | 0.000 | 0.049 | GPS course unobservable at low speed |
| static-low | local_5s_moderate | 0.020 | 0.000 | 0.020 | GPS course unobservable at low speed |

GPS validity is high, but GPS course is usually unavailable at the rover's low speed. This makes the 1 s, 5 s, and 10 s position-plus-heading contracts mostly unobservable. The global contract is evaluated more often, but it is usually withdrawn because GPS-to-twin disagreement exceeds the frozen tolerance.

## Claims Supported Now

- The complete stack ran online on physical UGV01 hardware and produced reproducible sensor, twin, contract, policy, and resource traces.
- The four frozen policies produce different requested rates and resource-mode decisions.
- The system explicitly marks services unobservable when the operational reference cannot support evaluation.
- Five repetitions per policy support descriptive variability estimates for this smooth-floor, baseline-Wi-Fi condition.

## Claims Not Supported By This Dataset Alone

- Contract-aware operation reduces delivered request rate, bytes, latency, or energy relative to static-high.
- Contract-aware operation preserves service satisfaction better than static-low or AoI-only.
- The result generalizes across surfaces, motion regimes, or wireless conditions.
- GPS-referenced online disagreement is independent physical ground truth.
- The paired videos provide independent-reference source material, but they are not synchronized AprilTag ground-truth trajectories in this package yet.

## Paper Readiness

Use this dataset as a live prototype and policy-execution study in the paper. Pair it with the existing AprilTag physical-validation results and the i2Nav condition/fidelity analyses. Describe resource adaptation using requested rates and policy states. Do not describe the near-identical delivered rates as demonstrated communication savings.

For the stronger policy-effectiveness claim, the minimum additional evidence is a controlled transport or firmware path that actually delivers distinct update rates, followed by the same four policies under matched motion. Improving low-speed heading observability is also required if all finite-horizon live contracts remain part of the claim.
