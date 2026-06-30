# Digital Twin Divergence Roadmap

This roadmap reflects the current codebase state: the EKF, telemetry packet,
attack injector, eigenvalue detectability math, confidence envelopes, logging,
plotting, and synthetic experiment runner exist. The remaining work is now
mostly integration, calibration, learned uncertainty estimation, and empirical
validation.

## Current Status

Already implemented:

- Packed Arduino/Python telemetry protocol with CRC.
- Python telemetry deserializer and UDP receiver.
- Differential-drive kinematics.
- EKF prediction and GPS update.
- Mahalanobis innovation detector.
- Proposal-faithful eigenvalue bound:

```text
epsilon_min = sqrt(lambda_star * lambda_max(S_k))
```

- Instantaneous stealth bound:

```text
epsilon_stealth_max = sqrt(gamma_star * lambda_max(S_k))
```

- Rolling uncertainty feature contract:

```text
Q_k = g(r_k, sigma_IMU, sigma_v, Delta t_k)
```

- Step, replay, freeze, and random-drift attack injection.
- CSV logging and plotting.
- Quick synthetic experiment automation.

Latest Phase 0 additions:

- Synthetic buffered latency now changes edge timing stress, `Q_k`, `S_k`, and
  `epsilon_min`.
- The experiment runner supports the full `2^3` matrix.
- Step-bias sweeps from `0.5 m` to `10 m` are implemented.
- Empirical `P_D` summaries are implemented.
- ROC and detection-probability plotting are implemented.
- Threshold locking from nominal CSVs is implemented.
- A Random Forest uncertainty-training pipeline stub is implemented.

## Phase 0: Before UGV01 Arrives

Goal: make the software pipeline stronger while hardware is unavailable.

Tasks:

1. Use the synthetic tools to run a longer full-matrix rehearsal.
2. Review the generated `P_D`, ROC, and threshold-locking outputs for sanity.
3. Decide whether the confidence-envelope heuristic needs a clearer formula
   before real experiments.
4. Keep equations frozen unless a real data failure proves a specific change is
   necessary.

Exit criteria:

- Synthetic latency visibly increases `lambda_max(S_k)` and `epsilon_min`.
- Synthetic step sweeps show detection probability increasing with attack size.
- Full synthetic pipeline can run without manual file edits.

## Phase 1: Hardware Bring-Up

Goal: prove the UGV01, GPS, encoders, IMU, Arduino, and Python receiver can
communicate reliably.

Tasks:

1. Assemble the UGV01.
2. Wire the BN-220 GPS.
3. Wire wheel encoder and IMU signals.
4. Verify voltage levels and common ground.
5. Flash the telemetry firmware.
6. Start the receiver:

```powershell
python -m DigitalTwin.telemetry_receiver --port 5005
```

Exit criteria:

- CRC-valid packets arrive continuously.
- `seq` increments normally.
- Encoder ticks change with wheel motion.
- GPS fix type, satellites, HDOP, latitude, and longitude are plausible.
- IMU readings are stable when stationary and respond to motion.

## Phase 2: Calibration

Goal: make all physical units defensible before running research trials.

Tasks:

1. Measure wheel radius.
2. Measure wheel base.
3. Confirm encoder ticks per revolution.
4. Update `DifferentialDriveGeometry`.
5. Run straight-line motion over a measured distance.
6. Run in-place turns and verify heading sign.
7. Collect 10-15 minutes of stationary GPS data.
8. Estimate baseline GPS noise and HDOP behavior.

Exit criteria:

- Encoder distance roughly matches tape-measured distance.
- Turn direction and heading sign are correct.
- GPS local-frame conversion produces meter-scale displacement correctly.
- Stationary GPS variance is known.

## Phase 3: Benign Baseline Dataset

Goal: collect clean nominal data for threshold locking and uncertainty training.

Run the proposal's `2^3` matrix:

```text
velocity: 0.2 m/s, 0.8 m/s
terrain: smooth, rough
latency: baseline Wi-Fi, 200 ms buffered
trials: 5 benign trials per condition
trajectory: 2 m x 2 m square
```

Total benign runs:

```text
2 * 2 * 2 * 5 = 40 runs
```

Exit criteria:

- Every CSV contains GPS, encoder, IMU, EKF, innovation, `S_k`, `Q_k`,
  `lambda_max_s`, `epsilon_min_m`, confidence, envelope region, and packet
  timing.
- No attack labels appear in baseline data.
- Runs are named by speed, terrain, latency, and trial.

## Phase 4: Threshold Locking

Goal: replace the theoretical detector threshold with an empirical threshold
from clean nominal data.

Tasks:

1. Load all benign baseline CSVs.
2. Compute the nominal Mahalanobis distribution.
3. Choose `gamma_star` so empirical false alarm probability satisfies:

```text
P_FA <= 0.05
```

4. Save threshold metadata:

```text
threshold value
date
dataset names
number of samples
false alarm estimate
```

Exit criteria:

- A locked threshold file exists.
- The baseline false alarm rate is measured, not assumed.
- Future attack trials use the locked threshold.

## Phase 5: Learned Uncertainty Estimator

Goal: replace the deterministic placeholder with the proposal's learned
self-calibrating uncertainty model.

Inputs:

```text
r_k          dead-reckoning residual
sigma_IMU    rolling IMU vertical/yaw variability
sigma_v      rolling velocity variance
Delta t_k    packet inter-arrival time
```

Tasks:

1. Build training rows from benign baseline trials.
2. Train a Random Forest regressor first.
3. Predict the diagonal of `Q_k`.
4. Compare learned adaptive EKF against fixed-covariance EKF.
5. Measure false alarm rate under terrain and latency changes.
6. Save model artifact and feature schema.

Exit criteria:

- Learned `g(.)` improves EKF consistency or false alarm behavior.
- The model does not hide obvious attacks.
- Adaptive vs fixed covariance comparison is logged and plotted.

## Phase 6: Attack Campaign

Goal: empirically validate the detectability boundary.

Use the same `2^3` operating matrix as the benign phase.

Attack types:

```text
step bias
telemetry replay
coordinate freeze
```

Step-bias magnitudes:

```text
0.5 m, 1 m, 2 m, 3 m, 5 m, 7.5 m, 10 m
```

Attack start time:

```text
t = 30 s
```

Exit criteria:

- Detection probability is computed for each attack size and condition.
- Detection delay is logged.
- False negatives are identified.
- Empirical detection probability is compared against `epsilon_min`.

## Phase 7: Detectability Maps

Goal: produce the figures that prove or disprove the core hypotheses.

Required plots:

- `epsilon_min(v, tau, l)`.
- `lambda_max(S_k)` over time.
- Mahalanobis distance vs locked threshold.
- Empirical `P_D` vs attack magnitude.
- ROC curves.
- Detection delay by condition.
- Safe/warning/blind envelope timelines.
- Adaptive covariance vs fixed covariance false alarm comparison.

Exit criteria:

- You can clearly answer whether velocity, terrain roughness, and latency
  expand `epsilon_min`.
- Blind regions correspond to weak detector conditions.
- Learned uncertainty is either supported by evidence or honestly shown to be
  limited.

## Phase 8: Results and Paper Writing

Goal: turn experiment logs into defensible research claims.

Write:

1. Hardware and telemetry setup.
2. Calibration procedure.
3. EKF and detectability formulation.
4. Threshold-locking method.
5. Learned uncertainty training method.
6. Attack methodology.
7. Detectability-bound validation.
8. Confidence-envelope results.
9. Limitations and failure cases.
10. Future work.

Exit criteria:

- Every claim is backed by a dataset, plot, or equation.
- The paper distinguishes theory, synthetic validation, and real hardware
  evidence.
- Any mismatch between theory and real data is documented rather than hidden.

## Practical Order of Operations

Do not start with attacks when the hardware arrives. The correct order is:

```text
packets -> calibration -> benign baselines -> threshold locking
        -> learned uncertainty -> attacks -> detectability maps -> paper
```

The project becomes much easier if the baseline data is clean. Bad baseline data
will poison the threshold, the uncertainty model, and the attack conclusions.
