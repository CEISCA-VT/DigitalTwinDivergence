# Digital Twin Divergence Roadmap

This roadmap reflects the current project state after UGV01 bench bring-up on
July 9, 2026. The software stack, embedded GPS telemetry, bench logger, timing
schema, and digital-twin replay path are now working. The remaining work is
mostly field power, tracked-rover calibration, real baseline collection,
threshold locking, learned uncertainty training, attack trials, and final
results writing.

## Current Completion Estimate

Overall project: about 60-65% complete.

By area:

- Synthetic digital-twin software: 85-90% complete.
- Embedded UGV01 GPS/base/IMU telemetry: 80-85% complete.
- Bench logging and timing instrumentation: 85-90% complete.
- Digital-twin replay of real bench logs: 75-80% complete.
- Tracked-rover calibration: 10-20% complete.
- Real field baseline dataset: 0-10% complete.
- Real attack campaign: 0% complete.
- Final paper/results: 25-35% complete.

The main external blocker is field power: batteries are still needed before
moving deployment, track calibration, baseline runs, and attack trials.

## Completed

### Core Digital Twin

- Differential-drive / tracked-drive-compatible kinematics scaffold.
- EKF prediction and GPS update.
- Mahalanobis innovation detector.
- Detectability bound:

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
- CSV logger, plotting, ROC, detection-probability summary, threshold-locking
  utility, and Random Forest uncertainty-training stub.
- Synthetic buffered latency / jitter emulator with queue-depth reporting.
- Full synthetic quick-run path verified.

### UGV01 Embedded Firmware

- `ugv01_gps_dev` contains the active embedded firmware.
- BN-220 GPS is integrated with `TinyGPSPlus`.
- GPS is kept on the original working UGV01 RX path:

```text
BN-220 white wire -> UGV01 RX
BN-220 red wire   -> UGV01 5V
BN-220 black wire -> UGV01 GND
```

- HTTP command `{"T":146}` returns GPS-only telemetry.
- HTTP command `{"T":147}` returns combined base, encoder, IMU, voltage, GPS,
  sequence, and firmware timing telemetry.
- `T:147` now includes:

```text
seq
sample_ms
send_ms
millis
L, R
enc_left, enc_right
voltage
IMU attitude and raw motion fields
gps_valid, gps_fix_type, lat, lon, sat, hdop
gps_chars, gps_sentences, gps_failed_checksums
```

### Bench Logging And Timing

- `bench_logger.py` logs `T:147` telemetry.
- The bench CSV includes:

```text
edge send timestamp
edge receive timestamp
edge midpoint timestamp
rover millis/sample time
session clock-offset estimate
clock calibration status
packet/drop count
stale-packet flag
queue depth
HTTP latency
GPS/base/IMU/encoder fields
```

- Successful bench run:

```text
raw_logs/telemetry/ugv_t147_telemetry_20260709_131731.csv
```

Run health:

```text
Rows: 379
Successful cycles: 379/379
Drops: 0
Seq: 0 -> 378, no gaps
GPS valid: true
Satellites: 8 -> 11
HDOP: 1.48 -> 0.92
GPS chars: 30152 -> 369862
Voltage: about 12.36 V
Encoders: 0/0, expected because rover was stationary
```

### Real Log Replay Into Digital Twin

- Real UGV01 bench logs can now be replayed through the EKF/detector pipeline:

```powershell
python -m DigitalTwin.analysis.replay_hardware_log raw_logs\telemetry\ugv_t147_telemetry_20260709_131731.csv --out DigitalTwin\datasets\hardware_bench\ugv_t147_telemetry_20260709_131731_digital_twin.csv
```

- The replayed CSV uses the standard digital-twin schema.
- Plotting and summary analysis accept the replayed hardware CSV.
- Hardware replay result for the successful bench run:

```text
Rows replayed: 379
Packet gaps: 0
Stale packets: 0
Detections: 0
Max Mahalanobis: 2.42 below threshold 5.99
Stationary GPS local span: about 7.1 m x 9.6 m
```

## What Is Left

### Blocked Until Batteries Arrive

1. Field-powered rover operation.
2. Straight-line tracked-drive calibration.
3. In-place turn calibration for effective track width.
4. Moving GPS/IMU/base validation.
5. Benign field baseline dataset.
6. Real attack campaign.

### Still Doable Before Batteries

1. Keep collecting stationary GPS logs in different placements:
   - indoors
   - window
   - outside if safe on bench power
2. Add a small stationary GPS summary script:
   - lat/lon local variance
   - HDOP distribution
   - satellite count distribution
   - checksum failure rate
3. Review the digital-twin replay plots from the bench log.
4. Confirm Arduino IDE can compile the current `ugv01_gps_dev` sketch.
5. Document the exact UGV01 firmware flashing procedure and board settings.

## Next Phase: Field Power And Safety Check

Goal: prove the rover can run under its own batteries without changing the
validated GPS wiring or telemetry protocol.

Tasks:

1. Install charged batteries.
2. Lift tracks off the ground for first powered command test.
3. Connect to UGV Wi-Fi.
4. Send stop command:

```powershell
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":1,"L":0,"R":0}'
```

5. Query base, IMU, GPS, and combined telemetry:

```powershell
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":130}'
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":126}'
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":146}'
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":147}'
```

6. Run a short stationary logger test:

```powershell
python bench_logger.py
```

Exit criteria:

- `T:147` returns continuously.
- `seq` increments without gaps.
- `gps_valid` is true outdoors or near sky view.
- Battery voltage looks healthy.
- IMU/yaw values update.
- Encoders remain stable while stationary.
- Stop command works.

## Phase 2: Tracked-Rover Calibration

Goal: estimate physical motion parameters for the tracked UGV01.

Use tracked-drive terms, not wheel-radius language. The important calibrated
parameters are:

```text
left_meters_per_tick
right_meters_per_tick
effective_track_width_m
heading_sign
```

Tasks:

1. Mark a straight 1-2 m path.
2. Drive slowly forward while logging `T:147`.
3. Compute left/right encoder-count deltas.
4. Estimate:

```text
left_meters_per_tick  = measured_distance_m / left_tick_delta
right_meters_per_tick = measured_distance_m / right_tick_delta
```

5. Repeat 3-5 times and average.
6. Run clockwise and counterclockwise in-place turns.
7. Estimate effective track width:

```text
effective_track_width_m = (right_distance_m - left_distance_m) / theta_rad
```

8. Update the digital-twin geometry/config.
9. Re-run straight and turn tests to verify.

Exit criteria:

- Encoder-derived distance roughly matches tape-measured distance.
- Turn direction and heading sign are correct.
- GPS local-frame conversion produces meter-scale displacement.
- Stationary GPS variance is known.

## Phase 3: Benign Baseline Dataset

Goal: collect clean nominal field data for threshold locking and uncertainty
training.

Target matrix:

```text
velocity: low, higher
terrain: smooth, rough
latency: baseline Wi-Fi, controlled buffered delay
trials: 5 benign trials per condition
trajectory: repeatable square or waypoint path
```

Initial minimum target:

```text
2 speeds * 2 terrains * 2 latency settings * 5 trials = 40 benign runs
```

Exit criteria:

- Every run has GPS, encoder, IMU, EKF, innovation, `S_k`, `Q_k`,
  `lambda_max_s`, `epsilon_min_m`, confidence, envelope region, packet timing,
  packet loss, stale flags, and queue depth.
- No attack labels appear in baseline data.
- Runs are named by speed, terrain, latency, and trial.

## Phase 4: Threshold Locking

Goal: lock the detector threshold from clean nominal field data.

Tasks:

1. Load benign baseline CSVs.
2. Compute nominal Mahalanobis distribution.
3. Choose `gamma_star` so empirical false-alarm probability satisfies:

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
- Baseline false alarm rate is measured from real UGV01 data.
- Attack trials use the locked threshold.

## Phase 5: Learned Uncertainty Estimator

Goal: replace or compare the deterministic covariance heuristic with a learned
uncertainty model trained only on benign data.

Inputs:

```text
r_k          dead-reckoning residual
sigma_IMU    rolling IMU vertical/yaw variability
sigma_v      rolling velocity variance
Delta t_k    edge-observed packet timing stress
```

Tasks:

1. Build training rows from benign baseline trials.
2. Train the Random Forest baseline.
3. Predict the diagonal of `Q_k`.
4. Compare learned adaptive EKF against fixed-covariance EKF.
5. Measure false alarm rate under terrain and latency changes.
6. Save model artifact and feature schema.

Exit criteria:

- Learned `g(.)` improves EKF consistency or false-alarm behavior, or the
  limitation is documented clearly.
- The learned model does not hide obvious attacks.
- Adaptive vs fixed covariance comparison is logged and plotted.

## Phase 6: Attack Campaign

Goal: empirically validate the detectability boundary.

Use the same operating matrix as the benign phase.

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

Exit criteria:

- Detection probability is computed for each attack size and condition.
- Detection delay is logged.
- False negatives are identified.
- Empirical detection probability is compared against `epsilon_min`.

## Phase 7: Figures And Results

Required outputs:

- `epsilon_min(v, tau, l)`.
- `lambda_max(S_k)` over time.
- Mahalanobis distance vs locked threshold.
- Empirical `P_D` vs attack magnitude.
- ROC curves.
- Detection delay by condition.
- Safe/warning/blind envelope timelines.
- Adaptive covariance vs fixed covariance false-alarm comparison.
- Stationary GPS noise and HDOP summary.
- Real hardware replay examples.

Exit criteria:

- You can answer whether velocity, terrain roughness, and latency expand
  `epsilon_min`.
- Blind regions correspond to weak detector conditions.
- Learned uncertainty is supported by evidence or honestly shown to be limited.

## Phase 8: Paper Writing

Write:

1. Hardware and telemetry setup.
2. Bench validation and GPS wiring.
3. Calibration procedure.
4. EKF and detectability formulation.
5. Threshold-locking method.
6. Learned uncertainty training method.
7. Attack methodology.
8. Detectability-bound validation.
9. Confidence-envelope results.
10. Limitations and failure cases.
11. Future work.

Exit criteria:

- Every claim is backed by a dataset, plot, equation, or logged artifact.
- The paper distinguishes synthetic validation, stationary bench hardware, and
  moving field hardware.
- Any mismatch between theory and real data is documented rather than hidden.

## Practical Order Of Operations

The correct remaining order is:

```text
batteries -> safety check -> track calibration -> moving validation
          -> benign baselines -> threshold locking -> learned uncertainty
          -> attacks -> detectability maps -> paper
```

Do not start attacks before calibration and benign baselines. Bad baseline data
will poison the threshold, the uncertainty model, and the attack conclusions.
