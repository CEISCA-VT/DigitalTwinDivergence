# Digital Twin Divergence Project Handoff

This file is a compact context package for another LLM or collaborator. It
summarizes the project state, repo layout, hardware setup, packet schema,
completed work, current motion scripts, and the remaining research roadmap as
of July 15, 2026.

## One-Sentence Project Summary

This project builds and validates a security-aware digital twin for a tracked
UGV01 rover, using GPS/encoder/IMU telemetry to estimate rover state, detect
sensor attacks with an EKF + Mahalanobis detector, and map when attacks become
detectable or hidden under uncertainty, latency, terrain, and speed changes.

## Current Practical Status

- Week 1 is complete: UGV01 + BN220 GPS integration, timestamped `T:147`
  telemetry, packet health logging, clock-offset calibration, bench logging,
  and real-log digital-twin replay are implemented.
- Week 2 is complete enough for now: powered basic movement, straight/reverse
  tests, 90 degree turn tuning, short square tests, route-reference thinking,
  and calibration/protocol docs are in place. Official Waveshare geometry is
  being used as the locked nominal reference where possible.
- Week 3 is next: collect benign baseline data only, using repeatable square
  loops across speed, surface, and network-delay conditions. No attack trials
  yet.
- Main practical lesson: host-side HTTP commands work but timing is not
  perfectly precise. Slow, settled motion primitives are more repeatable than
  fast encoder-targeted rotations.
- Current surface plan: `smooth = kitchen floor`, `rough = permeable concrete`.

## Repository Map

Important paths:

```text
bench_logger.py
bench_logger_square_0_5m.py
bench_logger_square_1m.py
tests/test_bench_logger.py
ugv01_gps_dev/General_Driver/
DigitalTwin/
DigitalTwin/analysis/
docs/
raw_logs/static/
raw_logs/telemetry/
presentations/
```

Important docs:

```text
docs/hardware_arrival_roadmap.md
docs/preregistered_protocol.md
docs/running.md
docs/telemetry_protocol.md
docs/log_data_dictionary.md
docs/week2_completion_tests.md
docs/calibration_ready.md
docs/ugv01_esp32_bringup.md
```

Important analysis scripts:

```text
DigitalTwin/analysis/analyze_stationary.py
DigitalTwin/analysis/analyze_bench_telemetry.py
DigitalTwin/analysis/replay_hardware_log.py
DigitalTwin/analysis/review_hardware_replay.py
DigitalTwin/analysis/calibration_prep.py
DigitalTwin/analysis/threshold_lock.py
DigitalTwin/analysis/train_uncertainty.py
DigitalTwin/analysis/plot_detection.py
DigitalTwin/analysis/summarize_pd.py
```

## Hardware Setup

Hardware:

- Waveshare UGV01 tracked rover.
- BN220 GPS module.
- ESP32-based UGV01 firmware in `ugv01_gps_dev`.

Verified GPS wiring, kept unchanged:

```text
BN220 white wire -> UGV01 RX
BN220 red wire   -> UGV01 5V
BN220 black wire -> UGV01 GND
```

Do not suggest rewiring GPS unless the user explicitly asks. The original RX
path worked and is trusted.

## Firmware Commands And Packet Schema

Active firmware path:

```text
ugv01_gps_dev/General_Driver
```

Relevant command definitions are in:

```text
ugv01_gps_dev/General_Driver/json_cmd.h
ugv01_gps_dev/General_Driver/gps_ctrl.h
```

Key HTTP JSON commands:

```json
{"T":1,"L":0,"R":0}
```

Stop or speed command. `L` and `R` are the left/right track command channels.

```json
{"T":146}
```

GPS-only telemetry. Firmware feedback `T` is `1006`.

```json
{"T":147}
```

Combined telemetry. Firmware feedback `T` is `1007`.

```json
{"T":404,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"your_ssid","sta_password":"password"}
```

Set AP+STA Wi-Fi configuration.

### `T:146` GPS-Only Telemetry

Emitted fields include:

```text
T
seq
sample_ms
valid
age_ms
fix_type
lat
lon
sat
hdop
alt_m
speed_mps
course_deg
chars
sentences
failed_checksums
send_ms
```

Notes:

- `seq` increments from the shared telemetry sequence counter.
- `sample_ms` is firmware `millis()` when the sample is created.
- `send_ms` is firmware `millis()` right before JSON serialization/send.
- Invalid GPS rows should not poison position-variation metrics.

### `T:147` Combined Telemetry

Emitted fields include:

```text
T
seq
sample_ms
millis
L
R
enc_left
enc_right
v
r
p
y
ax
ay
az
gx
gy
gz
mx
my
mz
temp
gps_valid
gps_age_ms
gps_fix_type
gps_lat
gps_lon
gps_sat
gps_hdop
gps_alt_m
gps_speed_mps
gps_course_deg
gps_chars
gps_sentences
gps_failed_checksums
lat
lon
sat
hdop
send_ms
```

Interpretation:

- `seq`: firmware packet sequence.
- `sample_ms` / `millis`: firmware clock at sample.
- `send_ms`: firmware clock near response send.
- `L`, `R`: base track speed feedback/command fields from firmware.
- `enc_left`, `enc_right`: left/right encoder counts.
- `v`: battery/load voltage.
- `r`, `p`, `y`: roll, pitch, yaw in degrees.
- `ax..mz`: IMU/magnetometer channels.
- `gps_*`: namespaced GPS fields from TinyGPSPlus.
- `lat`, `lon`, `sat`, `hdop`: convenience duplicates for common GPS fields.

### Edge-Side Logger Fields

`bench_logger.py` augments firmware fields with edge timing and health fields:

```text
sample_idx
cycle_ok
error
t_wall_unix_s
t_cycle_start_ns
t_edge_send_ns
t_edge_rx_ns
t_edge_mid_ns
http_latency_ms
rover_millis_s
source_sample_time_s
edge_arrival_time_s
estimate_time_s
alarm_time_s
clock_offset_s
clock_calibrated
clock_calibration_samples
packet_loss_count
stale_packet
queue_depth
```

Primary edge-pipeline latency definition:

```text
estimate_time_s - source_sample_time_s
```

Fallback:

```text
edge_arrival_time_s - source_sample_time_s
```

## Digital Twin Stack

Core modules:

```text
DigitalTwin/kinematics.py
DigitalTwin/ekf.py
DigitalTwin/detector.py
DigitalTwin/uncertainty.py
DigitalTwin/latency.py
DigitalTwin/attack.py
DigitalTwin/simulator.py
```

Main implemented concepts:

- Tracked/differential-drive-compatible kinematics.
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

- Synthetic attacks: step bias, replay, freeze, random drift.
- Latency/jitter emulator with queue-depth reporting.
- Replay of real UGV01 logs through the digital-twin path.

## Completed Pre-Battery / Bench Package

Implemented outputs:

- Stationary GPS analysis over `raw_logs/static/*.csv`.
- Bench telemetry latency and packet-health analysis over
  `raw_logs/telemetry/*.csv`.
- Hardware replay review through the digital twin.
- Firmware flashing/wiring docs.
- Telemetry data dictionary.
- Pre-registered protocol package.
- Calibration-prep utilities for tracked-drive parameters.

Analysis commands:

```powershell
python -m DigitalTwin.analysis.analyze_stationary
python -m DigitalTwin.analysis.analyze_bench_telemetry
python -m DigitalTwin.analysis.review_hardware_replay
```

Known clean bench reference from July 9:

```text
379/379 successful cycles
0 packet drops
seq 0 -> 378
GPS valid
8 -> 11 satellites
HDOP 1.48 -> 0.92
replay detections: 0
max Mahalanobis below threshold
```

Older/early logs may contain warnings, missing GPS-valid rows, or less complete
timing fields. Treat the newest clean logs as the primary evidence.

## Motion Scripts And Current Behavior

### `bench_logger.py`

Current motion plan:

```text
MOTION_SCRIPT_ENABLED = True
MOTION_PLAN = "validation_triplet"
STOP_WHEN_MOTION_COMPLETE = True
SEQUENCE_REPEAT_COUNT = 3
```

The validation triplet sequence is:

```text
forward 1 m
reverse 1 m
clockwise 360
forward 1 m
clockwise 180
forward 1 m back to start
clockwise 180 restore heading
```

This sequence repeats 3 times and then stops/logs automatically.

Important control lesson:

- Fast encoder-targeted 360 degree turns overshot badly because HTTP polling
  cadence was around 0.7 to 1.0 s and the rover coasted after stop.
- Current 360 and 180 turns are composed from the slower validated timed
  quarter-turn primitive.
- A 360 is four separate settled 90 degree turns.
- A 180 is two separate settled 90 degree turns.

Current relevant constants in `bench_logger.py`:

```text
ENCODER_COUNTS_PER_METER = 5632.373
TURN_CW_CMD = (0.32, -0.32)             # legacy fast turn constant; avoid for precise turns
SQUARE_STRAIGHT_FORWARD_CMD = (-0.14, -0.14)
SQUARE_TURN_CW_CMD = (0.076, -0.076)
SQUARE_TURN_SECONDS = 1.65
SQUARE_SIDE_HOLD_SECONDS = 1.0
SQUARE_CORNER_HOLD_SECONDS = 3.0
SQUARE_STRAIGHT_WARMUP_CMD = (-0.08, -0.08)
SQUARE_STRAIGHT_WARMUP_SECONDS = 0.35
SQUARE_STRAIGHT_SECONDS = 4.2
```

### Square Scripts

Wrappers:

```powershell
python bench_logger_square_0_5m.py --repeats 3
python bench_logger_square_1m.py --repeats 3
```

Behavior:

- One initial hold.
- Continuous square loops.
- Four sides and four turns per loop.
- The fourth turn closes the square so the rover faces the original direction.
- Output filename includes the repeat count, e.g. `_x3_`.
- Protocol trial repetitions should still be separate CSVs. Example: five
  separate `--repeats 3` runs for five valid trials.

Recent empirical observation from the user:

- 0.5 m square loop works better than 1 m square.
- 5 to 10 degree drift per corner is acceptable for Week 3 if route remains
  repeatable.
- About 5 cm rightward endpoint drift after three continuous 0.5 m loops was
  considered acceptable baseline repeatability.

## Week 1 Status

Week 1 objectives:

- Integrate BN220 GPS with UGV01 ESP32.
- Verify encoder, IMU, GPS, Wi-Fi, timestamped telemetry.
- Finalize packet schema with source/sample, send, edge-arrival, queue,
  estimate, and alarm timestamps.
- Record packet loss, stale packets, and queue depth.
- Implement session-level clock-offset calibration.
- Validate edge-side delay/jitter emulator without changing rover behavior.

Status: complete.

Remaining caveat: field deployment depends on battery-powered motion, but the
bench/instrumentation portion is done.

## Week 2 Status

Week 2 objectives:

- Calibrate tracked-drive parameters.
- Validate GPS local coordinates and IMU yaw behavior.
- Run stationary-to-moving and basic route tests.
- Establish route-reference setup.
- Lock experimental protocol.

Status: complete enough to move into Week 3.

Locked nominal/reference values used in docs and scripts:

```text
rail center distance: 170 mm
single track width: 44 mm
minimum turning radius: 0 m
firmware nominal encoder scale: 6646.16 counts/m
stock UGV01 firmware motion-model track width: 0.141 m
physical rail-center hardware dimension: 0.170 m
current logger counts-per-meter: 5632.373
```

The project explicitly uses tracked-rover language:

```text
left_meters_per_tick
right_meters_per_tick
effective_track_width_m
heading_sign
```

Avoid saying wheel radius/wheelbase for UGV01 unless discussing generic theory.

## Week 3 Plan

Week 3 goal: benign baseline collection only. Do not run attack campaigns yet.

Locked surface labels:

```text
surface-smooth_kitchen_floor
surface-rough_permeable_concrete
```

Recommended route:

```text
square_0p5_x3
```

because 0.5 m square loops have been more repeatable than 1 m loops.

Baseline matrix:

```text
2 speeds x 2 surfaces x 2 network conditions x 5 trials = 40 valid benign runs
```

Conditions:

```text
speed-low
speed-medium
surface-smooth_kitchen_floor
surface-rough_permeable_concrete
latency-wifi_baseline
latency-wifi_buffered_delay
route-square0p5x3
attack-none
trial-1..5
```

Suggested filename pattern:

```text
speed-low_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-1.csv
```

Valid run criteria:

- Three square loops complete.
- No manual rescue or change of route semantics.
- Logs close normally.
- `seq` is plausible and packet health is not badly corrupted.
- GPS, encoder, IMU, and timing fields are populated enough for replay.
- No attack labels or intentional attacks.
- Endpoint drift stays within a chosen corridor, currently about +/-10 cm is a
  reasonable working target.

Invalid runs should be archived, excluded from threshold locking, and repeated.

## Network Conditions

Use two latency conditions:

```text
wifi_baseline
wifi_buffered_delay
```

The buffered-delay condition should be created edge-side with the existing
latency emulator. Do not alter rover firmware just to create delay.

Synthetic defaults in the project have used 10 ms baseline and 200 ms buffered
delay, with jitter around 20 percent for larger delays.

## What Remains After Week 3

Week 4:

- Replay all benign runs through the digital twin.
- Lock Mahalanobis threshold to satisfy:

```text
P_FA <= 0.05
```

- Save threshold metadata with dataset names and false-alarm estimate.

Week 5:

- Train or compare learned uncertainty model using only benign data.
- Inputs include residual, IMU variability, velocity variability, and packet
  timing stress.
- Compare adaptive covariance versus fixed covariance.

Week 6:

- Run attack campaign only after threshold locking.
- Attack types:

```text
step bias
replay
freeze
```

- Step magnitudes:

```text
0.5, 1, 2, 3, 5, 7.5, 10 m
```

Week 7:

- Generate final figures:

```text
epsilon_min over time/condition
lambda_max(S_k)
Mahalanobis vs threshold
P_D vs attack magnitude
ROC curves
detection delay
safe/warning/blind envelope timelines
adaptive vs fixed uncertainty comparison
stationary GPS/HDOP summaries
real hardware replay examples
```

Week 8:

- Write paper/results, clearly separating synthetic, stationary bench hardware,
  benign moving hardware, and attack hardware claims.

## Main Technical Risks And Lessons

- HTTP command timing is not hard real-time. Browser/manual controls are
  especially shaky for research-quality routes.
- Host-side stop commands can arrive late because the ESP32 web server is
  synchronous and telemetry requests may occupy the command path.
- Fast turn commands overshoot. Slow timed quarter turns are currently the most
  repeatable method.
- 0.5 m square x3 is currently more reliable than 1 m square x3.
- A small amount of drift is normal for a tracked rover, especially on smooth
  floors where the rover can slide after the encoder stops.
- If precision turns become a blocking issue, the cleaner future fix is a
  firmware-side duration/deadline command, for example conceptually:

```json
{"T":1,"L":0.076,"R":-0.076,"D":1650}
```

where the ESP32 itself stops after the duration. This has been discussed but
should not be added casually while collecting current baselines.

## Testing State

`tests/test_bench_logger.py` covers:

- Current 1 m encoder target.
- Initial hold behavior.
- Distance-step advance.
- Validation triplet sequence labels.
- 24 quarter-turn timed primitives over 3 repeated validation cycles.
- Square sequences with 4 sides and 4 corners.
- 1.65 s timed square corner behavior.
- Repeated square loops as continuous loops.

Known validation:

- Direct test execution has passed for the bench logger tests.
- `py_compile` has passed.
- `pytest` package may not be installed in the environment.
- `git diff --check` previously showed only line-ending warnings.

Useful commands:

```powershell
python -m py_compile bench_logger.py bench_logger_square_0_5m.py bench_logger_square_1m.py
python -c "import runpy; runpy.run_path('tests/test_bench_logger.py')"
python -m pytest -q
```

## Suggested Immediate Next Actions

1. Collect Week 3 benign baseline runs with `bench_logger_square_0_5m.py --repeats 3`.
2. Use kitchen floor for smooth and permeable concrete for rough.
3. Keep every trial as a separate CSV and record metadata in the filename or a manifest.
4. Run bench telemetry analysis and replay review after each batch.
5. Only after 40 valid benign runs, lock the detector threshold.
6. Only after threshold locking, run attacks.

## Short Advisor-Facing Summary

The infrastructure is no longer just conceptual. The UGV01 emits combined
GPS/encoder/IMU/base telemetry with firmware and edge timestamps. The bench
logger records packet loss, stale packets, queue depth, clock-offset estimates,
and latency metrics. Real UGV01 logs can be replayed through the digital twin,
and the detector outputs are populated. The rover can execute repeatable
low-risk square-loop motion, with the 0.5 m x3 loop selected as the current
Week 3 benign baseline route. The next milestone is collecting the 40-run
benign baseline matrix across speed, surface, and network conditions, then
locking the false-alarm threshold before attack trials.

