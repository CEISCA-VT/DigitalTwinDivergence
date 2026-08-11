# Running the Project

This guide covers the current hardware, replay, analysis, and reproduction
workflows.

## Environment

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest -q
```

## Synthetic Experiments

Quick smoke test:

```powershell
python -m DigitalTwin.experiments.experiment --quick
```

Full benign `2^3` matrix:

```powershell
python -m DigitalTwin.experiments.experiment --full-matrix --nominal-only
```

Full matrix with step-bias sweep:

```powershell
python -m DigitalTwin.experiments.experiment --full-matrix --step-sweep
```

Shorter rehearsal runs can use:

```powershell
python -m DigitalTwin.experiments.experiment --full-matrix --step-sweep --trials 1 --duration 20
```

Generated datasets are written under `DigitalTwin/datasets/` and are ignored by
Git.

## Plotting

Plot a single CSV:

```powershell
python -m DigitalTwin.plotting.plot DigitalTwin/datasets/speed-0.20_terrain-0.0_latency-10_attack-step_eps-5.0_trial-0.csv
```

Plots are written under `DigitalTwin/datasets/plots/` or the analysis output
directory and are ignored by Git.

## Analysis

Summarize empirical detection probability:

```powershell
python -m DigitalTwin.analysis.summarize_pd "DigitalTwin/datasets/*.csv"
```

Lock the detector threshold from nominal runs:

```powershell
python -m DigitalTwin.analysis.threshold_lock "DigitalTwin/datasets/*attack-none*.csv"
```

Generate ROC and detection-probability plots:

```powershell
python -m DigitalTwin.analysis.plot_detection "DigitalTwin/datasets/*.csv"
```

Train the GPS-independent uncertainty candidate from the canonical benign-run
manifest:

```powershell
python -m DigitalTwin.analysis.train_uncertainty
```

The command uses complete-run grouped validation and records whether the model
beats the median target baseline. A rejected candidate remains disabled. The
locked threshold and trained model artifacts are ignored by Git because they
are generated from datasets.

Run the complete real-data attack campaign or regenerate its report from
existing artifacts:

```powershell
python -m DigitalTwin.analysis.real_data_study
python -m DigitalTwin.analysis.real_data_study --summarize-existing
```

Regenerate the complete paper-facing package with the primary and expanded
campaigns, revised mathematical diagnostics, statistical summaries, threshold
sweep, covariance analysis, runtime benchmark, and figures:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\regenerate_all_results.ps1
```

This is the long-running reproduction command. It copies the final
paper-facing artifacts to `results/` after every analysis succeeds.

Run the targeted paired covariance-poisoning analysis or regenerate its
statistics from the completed targeted replay:

```powershell
python -m DigitalTwin.analysis.covariance_poisoning
python -m DigitalTwin.analysis.covariance_poisoning --summarize-existing
```

## Pre-Battery Wrap-Up

Summarize stationary logs:

```powershell
python -m DigitalTwin.analysis.analyze_stationary
```

Summarize bench telemetry latency and packet health:

```powershell
python -m DigitalTwin.analysis.analyze_bench_telemetry
```

Replay bench telemetry logs through the digital twin and review consistency:

```powershell
python -m DigitalTwin.analysis.review_hardware_replay
```

Prepare tracked-rover calibration artifacts after the first powered runs:

```powershell
python -m DigitalTwin.analysis.calibration_prep straight raw_logs\telemetry\my_run.csv --distance-m 2.0 --out-prefix DigitalTwin\datasets\analysis\calibration\straight_run_01
python -m DigitalTwin.analysis.calibration_prep turn raw_logs\telemetry\my_turn.csv --turn-angle-deg 180 --left-distance-m -0.62 --right-distance-m 0.61 --out-prefix DigitalTwin\datasets\analysis\calibration\turn_run_01
python -m DigitalTwin.analysis.calibration_prep imu raw_logs\telemetry\my_turn.csv --expected-heading-change-deg 180 --out-prefix DigitalTwin\datasets\analysis\calibration\imu_turn_01
python -m DigitalTwin.analysis.calibration_prep gps raw_logs\telemetry\my_run.csv --out-prefix DigitalTwin\datasets\analysis\calibration\gps_run_01
python -m DigitalTwin.analysis.calibration_prep route-template --out-prefix DigitalTwin\datasets\analysis\calibration\route_reference_template
```

## Week 3 Square Loops

Run three continuous loops of either calibrated square route:

```powershell
python bench_logger_square_0_5m.py --surface smooth --speed low --network baseline --trial 1
python bench_logger_square_1m.py --surface smooth --speed low --network baseline --trial 1
```

The runner performs one initial hold, closes every loop with a fourth corner so
the rover finishes on its starting heading, and stops logging automatically
after the final hold. The loop count is part of the output filename. Protocol trial repetitions remain separate runs;
for example, collect five separate `--repeats 3` files for five benign trials.

The square runners are now set up for benign dataset collection. Defaults are:

```text
--repeats 3
--surface smooth
--speed low
--network baseline
attack_type metadata recorded as none
```

Surface profiles:

```text
smooth -> smooth_kitchen_floor, current 90 degree turn calibration
rough  -> rough_permeable_concrete, per-corner schedule 1.80, 1.90, 1.75, 1.80 s
```

Speed profiles:

```text
low    -> current low-speed square command
medium -> medium-speed square command; safety-check before formal trials
```

Network labels:

```text
baseline -> wifi_baseline
buffered -> wifi_buffered_delay
```

Example Week 3 benign commands:

```powershell
python bench_logger_square_0_5m.py --surface smooth --speed low --network baseline --trial 1
python bench_logger_square_0_5m.py --surface smooth --speed medium --network baseline --trial 1
python bench_logger_square_0_5m.py --surface rough --speed low --network baseline --trial 1
python bench_logger_square_0_5m.py --surface rough --speed medium --network buffered --trial 1
```

Generated filenames include the run labels, for example:

```text
speed-low_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-1_YYYYMMDD_HHMMSS.csv
```

Each row also records dataset metadata fields such as `run_id`, `route`,
`surface`, `surface_profile`, `speed_label`, `network_condition`, `trial_id`,
`attack_type`, `square_turn_profile`, `square_turn_seconds`, and the square
straight/turn commands used for that run.

If the rough surface needs a different 90 degree corner setting after the first
permeable-concrete tuning run, override it without editing firmware:

```powershell
python bench_logger_square_0_5m.py --surface rough --speed low --network baseline --trial 1 --turn-seconds 1.72
python bench_logger_square_0_5m.py --surface rough --speed low --network baseline --trial 1 --turn-left 0.078 --turn-right -0.078
```

For the July 20 rough-concrete adjustment, the rough profile keeps the first
corner lower than the good second turn, keeps the second turn, and reduces the
third turn after the latest outdoor run:

```text
corner turn schedule: 1.80, 1.90, 1.75, 1.80 seconds
```

You can test a one-off schedule without editing code:

```powershell
python bench_logger_square_0_5m.py --surface rough --speed low --network baseline --trial 1 --turn-schedule 1.80,1.90,1.75,1.80
```

For rough permeable-concrete collection after the July 20 accepted tuning run,
use the rough dataset collector. It pauses between trials by default so the
rover can be reset and the test area can be cleared:

```powershell
python collect_rough_dataset.py --speeds low --networks baseline --start-trial 5 --trials 1
```

To collect five low-speed baseline trials from trial 1 through trial 5:

```powershell
python collect_rough_dataset.py --speeds low --networks baseline --start-trial 1 --trials 5
```

To collect both low and medium rough baseline sets:

```powershell
python collect_rough_dataset.py --speeds low medium --networks baseline --start-trial 1 --trials 5
```

The collector uses the accepted rough settings unless overridden:

```text
turn command: 0.064, -0.064
turn schedule: 2.10, 2.10, 2.10, 2.10 seconds
```

Use `--dry-run` to print the queued commands without moving the rover.
Use `--auto` only when the area is controlled and it is safe to start each
trial after a countdown.

## Visual Digital-Twin Dashboard

Start the local replay dashboard from the repository root:

```powershell
python -m DigitalTwin.dashboard.server
```

Open `http://127.0.0.1:8765`, or launch the browser automatically:

```powershell
python -m DigitalTwin.dashboard.server --open
```

The first selection takes a few seconds while the accepted hardware log is
replayed through the revised digital twin. Replays are cached in memory after
loading. The interface provides:

- synchronized BN220 GPS, GPS-fused operational EKF, and GPS-independent
  security-predictor trajectory trails
- bias-corrected encoder/IMU yaw fusion and an encoder-IMU slip indicator
- current position, heading direction, and encoder-derived velocity
- GNSS satellite/HDOP, IMU, motor, voltage, and edge-transport state
- operational EKF-to-GPS agreement and security-branch NIS histories
- packet loss, stale-packet, queue, and latency summaries
- play, pause, step, scrub, and playback-speed controls

The displayed EKF-to-GPS distance is an internal sensor-agreement measure. It
must not be reported as physical localization error until an independent
camera/AprilTag trajectory is available.

Generate the complete benign-log agreement and consistency report with:

```powershell
python -m DigitalTwin.analysis.digital_twin_accuracy
```

## Hardware Receiver

When the UGV01 is streaming packets over UDP:

```powershell
python -m DigitalTwin.telemetry_receiver --port 5005
```

Before collecting research data, verify:

- CRC-valid packets arrive continuously.
- `seq` increments without unexplained drops.
- Encoder ticks change with wheel motion.
- GPS fix type, satellites, HDOP, latitude, and longitude are plausible.
- IMU readings are stable when stationary and respond to motion.

## Suggested Pre-Hardware Final Rehearsal

Run:

```powershell
python -m pytest -q
python -m DigitalTwin.experiments.experiment --full-matrix --step-sweep --trials 1 --duration 20
python -m DigitalTwin.analysis.summarize_pd "DigitalTwin/datasets/*.csv"
python -m DigitalTwin.analysis.plot_detection "DigitalTwin/datasets/*.csv"
```

If those pass, stop expanding scope until hardware arrives.
