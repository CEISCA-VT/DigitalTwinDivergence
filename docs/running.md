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

## AprilTag Carpet Development Calibration

The August 12 carpet recording uses a measured 2.0 m by 1.0 m reference
rectangle: ID 1 bottom-left, ID 2 bottom-right, ID 3 top-right, and ID 4
top-left. The video was recorded in the phone's 1x landscape mode, so use
`camera_calibration_landscape`, not the 0.6x calibration.

The dense tracking result covers 97.4% of the video. Apply the measured
8.112 cm rover-tag elevation before model fitting. The nominal rectangle and
camera model have a 6.919 px mean reference reprojection discrepancy, so the
correction is retained as a development artifact with that limitation stated:

```powershell
$env:PYTHONPATH='C:\tmp\codex-opencv'
python -m DigitalTwin.analysis.correct_apriltag_elevation `
  --input "DigitalTwin\datasets\analysis\apriltag_carpet_2x1_full_1x_calibrated\apriltag_still_summary.json" `
  --output-dir "DigitalTwin\datasets\analysis\apriltag_carpet_2x1_full_1x_elevation_corrected" `
  --calibration "DigitalTwin\datasets\analysis\camera_calibration_landscape\camera_calibration_charuco.json" `
  --tag-size-m 0.08 --tag-height-m 0.08112 `
  --maximum-reprojection-error-px 8.0
```

Audit synchronized pivot-turn events using the activity-derived video-minus-
telemetry offset of 9.75 s:

```powershell
$env:PYTHONPATH='C:\tmp\codex-opencv'
python -m DigitalTwin.analysis.apriltag_turn_event_audit `
  --tracking "DigitalTwin\datasets\analysis\apriltag_carpet_2x1_full_1x_elevation_corrected\apriltag_still_summary.json" `
  --telemetry "raw_logs\telemetry\ugv_t147_interactive_20260812_130620.csv" `
  --run-name carpet_2x1_development `
  --video-minus-telemetry-offset-s 9.75 `
  --output-dir "DigitalTwin\datasets\analysis\apriltag_carpet_2x1_turn_audit"
```

Fit distance scale, direction-specific effective track widths, and a bounded
bias-corrected IMU contribution on the first 75% of the run. The remaining 25%
is a temporal diagnostic and is not an independent final validation set:

```powershell
$env:PYTHONPATH='C:\tmp\codex-opencv'
python -m DigitalTwin.analysis.apriltag_temporal_calibration `
  --tracking "DigitalTwin\datasets\analysis\apriltag_carpet_2x1_full_1x_elevation_corrected\apriltag_still_summary.json" `
  --telemetry "raw_logs\telemetry\ugv_t147_interactive_20260812_130620.csv" `
  --dataset-name carpet_2x1_development `
  --train-fraction 0.75 `
  --include-imu-grid `
  --output-dir "DigitalTwin\datasets\analysis\apriltag_carpet_2x1_temporal_calibration"
```

Do not replace the frozen global motion parameters from this command alone.
Freeze a candidate only after inspecting the temporal holdout, then evaluate it
once on a newly recorded untouched run with the same camera and tag geometry.

The August 12 development fit selected distance scale `0.95`, clockwise width
`0.18 m`, counterclockwise width `0.20 m`, and bias-corrected gyro weight
`0.20`. On the temporal tail, heading MAE improved from `19.8 deg` to
`11.1 deg`, position RMSE from `1.964 m` to `1.841 m`, and path-length
agreement from `87.2%` to `92.9%`. These values are recorded as
`UGV01_CARPET_DEVELOPMENT_CANDIDATE`; the global `0.192 m` default remains
unchanged pending an untouched run. The event audit directly supports an
approximately `0.18 m` turn width but contains only two clean positive-turn
events, so the direction-specific split remains provisional.

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

## i2Nav-Robot Public Ground-Truth Study

Place the downloaded `playground00` text files under
`public_datasets/im2nav/playground00/`. Prepare the synchronized 10 Hz table:

```powershell
python -m DigitalTwin.analysis.prepare_i2nav_robot
```

Then train the GPS-independent random-forest and MLP covariance candidates and
evaluate both on the untouched chronological test segment:

```powershell
python -m DigitalTwin.analysis.i2nav_uncertainty_study
```

The study writes its report, CSV metrics, saved MLP, and four figures to
`DigitalTwin/datasets/analysis/i2nav_playground00/study/`. It selects global
`Q`, GNSS `R`, initial `P`, and turn/slip process-noise gain on the validation
split only, then reports both raw and calibrated consistency on the untouched
test split. The calibrated search uses GPS-independent speed/yaw-rate evidence
to inflate lateral and heading process noise during turns; GNSS residuals are
not used to authorize this adaptation.
The study also reports a GPS-bias EKF baseline for public data and skips GNSS
updates whose reported horizontal sigma exceeds `10 m`, recording the skipped
count as `gps_updates_skipped_quality`.
The full validation search is saved as `ekf_consistency_calibration.csv`, and
the raw/calibrated test comparison is in `ekf_test_metrics.csv`. Test metrics
include NIS, full-state NEES, position-only NEES, and heading-only NEES so that
position inconsistency can be separated from yaw inconsistency. The principal
files are `study_report.md`, `covariance_model_metrics.csv`,
`ekf_test_metrics.csv`, `trajectory_comparison.png`, and `ekf_consistency.png`.

Interpret acceptance conservatively. The MLP should beat the training-median
covariance predictor on all three held-out targets, move NIS and NEES coverage
toward 95%, and avoid worsening EKF position error. This public sequence is an
external feasibility check; it does not replace final UGV01 plus AprilTag
validation or authorize direct deployment of the learned model.

## UGV01 AprilTag Aligned Ground-Truth Export

The current UGV01 AprilTag pilot can be packaged in the same `aligned_samples`
style used by the i2Nav public-data preparation. This is useful for inspection,
plotting, and future tooling that expects an aligned table with ground-truth
pose fields.

```powershell
python -m DigitalTwin.analysis.prepare_ugv01_apriltag_ground_truth
```

The default export reads
`DigitalTwin/datasets/analysis/validation_carpet_142023_candidate/aligned_fidelity_samples.csv`
and writes:

- `DigitalTwin/datasets/analysis/ugv01_apriltag_carpet_142023/aligned_samples.csv`
- `DigitalTwin/datasets/analysis/ugv01_apriltag_carpet_142023/aligned_samples.npz`
- `DigitalTwin/datasets/analysis/ugv01_apriltag_carpet_142023/preparation_summary.json`
- `DigitalTwin/datasets/analysis/ugv01_apriltag_carpet_142023/README.md`

Interpret this export conservatively. The AprilTag trajectory is the physical
ground-truth reference for the pilot, but GPS is unavailable in this run and
camera/telemetry synchronization is inherited from the fidelity analysis rather
than a hardware sync pulse. The export is therefore i2Nav-like for tooling
compatibility, not a final synchronized GPS plus AprilTag validation set.

To compare stronger uncertainty models against the MLP, run the model bake-off:

```powershell
C:\Users\shrey\miniconda3\python.exe -m DigitalTwin.analysis.i2nav_model_bakeoff `
  --input DigitalTwin\datasets\analysis\i2nav_playground00\aligned_samples.npz `
  --output-dir DigitalTwin\datasets\analysis\i2nav_playground00\model_bakeoff `
  --ekf-top 4
```

The bake-off trains MLP, deep-ensemble MLP, Random Forest, ExtraTrees, gradient
boosting, histogram gradient boosting, and quantile gradient boosting. If
XGBoost, LightGBM, CatBoost, or NGBoost are installed, add `--include-optional`
to include them. Outputs are `model_bakeoff_report.md`,
`model_bakeoff_metrics.csv`, `model_bakeoff_coverage.csv`, and
`model_bakeoff_ekf_metrics.csv`.

The full paper-facing regeneration script now runs this bake-off automatically
for `i2nav_playground00`, `i2nav_parking00`, and `i2nav_street00` whenever
their `aligned_samples.npz` files are present:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\regenerate_all_results.ps1
```

Use `-SkipI2NavBakeoff` to omit those longer model comparisons, or
`-IncludeOptionalBakeoffModels` to include optional packages such as XGBoost,
LightGBM, CatBoost, and NGBoost when installed. A custom GRU/RNN uncertainty
model is the next natural upgrade because covariance is a temporal quantity;
it should be added as a separate bake-off model family after a deep-learning
dependency is selected and pinned.
