# Digital Twin Divergence Project Context For Another LLM

Current as of August 11, 2026. This file is intended as a compact handoff for
another LLM or collaborator. It explains what the project is, what has been
implemented, what is scientifically solid, what remains weak, and how the code
is organized.

## One-Sentence Summary

This project builds a security-aware digital twin for a Waveshare UGV01 tracked
rover, using encoder, IMU, GPS, timing, and AprilTag ground-truth data to study
when GPS should be trusted, down-weighted, or rejected under benign motion and
GPS-coordinate attack replay.

## Core Research Question

The project asks whether a rover digital twin can detect or bound divergence
when GPS measurements are unreliable or malicious, especially when adaptive
uncertainty policies can accidentally let attacked GPS influence the model's
own confidence.

The important distinction is:

- GPS is a sensor under test, not trusted ground truth.
- AprilTag/video tracking is the independent physical ground truth standard.
- Encoders and IMU provide protected relative-motion evidence.
- Edge timing and sequence counters provide packet-health evidence.
- The digital twin estimates rover state and decides whether GPS behavior is
  consistent with the protected motion/timing evidence.

## Hardware State

Hardware used:

- Waveshare UGV01 tracked rover.
- ESP32-based onboard controller using the repo firmware path
  `ugv01_gps_dev`.
- BN220 GPS module was integrated and worked earlier, but has recently been
  disconnected due connector/wiring instability. The digital-twin and AprilTag
  work can continue without live GPS until GPS attack experiments need fresh
  GPS data.
- AprilTag visual tracking is now the preferred ground-truth route-fidelity
  method.

Verified BN220 wiring when used:

```text
BN220 white -> UGV01 RX
BN220 red   -> 5V
BN220 black -> GND
```

Do not change this wiring assumption casually. Earlier working behavior used
the UGV01 RX path.

## Firmware And Telemetry

Active firmware path:

```text
ugv01_gps_dev
```

Important commands:

- `T:146`: GPS-only telemetry.
- `T:147`: combined UGV01 telemetry used by the edge logger.
- `T:404`: Wi-Fi station/AP configuration.

Example Wi-Fi update:

```json
{"T":404,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"YOUR_WIFI","sta_password":"YOUR_PASSWORD"}
```

The combined `T:147` packet includes:

- sequence counter and firmware timing: `seq`, `sample_ms`, `send_ms`,
  `millis`
- motor/command state: `L`, `R`
- encoders: `enc_left`, `enc_right`
- voltage: `v`
- IMU: roll/pitch/yaw `r`, `p`, `y`; acceleration `ax`, `ay`, `az`; gyro
  `gx`, `gy`, `gz`; magnetometer `mx`, `my`, `mz`; temperature `temp`
- GPS status: `gps_valid`, `gps_age_ms`, `gps_fix_type`, `lat`, `lon`,
  `sat`, `hdop`, `alt_m`, `speed_mps`, `course_deg`
- GPS parser counters: `gps_chars`, `gps_sentences`, `gps_failed_checksums`

The edge logger adds:

- local wall and monotonic timestamps
- edge send, edge receive, edge midpoint, estimate, and alarm timestamps
- HTTP latency
- session-level clock-offset calibration
- stale-packet flag
- sequence-gap and request-failure counts
- queue depth
- run metadata such as route, surface, speed, attack type, trial, and square
  calibration parameters

References:

- `docs/telemetry_protocol.md`
- `docs/log_data_dictionary.md`
- `docs/ugv01_esp32_bringup.md`

## Main Code Structure

Core digital-twin modules:

- `DigitalTwin/ekf.py`: EKF prediction and GPS update.
- `DigitalTwin/kinematics.py`: motion model support.
- `DigitalTwin/motion.py`: gyro-bias correction, encoder/IMU yaw diagnostics,
  and slip indicators.
- `DigitalTwin/security.py`: GPS-independent predictor, covariance bounds, and
  trusted/evidence-gated logic.
- `DigitalTwin/detector.py`: Mahalanobis/NIS scoring, detectability logic, and
  confidence fields.
- `DigitalTwin/alarm.py`: robust initialization, motion gating, and persistent
  alarm rules.
- `DigitalTwin/attack.py`: offline GPS attack injection.
- `DigitalTwin/analysis/real_data_study.py`: accepted-log study, benign
  thresholding, attack campaign, bootstrapping, plots, and reports.
- `DigitalTwin/analysis/digital_twin_accuracy.py`: current sensor-agreement and
  consistency report.
- `DigitalTwin/dashboard/`: visual replay dashboard for accepted logs.

Hardware/logging scripts:

- `bench_logger.py`: main `T:147` telemetry logger and scripted route support.
- `bench_logger_interactive.py`: terminal-driven manual motion commands.
- `bench_logger_square_0_5m.py`: 0.5 m square route runner.
- `bench_logger_square_1m.py`: 1 m square route runner.
- `bench_logger_curves.py`: figure-eight and S-curve timed route runner.
- `collect_rough_dataset.py`: helper for rough-surface benign data collection.

## Current Motion Script State

The UGV01 is a tracked rover, so turns are not perfectly repeatable. Recent
manual calibration showed that a slow encoder-count turn is better than
yaw-based or timed turning.

Current interactive turn settings:

```text
90 deg turn target: 575 average opposite-track encoder counts
turn command:       0.038 to 0.045 is physically best; slower reduces overshoot
```

Current square settings in `bench_logger.py`:

```text
SQUARE_STRAIGHT_FORWARD_CMD = (-0.10, -0.10)
SQUARE_TURN_CW_CMD          = (0.038, -0.038)
SQUARE_TURN_COUNTS_PER_90   = 575.0
SQUARE_SIDE_HOLD_SECONDS    = 1.5
SQUARE_CORNER_HOLD_SECONDS  = 4.0
SQUARE_STRAIGHT_WARMUP_CMD  = (-0.05, -0.05)
SQUARE_STRAIGHT_WARMUP_SECONDS = 0.6
```

Known issue:

- After a 90-degree turn, the rover sometimes pivots slightly before going
  straight.
- Logs show transient one-track or uneven commands such as `L=0, R=-0.093` or
  `L=-0.108, R=-0.151` before the clean straight segment.
- This is likely due to UGV01 motor ramping, HTTP command delay, track friction,
  and residual settling after the turn.
- This is annoying for visually clean square execution, but it is not fatal for
  the research because AprilTags measure the actual path.

For data collection inside a roughly `2 m x 1 m` area, prefer:

```powershell
python bench_logger_square_0_5m.py --surface smooth --speed low --repeats 1 --trial N
python bench_logger_curves.py --route figure8 --repeats 2 --surface smooth_kitchen_floor --speed low --trial N
python bench_logger_curves.py --route s_curve --repeats 2 --surface smooth_kitchen_floor --speed low --trial N
```

If curves are too large:

```powershell
--duration-scale 0.75
```

## Data Collected So Far

Bench and stationary data:

- `raw_logs/static/*.csv`: stationary GPS/static logs.
- `raw_logs/telemetry/*.csv`: combined `T:147` telemetry logs.
- Stationary GPS summary, telemetry latency/health reports, and replay review
  pipelines exist under `DigitalTwin/analysis/`.

Benign route corpus:

- A 20-run accepted corpus was collected from `square0p5x3` runs.
- Conditions were two surfaces and two speeds:
  smooth kitchen floor, rough permeable concrete, low speed, medium speed.
- These logs are useful for software replay, benign thresholding, and attack
  replay, but they should be treated as a design/evaluation corpus rather than
  final prospective validation.

AprilTag/video data:

- AprilTag ID 0 is mounted on the rover.
- Fixed world tags were placed around the route.
- ChArUco footage was collected for camera calibration.
- Important videos live under `docs/footage/`.
- Printable AprilTag and calibration files live under `docs/printables/`.

Existing high-confidence usable ground-truth motion data is limited. Earlier
estimates found only several minutes of useful synchronized AprilTag motion, so
more ground-truth footage is needed for final fidelity claims.

## AprilTag Ground Truth

AprilTags are needed because indoor GPS on a small route is not a reliable
physical accuracy reference. A 0.5 m square route can show around meter-level
EKF-to-GPS disagreement indoors, but that is sensor disagreement, not proof that
the rover's true path is wrong by a meter.

Ground truth role:

- AprilTag/world tags define a metric coordinate frame.
- The rover tag gives measured rover position and heading in video frames.
- The EKF output can be compared to this independent physical path.
- This allows reporting position error, heading error, trajectory-shape error,
  and local relative-pose error.

Current known tag geometry:

- Rover tag ID 0.
- Rover tag size: 8 cm.
- Rover tag is mounted near the center top of the rover.
- Tag elevation above floor: about 8.112 cm.
- Earlier reference layouts included a 1.5 m square and a measured trapezoid.

Useful accuracy metrics:

- ATE RMSE: global trajectory position error.
- Median ATE: typical global position error.
- P95 ATE: tail position error.
- RPE RMSE: short-window relative motion error.
- Heading MAE: heading-angle error.
- Path-length ratio: whether total traveled distance is close to truth.
- Percent within thresholds: samples within 0.10, 0.25, 0.50 m of ground truth.

Important caveat:

- Same-run calibration results are provisional.
- Final claims require training/calibration on some runs and evaluation on
  untouched held-out video/log runs.

## Current Digital Twin Model

The digital twin is primarily an EKF-based tracked-rover model.

State:

```text
x, y, heading
```

Tracked-drive update:

```text
dL = left_meters_per_tick  * delta_left_ticks
dR = right_meters_per_tick * delta_right_ticks
ds = (dL + dR) / 2
dtheta = (dR - dL) / effective_track_width_m
```

The project has moved away from "wheel radius/wheelbase" wording because UGV01
is tracked, not a normal wheeled differential-drive robot. Use:

```text
left_meters_per_tick
right_meters_per_tick
effective_track_width_m
heading_sign
```

Current tracked-turn findings:

- Vendor/nominal physical track width is not sufficient for turn prediction.
- AprilTag calibration suggests effective tracked-turn width around `0.192 m`.
- Earlier encoder-only estimate was around `0.1818 m`.
- The difference is expected because tracked vehicles slip laterally during
  turns.

IMU status:

- IMU yaw/gyro is useful as an independent evidence and slip-diagnostic signal.
- Direct high-weight gyro fusion degraded some trajectory fidelity in existing
  logs.
- Current direction is conservative: estimate gyro bias, use IMU for
  diagnostics/evidence, and only fuse carefully if validated against AprilTag
  ground truth.

## Accuracy Status

Do not overclaim digital-twin physical accuracy from GPS-only comparisons.

Existing EKF-to-GPS metrics quantify sensor agreement, not true physical
accuracy. They are useful for consistency checks but unreliable as final
fidelity evidence indoors.

Earlier sensor-agreement values included roughly:

```text
EKF-to-GPS RMSE:             about 1.08 m
Median EKF-to-GPS difference: about 0.67 m
Samples within 1 m of GPS:    about 70.6%
Samples within 2 m of GPS:    about 94.0%
```

These are not good enough as physical accuracy claims for a small indoor route
because indoor BN220 GPS can drift and wander. The right answer to an advisor
is:

> The existing indoor GPS logs are useful as degraded-GPS stress tests, not as
> digital-twin fidelity ground truth. Fidelity will be measured against
> AprilTag ground truth using position, heading, and trajectory-shape errors.

Current AprilTag-calibrated results are promising but provisional because they
partly use same-run calibration. They indicate the model can improve
substantially when tracked-turn geometry is calibrated, but final validation
needs held-out footage.

## Attack And Threat Model

The current attack campaign is offline replay injection, not live RF GPS
spoofing.

Main attacker assumption:

- The attacker can alter GPS coordinates delivered to the edge analysis stream.
- The attacker does not control encoders, IMU, sequence counters, or edge
  timing evidence in the core threat model.
- Raw physical logs and rover behavior remain unchanged; attacks are injected
  during replay.

Attack families:

- Step bias: sudden GPS position jump.
- Slow drift: GPS gradually drifts in along-track or cross-track direction.
- Freeze: GPS coordinate becomes stuck.
- Replay: earlier GPS segment is replayed.
- Strategic/offline attacker variants may choose timing or direction based on
  replay knowledge.

Important terminology:

- Do not call all replay rows "independent physical attacks."
- Use "attack-run-start combinations" and "detector-run evaluations" carefully.
- "Harmful-but-stealthy" should be phrased more cautiously as
  "tolerance-exceeding paired divergence before alarm" unless physical ground
  truth is present.

## Baselines

Baseline detector/model families used or discussed:

- GPS jump detector: detects sudden unrealistic jumps between GPS positions.
- Fixed NIS / chi-square Kalman innovation test: classic innovation
  consistency check.
- Innovation-gated EKF: rejects or down-weights GPS when innovation is too
  large.
- Huber EKF: robust update that reduces influence of large residuals.
- CUSUM: accumulates persistent smaller residuals.
- Naive adaptive covariance: changes uncertainty based on residuals, but can be
  vulnerable to GPS-residual feedback.
- GPS-independent or evidence-gated adaptation: limits adaptation using
  non-GPS evidence such as IMU/timing/sequence behavior.

The project's core argument is not "EKF beats all baselines." The stronger
argument is:

> Adaptive uncertainty can create a feedback path from attacked GPS to model
> confidence. A security-aware digital twin should separate GPS evidence from
> protected motion/timing evidence and explicitly bound when GPS can influence
> uncertainty.

## Novelty Position

The project is not novel just because it uses an EKF, GPS, or a tracked rover.
Those are established.

More defensible novelty:

- It frames adaptive uncertainty itself as a possible security vulnerability.
- It separates operational GPS fusion from a GPS-independent security branch.
- It combines rover telemetry, packet timing, and physical motion evidence in a
  digital-twin divergence detector.
- It uses a real low-cost tracked UGV with firmware-level timestamped telemetry,
  not only simulation.
- It moves toward independent AprilTag ground truth to quantify digital-twin
  fidelity before making attack claims.
- A future hybrid neural EKF could learn slip/turn corrections while preserving
  interpretable covariance and security constraints.

Weak novelty if left unfinished:

- GPS-only attack replay without independent ground truth is not enough for a
  top venue.
- Indoor small-route GPS disagreement cannot be presented as physical accuracy.
- A generic NN replacing EKF would be less defensible without much more data.

## Recommended Model Direction With Neural Networks

Do not replace the EKF with a full black-box neural net. The better path is a
hybrid neural EKF:

- Keep EKF physics and explicit covariance.
- Use a small MLP or GRU to learn tracked-rover slip/turn corrections from
  protected inputs.
- Inputs should be encoder deltas, `gz`, yaw diagnostics, commands, `dt`, and
  surface label.
- Outputs should be bounded corrections to distance/heading and/or process
  covariance scale.
- GPS should not be the training ground truth indoors.
- AprilTag trajectory should be the training/evaluation target.

Example target:

```text
true_delta_distance - encoder_delta_distance
true_delta_heading  - encoder_delta_heading
```

Security constraint:

```text
GPS may reduce its own trust, but should not be allowed to increase protected
model confidence without independent IMU/timing/encoder evidence.
```

This yields a cleaner research contribution:

> An evidence-gated, slip-aware neural EKF for secure low-cost tracked-rover
> digital twins.

## What Is Done

Implemented:

- Firmware path with `T:146`, `T:147`, and GPS/timing additions.
- Edge logger with timestamped telemetry, packet health, and session clock
  calibration.
- Bench telemetry and stationary GPS analysis scripts.
- Hardware replay path through the digital twin.
- EKF, detector, alarm, uncertainty-policy variants, and attack replay.
- Real-data study pipeline with manifesting, thresholding, attack campaign,
  bootstrapping, and plots.
- Dashboard for visual digital-twin replay.
- AprilTag printables, camera calibration work, overlay rendering, and initial
  ground-truth trajectory processing.
- Tracked-turn calibration showing nominal geometry over-rotates; effective
  width is larger.
- Motion scripts for interactive commands, square routes, figure-eight, and
  S-curve routes.
- Several advisor-facing docs, manuscripts, and roadmap files.

## What Still Needs To Be Done

Highest priority scientific work:

1. Collect cleaner synchronized AprilTag ground-truth runs.
2. Split data into train/development, validation, and untouched final test.
3. Quantify physical digital-twin fidelity against AprilTag ground truth.
4. Recalibrate tracked-turn/slip model only on train/development data.
5. Evaluate on held-out AprilTag runs without retuning.
6. Regenerate digital-twin accuracy, threshold, and attack reports after the
   final model is frozen.
7. Only then make strong claims about GPS attacks and detection performance.

High-value data collection:

- indoor smooth kitchen floor
- indoor carpet or rough indoor surface
- routes that stay within camera view: small squares, figure-eight, S-curve,
  random/freeform path
- low and medium speed
- clockwise and counter-clockwise turns
- straight, reverse, stop-start, arcs, and mixed trajectories

Approximate data need:

- prototype: current data plus a few more clean videos is enough
- strong paper: roughly 40-60 minutes of synchronized usable motion minimum
- better: 60-90 minutes with held-out validation/test, balanced across surfaces

## Commands To Know

Run interactive manual motion:

```powershell
python bench_logger_interactive.py --ip <UGV_IP>
```

Run a 0.5 m square:

```powershell
python bench_logger_square_0_5m.py --surface smooth --speed low --repeats 1 --trial N
```

Run a figure-eight:

```powershell
python bench_logger_curves.py --route figure8 --repeats 2 --surface smooth_kitchen_floor --speed low --trial N
```

Run an S-curve:

```powershell
python bench_logger_curves.py --route s_curve --repeats 2 --surface smooth_kitchen_floor --speed low --trial N
```

Open dashboard:

```powershell
python -m DigitalTwin.dashboard.server --open
```

Generate accepted-log study:

```powershell
python -m DigitalTwin.analysis.real_data_study
```

Summarize an existing campaign:

```powershell
python -m DigitalTwin.analysis.real_data_study --summarize-existing
```

Generate digital-twin accuracy report:

```powershell
python -m DigitalTwin.analysis.digital_twin_accuracy
```

Run tests:

```powershell
python -m pytest -q
```

## Publication Potential

Neutral assessment:

- As-is, without independent ground truth and prospective validation, this is
  more like a strong project/prototype or workshop-level paper.
- With AprilTag ground truth, held-out fidelity evaluation, clean methodology,
  and cautious attack replay claims, it can become a credible conference/journal
  submission.
- For a top/very strong venue, the paper needs a sharper contribution:
  security-aware uncertainty adaptation, evidence-gated GPS trust, and
  physically validated tracked-rover digital-twin fidelity.

The strongest paper framing is not:

> We built a rover EKF and detected GPS spoofing.

It is:

> We show that adaptive uncertainty in a digital twin can create a
> measurement-to-confidence feedback vulnerability, and we evaluate an
> evidence-gated digital-twin architecture on a real tracked rover with
> independent visual ground truth.

## Guidance For Future LLMs

When helping with this project:

- Do not overclaim GPS as ground truth indoors.
- Do not call same-run AprilTag calibration a held-out result.
- Prefer tracked-drive terminology over wheel-radius/wheelbase terminology.
- Keep firmware behavior stable unless explicitly asked to modify it.
- Preserve BN220 wiring notes.
- Be careful with raw logs; many are development/debug runs.
- Generated analysis artifacts under `DigitalTwin/datasets/analysis/` may be
  stale if model or policy code changed.
- If changing detector, model, or uncertainty policy, regenerate downstream
  reports before quoting numbers.
- If changing route scripts, first inspect newest logs and only make small,
  physically motivated changes.

## Current Bottom Line

The project has a substantial implementation: rover telemetry, edge logging,
EKF replay, attack injection, detector variants, dashboard, AprilTag tracking,
and route scripts all exist. The main remaining gap is not code volume; it is
scientific validation. The next major milestone is clean AprilTag-grounded
fidelity evaluation, followed by frozen-model attack replay and carefully
bounded claims.
