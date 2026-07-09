# Calibration Ready Checklist

This is the ready-to-run tracked-rover calibration package for the first
battery-powered session. Use tracked-drive terms throughout:

- `left_meters_per_tick`
- `right_meters_per_tick`
- `effective_track_width_m`
- `heading_sign`

## First Powered Session Order

1. Safety check with tracks lifted and stop command verified
2. Short stationary `T:147` logger capture
3. Straight-line encoder scale runs
4. In-place turn runs
5. IMU heading sign check
6. GPS local-frame review
7. Route-reference setup rehearsal

## Straight-Line Encoder Scale

Collect a slow straight run over a tape-measured distance, then run:

```powershell
python -m DigitalTwin.analysis.calibration_prep straight raw_logs\telemetry\straight_run.csv --distance-m 2.0 --out-prefix DigitalTwin\datasets\analysis\calibration\straight_run_01
```

Outputs:

- JSON summary
- Markdown note

Primary result:

```text
left_meters_per_tick
right_meters_per_tick
```

## In-Place Turn Track Width

Collect a clockwise and counterclockwise turn, then run:

```powershell
python -m DigitalTwin.analysis.calibration_prep turn raw_logs\telemetry\turn_run.csv --turn-angle-deg 180 --left-distance-m -0.62 --right-distance-m 0.61 --out-prefix DigitalTwin\datasets\analysis\calibration\turn_run_01
```

Primary result:

```text
effective_track_width_m
```

## IMU Heading Sign

Check whether positive yaw change matches the chosen turn convention:

```powershell
python -m DigitalTwin.analysis.calibration_prep imu raw_logs\telemetry\turn_run.csv --expected-heading-change-deg 180 --out-prefix DigitalTwin\datasets\analysis\calibration\imu_turn_01
```

Primary result:

```text
heading_sign
```

## GPS Local-Frame Validation

Review how the log maps into a local XY frame:

```powershell
python -m DigitalTwin.analysis.calibration_prep gps raw_logs\telemetry\straight_run.csv --out-prefix DigitalTwin\datasets\analysis\calibration\gps_run_01
```

This confirms the origin convention and gives a quick meter-scale span check.

## Route-Reference Template

Create the placeholder metadata artifact for the overhead-video or fiducial
reference path:

```powershell
python -m DigitalTwin.analysis.calibration_prep route-template --out-prefix DigitalTwin\datasets\analysis\calibration\route_reference_template
```

## Acceptance Criteria

Calibration is ready to lock when:

- straight-run encoder distance is repeatable
- clockwise and counterclockwise turn estimates agree closely
- heading sign is unambiguous
- GPS local-frame displacements are meter-scale and plausible
- route-reference metadata is defined before the first formal route trial
