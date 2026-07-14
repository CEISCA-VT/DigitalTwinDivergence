# Week 2 Completion Tests

This is the minimum powered test package needed to call Week 2 complete for the
tracked UGV01.

## Locked Geometry References

Use these as the nominal hardware references from the official Waveshare UGV01
documentation:

- rail center distance: `170 mm`
- single track width: `44 mm`
- minimum turning radius: `0 m` (in-place rotation supported)

Use these as the stock official/vendor defaults:

- firmware nominal encoder scale: `6646.16 counts/m`
- stock UGV01 firmware motion-model track width: `0.141 m`
- physical rail-center hardware dimension: `0.170 m`

The nominal hardware dimensions and the effective calibrated dimensions are not
the same thing. For this project, if you are staying with official vendor
values only, use `0.141 m` in the control/motion model and keep `0.170 m` as a
hardware geometry note.

## Test 1: Powered Safety And Telemetry Bring-Up

Goal: verify that `T:147` logging, stop commands, and BN220 telemetry still work
under self-powered operation.

Run:

1. tracks lifted off the ground
2. battery powered, charger disconnected
3. 30 to 60 seconds of `bench_logger.py` with motion script disabled
4. one short forward and one short turn command, then stop

Pass if:

- `T:147` remains responsive
- stop command works immediately
- encoder and IMU fields change during commanded motion
- no unexpected packet drops or stale-packet bursts appear

## Test 2: Straight Encoder-Scale Validation

Goal: confirm `left_meters_per_tick` and `right_meters_per_tick`.

Run:

1. tape-measure a `1.0 m` or `2.0 m` straight lane
2. perform `3 to 5` slow forward runs
3. perform `2 to 3` slow backward runs over the same lane
4. log each run separately

Pass if:

- encoder-derived distance agrees with measured distance within a tight, repeatable band
- left/right scale estimates are stable across runs
- forward and backward estimates do not diverge badly

Outputs to lock:

- `left_meters_per_tick`
- `right_meters_per_tick`
- final counts-per-meter value used by the logger and digital twin

## Test 3: In-Place Turn Calibration

Goal: lock `effective_track_width_m` and yaw sign convention.

Run:

1. mark a starting heading on the floor
2. perform `2` clockwise turns of known angle
3. perform `2` counterclockwise turns of known angle
4. use `180 deg` or `360 deg` turns with a clearly observed final heading

Pass if:

- clockwise and counterclockwise estimates agree closely
- IMU yaw changes with the expected sign
- the track-width estimate stays in a narrow band

Outputs to lock:

- `effective_track_width_m`
- `heading_sign`

## Test 4: Stationary-To-Moving Transition Test

Goal: compare GPS, encoder, and IMU behavior before, during, and after motion.

Run one structured log:

1. stationary for `10 s`
2. straight forward for `1 m`
3. stationary for `10 s`
4. backward for `1 m`
5. stationary for `10 s`
6. one clockwise turn
7. one counterclockwise turn
8. stationary for `10 s`

Pass if:

- encoders are flat while stationary and move cleanly during motion
- IMU yaw is quiet while stationary and changes during turns
- GPS does not produce impossible jumps relative to the indoor setting
- packet timing remains healthy across state transitions

## Test 5: Route-Reference Setup

Goal: establish an independent path reference for later route evaluation.

Choose one:

1. marked waypoints with tape measure
2. overhead smartphone video
3. fiducial markers

Minimum requirement:

- define origin
- define route geometry
- define how time alignment will be done
- record enough metadata to reproduce the reference path later

Pass if:

- the reference method is documented
- the first straight and square route can be compared against it

## Test 6: Repeatable Short Routes

Goal: verify route repeatability and controllability before full experiments.

Run:

1. `2 to 3` straight runs
2. `2` short square runs

Pass if:

- logs are complete
- timing fields stay populated
- route shape is recognizable in encoder/IMU/GPS traces
- the rover is controllable enough to repeat the route without major drift or aborts

## Week 2 Done When

Week 2 is complete when all of these are true:

- tracked-drive calibration values are locked from powered tests
- stationary-to-moving transition behavior has been reviewed
- an independent route-reference method exists
- straight and short-square validation runs have been completed and logged
- experiment defaults are preregistered and consistent with the hardware behavior
