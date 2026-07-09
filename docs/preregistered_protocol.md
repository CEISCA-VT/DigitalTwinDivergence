# Pre-Registered Protocol Package

This document locks the default post-battery experimental configuration for the
tracked UGV01 unless a later note explicitly overrides it.

## Separation Of Phases

1. **Bench validation**
   - stationary or lifted-track checks
   - verifies firmware, GPS path, timing, sequence health, and logging
2. **Stationary hardware validation**
   - powered but non-driving captures
   - characterizes GPS fix behavior, HDOP, update rate, and timing stability
3. **Moving benign field runs**
   - nominal motion only
   - used for calibration, baseline statistics, threshold locking, and
     uncertainty-model fitting
4. **Moving attack runs**
   - performed only after calibration and benign baselines are accepted

## Default Conditions

### Routes

- `straight`
- `square`
- `figure8`

### Surfaces

- `smooth_pavement`
- `rough_surface`

Use a rough local surface such as rough pavement, uneven concrete, or a
grass-edge equivalent that is repeatable and safe.

### Speed Settings

- `low`
- `medium`

These are fixed command settings on the rover side and are mapped to measured
meters-per-second during calibration. Do not rename runs using only command
values; keep the semantic names `low` and `medium`.

### Network Conditions

- `wifi_baseline`
- `wifi_buffered_delay`

The buffered delay condition should use the edge-side latency emulator only.
Do not alter rover firmware behavior to create latency.

### Attack Types

- `step`
- `replay`
- `freeze`

### Step Magnitudes

```text
0.5, 1, 2, 3, 5, 7.5, 10 m
```

### Detection And Mission Targets

- false-alarm target: `P_FA <= 0.05`
- evaluation horizon: full run plus first-detection delay
- mission tolerance: `epsilon_req = 5 m`

## Trial Matrix Defaults

Minimum benign matrix after calibration:

```text
2 speeds x 2 surfaces x 2 latency settings x 5 trials = 40 benign runs
```

Attack runs should reuse the same speed/surface/latency matrix after threshold
locking is complete.

## Run Naming Convention

Use the following naming structure for all replayable CSV outputs and notebooks:

```text
speed-{low|medium}_surface-{smooth_pavement|rough_surface}_latency-{wifi_baseline|wifi_buffered_delay}_route-{straight|square|figure8}_attack-{none|step|replay|freeze}{_eps-X.Y}_trial-N
```

Examples:

```text
speed-low_surface-smooth_pavement_latency-wifi_baseline_route-square_attack-none_trial-0
speed-medium_surface-rough_surface_latency-wifi_buffered_delay_route-figure8_attack-step_eps-5.0_trial-3
```

## Valid Run Criteria

A run is valid when:

- the rover follows the intended route without manual rescue
- telemetry remains continuous enough to reconstruct the run
- timestamps, sequence values, and GPS validity are plausible
- benign runs contain no intentional attack labels
- attack runs contain exactly one preregistered attack condition

A run is invalid when:

- motion deviates materially from the route
- telemetry logging fails or becomes badly corrupted
- a stop/restart or operator intervention changes the trial semantics
- surface, speed, or latency condition differs from the preregistered label

Invalid runs should be archived but excluded from threshold locking and final
effect estimates.

## Calibration Dependencies

Before moving benign runs, estimate and record:

- `left_meters_per_tick`
- `right_meters_per_tick`
- `effective_track_width_m`
- `heading_sign`

Use the scripts under `DigitalTwin.analysis.calibration_prep` to produce the
first machine-readable calibration artifacts as soon as powered runs begin.
