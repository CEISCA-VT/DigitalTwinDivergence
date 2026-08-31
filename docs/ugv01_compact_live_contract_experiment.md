# UGV01 Compact Live Contract Experiment

Updated: August 31, 2026

This runbook is for the final UGV01 live experiment that connects the rover
firmware backend to the digital-twin dashboard and records service-contract and
resource-control decisions. It is written for Windows PowerShell.

## Purpose

The experiment tests whether service-relative digital-twin contracts can guide
IoT update-rate decisions during live operation. The dashboard polls the UGV01,
propagates the sensor-lightweight twin, compares it with the live operational
GPS reference when GPS is valid, evaluates the 1 s, 5 s, 10 s, and global
contracts, and logs every decision.

The paper-facing claim should be based on repeated matched runs across policies,
not on a single dashboard smoke test.

## Hardware Setup

Required:

- UGV01 powered on.
- Laptop connected to the rover access point or the same station-mode Wi-Fi.
- Firmware backend reachable at `/js`.
- T:147 telemetry working.
- GPS connected and valid for observable live contract states.

Not required during the live dashboard run:

- AprilTags.
- ChArUco board.
- Overhead video.

AprilTags are used for independent offline physical-validation metrics. GPS is
the live operational reference used by the dashboard contracts.

## Firmware Backend

Default access-point URL:

```powershell
http://192.168.4.1/js
```

Station-mode URL:

```powershell
http://<ROVER_IP_FROM_DISPLAY>/js
```

The dashboard polls combined telemetry with:

```json
{"T":147}
```

Movement commands from the browser use the existing rover command style:

```json
{"T":1,"L":...,"R":...}
```

## Frozen Service Contracts

Contract definitions live in:

```text
DigitalTwin/configs/ugv01_live_service_contracts.json
```

| Service | Horizon | Position tolerance | Heading tolerance | AoI limit |
|---|---:|---:|---:|---:|
| Immediate motion | 1 s | 0.10 m | 2 deg | 0.60 s |
| Short prediction | 5 s | 0.20 m | 5 deg | 1.00 s |
| Planning support | 10 s | 0.50 m | 10 deg | 1.50 s |
| Global asset tracking | global | 1.00 m | 5 deg | 1.00 s |

Contract states:

| State | Meaning |
|---|---|
| `qualified` | Observable and inside the declared tolerance. |
| `at_risk` | Observable and still inside tolerance, but close to a limit. |
| `withdrawn` | Observable but outside position, heading, or AoI limits. |
| `unobservable` | GPS position/course, quality, or required history is unavailable. |

## Policies To Compare

| Policy | Behavior |
|---|---|
| `static-low` | Fixed economy update mode, 2 Hz. |
| `static-high` | Fixed high update mode, 10 Hz. |
| `aoi-only` | Uses only age-of-information to choose 2/5/10 Hz. |
| `contract-aware` | Uses service state and AoI to choose 2/5/10 Hz. |

The strongest IoT-J result is not simply that `contract-aware` changes rate. It
should show similar service satisfaction to `static-high` with lower request or
byte cost, while avoiding the service violations expected from `static-low` or
`aoi-only`.

## Recommended Final Matrix

Use the same motion script for every policy inside a condition.

Minimum useful final matrix:

```text
3 physical conditions x 4 policies x 3 repetitions = 36 runs
```

Stronger matrix:

```text
3 physical conditions x 4 policies x 5 repetitions = 60 runs
```

Recommended physical conditions:

| Condition | Intended stress |
|---|---|
| `stationary_low_motion` | Low motion, telemetry/freshness sanity. |
| `surface_transition` | Carpet to tile or similar physical condition change. |
| `turning_intensive` | Repeated turns, curves, and yaw-rate disagreement. |

Use `wifi_baseline` unless a real delay/jitter emulator is active. Do not label
a run `buffered_delay` unless controlled delay is actually applied to the live
path.

## One CSV Rehearsal

Use this first to check the dashboard without touching the rover:

```powershell
python -m DigitalTwin.dashboard.server --mode csv --csv raw_logs\telemetry\<YOUR_T147_LOG>.csv --host 127.0.0.1 --port 8765 --policy contract-aware --run-label csv_rehearsal --physical-condition replay --wireless-condition recorded --trial 1 --duration-s 60 --open
```

CSV mode is useful for UI and contract debugging. It is not a live
resource-control experiment because the samples were already recorded.

## One Live Smoke Test

Run this before the policy comparison:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --run-label live_smoke_gps_check --physical-condition stationary_low_motion --wireless-condition wifi_baseline --trial 0 --duration-s 60 --open
```

After it finishes, analyze logs:

```powershell
python -m DigitalTwin.analysis.analyze_live_contract_logs
```

Open:

```text
results/ugv01_live_contract_experiment/live_contract_experiment_report.md
```

The smoke test is acceptable only if the log shows samples flowing and GPS-valid
fraction is high enough for service contracts to become observable.

## Run One Complete Policy Set

This command runs all four policies sequentially for one physical condition and
one repetition. Each policy arm stops automatically after the chosen duration.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ugv01_live_policy_set.ps1 -RoverUrl "http://192.168.4.1/js" -PhysicalCondition "turning_intensive" -WirelessCondition "wifi_baseline" -Trial 1 -DurationSeconds 120 -Open
```

Repeat with `-Trial 2`, `-Trial 3`, and so on.

For station-mode Wi-Fi:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ugv01_live_policy_set.ps1 -RoverUrl "http://<ROVER_IP>/js" -PhysicalCondition "turning_intensive" -WirelessCondition "wifi_baseline" -Trial 1 -DurationSeconds 120 -Open
```

## Motion Script

Keep motion matched across policies. For each run:

1. keep the rover still for 5-10 s,
2. drive forward slowly,
3. reverse slowly,
4. perform gentle left and right curves,
5. include repeated turns for `turning_intensive`,
6. stop for 5-10 s at the end.

Keep the same approximate timing and route shape for every policy arm in the
same condition. Manual control is acceptable for a compact experiment, but
matched motion matters because otherwise policy comparisons are confounded by
different rover behavior.

## Analyze Completed Runs

Run:

```powershell
python -m DigitalTwin.analysis.analyze_live_contract_logs
```

Custom input/output:

```powershell
python -m DigitalTwin.analysis.analyze_live_contract_logs --input-dir raw_logs\live_validation --output-dir results\ugv01_live_contract_experiment
```

Generated outputs:

```text
results/ugv01_live_contract_experiment/live_run_summary.csv
results/ugv01_live_contract_experiment/live_service_summary.csv
results/ugv01_live_contract_experiment/live_policy_summary.csv
results/ugv01_live_contract_experiment/live_contract_experiment_report.md
results/ugv01_live_contract_experiment/analysis_manifest.json
```

The main paper table should come from `live_policy_summary.csv` and
`live_service_summary.csv`.

## Metrics To Report

Per policy and service:

- observable fraction,
- qualified / at-risk / withdrawn / unobservable fractions,
- p95 AoI,
- p95 position disagreement,
- p95 heading disagreement when GPS course is valid,
- request count,
- payload bytes and bytes/s,
- actual and requested update rate,
- p95 latency,
- stale-packet rate,
- queue-depth p95.

## Valid Run Rules

Keep a run in the final analysis if:

- the intended policy, condition, and trial metadata are present,
- the rover stayed powered for the full planned duration,
- T:147 telemetry was available,
- GPS validity is sufficient for the planned live contract question,
- no manual emergency interruption changed the intended condition.

Mark a run as a smoke/debug run, not a final run, if:

- GPS was disconnected or invalid for most of the run,
- the run used the wrong policy,
- the rover was turned off early,
- the wireless condition label does not match the actual setup,
- the operator changed the motion script substantially compared with other
  policy arms.

## Final Interpretation

Use cautious wording:

- GPS is the live operational reference, not independent ground truth.
- AprilTag validation is the independent offline physical reference.
- A single successful run shows live feasibility.
- Repeated matched policy runs are required to claim resource savings with
  preserved service satisfaction.
