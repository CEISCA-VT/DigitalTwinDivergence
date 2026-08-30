# UGV01 Compact Prospective Live Contract Experiment

Updated: August 29, 2026

This document explains how suitable the current repository is for the compact
UGV01 live contract experiment and gives the exact Windows PowerShell commands
to run it.

The experiment goal is to show the paper's live service-contract idea on the
real UGV01:

1. the rover streams live telemetry from the firmware backend,
2. the edge dashboard propagates the sensor-lightweight digital twin,
3. GPS acts as the live operational reference,
4. service contracts are evaluated online,
5. the policy chooses a telemetry/update mode,
6. every sample and decision is logged for later review.

## Suitability Of The Current Codebase

The current codebase is suitable for a working prototype of the compact live
contract experiment.

Implemented now:

- Live dashboard server: `DigitalTwin.dashboard.server`
- CSV rehearsal mode: `--mode csv`
- Live UGV01 mode: `--mode live`
- UGV01 movement commands from the dashboard
- Live T:147 polling from the rover backend
- Sensor-lightweight UGV01 twin propagation from encoder and IMU evidence
- GPS-to-twin operational agreement in a common initialized frame
- Service-contract engine with 1 s, 5 s, 10 s, and global services
- Resource policies: `static-low`, `static-high`, `aoi-only`, and
  `contract-aware`
- Rolling contract-state timelines and policy reasoning in the browser UI
- JSONL live-session logs under `raw_logs/live_validation/`
- Per-run experiment metadata in the live logs:
  `run_label`, `physical_condition`, `wireless_condition`, `trial`, and
  `notes`

Still limited:

- This is an operational live demo, not independent physical ground truth.
- AprilTags are not used by the live dashboard.
- GPS must be valid for live contract states to become fully observable.
- Camera/AprilTag fidelity metrics remain a separate offline validation path.
- The dashboard changes telemetry request rate, but it does not autonomously
  drive the robot or override user movement.

## Hardware Needed For The Live Experiment

Required:

- UGV01 rover powered on
- Laptop connected to UGV01 Wi-Fi or station-mode Wi-Fi
- Firmware backend reachable at `/js`
- Encoder telemetry working
- IMU telemetry working
- GPS connected if you want meaningful live contract states

Not required for the live dashboard:

- AprilTags
- ChArUco board
- overhead video

Use AprilTags later when you want independent offline ATE/RPE/heading metrics.

## Firmware Backend

Default access-point backend:

```text
http://192.168.4.1/js
```

Station-mode backend:

```text
http://<OLED_ST_IP>/js
```

The dashboard polls telemetry with:

```json
{"T":147}
```

Movement commands use the existing Waveshare-style speed command:

```json
{"T":1,"L":...,"R":...}
```

## Service Contracts

The live service contracts are configured in:

```text
DigitalTwin/configs/ugv01_live_service_contracts.json
```

Current service definitions:

| Service | Horizon | Position tolerance | Heading tolerance | AoI limit |
|---|---:|---:|---:|---:|
| Immediate motion | 1 s | 0.10 m | 2 deg | 0.60 s |
| Short prediction | 5 s | 0.20 m | 5 deg | 1.00 s |
| Planning support | 10 s | 0.50 m | 10 deg | 1.50 s |
| Global asset tracking | global | 1.00 m | 5 deg | 1.00 s |

Contract states:

- `qualified`: the service is observable and inside its tolerance.
- `at_risk`: the service is still inside tolerance but close to the limit.
- `withdrawn`: the observable service exceeds its tolerance or AoI limit.
- `unobservable`: GPS quality, GPS course, or required history is unavailable.

## Resource Policies

Use one policy per run:

| Policy | Meaning |
|---|---|
| `static-low` | Fixed economy mode, 2 Hz |
| `static-high` | Fixed high mode, 10 Hz |
| `aoi-only` | Uses only age of information to select 2/5/10 Hz |
| `contract-aware` | Uses contract state plus AoI to select 2/5/10 Hz |

The policy controls the dashboard polling/update rate. It does not change the
rover's motion command behavior.

## Step 1: Start In The Repository

```powershell
cd C:\Users\shrey\Documents\DigitalTwinDivergence
```

Run the focused tests:

```powershell
python -m pytest -q tests\test_dashboard.py tests\test_live_contracts.py
```

## Step 2: Dummy CSV Rehearsal

Use this before touching the rover:

```powershell
python -m DigitalTwin.dashboard.server --mode csv --csv raw_logs\telemetry\ugv_t147_bench_20260814_143729.csv --host 127.0.0.1 --port 8765 --policy contract-aware --run-label csv_rehearsal --physical-condition rehearsal --wireless-condition offline --trial 1 --open
```

Then open:

```text
http://127.0.0.1:8765
```

Stop with `Ctrl+C` in PowerShell.

## Step 3: Live Rover Smoke Test

Connect your laptop to the rover Wi-Fi and verify the normal page loads:

```text
http://192.168.4.1
```

Then run a short live smoke test:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --run-label smoke_contract --physical-condition stationary --wireless-condition wifi_baseline --trial 1 --notes "short stationary and small motion smoke test" --open
```

If you accidentally paste a Markdown-style URL such as
`[http://192.168.4.1/js](http://192.168.4.1/js)`, the current server sanitizes
it.

## Step 4: Compact Prospective Run Matrix

A compact matrix can be:

- physical condition: `stationary_low_motion`, `surface_transition`,
  `turning_intensive`
- wireless condition: `wifi_baseline`, `buffered_delay`
- policy: `static-low`, `static-high`, `aoi-only`, `contract-aware`
- repetitions: start with 1 smoke repetition, then increase to the desired
  count after the workflow is stable

Recommended paper-facing condition names:

```text
stationary_low_motion
surface_transition
turning_intensive
wifi_baseline
buffered_delay
```

## Step 5: Run One Policy

Template:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy <POLICY> --run-label <RUN_LABEL> --physical-condition <PHYSICAL_CONDITION> --wireless-condition <WIRELESS_CONDITION> --trial <TRIAL_NUMBER> --open
```

Example:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --run-label carpet_turning_contract_trial1 --physical-condition turning_intensive --wireless-condition wifi_baseline --trial 1 --open
```

## Step 6: Run The Four Policy Arms

For one condition and one repetition, run these one at a time.

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy static-low --run-label turning_baseline_static_low_t1 --physical-condition turning_intensive --wireless-condition wifi_baseline --trial 1 --open
```

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy static-high --run-label turning_baseline_static_high_t1 --physical-condition turning_intensive --wireless-condition wifi_baseline --trial 1 --open
```

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy aoi-only --run-label turning_baseline_aoi_only_t1 --physical-condition turning_intensive --wireless-condition wifi_baseline --trial 1 --open
```

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --run-label turning_baseline_contract_t1 --physical-condition turning_intensive --wireless-condition wifi_baseline --trial 1 --open
```

## Step 7: Suggested Motion Script For A Manual Run

Use the dashboard controls.

For each run:

1. keep the rover still for 5 to 10 s,
2. drive forward slowly,
3. reverse slowly,
4. perform gentle left and right turns,
5. for `turning_intensive`, include repeated controlled turns,
6. stop for 5 to 10 s at the end,
7. stop the server with `Ctrl+C`.

Keep each run short at first: 60 to 120 s is enough for a smoke test. After it
is stable, use the protocol duration you want for the final experiment.

## Step 8: Optional Buffered Wireless Condition

The current dashboard supports the `buffered_delay` label in metadata. It does
not itself emulate Wi-Fi delay in the live rover path.

For the final controlled buffered condition, use one of these defensible
options:

- run a local network-delay wrapper/proxy if available,
- use the existing edge-side delay emulator only if it is configured for this
  live path,
- otherwise label the run `wifi_baseline` and do not claim buffered-delay
  results.

Do not create a `buffered_delay` paper claim unless the run truly used a
controlled delay condition.

## Step 9: Check Outputs

List newest live logs:

```powershell
Get-ChildItem .\raw_logs\live_validation | Sort-Object LastWriteTime -Descending | Select-Object -First 10
```

View the last records:

```powershell
Get-Content .\raw_logs\live_validation\<LOG_FILE>.jsonl -Tail 5
```

Each JSONL row includes:

- `schema`
- `policy`
- `experiment`
- `point`
- `events`

The `experiment` object records:

- `run_label`
- `physical_condition`
- `wireless_condition`
- `trial`
- `notes`

## Step 10: What To Save For The Paper

For each completed run, keep:

- the JSONL log in `raw_logs/live_validation/`
- screenshots of the Live Twin view
- screenshots of the `C_s Contracts` view
- screenshots of the Resource Policy view
- notes about GPS quality, surface, lighting, and any manual interruption

Useful paper metrics from the live run:

- service qualified fraction
- service withdrawn fraction
- unobservable fraction
- p95 AoI
- p95 latency
- requested update-rate distribution
- relative resource cost
- GPS valid fraction
- GPS-to-twin operational disagreement
- policy transitions

## Step 11: CSV Mode For Any Recorded CSV

If you already recorded a T:147 telemetry CSV and want to replay it through the
same live dashboard:

```powershell
python -m DigitalTwin.dashboard.server --mode csv --csv raw_logs\telemetry\<YOUR_FILE>.csv --host 127.0.0.1 --port 8765 --policy contract-aware --run-label csv_replay_trial1 --physical-condition replay --wireless-condition recorded --trial 1 --open
```

This is useful for debugging the frontend and contract logic. It is not a live
resource-control experiment because the data are already recorded.

## Recommended Next Step

Run one CSV rehearsal and one stationary live smoke test first. If both logs
show samples flowing, GPS observability, service-contract updates, and policy
decisions, then proceed to the compact policy comparison.
