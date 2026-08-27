# UGV01 Live Service-Contract Digital Twin

This dashboard is the prototype interface for the final physical validation
experiment. It uses the same firmware backend as the Waveshare page at
`http://192.168.4.1`, but presents the left side as the paper's digital-twin
and fidelity view and the right side as movement control.

## Purpose

The display follows the paper framing:

1. UGV01 firmware emits tracked-drive, IMU, GPS, timing, and packet fields.
2. The Python edge process converts native `T:147` fields into canonical motion
   evidence.
3. A sensor-lightweight UGV01 twin propagates virtual pose from encoder and IMU
   evidence.
4. GPS is converted to the same initialized ENU frame and used as the live
   operational reference for position and moving-course agreement.
5. The same page sends safe bounded movement commands through the UGV01 backend.
6. A causal contract engine evaluates four frozen service-relative definitions,
   including quantity, horizon, tolerance, and maximum age of information.
7. A resource policy selects a `2`, `5`, or `10 Hz` telemetry mode without
   changing the twin model or overriding rover movement.
8. Every displayed update, contract state, and policy transition is written to
   `raw_logs/live_validation/*.jsonl`.

## Dashboard Views

The prototype is organized around four paper-facing views:

| View | Purpose |
|---|---|
| Live Twin | Shows the common-frame GPS reference, sensor-lightweight twin trajectory, current virtual pose, and UGV01 movement controls. |
| `C_s` Contracts | Shows the formal service-contract object `C_s = (Q_s, H_s, tau_s, C_s, A_s)`, per-service tolerances, current margins, lifecycle state, and rolling qualification timelines. |
| Resource Policy | Shows why the selected policy arm requested the current `2`, `5`, or `10 Hz` telemetry mode, including AoI triggers, relative cost, and contract states. |
| Fidelity Audit | Shows live `D_p(t)`, `D_theta(t)`, and the contract/policy event trail. |

## Frozen Service Contracts

The position and heading tolerances come from
`results/e1_e2_service_contract_publication/analysis_manifest.json`. Runtime
freshness and lifecycle defaults are predeclared in
`DigitalTwin/configs/ugv01_live_service_contracts.json` for the prospective
experiment; they are not universal safety limits.

| Service | Horizon | Position | Heading | Maximum AoI |
|---|---:|---:|---:|---:|
| Immediate motion | `1 s` | `0.10 m` | `2 deg` | `0.60 s` |
| Short prediction | `5 s` | `0.20 m` | `5 deg` | `1.00 s` |
| Planning support | `10 s` | `0.50 m` | `10 deg` | `1.50 s` |
| Global asset tracking | global | `1.00 m` | `5 deg` | `1.00 s` |

Each service is reported as `qualified`, `at risk`, `withdrawn`, or
`unobservable`. The last state means GPS quality, moving course,
synchronization, or required history is unavailable, so the system does not
manufacture a pass/fail result.

## Dummy CSV Prototype

Use this before the rover is connected, or when testing the frontend layout:

```powershell
python -m DigitalTwin.dashboard.server --mode csv --csv raw_logs\telemetry\ugv_t147_bench_20260814_143729.csv --host 127.0.0.1 --port 8765 --policy contract-aware
```

Then open:

```text
http://127.0.0.1:8765
```

Any `T:147` CSV with the standard bench-logger fields can be used with
`--csv`.

## Live Rover Prototype

Connect to the UGV01 Wi-Fi or station-mode network, keep the normal rover
control panel available at `http://192.168.4.1`, and run:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware
```

Then open:

```text
http://127.0.0.1:8765
```

The live dashboard repeatedly sends:

```json
{"T":147}
```

to the firmware backend as:

```text
http://192.168.4.1/js?cmd={"T":147}
```

Use this dashboard for both movement and live twin/fidelity monitoring. The
original UGV01 page can remain open as a backup control panel.

Select one frozen experiment policy before each run with `--policy`:

```text
static-low      fixed 2 Hz
static-high     fixed 10 Hz
aoi-only        rate selected from calibrated source age
contract-aware  rate selected from contract state and source age
```

The adaptive policies include frozen escalation, downshift, and dwell behavior.
They change the telemetry request rate only; they do not alter the twin model or
send autonomous motion commands.

## Movement Controls

The UI sends movement commands to:

```text
http://192.168.4.1/js?cmd={"T":1,"L":...,"R":...}
```

Control mapping:

| UI command | Firmware payload at medium speed |
|---|---|
| Stop | `{"T":1,"L":0.0,"R":0.0}` |
| Forward | `{"T":1,"L":-0.28,"R":-0.28}` |
| Reverse | `{"T":1,"L":0.28,"R":0.28}` |
| Left | `{"T":1,"L":0.22,"R":-0.22}` |
| Right | `{"T":1,"L":-0.22,"R":0.22}` |

Speed buttons scale the non-stop payloads:

| Speed | Scale |
|---|---:|
| Slow | `0.65` |
| Middle | `1.00` |
| Fast | `1.35` |

Pressing and holding a direction sends that motion. Releasing or leaving the
button sends `Stop`.

## What The Live View Can And Cannot Claim

The live view can show:

- virtual twin trajectory in the initialized local frame;
- encoder-derived forward speed;
- blended encoder/IMU yaw rate;
- firmware yaw and IMU yaw-rate behavior;
- wheel-IMU disagreement and a slip-like proxy;
- GPS-to-twin position divergence and running RMS/p95 agreement;
- GPS-course heading disagreement while GPS speed is at least `0.30 m/s`;
- operational 1/5/10 s displacement disagreement when enough fixes exist;
- local/global contract errors, margins, and lifecycle state;
- calibrated relative AoI, arrival jitter, bandwidth, and evaluation time;
- resource-policy transitions and requested telemetry rate;
- edge latency, stale packets, queue depth, and sequence gaps.

The live metrics describe operational GPS-to-twin agreement. They are not
independent localization accuracy because GPS is itself a measured sensor. The
paper's independent physical validation remains a separate offline experiment.

The GPS frame uses a single initialization, not trajectory fitting: the first
valid fix establishes translation, and the first valid course sample above
`0.30 m/s` establishes heading. NMEA course is converted from clockwise-from-
north to the local ENU convention.

The live reference gate requires at least four satellites, HDOP at most `2.5`,
GPS age at most `1.5 s`, and moving course at least `0.30 m/s`. When those
conditions are absent, affected contracts are `unobservable`. AprilTag/ChArUco
remains the independent offline accuracy reference; it is intentionally not a
dependency of the live service.

## Current UGV01 Twin Parameters

The live prototype uses the current UGV01 development candidate already recorded
in `DigitalTwin/kinematics.py`:

| Parameter | Value |
|---|---:|
| Distance scale | `0.975` |
| CW effective track width | `0.200 m` |
| CCW effective track width | `0.190 m` |
| Gyro blend weight | `0.20` |
| Gyro scale | `1.0` |

These parameters are for the current UGV01 prototype and should be treated as
asset-specific experimental settings, not universal tracked-rover constants.
