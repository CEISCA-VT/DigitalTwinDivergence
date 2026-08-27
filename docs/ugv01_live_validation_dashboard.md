# UGV01 Live Digital-Twin Validation Dashboard

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
6. The interface reports live GPS-to-twin `Dp`, RMS/p95 agreement,
   course-to-heading disagreement, and 1/5/10 s displacement disagreement.

## Dummy CSV Prototype

Use this before the rover is connected, or when testing the frontend layout:

```powershell
python -m DigitalTwin.dashboard.server --mode csv --csv raw_logs\telemetry\ugv_t147_bench_20260814_143729.csv --host 127.0.0.1 --port 8765
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
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --poll-hz 5
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
- edge latency, stale packets, queue depth, and sequence gaps.

The live metrics describe operational GPS-to-twin agreement. They are not
independent localization accuracy because GPS is itself a measured sensor. The
paper's independent physical validation remains a separate offline experiment.

The GPS frame uses a single initialization, not trajectory fitting: the first
valid fix establishes translation, and the first valid course sample above
`0.30 m/s` establishes heading. NMEA course is converted from clockwise-from-
north to the local ENU convention.

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
