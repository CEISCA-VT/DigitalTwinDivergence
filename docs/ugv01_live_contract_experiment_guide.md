# UGV01 Live Contract Experiment Guide

Updated: August 29, 2026

This guide explains how to run the **live contract experiment** with the
current repository and current UGV01 workflow.

It is intentionally practical. The goal is to answer:

- Do I need GPS connected?
- Do I need AprilTags connected?
- What do I run first?
- What does the live experiment actually produce?

## 1. What This Experiment Is

The live contract experiment is the **online digital-twin dashboard**
demonstration in which:

- the rover is controlled from the dashboard,
- the digital twin is updated from onboard telemetry,
- service contracts are evaluated online,
- the resource policy chooses `2 Hz`, `5 Hz`, or `10 Hz`,
- the session is logged to `raw_logs/live_validation/`.

The current dashboard entrypoint is:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --open
```

## 2. What Must Be Connected

### Required for the live contract experiment

- UGV01 rover powered on
- Wi-Fi connection to the rover or its station-mode IP
- Firmware sending live telemetry at `/js`
- IMU working
- Encoders working

These are the minimum signals needed for the live twin itself.

### Strongly recommended for the live contract experiment

- **GPS connected and functioning**

Reason:
the current live dashboard uses GPS as the live operational reference for
service-agreement and contract-style monitoring. Without GPS, the dashboard can
still run, but some live fidelity/contract states become much weaker or
effectively unobservable.

So the practical rule is:

- for a meaningful live contract demo, **keep GPS connected**
- for offline AprilTag validation, GPS is not required

### Not required for the live contract experiment

- **AprilTags**
- **ChArUco board**
- overhead phone video

These are **not** part of the live online demo itself.

They are only used later for **independent offline physical-validation
metrics** such as ATE, RPE, and heading error.

## 3. Short Answer: GPS vs AprilTags

### Live dashboard experiment

- GPS: **Yes, keep it connected**
- AprilTags: **No**

### Offline physical-validation experiment

- GPS: optional for the current fidelity paper
- AprilTags: **Yes**, if you want independent physical ground truth

## 4. Wiring and Rover Bring-Up

Active firmware path:

```text
ugv01_gps_dev/General_Driver
```

BN220 wiring:

```text
BN-220 white -> UGV01 RX
BN-220 red   -> UGV01 5V
BN-220 black -> UGV01 GND
```

Bring-up checklist:

1. Power on the rover.
2. Connect your laptop to the rover Wi-Fi or station-mode Wi-Fi.
3. Open:

```text
http://192.168.4.1
```

4. Confirm the stock control page responds.
5. Confirm telemetry updates.
6. Confirm GPS fields are updating if GPS is attached.

If the rover is on station-mode Wi-Fi, replace `192.168.4.1` with the IP shown
on the rover display.

## 5. Recommended Experiment Order

Use this exact order.

### Step 1: Rehearse the dashboard without the rover

```powershell
python -m DigitalTwin.dashboard.server --mode csv --csv raw_logs\telemetry\ugv_t147_bench_20260814_143729.csv --host 127.0.0.1 --port 8765 --policy contract-aware --open
```

This checks:

- UI layout
- contract panels
- live charts and status logic
- no browser/runtime surprises

### Step 2: Run the actual live rover dashboard

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --open
```

If needed, also test the comparison policies:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy static-low --open
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy static-high --open
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy aoi-only --open
```

### Step 3: Drive one controlled live session

Use a simple motion plan:

- 5 s stationary start
- straight forward
- straight reverse
- one or two gentle turns
- a short turning-intensive segment
- 5 s stationary end

This is enough to show:

- live twin response
- contract qualification/violation changes
- policy reasoning
- resource-mode switching

### Step 4: Save the live log

The dashboard automatically writes a JSONL file under:

```text
raw_logs/live_validation/
```

Check the newest file:

```powershell
Get-ChildItem .\raw_logs\live_validation | Sort-Object LastWriteTime -Descending | Select-Object -First 10
```

Read its last few lines:

```powershell
Get-Content .\raw_logs\live_validation\YOUR_FILE.jsonl -Tail 5
```

## 6. What The Live Experiment Produces

The live experiment produces:

- the dashboard visualization,
- screenshots/screen recording,
- live contract decisions,
- live policy decisions,
- JSONL session logs.

It does **not** by itself produce:

- AprilTag ground-truth metrics,
- final ATE / RPE / heading-validation plots,
- a benchmark-style fidelity CSV.

Those come from the offline validation path.

## 7. When AprilTags Are Needed

Use AprilTags only if you want to answer:

- how accurate was the twin versus an independent physical reference?
- what were ATE, RPE, and heading error?
- how good was the UGV01 physical fidelity offline?

Then you run a **separate paired experiment**:

1. collect telemetry CSV,
2. record overhead AprilTag video,
3. run the offline AprilTag analysis pipeline.

AprilTags are for **validation metrics**, not for the live online contract demo.

## 8. Should GPS Stay Connected?

Yes, for the live contract experiment, **GPS should stay connected**.

Why:

- the current live dashboard is built around live GPS-versus-twin agreement
- GPS helps make the service panels meaningful in real time
- the later follow-up work also needs GPS again

If GPS is disconnected:

- the rover can still stream IMU and encoder telemetry
- the twin can still propagate
- but the live contract result becomes less persuasive because the dashboard
  loses its main live external reference

So:

- **live demo:** keep GPS connected
- **offline AprilTag metric study:** GPS optional

## 9. Important Current Limitation

Do **not** run multiple simultaneous telemetry pollers against the rover unless
you intentionally accept extra network load.

In particular:

- if the live dashboard is polling `/js`,
- do not also run another aggressive logger at the same time unless that is
  part of a deliberate stress test.

Safest workflow:

1. run the live dashboard experiment by itself,
2. then run a separate offline telemetry/AprilTag validation trial.

## 10. Minimal “Do This” Checklist

If you want the shortest version:

1. Keep **GPS connected**.
2. Do **not** worry about AprilTags for the live demo.
3. Power the rover and confirm `http://192.168.4.1`.
4. Run:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --open
```

5. Drive one controlled session.
6. Save screenshots and the JSONL log.
7. If you later want physical fidelity metrics, run a separate AprilTag session.

## 11. Related Repo Files

- `docs/ugv01_live_validation_runbook.md`
- `docs/ugv01_live_validation_dashboard.md`
- `docs/ugv01_esp32_bringup.md`
- `DigitalTwin/configs/ugv01_live_service_contracts.json`

