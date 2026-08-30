# UGV01 Live Validation Runbook

Updated for the current repository state.

This document explains how to run the current UGV01 live-validation experiment
with the code that actually exists in this repo today. It covers:

1. environment setup on Windows,
2. rover bring-up,
3. the live digital-twin dashboard experiment,
4. the offline AprilTag reference pipeline,
5. where outputs are written,
6. what is implemented now versus what is still manual.

It is intentionally repo-grounded. Older notes that mention missing helpers such
as `firmware/python/ugv01_http_ctrl.py` are not the current path.

## 1. What The Current Repo Supports

The current repo has two closely related but distinct experiment paths.

### Path A: Live service-contract demonstration

This is the browser-based digital-twin interface driven by:

- `python -m DigitalTwin.dashboard.server --mode live ...`

It provides:

- live UGV01 movement control,
- a sensor-lightweight twin propagated from encoder and IMU evidence,
- GPS-to-twin operational agreement,
- online service contracts,
- online resource-policy selection,
- JSONL logging to `raw_logs/live_validation/`.

### Path B: Offline AprilTag-referenced fidelity evaluation

This is the camera/telemetry processing path driven by:

- `bench_logger_interactive.py` or another `bench_logger*.py` route script,
- `DigitalTwin.analysis.analyze_apriltag_still_video`,
- `DigitalTwin.analysis.repair_apriltag_continuity`,
- `DigitalTwin.analysis.correct_apriltag_elevation`,
- `DigitalTwin.analysis.apriltag_fidelity`.

It provides:

- independent physical reference from AprilTags,
- synchronized physical-versus-virtual fidelity plots and metrics,
- outputs under `DigitalTwin/datasets/analysis/...`.

## 2. Important Current Limitation

The live dashboard and the offline AprilTag evaluator are **not yet one single
end-to-end button** in this repo.

Today:

- the live dashboard logs `JSONL` records to `raw_logs/live_validation/`,
- the offline fidelity tools expect a `T:147` telemetry CSV such as one written
  by `bench_logger.py` or `bench_logger_interactive.py`.

So the safest current workflow is:

1. use the live dashboard for the prospective online experiment and screenshots,
2. use a separate paired `T:147` CSV run for offline AprilTag fidelity
   evaluation.

Do **not** run multiple active `T:147` pollers against the rover at the same
time unless you intentionally accept extra network load and timing interference.

## 3. Windows Environment Setup

Use the repo root:

```powershell
cd C:\Users\shrey\Documents\DigitalTwinDivergence
```

General Python setup:

```powershell
python -m pip install -r requirements.txt
```

Quick code sanity check:

```powershell
python -m pytest -q
```

### Recommended Python split

In the current machine setup, there are two practical Python entrypoints:

- `python`
  - good for the dashboard, bench logger, and most non-OpenCV utilities
- `C:\Users\shrey\miniconda3\python.exe`
  - recommended for OpenCV / AprilTag / ChArUco tools
  - also works for printable generation

## 4. Rover Bring-Up

The active embedded firmware path for this project is:

```text
ugv01_gps_dev/General_Driver
```

BN220 wiring should remain:

```text
BN-220 white -> UGV01 RX
BN-220 red   -> UGV01 5V
BN-220 black -> UGV01 GND
```

Bring-up steps:

1. Power on the rover.
2. Connect your laptop to the rover Wi-Fi or its station-mode network.
3. Open:

```text
http://192.168.4.1
```

4. Confirm the stock control panel loads.
5. Confirm you can stop the rover and that telemetry is updating.

The current live dashboard polls the firmware backend at:

```text
http://192.168.4.1/js
```

If the rover is in station mode, replace `192.168.4.1` with the IP shown on the
OLED `ST` line.

## 5. Printable Assets

Generate the current AprilTag printables:

```powershell
C:\Users\shrey\miniconda3\python.exe .\scripts\generate_apriltag_printables.py --tag-size-mm 200
```

This produces, among others:

- `docs/printables/apriltag_rover_id0_200mm_letter.pdf`
- `docs/printables/apriltag_world_reference_ids_1_to_6_200mm_letter.pdf`

Generate the ChArUco board:

```powershell
C:\Users\shrey\miniconda3\python.exe .\scripts\generate_charuco_board.py
```

The authoritative calibration board is:

- `docs/printables/charuco_7x5_30mm_letter_landscape.pdf`

## 6. Live Dashboard Experiment

### 6.1 Dummy CSV rehearsal

Use this first to confirm the UI loads and behaves correctly before touching the
rover:

```powershell
python -m DigitalTwin.dashboard.server --mode csv --csv raw_logs\telemetry\ugv_t147_bench_20260814_143729.csv --host 127.0.0.1 --port 8765 --policy contract-aware --open
```

Then open:

```text
http://127.0.0.1:8765
```

### 6.2 Live rover dashboard

Run the live dashboard against the rover:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --open
```

If the rover is in station mode, replace `192.168.4.1` with the current station
IP.

Supported policies:

- `static-low`
- `static-high`
- `aoi-only`
- `contract-aware`

Examples:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy static-low --open
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy static-high --open
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy aoi-only --open
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --host 127.0.0.1 --port 8765 --policy contract-aware --open
```

### 6.3 What the live dashboard records

Every live session writes a JSONL log to:

```text
raw_logs/live_validation/
```

Example file:

- `raw_logs/live_validation/ugv01_live_contract_20260827_192535.jsonl`

The schema written by the server is:

- `ugv01_live_contract_record_v1`

Each record includes:

- current point values,
- live contract states,
- resource mode,
- policy events.

### 6.4 How to check the newest live log

List recent files:

```powershell
Get-ChildItem .\raw_logs\live_validation | Sort-Object LastWriteTime -Descending | Select-Object -First 10
```

Read the last few records of a chosen file:

```powershell
Get-Content .\raw_logs\live_validation\ugv01_live_contract_20260827_192535.jsonl -Tail 5
```

## 7. Suggested Motion Plans For The Live Dashboard

The live dashboard itself handles motion through the browser controls. For the
current repo, use one of these practical motion plans.

### Plan A: Smooth low-turning

- 5 s stationary start
- slow forward segment
- slow reverse segment
- one or two gentle curves
- 5 s stationary end

### Plan B: Surface transition

- start on carpet or smooth floor
- drive across one clear surface boundary
- include a straight segment before and after transition
- include one shallow turn after the transition

### Plan C: Turning-intensive

- repeated 90 degree turns,
- square-like loop,
- or manual slalom/figure-eight using the live controls

The live experiment is best for:

- contract-state behavior,
- policy switching,
- AoI and latency behavior,
- live visualization,
- screenshots and demonstration.

## 8. Offline Telemetry CSV Run For AprilTag Fidelity

If you need a `T:147` CSV for the existing AprilTag fidelity tools, run a
separate logger-driven trial instead of the live dashboard.

### 8.1 Interactive logger

This is the most flexible current path:

```powershell
python bench_logger_interactive.py --ip 192.168.4.1 --speed low --turn-control encoder --turn-counts-per-90 575 --output raw_logs\telemetry\ugv_t147_live_validation_trial01.csv
```

This lets you choose motions from the terminal after each segment.

### 8.2 Square route

For a scripted square route:

```powershell
python bench_logger_square_0_5m.py --surface smooth --speed low --network baseline --trial 1
python bench_logger_square_1m.py --surface smooth --speed low --network baseline --trial 1
```

### 8.3 Curves

For figure-eight and S-curve collection:

```powershell
python bench_logger_curves.py --route figure8 --repeats 2 --surface smooth_kitchen_floor --speed low --trial 1
python bench_logger_curves.py --route s_curve --repeats 2 --surface smooth_kitchen_floor --speed low --trial 1
```

### 8.4 Recommended current telemetry settings

The current square-motion defaults in `bench_logger.py` are:

- distance scale: `0.975`
- effective track width candidate: approximately `0.200 m` clockwise and
  `0.190 m` counterclockwise for the current live twin
- smooth square turn counts per 90 degrees: `575`
- smooth square turn command: `(0.038, -0.038)`

These are the current asset-specific prototype settings in the repo. They are
not universal rover constants.

## 9. Camera Calibration And AprilTag Tracking

Use the Miniconda Python for these commands.

### 9.1 Calibrate the phone camera from ChArUco video

```powershell
C:\Users\shrey\miniconda3\python.exe -m DigitalTwin.analysis.calibrate_charuco_video docs\footage\charuco_run.mp4 --output-dir DigitalTwin\datasets\analysis\camera_calibration_live
```

Outputs go under:

- `DigitalTwin/datasets/analysis/camera_calibration_live/`

### 9.2 Track the AprilTag run video

Use one of the built-in layouts that the current script already supports:

- `rectangle`
- `rectangle_2p0x1p0_1234`
- `square_1p5`
- `square_2p0`
- `square_2p0_1236`
- `trapezoid`

Example:

```powershell
C:\Users\shrey\miniconda3\python.exe -m DigitalTwin.analysis.analyze_apriltag_still_video docs\footage\run01.mp4 --output-dir DigitalTwin\datasets\analysis\run01_tracked_full --calibration DigitalTwin\datasets\analysis\camera_calibration_live\camera_calibration_charuco.json --sample-stride 1 --max-frames 30000 --preview-count 12 --world-layout rectangle_2p0x1p0_1234
```

If the fixed reference tags never move and you recorded a dedicated reference
clip of the same setup, you can also provide:

```text
--static-reference-video <path>
```

### 9.3 Repair short tracking gaps

```powershell
C:\Users\shrey\miniconda3\python.exe -m DigitalTwin.analysis.repair_apriltag_continuity --input DigitalTwin\datasets\analysis\run01_tracked_full\apriltag_still_summary.json --output-dir DigitalTwin\datasets\analysis\run01_continuity_repaired --max-short-gap-s 6 --max-long-gap-s 30
```

### 9.4 Correct rover-tag elevation

For the current rover-tag setup, use the measured tag height and the actual tag
size you mounted. If you use the new 20 cm rover tag, set `--tag-size-m 0.20`.

Example:

```powershell
C:\Users\shrey\miniconda3\python.exe -m DigitalTwin.analysis.correct_apriltag_elevation --input DigitalTwin\datasets\analysis\run01_continuity_repaired\apriltag_still_summary.json --output-dir DigitalTwin\datasets\analysis\run01_elevation_corrected --calibration DigitalTwin\datasets\analysis\camera_calibration_live\camera_calibration_charuco.json --tag-size-m 0.20 --tag-height-m 0.08112 --maximum-reprojection-error-px 8.0
```

If the correction needs a separate unchanged reference clip:

```text
--reference-video <path>
```

## 10. AprilTag Fidelity Evaluation

Once you have:

- one repaired/corrected AprilTag summary, and
- one matching `T:147` telemetry CSV,

run the fidelity comparison:

```powershell
python -m DigitalTwin.analysis.apriltag_fidelity --tracking DigitalTwin\datasets\analysis\run01_elevation_corrected\apriltag_still_summary.json --telemetry raw_logs\telemetry\ugv_t147_live_validation_trial01.csv --output-dir DigitalTwin\datasets\analysis\run01_fidelity --sync-mode onset --distance-scale 0.975 --clockwise-track-width-m 0.200 --counterclockwise-track-width-m 0.190 --gyro-weight 0.20 --gyro-scale 1.0
```

If the video starts after telemetry or during motion, try:

```powershell
python -m DigitalTwin.analysis.apriltag_fidelity --tracking DigitalTwin\datasets\analysis\run01_elevation_corrected\apriltag_still_summary.json --telemetry raw_logs\telemetry\ugv_t147_live_validation_trial01.csv --output-dir DigitalTwin\datasets\analysis\run01_fidelity --sync-mode activity --distance-scale 0.975 --clockwise-track-width-m 0.200 --counterclockwise-track-width-m 0.190 --gyro-weight 0.20 --gyro-scale 1.0
```

Primary outputs:

- `fidelity_summary.json`
- `fidelity_report.md`
- `trajectory_fidelity.png`
- `fidelity_diagnostics.png`
- `aligned_fidelity_samples.csv`

## 11. Optional: Package The AprilTag Run In The i2Nav-Like Format

If you want the aligned export format used elsewhere in the repo:

```powershell
python -m DigitalTwin.analysis.prepare_ugv01_apriltag_ground_truth --input DigitalTwin\datasets\analysis\run01_fidelity\aligned_fidelity_samples.csv --summary DigitalTwin\datasets\analysis\run01_fidelity\fidelity_summary.json --output-dir DigitalTwin\datasets\analysis\ugv01_apriltag_run01
```

Outputs:

- `aligned_samples.csv`
- `aligned_samples.npz`
- `preparation_summary.json`
- `README.md`

## 12. Current Service-Contract Configuration

The live dashboard uses:

- `DigitalTwin/configs/ugv01_live_service_contracts.json`

Current frozen prototype service settings are:

| Service | Horizon | Position tolerance | Heading tolerance | Max AoI |
|---|---:|---:|---:|---:|
| Immediate motion | `1 s` | `0.10 m` | `2 deg` | `0.60 s` |
| Short prediction | `5 s` | `0.20 m` | `5 deg` | `1.00 s` |
| Planning support | `10 s` | `0.50 m` | `10 deg` | `1.50 s` |
| Global asset tracking | global | `1.00 m` | `5 deg` | `1.00 s` |

Current resource modes are:

| Mode | Update rate | Relative cost |
|---|---:|---:|
| economy | `2 Hz` | `0.2` |
| normal | `5 Hz` | `0.5` |
| high | `10 Hz` | `1.0` |

## 13. Recommended End-To-End Order

For the current repo, the cleanest order is:

1. rehearse the UI with dummy CSV mode,
2. run one live dashboard trial and save screenshots/video of the interface,
3. run one separate paired `T:147` CSV trial with the same style of motion,
4. record the AprilTag overhead video for that paired trial,
5. process the video through tracking, repair, and elevation correction,
6. run `apriltag_fidelity`,
7. use the live JSONL session for online-contract analysis and the paired CSV
   plus AprilTag run for offline physical fidelity.

## 14. What Is Already Implemented Versus Still Manual

### Already implemented

- live UGV01 dashboard polling `T:147`
- browser movement controls
- online contract evaluation
- online resource-mode selection
- JSONL logging of live state and policy events
- offline AprilTag video analysis
- offline continuity repair
- offline elevation correction
- offline AprilTag-versus-twin fidelity analysis


## 15. Troubleshooting

### The dashboard command says `unrecognized arguments`

That usually means the other machine is using an older copy of
`DigitalTwin/dashboard/server.py`. The current repo version supports:

```powershell
python -m DigitalTwin.dashboard.server --mode live --rover-url http://192.168.4.1/js --policy contract-aware
```

### `cv2` is missing

Use:

```powershell
C:\Users\shrey\miniconda3\python.exe
```

for the AprilTag and ChArUco commands.

### `rover=None` in AprilTag tracking

That means the rover tag ID was not reliably decoded. Use:

- larger rover tag,
- clearer top-down framing,
- less blur,
- more lighting,
- slower motion,
- or the continuity-repair path only if the tag is intermittently visible.

### GPS is weak indoors

That is expected. In the current live dashboard, GPS acts as the operational
reference for the live service view, not as final independent physical truth.
Independent physical fidelity still comes from the offline AprilTag pipeline.
Additionally for the fidelity contract GPS isnt needed but as it is there for the follow up it will be used.
## 16. Related Repo Documents

For supporting details, also see:

- [docs/ugv01_live_validation_dashboard.md](/C:/Users/shrey/Documents/DigitalTwinDivergence/docs/ugv01_live_validation_dashboard.md)
- [docs/ugv01_esp32_bringup.md](/C:/Users/shrey/Documents/DigitalTwinDivergence/docs/ugv01_esp32_bringup.md)
- [docs/printables/README.md](/C:/Users/shrey/Documents/DigitalTwinDivergence/docs/printables/README.md)
- [docs/running.md](/C:/Users/shrey/Documents/DigitalTwinDivergence/docs/running.md)
