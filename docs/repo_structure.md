# Repository Structure

Current as of August 11, 2026. This repo contains firmware, rover logging
scripts, digital-twin code, analysis pipelines, raw hardware logs, and writing
artifacts. The project is intentionally data-heavy, so the main cleanup rule is:
do not delete raw logs unless their exclusion rationale is recorded.

## Top-Level Layout

| Path | Purpose | Notes |
|---|---|---|
| `DigitalTwin/` | Core Python package | EKF, motion model, security logic, dashboard, analysis scripts. |
| `ugv01_gps_dev/` | Active UGV01 firmware tree | Contains the `T:146` and `T:147` firmware additions. |
| `raw_logs/` | Raw rover and GPS logs | Treat as immutable evidence. Use audits/manifests to include/exclude logs. |
| `docs/` | Human-facing documentation and manuscript notes | Roadmaps, protocol, firmware docs, handoffs, AprilTag notes. |
| `docs/footage/` | Local video evidence | AprilTag/ChArUco footage. Usually too large/noisy for code review. |
| `docs/printables/` | Generated AprilTag and calibration-board files | Reproducible from `scripts/` where possible. |
| `scripts/` | Utility generators | AprilTag and ChArUco printable generators. |
| `tests/` | Unit and smoke tests | Run with `python -m pytest -q`. |
| `presentations/` | Advisor/status decks | Presentation artifacts. |
| `results/` | Exported analysis copies | Ignored by Git. |

## Important Root Scripts

| Script | Purpose | Current Role |
|---|---|---|
| `bench_logger.py` | Main `T:147` logger and scripted route engine | Shared route primitives and CSV schema. |
| `bench_logger_interactive.py` | Terminal-driven manual motion | Best for calibration and ad hoc commands. |
| `bench_logger_square_0_5m.py` | 0.5 m square runner | Current indoor small-area square route. |
| `bench_logger_square_1m.py` | 1 m square runner | Larger route; use only if camera/space permit. |
| `bench_logger_curves.py` | Figure-eight and S-curve runner | Useful for non-square AprilTag trajectories. |
| `collect_rough_dataset.py` | Rough-surface collection helper | Earlier benign dataset helper. |

The route scripts remain at repo root because they import `bench_logger.py`
directly and are run from PowerShell during rover sessions.

## Core DigitalTwin Modules

| Path | Purpose |
|---|---|
| `DigitalTwin/ekf.py` | EKF state, prediction, and GPS update. |
| `DigitalTwin/motion.py` | Gyro bias, encoder/IMU diagnostics, slip indicators. |
| `DigitalTwin/kinematics.py` | UGV01 tracked-drive geometry. |
| `DigitalTwin/security.py` | GPS-independent predictor and evidence-gated security logic. |
| `DigitalTwin/uncertainty.py` | Fixed/adaptive/learned uncertainty helpers. |
| `DigitalTwin/detector.py` | Mahalanobis/NIS and detectability metrics. |
| `DigitalTwin/alarm.py` | Motion gating and persistent alarm logic. |
| `DigitalTwin/attack.py` | Offline GPS attack injection. |
| `DigitalTwin/dashboard/` | Visual replay dashboard. |
| `DigitalTwin/analysis/` | Reports, training, replay, attack campaign, AprilTag processing. |

## Generated Outputs

These are intentionally ignored or treated as generated:

- `DigitalTwin/datasets/analysis/`
- `results/`
- `DigitalTwin/configs/*_model.pkl`
- `.pytest_cache/`
- `__pycache__/`
- `ugv01_gps_dev/General_Driver/build/`
- device-local Wi-Fi config JSON files under firmware `data/`

Regenerate analysis outputs with documented commands rather than hand-editing
them.

## Raw Log Discipline

Raw logs should not be deleted during cleanup. Instead:

1. Keep formal accepted logs in manifests.
2. Mark debug/calibration logs in `docs/log_quality_audit.md`.
3. Exclude low-value logs in analysis scripts by manifest or filename rule.
4. Preserve exploratory logs if they explain calibration history.

For the current formal benign attack-replay corpus, the useful set is the July
`speed-*_surface-*_route-square0p5x3_attack-none_trial-*` logs accepted by the
`real_data_study` manifest. August 2026 logs are mostly route tuning,
interactive calibration, AprilTag setup, and GPS troubleshooting.

## Current Research Focus

The codebase is now beyond basic rover bring-up. The main research bottleneck is
scientific validation:

- collect cleaner AprilTag ground-truth motion;
- split train/validation/test by complete runs;
- validate physical digital-twin fidelity against AprilTags;
- freeze the model and regenerate threshold/attack reports;
- treat GPS indoors as degraded sensor input, not ground truth.

## Quick Commands

```powershell
python -m pytest -q
python -m DigitalTwin.dashboard.server --open
python -m DigitalTwin.analysis.digital_twin_accuracy
python -m DigitalTwin.analysis.real_data_study --summarize-existing
python bench_logger_square_0_5m.py --surface smooth --speed low --repeats 1 --trial N
python bench_logger_curves.py --route figure8 --repeats 2 --surface smooth_kitchen_floor --speed low --trial N
```
