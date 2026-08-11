# Raw Logs

This directory contains raw hardware logs. Treat these as immutable evidence:
do not delete or rewrite CSVs during ordinary cleanup.

## Subdirectories

| Path | Contents |
|---|---|
| `telemetry/` | Combined UGV01 `T:147` telemetry, route runs, debug runs, and motion calibration logs. |
| `static/` | Stationary/static GPS and IMU captures. |

## Formal Versus Debug Logs

The formal benign corpus is selected by analysis manifests, not by everything in
this directory. Many logs are useful only as development evidence:

- `ugv_t147_interactive_*`: manual turn/route calibration.
- `ugv_t147_bench_20260804_*`: GPS and bench troubleshooting.
- `ugv_t147_bench_20260805_*`: AprilTag/bench diagnostic runs.
- August 10 `route-*` logs: route-shape tuning after the main July dataset.
- manually named logs such as `basic_test.csv`: exploratory calibration history.

See `docs/log_quality_audit.md` for a table of low-value formal-result logs and
why they should be excluded from paper metrics.

## Rule

Keep raw logs, but use manifests and reports to decide which logs are included
in formal results.
