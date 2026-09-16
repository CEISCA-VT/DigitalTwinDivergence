# Offline UGV01 Position-Only Contracts

This is a retrospective service redefinition. It does not change the original online contracts or measured policies.
GPS position is an operational reference, not independent physical ground truth. Local displacement-vector comparison is in a common world frame and does not require course.
Pass and fail are counted only after the quality, history, and synchronization gates. Satisfaction uses observable windows; observability uses all windows.

## Paired replay on five physical source traces

| Policy | Global observable | Global satisfaction when observable | Requests/s | Bytes/s |
|---|---:|---:|---:|---:|
| static-low | 1.000 | 0.215 | 1.536 | 951.8 |
| static-high | 1.000 | 0.215 | 1.536 | 951.8 |
| aoi-only | 1.000 | 0.215 | 1.536 | 951.8 |
| contract-aware | 1.000 | 0.215 | 1.536 | 951.8 |

The replay recomputes position-only outcomes at delivered samples; it does not create fresh high-rate measurements. The physical trace is the comparison unit.
