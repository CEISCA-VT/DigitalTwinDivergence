# UGV01 Live Contract Experiment Summary

This report summarizes JSONL logs produced by the live UGV01 service-contract dashboard.
Unobservable service windows are retained as unobservable; they are not counted as successful contract satisfaction.
Duration and delivered response rates use the source clock when monotone, falling back to edge arrival time. Request counts include successful responses only.
Historical contract states and GPS disagreement were recorded before the frame/age corrections; these columns are retained as historical observations, not retroactively validated outcomes.
The saved AoI is excess over the minimum observed transport/clock offset, not independently verified absolute age.

## Campaign Status

- Runs analyzed: 1
- Policies present: contract-aware
- Runs with any observable contract samples: 0
- Required final policy set: static-low, static-high, aoi-only, contract-aware

## Policy-Level Summary

| Physical condition | Wireless condition | Policy | Runs | Observable | Qualified | Withdrawn | p95 AoI (s) | Bytes/s | Requests/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  | contract-aware | 1 | 0.000 | 0.000 | 0.000 | 0.184 | 1116.6 | 0.65 |

## Per-Service Summary

| Run | Policy | Service | Observable | Qualified | At risk | Withdrawn | Unobservable | p95 position (m) | p95 heading (deg) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ugv01_live_contract_20260827_192535 | contract-aware | Immediate motion | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| ugv01_live_contract_20260827_192535 | contract-aware | Short prediction | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| ugv01_live_contract_20260827_192535 | contract-aware | Planning support | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |
| ugv01_live_contract_20260827_192535 | contract-aware | Global asset tracking | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |  |  |

## Interpretation Rules

- `qualified` and `at_risk` are observable service windows that remain within the declared contract.
- `withdrawn` means an observable contract exceeded position, heading, or AoI limits.
- `unobservable` usually means GPS position/course quality was unavailable for that service.
- The final IoT-J claim requires repeated matched runs across all four policies, not a single dashboard smoke test.
