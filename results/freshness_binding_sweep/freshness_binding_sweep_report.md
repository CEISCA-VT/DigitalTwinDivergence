# Freshness-binding sweep

This analysis replays the frozen 10-Hz V2 states under tighter AoI limits and bursty outages.
It is a sensitivity study over saved trajectories, not a new model-training or live-network result.

## Main answer

- Freshness binds under the tight/bursty sweep: **308** sequence-service-condition cases have `freshness_only` or `physical_and_freshness` failures at AoI 0.2-0.3 s.
- `freshness_only` means physical satisfaction reaches the 0.80 requirement but freshness does not.
- `physical_and_freshness` means both physical discrepancy and freshness fail.

## Component counts

| aoi_setting | condition | freshness_only | none | physical_and_freshness | physical_only |
| --- | --- | --- | --- | --- | --- |
| aoi_0.2s | burst_2hz_2s_every_10s | 0 | 0 | 40 | 0 |
| aoi_0.2s | burst_5hz_1s_every_8s | 5 | 0 | 35 | 0 |
| aoi_0.2s | degraded_2hz | 2 | 0 | 38 | 0 |
| aoi_0.2s | ideal_10hz | 0 | 34 | 0 | 6 |
| aoi_0.2s | nominal_5hz | 19 | 12 | 9 | 0 |
| aoi_0.2s | stress_1hz | 0 | 0 | 40 | 0 |
| aoi_0.3s | burst_2hz_2s_every_10s | 0 | 0 | 40 | 0 |
| aoi_0.3s | burst_5hz_1s_every_8s | 0 | 5 | 0 | 35 |
| aoi_0.3s | degraded_2hz | 2 | 0 | 38 | 0 |
| aoi_0.3s | ideal_10hz | 0 | 34 | 0 | 6 |
| aoi_0.3s | nominal_5hz | 0 | 31 | 0 | 9 |
| aoi_0.3s | stress_1hz | 0 | 0 | 40 | 0 |
| aoi_0.5s | burst_2hz_2s_every_10s | 0 | 0 | 40 | 0 |
| aoi_0.5s | burst_5hz_1s_every_8s | 0 | 5 | 0 | 35 |
| aoi_0.5s | degraded_2hz | 2 | 0 | 38 | 0 |
| aoi_0.5s | ideal_10hz | 0 | 34 | 0 | 6 |
| aoi_0.5s | nominal_5hz | 0 | 31 | 0 | 9 |
| aoi_0.5s | stress_1hz | 0 | 0 | 40 | 0 |
| original | burst_2hz_2s_every_10s | 0 | 0 | 10 | 30 |
| original | burst_5hz_1s_every_8s | 0 | 5 | 0 | 35 |
| original | degraded_2hz | 0 | 2 | 7 | 31 |
| original | ideal_10hz | 0 | 34 | 0 | 6 |
| original | nominal_5hz | 0 | 31 | 0 | 9 |
| original | stress_1hz | 0 | 0 | 30 | 10 |

## Transport summary

| condition | requested_rate_hz | realized_rate_hz | loss_fraction | burst_loss_fraction | delay_p95_ms |
| --- | --- | --- | --- | --- | --- |
| burst_2hz_2s_every_10s | 2 | 1.52 | 0.24 | 0.2004 | 282.2 |
| burst_5hz_1s_every_8s | 5 | 4.33 | 0.134 | 0.1254 | 82.85 |
| degraded_2hz | 2 | 1.897 | 0.05173 | 0 | 282 |
| ideal_10hz | 10 | 10 | 0 | 0 | 0 |
| nominal_5hz | 5 | 4.949 | 0.0103 | 0 | 83.07 |
| stress_1hz | 1 | 0.9027 | 0.09748 | 0 | 464.7 |
