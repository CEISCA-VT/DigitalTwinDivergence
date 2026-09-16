# ugv01_live_contract

This dataset packages the UGV01 smooth-floor live-contract validation trials.
It is organized as paired video and dashboard telemetry for each policy/trial arm.

## Layout

- `trials/<policy>/trial_XX/video.mp4`: phone video used for AprilTag/visual reference.
- `trials/<policy>/trial_XX/telemetry.csv`: dashboard-exported UGV01 telemetry.
- `trials/<policy>/trial_XX/live_contract.jsonl`: live dashboard contract and resource-policy trace.
- `analysis/`: generated live-contract summaries copied from the source trial analysis folder.
- `dataset_manifest.csv`: one row per policy/trial with source paths and basic quality metadata.
- `dataset_summary.json`: machine-readable aggregate summary.

## Trial Matrix

| Policy | Trials | Video duration mean (s) | Telemetry rows mean | GPS valid fraction mean |
| --- | ---: | ---: | ---: | ---: |
| `aoi-only` | 5 | 100.5 | 131.0 | 1.000 |
| `contract-aware` | 5 | 98.7 | 131.2 | 1.000 |
| `static-high` | 5 | 101.6 | 130.8 | 1.000 |
| `static-low` | 5 | 114.8 | 123.0 | 1.000 |

## Intended Use

Use the videos plus telemetry CSV files for UGV01 physical-reference/digital-twin fidelity analysis.
Use live-contract JSONL logs as the primary evidence for online contract state, policy decisions, AoI, and requested update-rate behavior.

## Limitations

- The current package is a single indoor smooth-floor dataset.
- The packaged CSV telemetry is the stable source for the current AprilTag/fidelity tooling.
- Matching per-run JSONL files are present for 20/20 trials.
- The canonical paper dataset contains only the matched final-trial logs; development/debug logs are excluded.
- Requested policy rates were logged, but the delivered stream rate remained close to 1.5 Hz across policies; this package alone does not demonstrate delivered communication savings.
- GPS was valid in the telemetry exports, but GPS is an operational sensor rather than independent ground truth.
