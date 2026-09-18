# MAGNET thermal timing study: provenance stop gate

**Decision: causal delivery replay is not identifiable from the local paired MAGNET archive.** No timing or remediability result was computed. This is a deliberate stop under the study protocol, not a negative thermal result.

## Sources and correspondence

The local files are `results/magnet_tfp/data/MAGNET_Heat_Pipe_2022-03-30.csv` (physical, SHA-256 `bdc139c1a2d368308eea85bd487e640f84d7bc1eff495750f629c6efecb0e9ac`) and `results/magnet_tfp/data/ML_MAGNET_2022-03-30.csv` (virtual forecasts, SHA-256 `2eff12a7fe80c501d5b4f0e4df706c08829f21bc7f4b1b995b89bf5b8feccffa`). They correspond by ten identically named `Heat Pipe TC-01` through `TC-10` thermowell channels. Existing MAGNET analysis treats these as temperatures in degrees Celsius; the CSV header itself does not encode a unit. `Heater Temp` exists only in the physical file and is not paired to a virtual output here.

The [official INL dataset README](https://github.com/IdahoLabResearch/MAGNET-Heat-Pipe-Data) describes the `Machine_Learning/Multiple_Files` entries as forecasts of the **next ten minutes**, and `Single_File` as their combined forecast data. The [published experiment description](https://doi.org/10.1016/j.pnucene.2023.104813) says the thermal twin used LASSO variable selection followed by VAR multivariate forecasting. Thus the local virtual rows are **forecast values for target times**, not a 1-Hz stream of current-state estimates.

| Property | Physical file | Virtual single file |
|---|---:|---:|
| Rows | 80,104 | 66,600 |
| Clock column | `Time (s)` | `Time (s)` |
| Time range, s | 20-80,123 | 3,799-23,228 |
| Native within-window cadence | 1 s | 1 s |
| Forecast windows | n/a | 111 x 600 rows |
| Unique target times | 80,104 | 17,160 |
| Missing per TC-01...09 | 0 | 7,800 |
| Missing TC-10 | 0 | 8,400 |
| Issue-time column | n/a | **absent** |
| Arrival/receipt-time column | n/a | **absent** |

The physical `Date/Time` column is a physical-record timestamp and cannot supply the forecast issue time. The virtual `Time (s)` column advances within each 600-row forecast window and identifies target/valid time. It **resets between overlapping windows** (105 nonpositive row-to-row time changes). Example: target second 4,000 occurs in both the first and second forecast windows, but the TC-06 predictions differ (372.245 versus 369.781 C). Without issue times, there is no defensible ordering saying which value was known at second 4,000. Window filenames in the remote `Multiple_Files` directory display target-time ranges, not independently documented issue/arrival timestamps. The local archive contains only the two single files.

## Prior MAGNET work and preserved limitations

The existing `results/magnet_tfp/results/` analysis compares forecast target values with physical values at corresponding target seconds. Its eligibility audit reports 97 strict-eligible windows and 23 greedily selected non-overlapping windows for dependence-reduced horizon analysis. It uses physical p95-p05 scaling, an exploratory 5 C persistence threshold, and early-horizon robust-error envelopes. The cross-domain contract study uses descriptive 60/300/600 s horizons and a fixed normalized tolerance grid. These are **forecast-fidelity** analyses, not evidence that any particular forecast was available to a live monitor by its target time. The local paired files do not include a documented train/validation/test split or forecast-generation log. No prior calculation in the repository resolves issue or arrival time.

## Why the requested replay must stop

At evaluation time `t`, selecting the newest *arrived* forecast requires at least an independently recorded issue time for each forecast window, a documented target-time mapping, and an arrival time or a predeclared delivery-delay model applied to that issue time. Assigning `issue_time = first_target_time`, `first_target_time - 1`, or a file-order-derived time would be an unverified assumption prohibited by the protocol. Selecting a forecast by target time alone would leak unavailable future predictions and would give a meaningless age of represented physical information.

Consequently, none of the following is identifiable from this archive: a causal ideal-versus-held thermal discrepancy, a scalar age-versus-discrepancy comparison, qualification under degraded/practical delivery, remediable-versus-persistent case counts, or chronological transfer of a timing surface. `protocol_manifest.json` records these as blocked rather than inventing rate, delay, freshness, or service thresholds. No timing figures were generated; existing MAGNET forecast-horizon figures must not be relabeled as delivery-replay figures.

This does **not** invalidate the earlier target-aligned thermal forecast-fidelity analysis. It means that analysis answers a different question from the proposed causal synchronization study.

## Minimum missing provenance

An authoritative per-window record of `(window_id, issue_time_s, first_target_time_s, last_target_time_s)` plus either receipt time or a defensible arrival model is needed. The issue-time field must reflect when all forecast values in a window became available, including computation latency if relevant. A source document showing that a specific filename field is the issue time could also suffice; a timestamp range alone cannot. With such evidence, the existing physical/forecast files could be reused without retraining or new thermal measurements. Overlapping forecast windows and correlated thermowells would still require blocked temporal evaluation, not independent-window inference.

## Reproduction

From repository root:

```powershell
python -m DigitalTwin.analysis.magnet_timing_provenance
python -m pytest tests/test_magnet_timing_provenance.py -q -p no:cacheprovider
```

Outputs: `provenance_manifest.json`, `protocol_manifest.json`, and `forecast_window_audit.csv`. Source files are read-only. The guard tests verify that target time is never silently promoted to issue time and absent arrival time is not fabricated.

## Manuscript recommendation

Do **not** add a MAGNET timing/remediability figure to the main paper or supplement. At most retain one existing target-aligned forecast-horizon fidelity figure in the supplement, explicitly labeled as descriptive forecast fidelity. The main timing mechanism remains supported by the i2Nav replay only until actual MAGNET forecast-availability provenance is obtained.
