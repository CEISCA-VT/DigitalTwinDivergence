# Statistical Attack Campaign

Status: revised-model campaign implementation prepared for the current 20-run
offline replay corpus. The frozen specification is
`DigitalTwin/configs/attack_campaign.json`.

## Threat Boundary

Attacks modify only the GPS coordinates supplied to the digital twin during
offline replay. Raw UGV01 logs, firmware, commands, and physical rover behavior
remain unchanged. Each attacked replay is paired with the clean replay of the
same physical run.

## Campaign Matrix

- 20 accepted benign physical runs
- thirteen detector/model variants after the revised-model update
- injection at 25%, 50%, and 70% of each post-motion run horizon
- along-track and cross-track step offsets of `0.5, 1, 2, 3, 5, 7.5, 10 m`
- along-track and cross-track drift rates of `0.01, 0.03, 0.05 m/s`
- along-track and cross-track strategic drift at `0.03 m/s`
- coordinate freeze and five-second value replay

This produces 24 attack profiles and 1,440 unique attack-run-start
combinations. With 13 detector/model variants, the baseline-transport campaign
contains 18,720 detector-run evaluations. The generated
`campaign_validation.json` confirms that every detector/attack condition group
contains 20 physical runs and three starts.

## Statistical Evaluation

The physical run is the independent sampling unit. The three starts within a
run are kept together using stratified physical-run cluster bootstrap
resampling with 2,000 iterations. Reported outputs include:

- detection probability with 95% confidence intervals
- detected-case median and p95 alarm delay, with non-detections reported as
  censored
- harmful-but-stealthy probability relative to the `5 m` mission tolerance
- maximum undetected paired state deviation and time above tolerance
- attacked/clean `Q` and `S` trace ratios
- directional `epsilon_50`, `epsilon_90`, and `epsilon_95`

Step detection curves are made monotone with weighted isotonic regression
before interpolating epsilon values. Estimates above the largest tested step
are reported as `>10 m`; confidence limits that extend outside the tested grid
are explicitly marked as censored.

## Current Headline Results

After rerunning the campaign, quote results from
`DigitalTwin/datasets/analysis/real_data_study/real_data_study_report.md`,
`post_campaign_report.md`, and `epsilon_summary.csv`. Older headline numbers
remain provenance only because the revised GPS-bias variants change the
detector/model count and the comparison set.

These results are for the existing design corpus, not an independent
prospective test. They support analysis of this dataset but must not be framed
as population-level deployment guarantees.

The follow-on paired covariance analysis found statistically measurable Q
inflation and NIS suppression from residual-coupled adaptation, but no
established operational detection or harmful-probability disadvantage against
the frozen-clean control. See `docs/covariance_poisoning_analysis.md`.

## Reproduction

Run the full campaign:

```powershell
python -m DigitalTwin.analysis.real_data_study
```

Run the full paper-facing regeneration pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\regenerate_all_results.ps1
```

Regenerate the report and figures from completed CSV artifacts:

```powershell
python -m DigitalTwin.analysis.real_data_study --summarize-existing
```

The generated outputs are under
`DigitalTwin/datasets/analysis/real_data_study/`.
