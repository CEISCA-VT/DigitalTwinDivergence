# Statistical Attack Campaign

Status: complete for the current 20-run offline replay corpus on July 21,
2026. The frozen specification is
`DigitalTwin/configs/attack_campaign.json`.

## Threat Boundary

Attacks modify only the GPS coordinates supplied to the digital twin during
offline replay. Raw UGV01 logs, firmware, commands, and physical rover behavior
remain unchanged. Each attacked replay is paired with the clean replay of the
same physical run.

## Campaign Matrix

- 20 accepted benign physical runs
- five frozen detector/uncertainty variants
- injection at 25%, 50%, and 70% of each post-motion run horizon
- along-track and cross-track step offsets of `0.5, 1, 2, 3, 5, 7.5, 10 m`
- along-track and cross-track drift rates of `0.01, 0.03, 0.05 m/s`
- along-track and cross-track strategic drift at `0.03 m/s`
- coordinate freeze and five-second value replay

This produces 24 attack profiles and 7,200 paired attacked scenarios. The
generated `campaign_validation.json` confirms that all 120 detector/attack
condition groups contain 20 physical runs and three starts.

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

Directional `epsilon_50` is approximately `6.61-8.06 m`, depending on detector
and direction. Most adaptive variants do not reach 90% or 95% detection within
the `10 m` step grid. At `0.05 m/s` cross-track drift, detection probability is
zero for the evaluated adaptive variants while harmful-but-stealthy probability
ranges from `0.10` to `0.15`; the fixed variant is harmful-but-stealthy in
approximately `0.017` of scenarios.

These are results for the existing design corpus, not an independent
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

Regenerate the report and figures from completed CSV artifacts:

```powershell
python -m DigitalTwin.analysis.real_data_study --summarize-existing
```

The generated outputs are under
`DigitalTwin/datasets/analysis/real_data_study/`.
