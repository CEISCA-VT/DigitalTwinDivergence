# Covariance-Poisoning Analysis

Status: complete for the current 20-run design corpus on July 21, 2026. The
analysis specification is
`DigitalTwin/configs/covariance_poisoning_analysis.json`.

## Question

Does allowing attacked GPS residuals to increase adaptive process covariance
help a slow-drift attacker remain hidden or cause more state error?

The primary comparison is naive residual-coupled adaptation versus
frozen-clean covariance. Both use the same clean covariance schedule; during
the attacked replay, only the naive variant can change covariance in response
to the attacked GPS residual. Fixed, GPS-independent, and evidence-gated
variants are secondary controls.

## Method

The targeted matrix contains all 20 physical runs, three injection times,
along-track and cross-track drift at `0.01, 0.03, 0.05 m/s`, and strategic
drift at `0.03 m/s`. This produces 2,400 attacked replays and 1,920
scenario-matched naive-minus-control comparisons.

Effects are calculated only during the active attack window. Confidence
intervals use stratified physical-run cluster bootstrap resampling. Two-sided
sign-flip tests use each physical run's mean paired effect. The measured
outcomes are attacked/clean `Q`, `S`, and NIS ratios, residual-feedback gate
activation, maximum undetected paired state error, harmful-but-stealthy
probability, and detection probability.

## Primary Result

For pooled standard drift, naive adaptation compared with frozen-clean gives:

| Outcome | Paired effect with 95% CI |
|---|---:|
| Attacked/clean `Q` ratio | `+0.0243 [0.0185, 0.0300]` |
| Attacked/clean `S` ratio | `+0.0009 [0.0007, 0.0011]` |
| Attacked/clean NIS ratio | `-0.0135 [-0.0174, -0.0090]` |
| Maximum undetected state error | `+0.013 m [0.009, 0.017]` |
| Harmful-but-stealthy probability | `+0.006 [0.000, 0.010]` |
| Detection probability | `0.000 [0.000, 0.000]` |

Residual feedback therefore produces statistically measurable covariance
inflation and NIS suppression. The isolated state-error increase is about
`1.3 cm`, detection probability does not change, and the harmful-probability
interval includes zero.

## Conclusion

**The covariance-poisoning mechanism is supported, but operational attacker
advantage is not established on the current corpus.**

The operational decision requires lower detection probability, a clearly
higher harmful probability, or at least `0.10 m` additional undetected state
error. This practical threshold prevents a statistically detectable but tiny
effect from being presented as a safety failure.

The evidence gate admits residual feedback during approximately `36.6%` of
attack-window updates. Its activation rate is essentially identical in clean
and attacked replay, meaning the tested GPS drift does not directly open the
gate. The gate limits exposure to independently indicated motion or timing
windows, but it is not a drift classifier.

## Reproduction

```powershell
python -m DigitalTwin.analysis.covariance_poisoning
```

To regenerate statistics and figures from the targeted replay CSV:

```powershell
python -m DigitalTwin.analysis.covariance_poisoning --summarize-existing
```

Generated artifacts are under
`DigitalTwin/datasets/analysis/covariance_poisoning/`. These are offline paired
replay results against the clean digital-twin trajectory, not independent
overhead ground truth or prospective deployment validation.
