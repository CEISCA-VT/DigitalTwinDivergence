# Frozen Composite Security Policy

This document freezes the final replay-evaluation security policy used for the
UGV01 digital-twin attack campaign.

## Objective

The final policy is not a single narrow GPS-spoofing detector. It combines the
strongest pieces of the baseline suite into one secure adaptive digital-twin
policy:

- GPS jump guard for abrupt position changes.
- CUSUM-style memory for slow drift.
- Raw digital-twin residual monitoring for prediction disagreement.
- GPS-bias monitoring for persistent offset.
- Evidence-gated adaptive EKF uncertainty so GPS residuals cannot directly
  authorize covariance growth.

## Frozen Implementation

The implemented detector variant is:

```text
composite_ours
```

It is defined in:

```text
DigitalTwin/analysis/real_data_study.py
```

The composite policy uses the GPS-bias evidence-gated EKF as its state estimator.
Its alarm score is the maximum of these normalized evidence channels:

| Channel | Purpose |
|---|---|
| NIS | Standard Kalman innovation consistency check. |
| GPS jump | Detects abrupt GPS discontinuities. |
| Raw residual | Detects large prediction-versus-GPS disagreement. |
| CUSUM memory | Accumulates persistent small innovation evidence. |
| GPS-bias magnitude | Detects persistent offset absorbed by the bias state. |

The final threshold is locked using benign runs only. Attack labels are not used
for threshold selection.

## Evaluation Command

Run the final replay campaign with:

```powershell
python -m DigitalTwin.analysis.real_data_study --bootstrap-iterations 2000
```

For the expanded replay-only grid:

```powershell
python -m DigitalTwin.analysis.real_data_study --expanded-attack-grid --bootstrap-iterations 2000
```

For the full paper-facing result regeneration:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\regenerate_all_results.ps1 -BootstrapIterations 2000
```

## Reporting Language

Use this wording:

> The final policy combines abrupt-change detection, slow-drift memory,
> persistent GPS-bias monitoring, and evidence-gated adaptive EKF uncertainty.
> Thresholds are locked on benign data only, then evaluated against replayed GPS
> attacks using detection probability, false-alarm rate, detection delay, paired
> state divergence, and epsilon thresholds.
