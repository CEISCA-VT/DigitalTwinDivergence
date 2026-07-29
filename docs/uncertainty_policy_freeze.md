# Uncertainty Policy Freeze

Status: revised architecture and evidence gate frozen on the existing benign
development split July 29, 2026. Matched run-level alarm thresholds still
require regeneration before the attack campaign is rerun.

The machine-readable specification is
`DigitalTwin/configs/uncertainty_policies.json`. Changes to features,
coefficients, gates, targets, or acceptance rules require a schema-version
increment and a new benign-only validation pass before attack evaluation.

## Frozen Variants

| Variant | Process covariance | Measurement covariance | GPS residual feedback |
|---|---|---|---|
| Fixed | Constant position sigma `0.05 m/s` and heading sigma `0.01 rad/s`, scaled by update time | Constant GPS sigma `1.75 m` | No |
| Naive adaptive | Deterministic function of previous GPS/dead-reckoning residual, rolling IMU variation, velocity variance, and packet timing mismatch | Deterministic HDOP, satellite-count, and timing rule | Yes, without an independent evidence gate |
| GPS-independent | Same deterministic adaptive coefficients, with the GPS/dead-reckoning residual forced to zero | Same rule as naive adaptive | No |
| Evidence-gated | Bounded, smoothed GPS-independent proposal; rejected updates freeze the last accepted covariance | Same rule as GPS-independent | No |

The revised evidence gate passes only when:

- the packet is not stale and sequence continuity is intact
- edge/source update-time mismatch is at most `0.20 s`
- trusted NIS is at most `10.4806`
- the exponentially weighted trusted-whitened bias score is at most `155.3048`

These are the 95th per-update quantiles from 1,797 monitored updates in the 12
complete benign development runs. The relatively high bias threshold records
the existing encoder/GPS model mismatch rather than hiding it. Attack outcomes
were not used.

## Learned Target

The old placeholder target has been replaced by an offline benign
process-error covariance surrogate. For each raw successful GPS-valid `T:147`
run:

1. Convert GPS coordinates to a local metric frame.
2. Convert encoder ticks to traveled distance using the locked UGV01 geometry
   and combine it with IMU yaw.
3. Align encoder/IMU displacement to the GPS local frame using one rotation
   fitted per complete benign run. This alignment is used only to make labels.
4. Define `q_xx` and `q_yy` from future per-axis squared displacement error,
   after subtracting the initial stationary GPS step-noise estimate.
5. Define `q_tt` from future squared wrapped disagreement between IMU yaw
   increments and encoder-predicted yaw increments.
6. Use the median over the next five updates and clip target tails at the
   benign-corpus 1st and 99th percentiles.

Deployment features are causal and contain no GPS coordinate residual:
rolling vertical-IMU standard deviation, yaw-rate standard deviation, velocity
variance, packet interval, GPS HDOP, and satellite count. Complete runs, not
individual rows, form the grouped cross-validation folds. Attack rows are
forbidden from training and model selection.

Here, GPS-independent means independent of GPS coordinates and innovations;
HDOP and satellite count remain allowed as receiver-reported quality metadata.

## Current Model Decision

The current corpus contains 20 benign runs and 3,043 training rows. Five-fold
complete-run validation found that the Random Forest candidate was worse than
the training-fold median baseline:

| Target | Random Forest MAE | Median baseline MAE | Relative improvement |
|---|---:|---:|---:|
| `q_xx` | 0.02205 | 0.01602 | -37.6% |
| `q_yy` | 0.02217 | 0.01654 | -34.0% |
| `q_tt` | 0.03142 | 0.02675 | -17.5% |

The learned target is frozen, but this trained candidate is rejected and is
not enabled in the primary attack campaign. This prevents a weak model from
being presented as an improvement. The deterministic fixed, naive-adaptive,
GPS-independent, and evidence-gated variants remain the primary comparison.

Reproduce the training and decision with:

```powershell
python -m DigitalTwin.analysis.train_uncertainty
```

No additional rover run is required for this freeze. A future learned model
may be activated only if it improves complete-run cross-validated MAE for all
three targets and then survives a separate prospective benign validation set.
