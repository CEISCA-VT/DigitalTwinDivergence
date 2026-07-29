# Revised Mathematical Architecture

Status: implemented from `Digital_Twin_Divergence_draft.pdf` on July 29, 2026.

## State Separation

The replay now maintains three explicitly different quantities:

1. `SecurityPredictor`: encoder-driven EKF-style state and covariance
   propagation with no GPS correction during monitoring.
2. `RoverEKF`: the GPS-fused operational navigation estimate.
3. Independent camera/AprilTag ground truth: not yet collected and therefore
   not available to either branch.

The security innovation is

```text
nu_k = z_k - h(x_security,k|k-1)
S_k  = H P_security,k|k-1 H' + R_k
NIS  = nu_k' inv(S_k) nu_k
```

This prevents an attacked GPS posterior from becoming the reference used to
judge the next attacked GPS coordinate.

## Covariance Adaptation

`BoundedCovarianceAdapter` clips every diagonal process-covariance proposal
between configured `Q_min` and `Q_max`, then applies

```text
Q_k = phi_Q Q_(k-1) + (1 - phi_Q) Q_proposal,k
```

with `phi_Q = 0.9`. Naive adaptation may still use the rolling GPS residual as
the deliberate vulnerability baseline. GPS-independent and evidence-gated
adaptation do not.

## Trusted Evidence Gate

The gate whitens the security innovation using the GPS-independent trusted
covariance, not a residual-inflated covariance:

```text
w_k = S_trusted^(-1/2) nu_k
c_k = lambda_c c_(k-1) + (1 - lambda_c) w_k
T_bias = ||c_k||^2 / sigma_c,k^2
```

An update is accepted only when protected packet/timing evidence is healthy,
trusted NIS is below the soft limit, and `T_bias` is below the persistent-bias
limit. Otherwise the last accepted covariance is frozen.

## Implementation Map

- `DigitalTwin/security.py`: security predictor, covariance bounds/smoothing,
  and trusted-whitened gate.
- `DigitalTwin/analysis/real_data_study.py`: paired security/operational replay
  and exported gate, innovation, covariance, and state fields.
- `DigitalTwin/analysis/digital_twin_accuracy.py`: benign sensor-agreement,
  NIS-calibration, covariance-mismatch, and loop-closure report.
- `DigitalTwin/dashboard/`: visualizes GPS, operational EKF, and security
  predictor together.

## Gate Freeze and Remaining Alarm Re-Freeze

The gate limits were frozen at the 95th per-update quantiles from 1,797
monitored updates in the 12 complete benign development runs:

```text
gamma_soft = 10.480551254279684
h_c        = 155.30477241316595
lambda_c   = 0.90
```

The architecture change invalidates the numerical comparability of the old
alarm thresholds and attack-campaign results. Before rerunning attacks:

1. lock matched run-level alarm thresholds under the new security NIS;
2. verify held-out benign false alarms;
3. regenerate the attack campaign and figures.

No new rover run is required for this software re-freeze. Physical accuracy
still requires the planned synchronized AprilTag reference.
