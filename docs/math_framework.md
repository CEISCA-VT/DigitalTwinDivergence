# Digital Twin Divergence Mathematical Framework

This document is the Week 0 freeze point for the math-facing interfaces. It is
aligned with the attached proposal: the core contribution is not merely EKF
localization, but the online characterization of mobility-induced attack
detectability limits.

## State and Inputs

The EKF state is

```text
x_k = [p_x, p_y, theta]^T
```

where `p_x` and `p_y` are local tangent-plane east/north position in meters and
`theta` is rover heading in radians.

Encoder ticks are converted to differential-drive controls:

```text
Delta s_L = r * 2*pi * Delta ticks_L / N
Delta s_R = r * 2*pi * Delta ticks_R / N
v_k       = (Delta s_R + Delta s_L) / (2*Delta t)
omega_k   = (Delta s_R - Delta s_L) / (b*Delta t)
```

where `r` is wheel radius, `N` is ticks per revolution, and `b` is wheel base.

## EKF Prediction

The nonlinear transition is

```text
theta_mid = theta_k + 0.5 * omega_k * Delta t
p_x'      = p_x + v_k*cos(theta_mid)*Delta t
p_y'      = p_y + v_k*sin(theta_mid)*Delta t
theta'    = wrap(theta_k + omega_k*Delta t)
```

The covariance prediction is

```text
P_k|k-1 = F_k P_k-1|k-1 F_k^T + Q_k
```

with local Jacobian

```text
F_k =
[1, 0, -v_k*sin(theta_mid)*Delta t]
[0, 1,  v_k*cos(theta_mid)*Delta t]
[0, 0,  1                         ]
```

## GPS Update

For local GPS coordinates,

```text
z_k = [gps_x, gps_y]^T
h(x_k) = H x_k
H = [1, 0, 0]
    [0, 1, 0]
```

The innovation, innovation covariance, and Kalman gain are

```text
nu_k = z_k - H x_k|k-1
S_k  = H P_k|k-1 H^T + R_k
K_k  = P_k|k-1 H^T S_k^-1
```

The corrected state and covariance are

```text
x_k|k = x_k|k-1 + K_k nu_k
P_k|k = (I - K_k H) P_k|k-1 (I - K_k H)^T + K_k R_k K_k^T
```

The Joseph covariance update is used in code for numerical stability.

## Null and Alternative Hypotheses

Under nominal operation,

```text
nu_k ~ N(0, S_k)
delta_k = nu_k^T S_k^-1 nu_k
delta_k | H0 approximately chi-square_m
```

For GPS position updates, `m = 2`.

Under a semantic GPS spoofing attack with injected vector `a_k`,

```text
z_k^a  = z_k + a_k
nu_k^a = nu_k + a_k
nu_k^a ~ N(a_k, S_k)
delta_k^a | H1 approximately noncentral chi-square_m(mu)
mu = a_k^T S_k^-1 a_k
```

## Threshold and Noncentrality Requirement

For an allowable false alarm probability `P_FA = alpha`, the detector threshold
is

```text
gamma_star = F_chi-square_m^-1(1 - alpha)
```

The default implementation uses `alpha = 0.05`, matching the proposal's
threshold-locking phase.

To guarantee a target detection probability `P_D = beta`, define `lambda_star`
as the noncentrality parameter satisfying

```text
1 - F_noncentral-chi-square_m(lambda_star)(gamma_star) = beta
```

The code uses SciPy for this solve when available and falls back to a
conservative analytic approximation otherwise.

## Minimum Detectability Boundary

The proposal's structural detectability boundary is eigenvalue-based:

```text
epsilon_min(v, tau, l) = sqrt(lambda_star * lambda_max(S_k(v, tau, l)))
```

This follows from the Rayleigh quotient for the positive-definite innovation
covariance matrix. As velocity, terrain roughness, or packet latency increase,
the learned process covariance `Q_k` inflates, which propagates into `S_k` and
therefore increases `epsilon_min`.

## Instantaneous Maximum Stealth Bound

The same eigenvalue relationship gives the analytical stealth bound when the
noncentrality limit is replaced by the detector threshold:

```text
epsilon_stealth_max = sqrt(gamma_star * lambda_max(S_k))
```

Attack vectors larger than this bound force the noncentrality parameter beyond
the surrogate stealth constraint used in the proposal.

## Frozen Uncertainty Variants

The study compares four preregistered uncertainty definitions. The fixed
variant uses constant `Q` and `R`. The naive-adaptive variant treats process
noise as a deterministic function of rolling telemetry statistics:

```text
Q_k = g(r_k, sigma_IMU, sigma_v, Delta t_k)
```

The naive-adaptive feature vector is

```text
phi_k = [
  dead_reckoning_residual_m,
  imu_vertical_std,
  imu_yaw_std,
  velocity_variance,
  packet_dt_s
]
```

where:

```text
r_k        = sliding-window norm between GPS position and encoder prediction
sigma_IMU  = rolling vibration/yaw-rate variability
sigma_v    = rolling velocity variance
Delta t_k  = edge-observed timing stress: arrival gap plus packet age
```

The GPS-independent variant uses the same mapping with `r_k = 0`. The
evidence-gated variant admits residual feedback only when previous NIS is below
threshold, the packet is not stale, and independent IMU or timing evidence is
present. Exact coefficients and gates are frozen in
`DigitalTwin/configs/uncertainty_policies.json`.

The learned candidate uses no GPS coordinate or innovation residual as an
input. Its six deployment features are rolling vertical-acceleration standard
deviation, yaw-rate standard deviation, velocity variance, packet interval,
HDOP, and satellite count. The frozen offline target is the median
five-update-ahead process-error covariance surrogate derived from aligned
GPS-versus-encoder/IMU displacement error and IMU-versus-encoder heading error.

Complete-run grouped validation on 20 benign runs rejected the current Random
Forest because it was worse than the training-fold median baseline for all
three covariance targets. It is therefore not active in the primary campaign.
See `docs/uncertainty_policy_freeze.md` for the target construction, metrics,
and activation rule.

## Confidence Envelopes

The digital twin maps structural sensitivity and divergence into an operational
confidence score `C_k in [0, 1]`.

```text
Safe:    C_k >= 0.90
Warning: 0.50 <= C_k < 0.90
Blind:   C_k < 0.50
```

The implementation decreases confidence when the Mahalanobis statistic
approaches the detector threshold or when `epsilon_min` approaches the 5 meter
blindness criterion described in the proposal.

## Research Hypotheses

H1 Detectability Boundary Scaling: `epsilon_min` scales monotonically with
velocity `v`, terrain roughness `tau`, and communication latency `l`.

H2 Learned Uncertainty Superiority: an online uncertainty regressor `g(.)`
trained on sliding-window telemetry statistics maintains the locked false alarm
probability across changing environments better than a static hand-tuned filter.

H3 Operational Envelope Reliability: safe, warning, and blind regions based on
the structural scale of `S_k` identify intervals where `epsilon_min > 5 m` and
trustworthy attack detection becomes theoretically weak.
