# Frozen protocol: causal online service-risk estimation

## Question
Can online-observable context predict whether a frozen sensor-lightweight digital twin will violate a **specific service requirement**, without using ground truth at decision time?

## Frozen primary services
- Local 1 s: position <= 0.10 m AND heading <= 2 deg.
- Local 5 s: position <= 0.20 m AND heading <= 5 deg.
- Local 10 s: position <= 0.50 m AND heading <= 10 deg.
- Global synchronized state: position <= 1.0 m AND heading <= 5 deg.

These are stress-test service definitions, not safety standards.

## Causality
For local horizon h, features are evaluated at or before the window start t; the label uses physical-reference error at t+h. For the global service, features and the current synchronized-state validity label share t, but no physical-reference error is supplied as a feature.

Forbidden features include GT/reference pose, future samples, local/global error, true learned-correction targets, and remaining prediction error.

## Models
- Constant-risk baseline: development-sequence failure prevalence only.
- M0 elapsed-only logistic model.
- M1 instantaneous causal context logistic model.
- M2 instantaneous + causal-history logistic model.

M2 history emphasizes signed wheel-IMU disagreement persistence. No neural network is introduced.

## Nested evaluation
Outer loop: leave one complete physical sequence untouched for final testing.
Inner loop: three group-disjoint folds over the other nine physical sequences calibrate the operational risk cutoff. Logistic regularization is frozen at C=1.0 before the signed-context run. No timestamp-random split is permitted.

## Primary evaluation
Continuous: sequence-macro Brier, ROC-AUC/PR-AUC (secondary), and risk-coverage area (AURC; lower is better).
Operational: support rate, unsafe-among-supported, false-safe, false-reject, and valid-captured fraction at inner-selected target unsafe rates.

Risk-coverage comparisons are matched by coverage, preventing a monitor from winning by rejecting nearly everything.

## GO/NO-GO boundary
A `GO` requires the history model to improve AURC over current-context and constant-risk baselines across at least 3/4 services, with sequence-level majority support on at least 3 services, plus at least 5% relative AURC gain vs constant risk on those services. Otherwise the online-monitor claim is not promoted.

The service-relative fidelity specification remains a separate contribution regardless of this verdict.
