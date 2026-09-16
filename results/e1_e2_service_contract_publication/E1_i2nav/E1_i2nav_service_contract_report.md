# E1 — Deep service-relative fidelity on frozen i2Nav

## Frozen evidence guard

- Input sequences: 10; expected frozen set matched.
- Source full-LOSO commit: `6540c01f90f3c1074de0d8dae9964a5276fbbc91`.
- No model training, threshold optimization, or post-hoc service redefinition is performed here.

## Full-grid parking00 ↔ parking02 inversion

- global: parking02 wins 0/36, parking00 wins 36/36; mean pass-rate difference -0.385.
- local 1 s: parking02 wins 25/25, parking00 wins 0/25; mean pass-rate difference +0.015.
- local 5 s: parking02 wins 25/25, parking00 wins 0/25; mean pass-rate difference +0.158.
- local 10 s: parking02 wins 25/25, parking00 wins 0/25; mean pass-rate difference +0.223.

This is stronger than a single selected operating point: the local/global reversal persists over every tested point of the predeclared tolerance sweep.

## Why one scalar fidelity number is inadequate

- Global-service pass rate vs ATE: median Kendall tau = 0.708.
- Global-service pass rate vs RPE10: median Kendall tau = 0.151.
- Local-1s pass rate vs ATE: median Kendall tau = 0.200; vs RPE1 = 0.692.
- Local-5s pass rate vs ATE: median Kendall tau = 0.277; vs RPE5 = 0.674.
- Local-10s pass rate vs ATE: median Kendall tau = 0.333; vs RPE10 = 0.762.

Interpretation: ATE is informative for global synchronized-state validity but weakly ranks local service validity; finite-horizon RPE is substantially more aligned with local service validity but weak for global validity. The metrics are not 'wrong'—they answer different service questions.

## Threshold-robust sequence ordering

- Local 1 s vs global: 17/45 sequence-pair orderings invert; grid-average Kendall tau 0.244.
- Local 5 s vs global: 17/45 sequence-pair orderings invert; grid-average Kendall tau 0.244.
- Local 10 s vs global: 15/45 sequence-pair orderings invert; grid-average Kendall tau 0.333.

## E1 claim boundary

E1 supports a service-relative interpretation of fidelity. It does not claim that local/global drift is a newly discovered mathematical phenomenon or that ATE/RPE are invalid metrics.
