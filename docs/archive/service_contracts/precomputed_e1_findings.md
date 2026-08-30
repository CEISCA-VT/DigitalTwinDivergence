# Frozen E1 findings already verified from current service-relative outputs

These findings are computed from the already-audited 10-sequence frozen service grid and should be reproduced by the package.

## Full-grid parking inversion
- Global synchronized-state service: parking00 beats parking02 at 36/36 tolerance-grid points; mean parking02-minus-parking00 service-pass difference = -0.385.
- Local 1 s service: parking02 beats parking00 at 25/25 grid points; mean difference = +0.015.
- Local 5 s service: parking02 beats parking00 at 25/25 grid points; mean difference = +0.158.
- Local 10 s service: parking02 beats parking00 at 25/25 grid points; mean difference = +0.223.

Thus the parking00/parking02 reversal is not caused by one illustrative threshold.

## Scalar metric/service alignment over the complete grid
Median Kendall tau between service pass rate and negative error (higher agreement is better):
- global service vs ATE: 0.708
- global service vs RPE10: 0.151
- local 1 s vs ATE: 0.200; vs RPE1: 0.692
- local 5 s vs ATE: 0.277; vs RPE5: 0.674
- local 10 s vs ATE: 0.333; vs RPE10: 0.762

Interpretation: ATE and finite-horizon RPE are useful, but they rank different service claims differently. This supports service-relative validity rather than invalidating either metric.

## Threshold-robust sequence-order inversions
Using each sequence's grid-average service validity:
- local 1 s vs global: 17/45 pairwise sequence orderings invert; Kendall tau = 0.244.
- local 5 s vs global: 17/45 invert; tau = 0.244.
- local 10 s vs global: 15/45 invert; tau = 0.333.
