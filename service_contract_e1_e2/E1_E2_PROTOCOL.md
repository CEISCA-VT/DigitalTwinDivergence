# Frozen E1/E2 protocol

## E1: i2Nav depth
E1 consumes the already-audited `results/service_relative_fidelity` outputs. It does not rerun or tune Twin V2.

Primary tests:
1. Full-grid parking00↔parking02 dominance, not a single selected threshold.
2. Grid-average local-vs-global sequence ordering and pairwise inversions.
3. Kendall rank alignment between service validity and conventional scalar errors (ATE, RPE1/5/10, Dp/Dtheta tails).

Interpretation rule: conventional errors are not called wrong. Evidence that different metrics rank different service claims differently supports service-relative validity.

## E2: TerraSentia transfer
E2 reuses the frozen AIFARMS/TerraSentia full-study outputs. The exact i2Nav contract structure is transferred unchanged:
- synchronized physical↔virtual correspondence;
- SE(2) finite-horizon local relative motion at 1/5/10 s;
- synchronized global-state error;
- the same tolerance grids.

No target-domain normalization, V2 tuning, checkpoint selection, or threshold tuning is allowed. All 30 frozen V2 checkpoints are summarized within each physical sequence.

Primary transfer reference: RTK position. Fused-EKF heading is secondary because it is not independent ground truth.

E2 success means protocol/contract portability, not high frozen-V2 target performance.
