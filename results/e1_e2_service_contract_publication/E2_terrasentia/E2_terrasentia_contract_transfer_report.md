# E2 — TerraSentia cross-platform service-contract transfer

## Protocol

- Accepted physical sequences evaluated: 5/5.
- The exact i2Nav local horizons (1/5/10 s), tolerance grids, SE(2) relative-pose convention, and synchronized-global convention are reused.
- No TerraSentia normalization refit, model tuning, checkpoint selection, or service-threshold selection is performed.
- All 30 frozen V2 checkpoints are evaluated and summarized within each physical sequence.
- RTK position is the primary transfer reference. Dataset fused-EKF heading is secondary and must not be described as independent ground truth.

## Protocol-transfer result

- Structural transfer criterion: PASS.
- PASS means the unchanged predeclared contract grid is computed completely on at least four accepted physical sequences; it does not mean the frozen i2Nav V2 is a high-fidelity TerraSentia twin.
- Non-degenerate frozen-V2 position surfaces: 20/20 sequence/service surfaces (reported descriptively, not used as the pass gate).

## Frozen V2 macro position-service validity

- global: sequence-mean grid-average RTK-position validity 0.083 (median 0.086).
- local 1 s: sequence-mean grid-average RTK-position validity 0.778 (median 0.806).
- local 5 s: sequence-mean grid-average RTK-position validity 0.418 (median 0.451).
- local 10 s: sequence-mean grid-average RTK-position validity 0.223 (median 0.246).

## Claim boundary

E2 tests portability of the fidelity-contract structure, not target-domain model superiority. TerraSentia motor/IMU provenance and fused-reference heading limitations remain inherited from the frozen external study.
