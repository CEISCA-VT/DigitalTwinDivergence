# TerraSentia LOSO-Calibrated Physics Study

- Completed held-out folds: 5/5.
- Each fold fits three bounded deterministic parameters on the other four physical sequences only.
- Primary evaluation is untouched held-out RTK position; fused-EKF heading remains secondary.
- The position-only ledger compares 2 Hz/200 ms held delivery with 10 Hz/0 ms ideal delivery.
- This is platform-specific deterministic calibration, not learned-V2 retraining.

- loso_calibrated_physics: sequence-mean ATE 44.430 m; RPE1 0.341 m; RPE5 1.835 m; RPE10 3.967 m.
- nominal_physics: sequence-mean ATE 47.126 m; RPE1 0.344 m; RPE5 1.911 m; RPE10 4.218 m.
- loso_calibrated_physics ledger: {'persistent': 20}.
- nominal_physics ledger: {'persistent': 17, 'delivery_remediable': 3}.

## Claim boundary

Five physical sequences support a preliminary platform-specific LOSO study, not a population-wide TerraSentia claim. Motor/IMU semantic provenance remains unresolved and is not corrected using held-out RTK performance.
