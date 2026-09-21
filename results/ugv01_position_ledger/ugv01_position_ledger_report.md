# UGV01 Position-Only Remediability Ledger

The frozen ledger contains 12 run-service cases across three physical runs and four services.
Partition: **already_qualified=5, delivery_remediable=3, persistent=4**.

The primary comparison is 1 Hz/200 ms degraded delivery versus 2 Hz/0 ms ideal delivery. The 10/5 Hz rows for run 1 are diagnostic only and do not enter the ledger.

## Interpretation

This small retrospective study tests whether the taxonomy computes on UGV01; it does not estimate population-level category rates. AprilTag position is the qualification reference. Heading is excluded, and windows crossing the frozen reference-gap limit are removed from the observable denominator.

## Limitations

- Three physical runs are available, and run 1 is an asset-calibration run rather than an untouched holdout.
- The common source-rate ceiling is 2 Hz because the two holdout exports cannot support 10/5 Hz replay.
- Thresholds are platform-scaled and frozen before this ledger computation, but informed by prior UGV01 characterization rather than prospectively blinded.
- This is offline delivery replay over saved physical trajectories, not an online communication-savings experiment.
