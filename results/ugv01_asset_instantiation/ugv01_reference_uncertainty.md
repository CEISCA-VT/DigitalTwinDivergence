# UGV01 AprilTag Reference Uncertainty

The accepted UGV01 reference is the repaired AprilTag trajectory in
`DigitalTwin/datasets/analysis/validation_carpet_142023_four_tag_continuity_repaired/apriltag_still_summary.json` paired with telemetry `raw_logs/telemetry/ugv_t147_interactive_20260812_142023.csv`.

Known uncertainty sources:

- Camera/telemetry synchronization is motion-correlation based, not a hardware-visible sync pulse.
- Estimated video-minus-telemetry offset: `20.000 s`.
- Synchronization correlation: `0.933`.
- Synchronization uncertainty: `0.025 s`.
- Directly decoded evaluation fraction: `0.281`.
- Recovered evaluation samples: `1549` of `2153`.
- Camera stationary jitter RMSE / p95: `0.0014 m / 0.0039 m`.
- The accepted docs also record a measured reference-geometry/camera-model reprojection discrepancy on the order of several pixels for the development setup.

Interpretation:

The reference is strong enough for an asset-instantiation development audit on
low-speed carpet, because the observed improvement in heading and full-window
path consistency is much larger than camera stationary jitter. Smaller position
changes of only a few millimeters to centimeters should not be overstated,
because they can be comparable to synchronization, reference-geometry, and
recovered-track uncertainty. The current reference supports a low-speed carpet
UGV01 claim, not a general all-surface/all-speed claim.
