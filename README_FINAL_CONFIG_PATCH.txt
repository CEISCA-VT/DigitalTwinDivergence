FINAL CONFIG PATCH

Extract this ZIP at the repository root:
  C:\Users\shrey\Documents\DigitalTwinDivergence

It overwrites only:
  baseline_suite_config.json

Changes:
- Enables frozen Twin V1 at:
    results/i2nav_v1_frozen/canonical_predictions
- Requires 30 V1 files.
- Adds Fixed Physics (recomputed) to generated baseline models.
- Keeps frozen official Fixed Physics disabled because no saved frozen trajectory archive was discovered.
- Keeps EKF-IW, LWOI-IMU adaptation, YNet-style reduced-input baseline, V2, TFP, Bergs/Hausdorff, and Muñoz evaluation enabled by the existing runner.

Before publication:
- The 30/30 V1-V2 check establishes matched evaluation timestamps/GT, not by itself every detail of the original training protocol.
- Label LWOI-IMU and YNet-style as adaptations, not exact reproductions.
- Label Fixed Physics (recomputed) separately from the frozen official benchmark until equivalence is verified.
