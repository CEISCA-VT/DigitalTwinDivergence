BASELINE SUITE HOTFIX 01
========================

Fixes:
1. EKF-IW NumPy scalar conversion failure in update_bias_from_yaw_rate.
   Old: float(1x1 ndarray)
   New: explicit scalar extraction with .item().
   This is compatible with NumPy versions where converting a non-0D array
   directly to float raises TypeError.

2. Adds DigitalTwin/analysis/validate_v1_v2_protocol.py to compare ALL 30
   frozen V1/V2 trajectory pairs by sequence + replicate/seed. It verifies:
   row count, time_s, gt_east_m, gt_north_m, gt_heading_rad.

After extracting at repository root:

  python -m DigitalTwin.analysis.validate_v1_v2_protocol
  .\run_i2nav_baseline_suite.ps1 -Smoke

Do not enable V1 in the publication manifest until the protocol validator PASSes.
