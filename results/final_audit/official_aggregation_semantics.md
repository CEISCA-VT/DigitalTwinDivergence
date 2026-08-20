# Official Aggregation Semantics

The public evaluator computes per-trajectory metrics. The i2Nav-Robot README reports aggregate table rows labeled RMS, so this audit preserves both arithmetic sequence macro means and sequence-RMS aggregates. The existing manuscript-facing 1.635 m remains the arithmetic macro mean and is not replaced.

| metric | arithmetic macro mean | sequence RMS |
|---|---:|---:|
| official_ape_translation_rmse_m | 1.635 | 2.187 |
| official_ape_rotation_rmse_deg | 3.011 | 3.706 |
| official_rpe_50m_translation_rmse_m | 1.310 | 1.365 |
| official_rpe_100m_translation_rmse_m | 2.217 | 2.272 |
| official_rpe_300m_translation_rmse_m | 3.635 | 3.817 |
