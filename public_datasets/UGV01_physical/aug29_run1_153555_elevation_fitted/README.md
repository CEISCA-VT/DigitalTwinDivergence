# UGV01 AprilTag Aligned Ground-Truth Export

- Rows: **1664**
- Duration: **164.64 s**
- Median rate: **10.10 Hz**
- GPS updates: **0**
- Position RMSE: **0.149 m**
- Heading MAE: **21.2 deg**

This package mirrors the i2Nav `aligned_samples` layout for convenient
inspection and downstream tooling. It is not a replacement for a final
synchronized UGV01 run with telemetry, GPS, AprilTag video, and a hardware
or visible sync event.

## Files

- `aligned_samples.csv`: human-readable aligned table.
- `aligned_samples.npz`: NumPy package with the same columns and extra metadata arrays.
- `preparation_summary.json`: machine-readable provenance and limitations.
