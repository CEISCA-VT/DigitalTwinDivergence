# Verified existing evidence before the new experiment

Source: frozen `full_loso_per_sequence.csv` from the completed 10-sequence × 3-seed i2Nav V2 study.

| Metric | parking00 | parking02 | parking02 / parking00 |
|---|---:|---:|---:|
| V2 ATE (m) | 2.106805 | 11.350361 | 5.387x |
| V2 heading MAE (deg) | 0.613552 | 16.719753 | 27.251x |
| V2 RPE1 (m) | 0.132211 | 0.019314 | 0.146x |
| V2 RPE5 (m) | 0.318032 | 0.054970 | 0.173x |
| V2 RPE10 (m) | 0.453027 | 0.097139 | 0.214x |
| Dp p95 (m) | 3.085755 | 22.344767 | 7.241x |
| Dtheta p95 (deg) | 2.395108 | 30.415317 | 12.699x |

This verifies the remembered **trajectory-sticking/local-vs-global inversion**: parking02 is better than parking00 at all three reported local translation horizons, especially RPE10, while being much worse in global position and heading synchronization.

The new experiment must not claim that this basic mathematical possibility is novel. It tests the stronger proposition that a synchronized twin should be certified relative to the service being requested, rather than given one undifferentiated fidelity label.
