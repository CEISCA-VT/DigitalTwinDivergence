# UGV01 Frozen Contract and Parameter Sensitivity

The fitted parameters were frozen before the two holdouts. Contract satisfaction is evaluated against AprilTag reference samples.

## Frozen fitted result

| Run | ATE (m) | Heading MAE (deg) | Global satisfaction |
|---|---:|---:|---:|
| calibration | 0.149 | 21.2 | 0.207 |
| holdout_1 | 0.505 | 76.3 | 0.027 |
| holdout_2 | 1.136 | 108.7 | 0.030 |

## Track-width interval

Tested 0.180--0.210 m in 0.005-m steps without selecting a width per holdout.
Common widths satisfying S >= 0.8: none.
A poor holdout is evidence that the frozen contract withdraws the service claim only if the predeclared contract is applied; it is not, by itself, proof that the model was prevented from deployment.

A first-10-second recalibration was not reported because track width is not identifiable from a window lacking sufficient differential turning excitation. No positive restoration result is assumed.
