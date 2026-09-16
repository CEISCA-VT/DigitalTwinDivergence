# Cross-domain operational-fidelity contract audit

**Structural transfer verdict: PASS (3/3 required datasets passed; 3/3 completed)**

## Frozen question

Does the same *quantity × horizon × tolerance* fidelity-contract structure remain computable and informative in independent non-robot digital-twin domains without tuning thresholds to outcomes?

## Frozen contract

For each physical quantity, normalize absolute physical–virtual discrepancy by that quantity's physical p95−p05 range (with only a small scale floor for near-constant channels). For a service horizon h, compute the p95 normalized discrepancy inside each non-overlapping h-second service window. A contract passes at normalized tolerance τ when that window statistic is ≤ τ.

- Horizons: 60 s, 300 s, 600 s.
- Normalized tolerance sweep: 0.01, 0.02, 0.05, 0.1, 0.2, 0.5.
- Within-window error statistic: q=0.95.
- The numerical tolerance grid is dimensionless; no dataset-specific error threshold is tuned after observing results.

## Dataset transfer gates

| dataset         |   n_components |   horizons_meeting_min_units |   required_horizons | contract_units_by_horizon   | surface_nondegenerate   | structural_transfer_pass   |
|:----------------|---------------:|-----------------------------:|--------------------:|:----------------------------|:------------------------|:---------------------------|
| FreeTwinEV_1S4P |              3 |                            3 |                   3 | 60s:144;300s:33;600s:18     | True                    | True                       |
| MAGNET          |             10 |                            3 |                   3 | 60s:230;300s:230;600s:230   | True                    | True                       |
| TUWien_SNG      |              7 |                            3 |                   3 | 60s:3930;300s:798;600s:399  | True                    | True                       |

## Grid-average descriptive validity

| dataset         |   horizon_s |   grid_average_validity |   total_contract_units |
|:----------------|------------:|------------------------:|-----------------------:|
| FreeTwinEV_1S4P |          60 |                  0.3438 |                    144 |
| FreeTwinEV_1S4P |         300 |                  0.3131 |                     33 |
| FreeTwinEV_1S4P |         600 |                  0.2685 |                     18 |
| MAGNET          |          60 |                  0.9428 |                    230 |
| MAGNET          |         300 |                  0.8188 |                    230 |
| MAGNET          |         600 |                  0.6594 |                    230 |
| TUWien_SNG      |          60 |                  0.3851 |                   3930 |
| TUWien_SNG      |         300 |                  0.3243 |                    798 |
| TUWien_SNG      |         600 |                  0.2975 |                    399 |

## Dataset execution status

- All three required datasets completed without loader errors.

## Interpretation boundary

A structural PASS means the unchanged contract abstraction produces complete, non-degenerate validity surfaces over all frozen horizons in that dataset. It is **not** a claim that the underlying domain model is superior, that normalized tolerances are safety standards, or that all domains should share identical physical-unit tolerances.

MAGNET remains the strongest inferential non-robot transfer because its existing hardened study includes dependence-reduced forecast windows and horizon statistics. FreeTwinEV and TU Wien SNG are used here to test broader contract portability across battery electro-thermal and industrial-process digital-twin data.

## Dataset-specific semantics

- **MAGNET:** released 10-thermowell physical experiment versus digital-twin forecast windows; dependence-reduced forecast windows are used.
- **FreeTwinEV 1S4P:** released ID22 experiment versus the released identification/validation 3D CFD cooldown/discharge simulation; thermal aggregate quantities are paired without calibrating the simulation in this script.
- **TU Wien SNG:** measured versus Kalman-estimated DFB process states plus measured product-gas composition versus the released soft-sensor outputs from the same 9.5 h DT campaign.

## Publication use

If all three transfer gates pass, the defensible contribution is **cross-domain portability of the service-contract representation**. Do not claim universal fidelity, universal safety tolerances, or model transfer superiority.
