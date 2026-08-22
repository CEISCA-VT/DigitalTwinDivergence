# Cross-domain transfer protocol for MAGNET

## Purpose

MAGNET is not a second primary benchmark and it is not used to claim universal digital-twin validity. Its purpose is to test whether the **same generic fidelity dimensions** remain meaningful when the physical state, units, dynamics, and twin construction differ radically from mobile robotics.

## Generic object being transferred

For physical component/state `i`, define a domain-appropriate discrepancy

`D_i(t,h) = d_i(x_P,i(t+h), x_T,i(t+h | t))`.

For a synchronized-state twin, `h=0` is the immediate physical–virtual discrepancy and additional temporal/spatial horizons summarize accumulated behavior. For a predictive twin such as MAGNET, `h>0` is the forecast lead time. The framework transfers the decomposition, not the units or a robot-specific trajectory metric.

The transferred dimensions are:

1. component/state discrepancy;
2. short/local versus longer-horizon fidelity;
3. persistent residual/divergence;
4. tail severity;
5. operating-condition dependence;
6. uncertainty/variation across independent or approximately independent evaluation episodes.

## MAGNET instantiation

- Physical state components: the ten released heat-pipe thermowell temperature traces.
- Digital state components: the corresponding ten quantities in the released ML forecast file.
- Discrepancy: absolute/signed temperature error in °C, with a robust component-normalized form for cross-component aggregation.
- Short horizon: 0–60 s.
- Intermediate horizon: 61–300 s.
- Long horizon: 301–599 s.
- Persistence: fraction of aligned component-time samples outside either a transparent fixed °C threshold or a robust early-horizon component envelope.
- Tail severity: p95, p99, and maximum absolute discrepancy.
- Operating condition: heating/steady/cooling labels derived from the slope of the physical hot-zone thermowell mean; this analysis is secondary because the labels are observational rather than randomized.

## Independence policy

The released forecasts overlap. All windows may be used for descriptive visualizations, but headline uncertainty/statistical tests must use temporally non-overlapping windows. Similar-RMSE counterexamples used in the manuscript should also use temporally disjoint forecast intervals.

## Robustness policy

A publishable MAGNET result should survive:

- removal of the most extreme forecast windows;
- multiple fixed persistence thresholds;
- multiple robust-envelope hyperparameters;
- reasonable changes to the heating/steady/cooling slope threshold.

## Claim boundary

Supported if the hardened analysis succeeds:

> The proposed fidelity decomposition transfers beyond mobile-robot state variables to an independently developed thermal digital twin and reveals horizon-, component-, persistence-, tail-, and condition-dependent behavior not represented by a single aggregate error statistic.

Not supported by this experiment alone:

> The framework is universally valid for all digital twins.
