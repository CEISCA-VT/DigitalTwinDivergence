# Empirical Benign Digital-Twin Fidelity Characterization

Script version: `2026-08-20-benign-fidelity-envelope-v1`
Frozen full-LOSO commit expected by context: `6540c01f90f3c1074de0d8dae9964a5276fbbc91`

This is a descriptive benign characterization, not a detector. Exceeding the p95 envelope does not mean attack, anomaly, failure, or untrustworthiness. It only means the observation is uncommon relative to the characterized benign data.

## Formal Object

The empirical object is `D(t) | H0, c ~ P_benign(D | c)`, where `c` is a benign operating context. The componentwise descriptive envelope is:

`E_benign^(q)(c) = {D : D_j <= Q_q[D_j | H0, c], for all j}` with `q = 0.95`.

Meters, degrees, m/s, and rad/s are kept as separate physical dimensions.

## Is Benign Divergence Sufficiently Bounded To Characterize Empirically?

Partly. The data are sufficient for a descriptive empirical characterization across the 10 i2Nav physical sequences and three seeds, especially for broad components such as `Dp_m`, `Dtheta_deg`, `Dv_mps`, and `Domega_radps`. The values should not be presented as universal stable operating limits because only 10 physical sequences define the dataset-level support.

## Does The Envelope Depend On Operating Condition?

Yes. For example, `Dp_m` p95 under the elapsed-time envelope changes from 5.828 m in the early-run bin to 17.118 m in the late-run bin.
The unconditional envelope therefore hides condition dependence, particularly for long-horizon position/heading dimensions and time-accumulating quantities.

## Stable Versus Highly Variable Dimensions

| Component | median relative LOSO p95 sensitivity |
|---|---:|
| `Dv_mps` | 0.078 |
| `Iomega_abs_deg` | 0.118 |
| `Domega_radps` | 0.184 |
| `Dp_m` | 0.587 |
| `Dtheta_deg` | 0.704 |

Lower relative LOSO sensitivity means the envelope component is more stable under leave-one-sequence-out perturbation. Larger values mean the p95 estimate is more dependent on which physical sequence is included.

## parking01/parking02 Influence

parking01 and parking02 materially influence long-horizon global dimensions, especially `Dp_m` and `Dtheta_deg`. The strongest `Dp_m` examples are:

| Context | Bin | Full p95 | Without parking01/02 p95 | Delta |
|---|---|---:|---:|---:|
| elapsed_time | late | 17.118 | 4.510 | 12.608 |
| turning | high | 16.551 | 4.365 | 12.185 |
| wheel_imu_disagreement | high | 15.989 | 4.348 | 11.641 |

## Would An Unconditional p95 Envelope Obscure Important Behavior?

Yes. A single unconditional p95 mixes easy street/playground behavior with hard parking behavior and hides the difference between local finite-horizon fidelity and accumulated global synchronization error. The conditioned envelope is more scientifically honest because it records where benign divergence is naturally larger.

## Publication-Grade Now Versus Preliminary

Publication-grade now:
- the Twin Fidelity Profile database for the 30 frozen V2 runs;
- the componentwise condition-dependent descriptive distributions;
- the conclusion that unconditional p95 values obscure condition-dependent behavior;
- the finding that parking01/parking02 strongly influence global envelope dimensions.

Preliminary/descriptive rather than final operating guarantees:
- per-condition p95 values with only a few effective physical-sequence units;
- any condition/bin with fewer than 8 sequence-level supports;
- any envelope dimension dominated by parking01/parking02;
- lateral/slip behavior, because the supported canonical lateral proxy is unavailable here.

## Application-Specific Trust Coverage Interface

`rho_j(c; delta_j) = P(D_j <= delta_j | H0, c)` is implemented as a parameterized function in the analysis script. No application-independent tolerances are invented in this report.

## Files Produced

- `twin_fidelity_profiles.csv`
- `benign_condition_distributions.csv`
- `benign_envelope_p95.csv`
- `benign_envelope_stability.csv`
- `benign_envelope_by_condition.png`
- `unconditional_vs_conditioned_envelope.png`
- `benign_fidelity_manifest.json`
