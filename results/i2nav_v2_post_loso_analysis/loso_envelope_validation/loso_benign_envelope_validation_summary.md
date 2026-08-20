# LOSO Benign Envelope Validation

Script version: `2026-08-20-loso-envelope-validation-v1`
Frozen full-LOSO commit expected by context: `6540c01f90f3c1074de0d8dae9964a5276fbbc91`

This analysis validates the empirical benign fidelity envelope by holding out one physical i2Nav sequence at a time, estimating p90/p95 envelopes from the remaining nine sequences, and evaluating coverage on the held-out sequence. It uses the frozen condition definitions and does not retrain, retune, or modify Twin V2.

Important interpretation constraint: p95 exceedance is not an attack, anomaly, failure, or untrustworthiness claim. It only means that a held-out observation is above the descriptive benign envelope learned from the other physical sequences.

## Does The Benign Envelope Generalize To An Unseen Physical Sequence?

Partially. Rate-domain components generalize well; global position and heading dimensions are much less stable because difficult parking sequences strongly affect the p95 envelope.

| Component | Mean conditioned p95 coverage | Mean unconditional p95 coverage | Mean conditioned exceedance | Median p95 sensitivity |
|---|---:|---:|---:|---:|
| `Dp_m` | 0.945 | 0.948 | 0.055 | 0.046 |
| `Dtheta_deg` | 0.915 | 0.916 | 0.085 | 0.055 |
| `Dv_mps` | 0.972 | 0.971 | 0.028 | 0.006 |
| `Domega_radps` | 0.975 | 0.979 | 0.025 | 0.004 |
| `Iomega_abs_deg` | 0.951 | 0.959 | 0.049 | 0.004 |

## parking01 and parking02 Predictability

parking02 is the severe held-out under-coverage case for global position/heading characterization. parking01 is much better covered in the held-out validation, but the parking01/parking02 family still materially affects the global p95 envelope because those sequences define the difficult long-horizon regime. For rate-domain quantities, held-out coverage is much closer to the nominal descriptive p95.

| Component | parking01/02 mean coverage | other sequences mean coverage |
|---|---:|---:|
| `Dp_m` | 0.723 | 1.000 |
| `Dtheta_deg` | 0.573 | 1.000 |
| `Dv_mps` | 0.977 | 0.971 |
| `Domega_radps` | 0.989 | 0.972 |
| `Iomega_abs_deg` | 0.920 | 0.958 |

## Do parking01 and parking02 Inflate Global p95?

Yes. The leave-one-sequence p95 sensitivity confirms the earlier benign-envelope finding: global `Dp_m` and `Dtheta_deg` envelopes are materially influenced by difficult parking behavior, with parking02 the dominant single held-out influence. Rate-domain quantities such as `Dv_mps` and `Domega_radps` are less affected by which physical sequence is left out.

## Does Conditioning Improve Held-Out Characterization?

Conditioning does not uniformly increase scalar coverage in every component, because the conditioned envelope is deliberately more specific and often tighter than the unconditional envelope. Its main benefit is interpretability: it exposes which operating contexts are naturally high-divergence instead of hiding them inside one broad unconditional p95.

## Stable Enough For Publication?

The current envelope is **partially stable**. Publication-grade descriptive claims are strongest for the existence of condition dependence, the local/global distinction, and the relative stability of rate-domain quantities. Exact global-position and global-heading p95 values should be presented as descriptive and sequence-sensitive, not as universal operating guarantees.

## Files Produced

- `loso_envelope_validation_per_sequence.csv`
- `loso_envelope_validation_summary.csv`
- `loso_envelope_stability.csv`
- `loso_conditioned_vs_unconditional_coverage.png`
- `loso_envelope_influence_by_sequence.png`
