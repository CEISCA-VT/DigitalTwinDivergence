# All-Sequence Twin V2 Mechanistic Fidelity Analysis

Script version: `2026-08-20-all-sequence-mechanism-v1`
Input root: `results/i2nav_v2_full_loso/`
Frozen full-LOSO commit expected by context: `6540c01f90f3c1074de0d8dae9964a5276fbbc91`

This analysis uses only frozen V2 full-LOSO outputs. It does not retrain, tune, or change the V2 architecture.

## Statistical Unit

Each run is one held-out sequence and one base seed. The three seeds are averaged within each held-out physical sequence before dataset-level interpretation. Timestamp correlations are reported only as descriptive diagnostics because timestamps are correlated and are not independent statistical replicates.

## Main Answer

parking02 is the unique worst global-divergence sequence in the frozen V2 LOSO set for ATE, Dp p95, and Dtheta p95. However, parking01 also shows the low-local/high-global pattern under the median-split criterion, so parking02 is best interpreted as an extreme point on a broader fidelity failure mode rather than an unsupported one-off anecdote.

The frozen V2 results support the local-vs-global fidelity distinction: finite-horizon RPE can remain small while persistent orientation mismatch accumulates into large heading and position divergence.

## parking02 Position in the 10-Sequence Set

- ATE rank, largest first: 1/10; ATE = 11.350 m.
- Dp p95 rank, largest first: 1/10; Dp p95 = 22.345 m.
- Dtheta p95 rank, largest first: 1/10; Dtheta p95 = 30.415 deg.
- RPE10 rank, smallest first: 2/10; RPE10 = 0.097 m.
- Max |Iomega| = 32.318 deg.

## Low-Local / High-Global Pattern

Using a simple median split, a sequence is counted as low-local/high-global when its RPE10 is at or below the sequence median while its Dp p95 is at or above the sequence median.

Sequences meeting that criterion: parking01, parking02.

This is a descriptive classification, not a tuned decision rule.

## Sequence-Level Mechanism Associations

| Association | Pearson r | Spearman r |
|---|---:|---:|
| persistent yaw mismatch -> accumulated yaw residual | 0.974 | 0.952 |
| Iomega -> heading divergence | 0.381 | -0.030 |
| heading divergence -> position divergence | 0.988 | 0.952 |
| short-horizon local fidelity vs global divergence | -0.187 | 0.285 |

The mechanistic chain is interpreted at sequence level after seed aggregation:

`persistent yaw mismatch -> Iomega -> Dtheta -> Dp`

The sequence-level evidence is strongest for persistent yaw mismatch -> Iomega and Dtheta -> Dp. The direct Iomega -> Dtheta association is weak across all 10 sequences, mainly because some sequences accumulate yaw-rate residual without the same large global heading divergence. This means the chain should be described as a measurable failure pathway, not a universal monotonic law.

## Files Produced

- `per_run_mechanism.csv`: 30 frozen run summaries.
- `per_sequence_mechanism.csv`: 10 sequence summaries after seed aggregation.
- `mechanism_sequence_associations.csv`: sequence-level association table.
- `mechanism_timestamp_correlations.csv`: descriptive timestamp-level correlations.
- `local_vs_global_fidelity.png`: local RPE versus global divergence.
- `persistent_yaw_vs_global_divergence.png`: sequence-level mechanism chain.

## Interpretation

parking02 should be presented as the extreme hard case in a broader local-vs-global fidelity pattern, not as an isolated anecdote and not as a solved sequence. The broader all-sequence analysis supports the paper's main fidelity argument: local relative-pose fidelity and long-horizon physical-virtual synchronization are different properties of a digital twin.

## Descriptive Timestamp Correlations

Timestamp-level correlation rows are retained to inspect the time evolution inside each run, but they must not be converted into dataset-level p-values.
