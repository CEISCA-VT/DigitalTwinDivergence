# Current Project Status

Date: 2026-08-20  
Scope: sensor-lightweight digital twin fidelity, before returning to the security/attack layer.  
Basis: repository artifacts only. No training or result regeneration was run for this audit.

## Repository Basis

- Current commit inspected: `6540c01f90f3c1074de0d8dae9964a5276fbbc91`.
- The frozen V2 LOSO context in `AGENTS.md` and `CODEX_CONTEXT_DT_FIDELITY_POST_LOSO.md` identifies this commit as the frozen V2 commit.
- Main completed LOSO artifact root: `results/i2nav_v2_full_loso/`.
- Main post-LOSO diagnostic root found: `results/i2nav_v2_post_loso_analysis/parking02_vs_parking00/`.
- The worktree contains many untracked/generated artifacts, so this report treats files on disk as current evidence, not only committed files.

## Status Summary

| Item | Status | Repository evidence | What remains |
|---|---:|---|---|
| 30/30 frozen V2 LOSO audit | Partial | `results/i2nav_v2_full_loso/i2nav_v2_full_loso_summary/full_loso_completion_audit.json` reports `all_30_complete: true`. All 30 run directories contain the expected V2 artifacts: model, histories, traces, evaluated trajectories, fidelity profiles, summaries, manifests, and completion markers. | A final post-LOSO audit package with explicit checksums/recomputed metric verification was not found under `results/i2nav_v2_post_loso_analysis/audit/`. Completion is proven; final reproducibility audit is not yet packaged. |
| Final Fixed Physics -> V1 -> V2 reproducibility statistics | Partial | V2 LOSO has `full_loso_run_results.csv`, `full_loso_per_sequence.csv`, and `full_loso_macro_summary.json`. V1 evidence exists under `results/i2nav_v1_frozen/evidence/`. A partial fixed-physics official table exists at `results/i2nav_final_model_study_repaired/phase1_official_fixed_v5.csv`. | Build one integrated Fixed Physics/V1/V2 table across the same folds/sequences and metrics. Current artifacts strongly support V1 -> V2; the Fixed Physics -> V1 -> V2 chain is not fully unified. |
| Sequence-level paired/bootstrap/sign-flip statistics | Partial | `full_loso_macro_summary.json` includes sequence-level V1/V2 macro deltas and bootstrap intervals. `results/i2nav_v1_frozen/evidence/final_summary.json` includes V1-vs-fixed sign-flip evidence for ATE. | Need one final paired-test artifact covering Fixed -> V1 and V1 -> V2 for ATE, heading, RPE1/RPE5/RPE10, and fidelity-divergence quantities. |
| parking00 vs parking02 local/global fidelity analysis | Complete | `results/i2nav_v2_post_loso_analysis/parking02_vs_parking00/` contains trajectory, metric, short-horizon, mechanism figures, aligned timeseries, and `parking02_vs_parking00_summary.txt`. | Nothing required for this specific two-sequence diagnostic. |
| Short-horizon RPE1/RPE5/RPE10 figure | Complete for parking00/parking02 | `parking02_vs_parking00_short_horizon_rpe.png` and `short_horizon_rpe.csv` exist. | Optional: create a full all-sequence short-horizon figure if the manuscript needs broader evidence. |
| Mechanistic yaw-divergence analysis | Partial | parking00/parking02 mechanism outputs exist, including `parking02_vs_parking00_mechanism.png` and `mechanism_correlations.csv`. Additional yaw-bias diagnostics exist in `results/i2nav_v2_bias_diagnostic/`. | Generalize into an all-sequence, sequence-level mechanism table/figure. Current evidence is strong as a focused diagnostic, not yet a final global mechanism analysis. |
| Condition-dependent fidelity | Requires work | No final `condition_fidelity` artifact directory was found. Existing parking00/parking02 diagnostics include some local/global context, but not a condition-wide study. | Define condition labels/bins and produce condition-dependent fidelity summaries and figures. |
| Twin Fidelity Profile database | Partial | Each of the 30 V2 LOSO runs has `fidelity_profile.json` and `fidelity_timeseries.csv`. `full_loso_per_sequence.csv` includes key profile-like metrics such as `Dp`, `Dtheta`, and `Iomega`. | Consolidate these into an explicit database, for example all-run, per-sequence, and macro profile tables. |
| Empirical benign fidelity envelopes | Not started | No envelope artifact was found. | Build benign envelopes from frozen V2 fidelity profiles, such as per-sequence and macro p50/p95/p99 bounds for position, heading, RPE, and divergence metrics. |
| UGV01 asset-specific instantiation | Partial | UGV01 AprilTag and fine-tuning artifacts exist under `DigitalTwin/datasets/analysis/`, with supporting docs such as `docs/apriltag_validation_split.md` and `docs/ugv01_apriltag_finetune_results.md`. | Create a formal asset-instantiation package: data inventory, quality report, staged model comparison, and clearly separated pilot/development vs validation claims. |
| NCLT portability | Requires work | `public_datasets/nclt/` exists and appears populated. | No NCLT portability evaluation artifacts were found. Need preprocessing, frozen-model application, and portability report. |
| RELLIS-3D portability | Requires work | `public_datasets/rellis3d/` exists. | No RELLIS-3D portability evaluation artifacts were found. Need preprocessing, frozen-model application, and portability report. |
| Official i2Nav evaluation | Partial | `results/i2nav_final_model_study_repaired/` contains official-protocol groundwork and partial fixed-physics outputs. | Need final official i2Nav evaluation/export for the frozen V2 model across the intended sequence set. Keep this separate from the digital-twin fidelity evaluation. |
| Sensing-fidelity comparison | Not started | No final sensing/fidelity comparison table or frontier figure was found. | Requires official/comparable metrics first, then a sensor-stack vs fidelity comparison. |
| Final manuscript integration | Requires work | Existing docs and reports exist, but no final manuscript-integrated package was found using the frozen V2 LOSO, post-LOSO diagnostics, and rewritten fidelity-first framing. | Integrate final tables, figures, claims, limitations, and method framing into the manuscript. |

## Key Current Results Already Supported

From `results/i2nav_v2_full_loso/i2nav_v2_full_loso_summary/full_loso_macro_summary.json`:

| Metric | V1 mean | V2 mean | V2 - V1 | Macro change | V2 better sequences |
|---|---:|---:|---:|---:|---:|
| ATE RMSE | 2.834 m | 2.398 m | -0.436 m | -15.38% | 6/10 |
| Heading MAE | 3.336 deg | 2.569 deg | -0.767 deg | -22.99% | 8/10 |
| RPE1 | 0.0606 m | 0.0611 m | +0.0005 m | +0.87% | 7/10 |
| RPE5 | 0.1670 m | 0.1603 m | -0.0067 m | -4.00% | 9/10 |
| RPE10 | 0.2714 m | 0.2532 m | -0.0182 m | -6.70% | 8/10 |

The most defensible current claim is:

> Frozen V2 is complete across 30/30 LOSO runs and improves macro ATE, heading, RPE5, and RPE10 over V1, while RPE1 is essentially unchanged. The strongest completed diagnostic explains a major local/global fidelity mismatch between parking00 and parking02 through yaw-divergence behavior.

## Recommended Next Action

Do the final reproducibility/statistics package before starting portability or manuscript expansion:

1. Create the missing post-LOSO audit package with checksums and metric verification.
2. Build one integrated Fixed Physics -> V1 -> V2 table on the same sequence-level unit.
3. Add paired/bootstrap/sign-flip statistics for the final metrics.
4. Consolidate the 30 `fidelity_profile.json` files into the Twin Fidelity Profile database.

This is the cleanest next step because it turns already-completed computation into paper-ready evidence without rerunning training or changing the frozen V2 model.
