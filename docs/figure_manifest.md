# Manuscript Figure Manifest

The paper package deliberately contains only figures cited by
`manuscript/main.tex`. Detailed diagnostics remain in their canonical
`results/` directories and are not duplicated into the submission package.

| Figure asset | Evidence role | Canonical source |
|---|---|---|
| `e1_parking_full_grid_inversion.png` | Local/global service-ranking inversion | `results/e1_e2_service_contract_publication/E1_i2nav/` |
| `e1_metric_service_rank_alignment.png` | ATE/RPE alignment with matching services | `results/e1_e2_service_contract_publication/E1_i2nav/` |
| `condition_dependent_fidelity.png` | Condition-dependent fidelity dimensions | Frozen post-LOSO condition analysis |
| `persistent_yaw_mechanism.png` | Persistent yaw and accumulated divergence | Frozen post-LOSO mechanism analysis |
| `E1_E2_cross_platform_position_contracts.png` | i2Nav/TerraSentia contract portability | `results/e1_e2_service_contract_publication/` |
| `cross_domain_horizon_profile.png` | Cross-domain horizon behavior | `results/cross_domain_contract_generalization/figures/` |
| `cross_domain_contract_heatmap.png` | Cross-domain contract map | `results/cross_domain_contract_generalization/figures/` |
| `timing_jitter_stage2.png` | Timestamp-jitter sensitivity | Synchronization-sensitivity analysis |
| `timing_delay_stage2.png` | Fixed-delay sensitivity | Synchronization-sensitivity analysis |
| `ugv01_aug29_validation_summary.png` | UGV01 calibration and frozen holdouts | UGV01 physical-validation analysis |
| `trace_replay_policy_comparison.png` | Measured-trace policy replay | `results/ugv01_live_contract_trace_replay/` |
| `transport_capacity_sensitivity.png` | Counterfactual capacity bounds | `results/ugv01_live_contract_trace_replay/` |

## Claim Boundaries

- E1 establishes service-relative decisions; it does not invalidate ATE or RPE.
- E2 establishes contract portability, not zero-shot model superiority.
- E3 establishes structural cross-domain use; normalized tolerances are not universal safety limits.
- UGV01 fitting is calibration evidence with adverse frozen holdouts.
- Live replay demonstrates policy execution and a measured transport bottleneck, not communication or energy savings.
