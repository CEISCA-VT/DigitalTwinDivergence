# Pre-GPS publication hardening suite

Extract the ZIP at the repository root.

It does **not** overwrite V1, V2, trained models, or existing evaluator results.

## What it adds
- Sequence-level paired statistics: bootstrap CIs, Wilcoxon, effect size, IQR/SD.
- Sequence-bootstrap uncertainty for TFP-vs-Muñoz correlations.
- Controlled symmetric yaw-rate-bias replay with a mandatory zero-bias replay gate.
- Manuscript claim audit, including the unresolved 1.635-vs-1.251 aligned-APE issue.
- Reproducibility snapshot.
- Cleaner publication figures.
- LaTeX additions and a pre-GPS protocol freeze.

## Run
```powershell
.\run_pre_gps_publication_hardening.ps1
```

If your manuscript file is not auto-detected:
```powershell
.\run_pre_gps_publication_hardening.ps1 -TexPath ".\iotj_sensor_lightweight_dt_fidelity_v2.tex"
```

## Key outputs
`results/publication_hardening/`

Most important:
- `publication_sequence_statistics.csv`
- `publication_sequence_statistics.tex`
- `munoz_tfp_bootstrap_correlations.csv`
- `munoz_tfp_bootstrap_correlations_midMAD.csv`
- `yaw_bias_zero_replay_audit.csv`
- `yaw_bias_macro_by_magnitude.csv`
- `yaw_bias_replay_status.txt`
- `claim_audit.csv`
- `claim_audit.md`
- `reproducibility_snapshot.json`
- `PRE_GPS_HARDENING_REPORT.md`
- `figures/v1_v2_local_vs_global_clean.png`
- `figures/parking02_tfp_global_tail.png`

## Publication guardrails
1. The physical sequence is the statistical unit.
2. Correlations with n=10 are exploratory even with bootstrap CIs.
3. Do not publish yaw-bias results unless the zero-bias reconstruction gate passes.
4. Do not use the failed EKF/recomputed Fixed Physics as exact published-method baselines.
5. Resolve the exact official aligned-APE aggregation/provenance before final freeze.
6. The framework has a domain-general mathematical structure, but current empirical validation is robotics-only.
