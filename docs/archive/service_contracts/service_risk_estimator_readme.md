# Nested service-risk estimator

Extract this ZIP **directly at the repository root**. It does not retrain Twin V2.

Preferred Windows runner (avoids PowerShell execution-policy issues):

```cmd
run_service_risk_estimator.cmd
```

PowerShell alternative:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_service_risk_estimator.ps1
```

The default run requires the frozen raw V2 trajectories/traces so that signed causal context can be reconstructed. It also consumes the prior service-relative window table at:

`results/service_relative_fidelity/physical_windows_seed_averaged.csv`

Outputs are written to:

`results/service_risk_estimator_nested_loso/`

The analysis is deliberately falsifiable. It prints `GO`, `NO-GO`, or `INCONCLUSIVE` from predeclared criteria; do not retune those criteria after seeing the result.
