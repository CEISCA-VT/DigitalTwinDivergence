# Service-relative fidelity drop-in

Extract this ZIP **at the `DigitalTwinDivergence` repository root**, preserving paths. This package does not overwrite or retrain Twin V2.

## Run

From PowerShell at the repository root:

```powershell
.\run_service_relative_fidelity.ps1
```

If the frozen result tree is elsewhere:

```powershell
.\run_service_relative_fidelity.ps1 -InputRoot "C:\path\to\i2nav_v2_full_loso"
```

The runner first executes a deterministic self-test, then the frozen-signature audit, then the full service-relative analysis.

## Expected input

The analysis auto-discovers the 30 frozen V2 runs under `results/i2nav_v2_full_loso` and expects one `v2_evaluated_trajectory.csv` per 10 held-out sequences × 3 base seeds. Adjacent `v2_prediction_trace.csv` files are used for wheel–IMU disagreement where available.

If only a summary ZIP is present, the full experiment cannot be reconstructed from that summary alone; the per-timestep frozen trajectories are required.

## Outputs

`results/service_relative_fidelity/`

The most important files are:

- `parking00_vs_parking02_verification.md` — confirms the pre-existing inversion before new analysis.
- `service_pass_rates_per_sequence.csv` — actual held-out service success across a tolerance sweep.
- `loso_monitor_decisions.csv` — per-sequence false-safe/false-reject/support results.
- `loso_monitor_macro.csv` — sequence-level macro comparison of evaluators.
- `service_relative_fidelity_report.md` — automatically bounded interpretation; it can explicitly say the new hypothesis is unsupported.
- `analysis_manifest.json` — source commit, configuration, hashes, and aggregation rules.

## Important limitation

This experiment does **not** magically make physical-vs-virtual ground-truth errors available online. It separates offline evaluation truth from the online-observable operating context used by the service-support envelope. Do not describe the GT-derived TFP metrics themselves as a deployable live monitor unless an independent online reference actually exists.
