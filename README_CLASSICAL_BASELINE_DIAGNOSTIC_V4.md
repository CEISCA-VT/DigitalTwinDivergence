# Classical baseline diagnostic v4

Extract this ZIP at the repository root:

`C:\Users\shrey\Documents\DigitalTwinDivergence`

It adds diagnostic files only; it does **not** overwrite the frozen V1/V2 trajectories or learned-baseline outputs.

## Why this patch exists

The first EKF repair improved the old result but the uploaded diagnostics still show a degenerate classical filter. Rather than tuning on held-out sequences, this patch compares several classical wheel/IMU fusion choices under strict LOSO and reports a strategy selected from training sequences only.

It also runs the public i2Nav evaluation convention with `evo`: timestamp association within 5 ms, SE(3) alignment without scale, then translation APE RMSE.

## Run

First:

```powershell
.\diagnose_classical_baselines_v4.ps1 -Smoke
```

Then:

```powershell
.\diagnose_classical_baselines_v4.ps1
```

If `evo` is missing:

```powershell
python -m pip install evo
```

## Send back

- `results\i2nav_fidelity_baselines\validation\classical_yaw_fusion_summary.csv`
- `results\i2nav_fidelity_baselines\validation\classical_yaw_fusion_selected_by_training.csv`
- `results\i2nav_fidelity_baselines\validation\i2nav_evo_ape_per_sequence.csv`
- `results\i2nav_fidelity_baselines\validation\i2nav_evo_ape_summary.csv`

Do not publish a held-out winner selected after looking at held-out results. If a new classical baseline is adopted, use the `selected_by_train_ate` protocol or predeclare one candidate based on the training-only diagnostic.
