# MAGNET cross-domain TFP audit

This folder is a drop-in exploratory validation extension for the INL **MAGNET Heat Pipe Digital Twin** dataset. It is designed to answer one question: **does a generic physical–virtual fidelity decomposition reveal useful horizon-, component-, persistence-, tail-, and condition-dependent behavior outside mobile robotics?**

## One-command Windows run

From the root of your repository after extracting the ZIP:

```powershell
.\run_magnet_tfp.ps1
```

The runner will:

1. create `magnet_tfp/.venv` if needed;
2. install `numpy`, `pandas`, and `matplotlib`;
3. download the two official INL single-file CSVs if they are not already present;
4. run the audit; and
5. place all outputs in `magnet_tfp/results/`.

To force a fresh data download:

```powershell
.\run_magnet_tfp.ps1 -ForceDownload
```

If you already downloaded MAGNET manually, put these exact files in `magnet_tfp/data/`:

- `MAGNET_Heat_Pipe_2022-03-30.csv`
- `ML_MAGNET_2022-03-30.csv`

Then run the normal command.

## What the code computes

- eligibility/missing-window audit;
- physical-to-forecast timestamp alignment;
- per-thermowell physical–virtual discrepancy;
- raw MAE/RMSE/bias and p95/p99/max tail error;
- 0–60 s, 61–300 s, and 301–599 s horizon fidelity;
- 60 s / 300 s / 599 s point-horizon summaries;
- persistence outside a robust early-horizon envelope;
- persistence above a transparent fixed 5 °C threshold;
- per-component robust normalization;
- heating/steady/cooling condition stratification using the physical TC06–TC10 temperature slope;
- a greedy non-overlapping subset for conservative bootstrap summaries;
- detection of pairs of forecast windows with similar aggregate RMSE but substantially different tail/persistence/long-horizon behavior;
- publication-oriented plots and a `SUMMARY.md` decision aid.

## Important interpretation

The generated 0–5 "evidence score" is only an exploratory decision aid. **Do not put that score in a paper.** The scientific evidence is in the actual error distributions, horizon behavior, component localization, condition dependence, and similar-RMSE diagnostic counterexamples.

The intended claim, if supported by the results, is not "TFP is universally valid." A defensible claim is that a fidelity decomposition developed and deeply validated for mobile-robot twins **transfers to an independently developed thermal digital twin with different state variables**.

## Data provenance

Dataset: Idaho National Laboratory, `IdahoLabResearch/MAGNET-Heat-Pipe-Data`, March 30, 2022. The repository describes a digital twin that ran alongside a physical single heat pipe and released full experimental and machine-learning forecast data. The repository is archived/read-only as of 2026 but remains publicly accessible. Respect the dataset's original license and attribution requirements.

## Publication-hardening stage (v2)

The same PowerShell command now runs a second stage after the base audit. This stage is intended to address likely reviewer concerns rather than add new TFP dimensions.

It adds:

- paired long-vs-short horizon statistics on the non-overlapping subset;
- bootstrap 95% intervals and a paired Wilcoxon signed-rank test;
- a paired rank-biserial effect size;
- temporally disjoint similar-RMSE counterexamples at both 5% and 10% RMSE tolerance;
- robustness after trimming extreme forecast windows;
- persistence-threshold sensitivity over 1, 2, 5, 10, 20 and 50 °C;
- robust-envelope sensitivity to multiple MAD multipliers/floors;
- operating-regime definition sensitivity;
- a non-overlapping exploratory regime significance/effect-size test;
- component-by-horizon localization summaries;
- figures specifically aimed at demonstrating that the result is not driven by overlap, one threshold, or a few numerically unstable forecasts.

Inspect `results/PUBLICATION_HARDENING_SUMMARY.md` first. The headline manuscript statistics should come from `paired_horizon_significance_nonoverlap.csv`, not from treating all overlapping windows as independent replicates.

### Muñoz on MAGNET

The package deliberately does **not** approximate the Muñoz trace-alignment baseline. The published Muñoz method uses an application-specific maximum admissible distance (MAD), normally related to measurement accuracy. A clean comparison should use the authors' official implementation and a defensible MAGNET measurement-accuracy/MAD choice. Unless that information is established, retain the Muñoz comparison in the primary mobile-robot experiments and use MAGNET for cross-domain transfer validation.


## V3 final publication checks
The runner now adds a strict same-RMSE counterexample audit within the 23-window non-overlapping subset, an explicit metric-redundancy claim boundary, a compact main-paper evidence table, and a publication-ready counterexample figure with separate units. Inspect `MAGNET_FINAL_PUBLICATION_DECISION.md` after each run.
