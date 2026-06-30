# Running the Project

This guide covers the current pre-hardware workflow and the commands to use
once the UGV01 starts streaming telemetry.

## Environment

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest -q
```

## Synthetic Experiments

Quick smoke test:

```powershell
python -m DigitalTwin.experiments.experiment --quick
```

Full benign `2^3` matrix:

```powershell
python -m DigitalTwin.experiments.experiment --full-matrix --nominal-only
```

Full matrix with step-bias sweep:

```powershell
python -m DigitalTwin.experiments.experiment --full-matrix --step-sweep
```

Shorter rehearsal runs can use:

```powershell
python -m DigitalTwin.experiments.experiment --full-matrix --step-sweep --trials 1 --duration 20
```

Generated datasets are written under `DigitalTwin/datasets/` and are ignored by
Git.

## Plotting

Plot a single CSV:

```powershell
python -m DigitalTwin.plotting.plot DigitalTwin/datasets/speed-0.20_terrain-0.0_latency-10_attack-step_eps-5.0_trial-0.csv
```

Plots are written under `DigitalTwin/datasets/plots/` or the analysis output
directory and are ignored by Git.

## Analysis

Summarize empirical detection probability:

```powershell
python -m DigitalTwin.analysis.summarize_pd "DigitalTwin/datasets/*.csv"
```

Lock the detector threshold from nominal runs:

```powershell
python -m DigitalTwin.analysis.threshold_lock "DigitalTwin/datasets/*attack-none*.csv"
```

Generate ROC and detection-probability plots:

```powershell
python -m DigitalTwin.analysis.plot_detection "DigitalTwin/datasets/*.csv"
```

Train the uncertainty-model stub from benign runs:

```powershell
python -m DigitalTwin.analysis.train_uncertainty "DigitalTwin/datasets/*attack-none*.csv"
```

The locked threshold and trained model artifacts are ignored by Git because
they are generated from datasets.

## Hardware Receiver

When the UGV01 is streaming packets over UDP:

```powershell
python -m DigitalTwin.telemetry_receiver --port 5005
```

Before collecting research data, verify:

- CRC-valid packets arrive continuously.
- `seq` increments without unexplained drops.
- Encoder ticks change with wheel motion.
- GPS fix type, satellites, HDOP, latitude, and longitude are plausible.
- IMU readings are stable when stationary and respond to motion.

## Suggested Pre-Hardware Final Rehearsal

Run:

```powershell
python -m pytest -q
python -m DigitalTwin.experiments.experiment --full-matrix --step-sweep --trials 1 --duration 20
python -m DigitalTwin.analysis.summarize_pd "DigitalTwin/datasets/*.csv"
python -m DigitalTwin.analysis.plot_detection "DigitalTwin/datasets/*.csv"
```

If those pass, stop expanding scope until hardware arrives.
