# Digital Twin Divergence

Week 0 software-first scaffold for rover/GPS attack-detection experiments.

Run a quick synthetic experiment:

```powershell
python -m DigitalTwin.experiments.experiment --quick
```

Run the full proposal matrix or a step-bias sweep:

```powershell
python -m DigitalTwin.experiments.experiment --full-matrix --nominal-only
python -m DigitalTwin.experiments.experiment --full-matrix --step-sweep
```

Plot one generated CSV:

```powershell
python -m DigitalTwin.plotting.plot DigitalTwin/datasets/speed-0.20_terrain-0.0_latency-10_attack-none_trial-0.csv
```

Summarize detection probability, lock thresholds, and plot ROC curves:

```powershell
python -m DigitalTwin.analysis.summarize_pd "DigitalTwin/datasets/*.csv"
python -m DigitalTwin.analysis.threshold_lock "DigitalTwin/datasets/*attack-none*.csv"
python -m DigitalTwin.analysis.plot_detection "DigitalTwin/datasets/*.csv"
python -m DigitalTwin.analysis.train_uncertainty "DigitalTwin/datasets/*attack-none*.csv"
```

Listen for UGV01 bridge packets later:

```powershell
python -m DigitalTwin.telemetry_receiver --port 5005
```

Core modules:

- `DigitalTwin/telemetry.py`: packet serializer/deserializer
- `DigitalTwin/telemetry_receiver.py`: UDP hardware packet listener
- `DigitalTwin/kinematics.py`: differential-drive model
- `DigitalTwin/ekf.py`: prediction and GPS update
- `DigitalTwin/uncertainty.py`: `Q` and `R` estimator
- `DigitalTwin/detector.py`: Mahalanobis, eigenvalue detectability bounds, confidence envelopes
- `DigitalTwin/attack.py`: step, freeze, replay, random drift
- `DigitalTwin/logger.py`: experiment CSV schema
- `DigitalTwin/experiments/experiment.py`: batch automation
- `DigitalTwin/plotting/plot.py`: trajectory/detection plots
- `DigitalTwin/analysis/`: threshold locking, P_D summaries, ROC plots, uncertainty training

The hardware-facing protocol is documented in `docs/telemetry_protocol.md`.
Running commands are documented in `docs/running.md`.
UGV01 firmware bring-up, wiring, protocol lock, and log dictionaries are in:

- `docs/ugv01_esp32_bringup.md`
- `docs/log_data_dictionary.md`
- `docs/preregistered_protocol.md`
- `docs/calibration_ready.md`

UGV01 Wi-Fi notes:

- default AP mode: SSID `UGV`, password `12345678`, IP `192.168.4.1`
- to connect the rover to another Wi-Fi network, use `T:404`:

```json
{"T":404,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"Shrayus","sta_password":"Giraffe007"}
```

After sending `T:404`, the OLED `ST` line should show the router-assigned IP.
