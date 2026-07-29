# Digital Twin Divergence

Security-aware UGV01 digital-twin implementation, hardware telemetry pipeline,
and reproducible offline GPS-attack evaluation.

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
python -m DigitalTwin.analysis.train_uncertainty
```

Run the accepted-hardware-log study (manifest, run-level split, benign-only
threshold lock, paired GPS attacks, summaries, and plots):

```powershell
python -m DigitalTwin.analysis.real_data_study
```

The July 29 security-predictor revision supersedes the previously generated
alarm thresholds and attack-campaign outputs. Regenerate them before quoting
security results. The existing rover logs are sufficient for this software
rerun.

Regenerate plots and the report from an already completed campaign without
rerunning all EKF replays:

```powershell
python -m DigitalTwin.analysis.real_data_study --summarize-existing
```

See `docs/real_data_study.md` for the threat boundary and interpretation
limits.

The previously completed statistical attack matrix is documented in
`docs/statistical_attack_campaign.md`.

Run or regenerate the paired covariance-poisoning analysis:

```powershell
python -m DigitalTwin.analysis.covariance_poisoning
python -m DigitalTwin.analysis.covariance_poisoning --summarize-existing
```

The causal comparison and current conclusion are documented in
`docs/covariance_poisoning_analysis.md`.

The frozen uncertainty variants, learned target, and current model decision are
documented in `docs/uncertainty_policy_freeze.md`.

Listen for UGV01 bridge packets:

```powershell
python -m DigitalTwin.telemetry_receiver --port 5005
```

Open the visual digital-twin replay dashboard:

```powershell
python -m DigitalTwin.dashboard.server --open
```

The dashboard reads the 20 accepted benign `T:147` logs and shows BN220 GPS,
the GPS-fused operational EKF, and the GPS-independent security predictor.
It also shows current rover pose, sensor state, uncertainty, security-branch
NIS, and edge packet health. EKF-to-GPS distance is labeled as sensor
agreement rather than physical accuracy because the current logs do not
contain independent ground truth.

Core modules:

- `DigitalTwin/telemetry.py`: packet serializer/deserializer
- `DigitalTwin/telemetry_receiver.py`: UDP hardware packet listener
- `DigitalTwin/kinematics.py`: tracked-drive-compatible motion model
- `DigitalTwin/ekf.py`: prediction and GPS update
- `DigitalTwin/security.py`: GPS-independent predictor, covariance bounds, and trusted evidence gate
- `DigitalTwin/uncertainty.py`: `Q` and `R` estimator
- `DigitalTwin/detector.py`: Mahalanobis, eigenvalue detectability bounds, confidence envelopes
- `DigitalTwin/alarm.py`: robust initialization, motion gating, and persistent alarms
- `DigitalTwin/attack.py`: step, freeze, replay, random drift
- `DigitalTwin/logger.py`: experiment CSV schema
- `DigitalTwin/experiments/experiment.py`: batch automation
- `DigitalTwin/plotting/plot.py`: trajectory/detection plots
- `DigitalTwin/analysis/`: replay, attack campaigns, uncertainty training, and statistical analysis
- `DigitalTwin/dashboard/`: visual accepted-log replay and sensor-health interface

Generate the benign digital-twin accuracy and consistency report:

```powershell
python -m DigitalTwin.analysis.digital_twin_accuracy
```

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
{"T":404,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"YOUR_WIFI","sta_password":"YOUR_PASSWORD"}
```

After sending `T:404`, the OLED `ST` line should show the router-assigned IP.
