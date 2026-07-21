# Real-Log Replay and Attack Study

This analysis consumes the accepted UGV01 benign square-route logs without
changing the raw CSVs, rover firmware, or motion commands. It implements the
offline stage under the primary A1 threat model: an attacker may alter GPS
coordinates delivered to the digital twin but may not alter encoder, IMU,
command, sequence, timing, detector, alarm, or stored raw-log fields.

## Run

```powershell
python -m DigitalTwin.analysis.real_data_study
```

For a quick dataset-selection check without replaying attacks:

```powershell
python -m DigitalTwin.analysis.real_data_study --manifest-only
```

Generated artifacts are written under
`DigitalTwin/datasets/analysis/real_data_study/`, which is ignored by Git.

## Reproducibility Contract

- The canonical manifest contains exactly five trials for each combination of
  smooth/rough surface and low/medium speed.
- Interrupted, request-failure, stale-packet, sequence-gap, short, and
  GPS-invalid candidates are excluded before duplicate selection.
- The original split remains recorded for provenance: trials 1-3 development,
  trial 4 validation, and trial 5 diagnostic in every surface/speed stratum.
- Each accepted raw file is identified by SHA-256 in `benign_manifest.csv`.
- Alarm validation and threshold freezing use benign complete runs only.
  Attack data do not influence threshold selection.

The original trial-5 subset was inspected while diagnosing the alarm, so it is
no longer presented as untouched confirmatory evidence. The frozen alarm uses
all 20 benign runs and reports complete-run leave-one-run-out false alarms.
Each fold derives its threshold from the other 19 runs, and the final deployed
threshold is frozen from all 20 runs. Attack data never tune the alarm.

## Frozen Alarm Policy

- Buffer five valid GPS positions and initialize the EKF from their robust
  median and observed spread.
- Enable mission monitoring after two consecutive tracked-drive motion
  updates, using `0.02 m/s` translation or `0.10 rad/s` yaw rate as motion.
- Keep raw NIS for every update, but alarm only when three of the latest five
  monitored updates exceed the variant threshold.
- Derive each threshold from the maximum benign 3-of-5 operational statistic.
- Preserve the naturally anomalous run; it becomes the single leave-one-run-out
  false alarm, yielding `1/20 = 0.05` for every variant.

The machine-readable policy is `DigitalTwin/configs/locked_alarm_policy.json`.

## Replay Model

Hardware replay uses the locked Waveshare UGV01 motion-model values:

- drive diameter: `0.0523 m`
- encoder counts/revolution: `1092`
- model track width: `0.141 m`
- raw encoder sign: negative count change is positive tracked-drive travel

The `T:147` IMU units are normalized before uncertainty estimation:
acceleration is converted from `mg` to `m/s^2`, and angular rate is converted
from `deg/s` to `rad/s`.

The paired variants are fixed covariance, naive residual-coupled adaptation,
frozen-clean covariance, GPS-independent adaptation, and evidence-gated
adaptation. The evidence gate admits GPS-residual feedback only when IMU or
timing evidence independently indicates a disturbance.

The four primary uncertainty definitions and evidence thresholds are frozen in
`DigitalTwin/configs/uncertainty_policies.json`; their rationale and learned
target are recorded in `docs/uncertainty_policy_freeze.md`.

## Attack Matrix

- Along-track and cross-track step offsets: `0.5, 1, 2, 3, 5, 7.5, 10 m`
- Coordinate freeze
- Five-second value replay
- Along-track and cross-track drift: `0.01, 0.03, 0.05 m/s`
- Cross-track strategic drift during the upper quartile of independent
  uncertainty evidence

Attacks begin at 30 percent of each recorded run. Impact is measured against
the paired clean replay of the same physical log. This is a counterfactual
replay reference, not independent position ground truth.

The held-out runs are also replayed through a deterministic edge-side buffered
condition with `200 ms` added delay and `40 ms` jitter. This transformation
changes delivery timing only and does not alter source packets or rover motion.

## Interpretation

The complete offline campaign now uses all 20 accepted runs at three injection
times and reports physical-run-clustered confidence intervals. Because all 20
runs were available during alarm design, the results characterize the current
design corpus rather than an independent prospective test. Publication claims
about covariance poisoning or evidence-gated mitigation still require the
paired ablations and a small untouched validation dataset.
