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

To regenerate the complete paper-facing analysis package, including the
expanded attack grid, covariance study, threshold sweep, revised mathematical
diagnostics, and figures copied under `results/`, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\regenerate_all_results.ps1
```

The optional combined GPS-content and buffered-transport campaign is excluded
by default. Add `-IncludeBufferedAttackCampaign` to include it.

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
- Keep the variant score for every update. The primary NIS-family detectors
  alarm when three of the latest five monitored scores exceed the variant
  threshold; GPS-jump and CUSUM use a one-of-one sequential threshold.
- Derive each threshold from the maximum benign operational statistic for that
  variant.
- Preserve any naturally anomalous run; leave-one-run-out false alarms are
  reported rather than removed post hoc. Because the final threshold is an
  order statistic from the design corpus, this result is not independent
  false-alarm validation.

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

The paired primary variants are fixed covariance, naive residual-coupled
adaptation, frozen-clean covariance, GPS-independent adaptation, and
evidence-gated adaptation. The expanded comparator suite adds GPS-jump, raw
digital-twin residual, robust innovation gate, Huber EKF, CUSUM whitened
innovation, and innovation-matching adaptive EKF baselines. The revised-model
variants add a GPS-bias EKF with fixed covariance and a GPS-bias EKF whose
covariance adaptation is evidence-gated. In the revised architecture, the
security score remains a GPS-vs-motion innovation check, so GPS bias modeling
can improve the operational twin without letting GPS residuals authorize their
own covariance growth.

The policy definitions are recorded in
`DigitalTwin/configs/uncertainty_policies.json`. Because the security-reference
architecture changed on July 29, 2026, the previous locked thresholds and
attack-campaign outputs are provenance artifacts and must be regenerated
before they are quoted as results.

## Revised Mathematical Diagnostics

The paired replay now retains each branch's pre-update innovation and
innovation covariance. For every attack window it records:

- the counterfactual attacked NIS under the paired reference covariance;
- normalization credit, reference-metric innovation change, and their exact
  NIS score decomposition;
- empirical innovation-covariance ordering frequency;
- directional and worst-direction detectability-loss factors;
- rolling residual gate-pass fraction and residual-cover bound checks.

Run only this summary stage after generating `campaign_summary.csv` with:

```powershell
python -m DigitalTwin.analysis.math_revision_analysis
```

The summary writes `math_mechanism_summary.csv`,
`math_revision_validation.json`, `math_revision_report.md`, and
`math_score_decomposition.png`.

## Attack Matrix

- Along-track and cross-track step offsets: `0.5, 1, 2, 3, 5, 7.5, 10 m`
- Coordinate freeze
- Five-second value replay
- Along-track and cross-track drift: `0.01, 0.03, 0.05 m/s`
- Cross-track strategic drift during the upper quartile of independent
  uncertainty evidence

Attacks begin at 25, 50, and 70 percent of each post-motion run horizon. Impact
is measured against the paired clean replay of the same physical log. This is a
counterfactual replay reference, not independent position ground truth.

The held-out runs are also replayed through a deterministic edge-side buffered
condition with `200 ms` added delay and `40 ms` jitter. This transformation
changes delivery timing only and does not alter source packets or rover motion.

## Interpretation

The complete offline campaign now uses all 20 accepted runs at three injection
times and reports physical-run-clustered confidence intervals. Because all 20
runs were available during alarm design, the results characterize the current
design corpus rather than an independent prospective test. Publication claims
about covariance poisoning or evidence-gated mitigation still require the
paired ablations and an untouched prospective validation corpus sized to the
strength of the intended false-alarm claim.
