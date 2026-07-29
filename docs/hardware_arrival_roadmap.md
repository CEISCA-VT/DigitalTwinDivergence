# Hardware Arrival Roadmap

Status: completed and retained as a hardware-stage summary. The authoritative
current research roadmap is `docs/current_roadmap_latex.tex`.

The campaign figures in this historical summary predate the July 29, 2026
GPS-independent security-predictor revision. Hardware and dataset completion
remain valid; thresholds, attack summaries, and figures require a software-only
regeneration.

## Current Completion

The original hardware and implementation roadmap is complete for the current
offline study, but strong-publication readiness is still limited by missing
independent validation:

- engineering implementation: approximately 98%
- offline analysis package: approximately 95-97%
- current manuscript package: approximately 80-85%
- strong top-venue readiness without new validation: approximately 60-68%

The project is no longer blocked on batteries, firmware bring-up, calibration,
or benign data collection.

## Completed Hardware Work

- Active firmware is under `ugv01_gps_dev`.
- BN220 GPS remains on the verified original wiring:

```text
BN220 white -> UGV01 RX
BN220 red   -> 5V
BN220 black -> GND
```

- `T:146` returns GPS telemetry.
- `T:147` returns combined encoder, IMU, voltage, GPS, sequence, and firmware
  timing telemetry.
- Edge logging records source, send, arrival, estimate, and alarm timing plus
  packet loss, stale packets, queue depth, and HTTP latency.
- Session clock-offset calibration and edge-side delay/jitter emulation are
  implemented.
- Powered straight, reverse, turn, and repeated-square tests were completed.
- Waveshare nominal geometry is locked for the current motion model:
  `0.0523 m` drive diameter, `1092` counts/revolution, and `0.141 m` model
  track width.

## Completed Dataset Work

The accepted benign corpus contains 20 physical runs:

```text
2 speeds x 2 surfaces x 5 trials = 20 runs
```

The route is a `0.5 m` square repeated three times. Surfaces are smooth kitchen
floor and rough permeable concrete. All accepted runs use baseline Wi-Fi;
buffered transport stress is generated at the edge without changing rover
behavior.

No additional rover operation is required for current offline analysis.

## Completed Analysis

- Real-log EKF and detector replay.
- Frozen robust alarm with motion gating and a three-of-five persistence rule.
- Frozen fixed, naive-adaptive, frozen-clean, GPS-independent, and
  evidence-gated uncertainty variants.
- Rejected learned Random Forest candidate documented without activating it.
- Complete frozen statistical GPS-attack campaign: 1,440 unique
  attack-run-start combinations and 15,840 detector-run evaluations.
- Expanded replay-only grid: 2,640 unique attack-run-start combinations and
  29,040 detector-run evaluations.
- Physical-run-clustered confidence intervals, detection delays,
  tolerance-exceeding paired-divergence probabilities, and directional epsilon
  estimates.
- Paired covariance-poisoning analysis over 2,400 targeted attacked replays.
- External comparator baselines are implemented in the real-data study:
  GPS-jump, raw digital-twin residual, fixed NIS, robust innovation gate,
  Huber EKF, CUSUM whitened innovation, and innovation-matching adaptive EKF.
- Post-campaign tables now include multiple tolerance thresholds, paired
  detector differences, gate-behavior summaries, runtime estimates, threshold
  sweep output, and result provenance hashes.

The covariance-poisoning mechanism is statistically measurable, but an
operational attacker advantage is not established against the primary
frozen-clean control.

## Remaining Work

1. For the no-new-validation path, finish paper polish: stronger theory,
   expanded related work, paired-difference tables, multiple-tolerance tables,
   threshold-sweep interpretation, and artifact packaging.
2. If later pursuing a stronger empirical claim, add independent
   camera/AprilTag ground truth and an untouched prospective benign dataset.
3. Optional extension: run the combined GPS-content plus buffered delay/jitter
   replay campaign as an A3 robustness section.

## Current References

- `docs/current_roadmap_latex.tex`
- `docs/statistical_attack_campaign.md`
- `docs/covariance_poisoning_analysis.md`
- `docs/uncertainty_policy_freeze.md`
- `docs/real_data_study.md`
- `docs/telemetry_protocol.md`
- `docs/log_data_dictionary.md`
