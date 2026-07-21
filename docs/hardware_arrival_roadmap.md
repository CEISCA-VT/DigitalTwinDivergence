# Hardware Arrival Roadmap

Status: completed and retained as a hardware-stage summary. The authoritative
current research roadmap is `docs/current_roadmap_latex.tex`.

## Current Completion

The project is approximately 90% complete overall:

- engineering implementation: approximately 95%
- experimental analysis: approximately 90%
- publication preparation: approximately 80-85%

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
- Complete 7,200-scenario statistical GPS-attack campaign.
- Physical-run-clustered confidence intervals, detection delays,
  harmful-but-stealthy probabilities, and directional epsilon estimates.
- Paired covariance-poisoning analysis over 2,400 targeted attacked replays.

The covariance-poisoning mechanism is statistically measurable, but an
operational attacker advantage is not established against the primary
frozen-clean control.

## Remaining Work

1. Complete the matched ordinary-versus-strategic drift comparison.
2. Expand buffered-delay/jitter transfer evaluation and capability maps.
3. Collect a small untouched prospective validation dataset.
4. Freeze final figures, tables, checksums, and manuscript artifacts.
5. Complete the paper results, discussion, limitations, and conclusion.

## Current References

- `docs/current_roadmap_latex.tex`
- `docs/statistical_attack_campaign.md`
- `docs/covariance_poisoning_analysis.md`
- `docs/uncertainty_policy_freeze.md`
- `docs/real_data_study.md`
- `docs/telemetry_protocol.md`
- `docs/log_data_dictionary.md`
