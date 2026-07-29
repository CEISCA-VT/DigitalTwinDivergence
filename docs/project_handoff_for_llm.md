# Digital Twin Divergence Project Handoff

Current as of July 29, 2026. This document gives another collaborator enough
context to continue without reconstructing the full development history.

## Project Goal

The project studies whether adaptive uncertainty in a tracked-rover digital
twin can become a security weakness when GPS measurements are manipulated. A
UGV01 sends encoder, IMU, GPS, power, sequence, and timing telemetry to an edge
logger. An EKF estimates rover state, a Mahalanobis/NIS detector evaluates GPS
consistency, and controlled offline attacks test when corrupted GPS remains
undetected.

## Current Status

- Engineering implementation: approximately 95% complete.
- Current experimental program: approximately 85-90% complete.
- Strong-publication readiness: approximately 55-60% complete.

Hardware bring-up, powered motion tests, and the 20-run benign corpus are
complete. The independent security predictor and revised evidence gate are
implemented. The pre-revision attack campaign remains available as provenance,
but its alarm lock, attack summaries, and figures must be regenerated before
being quoted as current results.

## Hardware and Firmware

- Rover: Waveshare UGV01 tracked platform.
- Active firmware: `ugv01_gps_dev`.
- GPS: BN220 using the original verified RX path.

```text
BN220 white -> UGV01 RX
BN220 red   -> 5V
BN220 black -> GND
```

- `T:146`: GPS-only telemetry.
- `T:147`: combined base, encoder, voltage, IMU, GPS, sequence, and firmware
  timing telemetry.
- `T:404`: Wi-Fi AP/station configuration.

Locked motion-model values:

```text
drive diameter: 0.0523 m
encoder counts/revolution: 1092
model track width: 0.141 m
raw encoder sign: negative count change is positive forward travel
```

## Telemetry and Edge Logging

The edge wrapper polls `T:147`, performs session clock-offset calibration, and
records source/sample, send, edge-arrival, estimate, and alarm times. Logs also
contain sequence gaps, request failures, stale-packet flags, queue depth, HTTP
latency, GPS quality, encoder values, IMU fields, and voltage.

Important references:

- `docs/telemetry_protocol.md`
- `docs/log_data_dictionary.md`
- `docs/ugv01_esp32_bringup.md`

## Benign Dataset

The canonical corpus contains 20 accepted physical runs:

```text
2 speeds x 2 surfaces x 5 trials = 20 runs
```

- Route: `square0p5x3`, a 0.5 m square repeated three times.
- Speeds: low and medium.
- Smooth surface: kitchen floor.
- Rough surface: permeable concrete.
- Live transport: baseline Wi-Fi.
- Buffered transport: deterministic edge-side replay transformation.

The checksum-backed manifest is generated under
`DigitalTwin/datasets/analysis/real_data_study/benign_manifest.csv`.

## Alarm and Uncertainty Policies

The alarm uses robust five-fix GPS initialization, monitoring after sustained
tracked-drive motion, and a three-of-five persistent NIS rule. The previous
numerical alarm lock is superseded because NIS now uses the independent
security predictor. A new benign-only run-level lock is required before attack
evaluation; prospective validation remains required.

Frozen uncertainty comparisons:

- fixed covariance
- naive GPS-residual-coupled adaptation
- frozen-clean covariance schedule
- GPS-coordinate-independent adaptation
- bounded GPS-independent adaptation with trusted-NIS and persistent-bias gating

The learned GPS-independent target is implemented, but the current Random
Forest candidate is disabled because grouped validation was 17-38% worse than
the median baseline.

References:

- `DigitalTwin/configs/locked_alarm_policy.json`
- `DigitalTwin/configs/uncertainty_policies.json`
- `docs/uncertainty_policy_freeze.md`

## Historical Statistical Attack Campaign

The numbers in this section describe the pre-security-predictor campaign. Run
the current regeneration workflow before using them in a manuscript.

Attacks are injected only into GPS coordinates during offline replay. Raw logs,
firmware, and rover behavior remain unchanged.

The complete matrix contains:

- 20 physical runs
- five detector/uncertainty variants
- attack starts at 25%, 50%, and 70% of the post-motion horizon
- along- and cross-track steps at `0.5, 1, 2, 3, 5, 7.5, 10 m`
- along- and cross-track drift at `0.01, 0.03, 0.05 m/s`
- strategic drift, coordinate freeze, and five-second replay

This produces 7,200 attacked scenarios. Outputs include physical-run-clustered
confidence intervals, detection delays, harmful-but-stealthy probability, and
directional `epsilon_50`, `epsilon_90`, and `epsilon_95`.

Headline result: `epsilon_50` is approximately `6.61-8.06 m`. At `0.05 m/s`
cross-track drift, adaptive variants have harmful-but-stealthy probabilities
of approximately `0.10-0.15` while producing no alarms.

Reference: `docs/statistical_attack_campaign.md`.

## Covariance-Poisoning Result

The primary causal comparison is naive residual-coupled adaptation versus the
frozen-clean covariance schedule on matched run, attack, direction, rate, and
start time.

Pooled standard-drift result:

```text
Delta attacked/clean Q ratio:   +0.0243 [0.0185, 0.0300]
Delta attacked/clean NIS ratio: -0.0135 [-0.0174, -0.0090]
Delta max undetected error:     +0.013 m [0.009, 0.017]
Delta harmful probability:      +0.006 [0.000, 0.010]
Delta detection probability:     0.000 [0.000, 0.000]
```

Conclusion: the covariance-poisoning mechanism is measurable, but operational
attacker advantage is not established on the current corpus. The evidence
gate admits residual feedback during approximately 36.6% of attack-window
updates and is not opened more often by the tested GPS drift.

Reference: `docs/covariance_poisoning_analysis.md`.

## Important Commands

```powershell
python -m DigitalTwin.analysis.real_data_study
python -m DigitalTwin.analysis.real_data_study --summarize-existing
python -m DigitalTwin.analysis.train_uncertainty
python -m DigitalTwin.analysis.covariance_poisoning
python -m DigitalTwin.analysis.covariance_poisoning --summarize-existing
python -m pytest -q
```

Generated analysis artifacts are ignored by Git and live under
`DigitalTwin/datasets/analysis/`.

## Remaining Roadmap

1. Correct the prospective false-alarm protocol before collecting new data.
2. Add independent overhead-video or fiducial trajectory ground truth.
3. Reproduce covariance poisoning on a standard adaptive Kalman method and add
   the missing robust and sequential detector baselines.
4. Compare ordinary and strategically scheduled drift under identical attack
   budgets.
5. Run the frozen detector live at the edge and expand buffered delay/jitter
   transfer evaluation.
6. Collect an untouched prospective benign corpus. A target of 60 independent
   runs supports a one-sided 95% upper false-alarm bound near 0.05 when no
   alarms occur; a smaller set must be reported with its wider interval.
7. Freeze publication figures, tables, checksums, dependencies, and the
   claims-versus-evidence map before assembling the results manuscript.

Do not claim prospective deployment validation from the current 20 runs. They
form the design corpus and the attacks are counterfactual offline replays.
