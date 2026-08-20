# Official i2Nav Benchmark Protocol Audit

Date: 2026-08-20

Status: **public-source protocol verified; ready for a later frozen V2 official evaluation**. No official benchmark export or scoring was performed in this task.

This audit separates the project-internal digital-twin fidelity framework from an official i2Nav benchmark evaluation. The frozen Twin V2 predictions were not retrained, retuned, edited, checkpoint-selected, or post-processed.

## Scope Checked

Repository instructions and context reviewed:

- `AGENTS.md`
- `CODEX_CONTEXT_DT_FIDELITY_POST_LOSO.md`

Completed post-LOSO analyses inspected:

- `results/i2nav_v2_post_loso_analysis/all_sequence_mechanism/`
- `results/i2nav_v2_post_loso_analysis/condition_fidelity/`
- `results/i2nav_v2_post_loso_analysis/benign_fidelity_characterization/`
- `results/i2nav_v2_post_loso_analysis/loso_envelope_validation/`
- `results/i2nav_v2_post_loso_analysis/parking02_vs_parking00/`

Frozen V2 prediction availability was verified:

- Source root: `results/i2nav_v2_full_loso/i2nav_v2_full_loso/`
- Replicates present: `replicate_01_base42`, `replicate_02_base1042`, `replicate_03_base2042`
- Held-out sequences per replicate: 10
- Total frozen evaluated V2 trajectories present: 30
- Internal trajectory file per run: `v2_evaluated_trajectory.csv`

## Audit Finding

The previous local-only audit stopped because the exact external i2Nav evaluation protocol had not been verified locally. This follow-up task cloned and inspected authoritative public i2Nav-WHU sources:

- `i2Nav-WHU/i2Nav-Robot` at commit `2ffdca6d56e6432d4daf27c070e570171548f32d`
- `i2Nav-WHU/evaluate_odometry` at commit `6553b1a1d1de79ea50686664ceaacfa02d4f2fe1`
- `i2Nav-WHU/LE-VINS` at commit `ad0b395f5675d17a35ed1a6386f411fd1d61ee19`

The metric/alignment protocol in the repo-local implementation is now verified against the public `evaluate_odometry` source. The active Python evaluator environment was also prepared with `evo v1.37.0` and `numpy-quaternion 2024.0.13`, and a minimal synthetic TUM trajectory sanity test passed.

No frozen V2 trajectories were exported or scored.

## Candidate Protocol Evidence Found

| Artifact | What It Contains | Audit Interpretation |
|---|---|---|
| `results/i2nav_final_model_study/official_protocol_manifest.json` | Manifest naming `i2Nav-WHU/evaluate_odometry`, SE(3) no-scale alignment, APE translation/rotation RMSE, distance-based RPE at 50/100/150/200/250/300 m. | Consistent with the authoritative public evaluator. |
| `results/i2nav_final_model_study_repaired/official_protocol_manifest.json` | Same official-style protocol record, with a clearer ENU/FLU to NED/FRD conversion note. | Consistent with the authoritative public evaluator. |
| `DigitalTwin/analysis/i2nav_final_model_study.py` | Implements `official_i2nav_evaluate`, `estimate_to_official_trajectory`, and official trajectory saving using `evo`. | Protocol-equivalent for metric mechanics. Project-specific ENU/FLU to NED/FRD export must be documented. |
| `DigitalTwin/analysis/i2nav_final_model_study_v2.py` | Contains the same official-style evaluator/converter functions for the later V2 study path. | Protocol-equivalent for metric mechanics. Project-specific ENU/FLU to NED/FRD export must be documented. |
| `results/i2nav_final_model_study*/phase1_official_fixed_v5.csv` | Prior fixed-physics official-style replay results. | Prior evidence only. It does not evaluate frozen V2 full-LOSO runs and should not be substituted for the requested V2 benchmark. |
| `public_datasets/im2nav/*_trajectory.csv`, `*_groundtruth.nav`, `*_F9P_GNSS.pos`, `*_ODO_SPEED.txt`, `calibration.yaml` | Dataset files and reference trajectories. | Data are present, but no official benchmark README/script was found locally in the dataset tree. |

## Exact Protocol Fields

Based on the repo-local manifest and code, the intended official-style protocol appears to be:

- Protocol source named in repo artifacts: `i2Nav-WHU/evaluate_odometry`
- Evaluator mechanics: `evo` trajectory association and metrics
- Required trajectory format in repo-local exporter: TUM-style rows with `t x y z qx qy qz qw`
- Coordinate convention in repo-local converter: i2Nav NED/FRD
- Internal-to-official conversion: internal ENU/FLU planar `east, north, yaw` is converted to official `north, east, down, yaw_ned`
- Timestamp convention in repo-local exporter: seconds, matched with maximum synchronization difference `0.005 s`
- Alignment in repo-local manifest/code: SE(3) alignment, no scale correction
- APE metrics in repo-local manifest/code: translation RMSE and rotation-angle RMSE in degrees
- RPE metrics in repo-local manifest/code: distance-based RPE at 50, 100, 150, 200, 250, and 300 m, all pairs, relative delta tolerance `0.002`

These protocol fields are now verified against the public `i2Nav-WHU/evaluate_odometry` repository. The official target trajectory convention is verified by `i2Nav-WHU/i2Nav-Robot`. The internal V2 state-to-official export remains a project-specific conversion because V2 stores planar ENU/FLU state while the benchmark consumes TUM-style local NED poses.

## Local Availability Check

The active interpreter for this repository is:

```text
C:\Python313\python.exe
```

Installed and verified:

```text
evo v1.37.0
numpy-quaternion 2024.0.13
```

A non-collinear synthetic TUM trajectory was evaluated through the cloned `evaluate_odometry/evaluate.py` path. The sanity test associated 401 poses and returned zero APE/RPE after SE(3) alignment for an offset copy, as expected.

## What Remains To Proceed

The benchmark protocol is now verified well enough for a later frozen V2 official evaluation. The next task should:

1. Export the 30 already frozen `v2_evaluated_trajectory.csv` files to official TUM/NED format.
2. Document the project-specific ENU/FLU to NED/FRD conversion for each exported run.
3. Run the verified evaluator without retraining, retuning, checkpoint selection, or post-hoc trajectory changes beyond the official SE(3) alignment.
4. Aggregate seeds within sequence first, then summarize across the 10 physical sequences.

## Why No Export Was Produced

Official-format export was intentionally not performed in this task because the user requested protocol verification and evaluator setup only. The status for the next task is:

```text
READY_FOR_FROZEN_V2_OFFICIAL_EVALUATION
```

## Answers To Required Questions

**What exactly is the official i2Nav protocol used here?**

The verified protocol is the public i2Nav-WHU `evaluate_odometry` evaluator using `evo`: TUM-style trajectory rows, local NED position for i2Nav-Robot ground truth, trajectory association at `0.005 s`, SE(3) alignment with scale correction disabled, APE translation/rotation RMSE, and distance-based RPE at 50/100/150/200/250/300 m using all pairs and relative delta tolerance `0.002`.

**What are the frozen V2 official benchmark results?**

Not computed. This task explicitly did not export or score frozen V2 trajectories.

**How do they compare with Fixed Physics and V1 where valid?**

Not computed in this task. The protocol is now verified for a later same-protocol comparison.

**Are the hard sequences still hard under official evaluation?**

Not evaluated under the official benchmark layer yet. Internally, parking02 is the strongest hard sequence for global divergence, but that statement remains separate from the official benchmark.

**Does official alignment materially change the interpretation relative to the DT-fidelity analysis?**

Not evaluated yet. The verified official protocol uses SE(3) alignment with no scale correction, so it may reduce or hide some accumulated global offset relative to the internal physical-virtual synchronization analysis. That effect should be measured in the next task.

**Are the results suitable for protocol-compatible comparison against heavier published systems?**

Not yet, because scores have not been generated. The protocol and evaluator setup are now sufficient to generate protocol-compatible numbers in the next task.

## Recommended Next Action

Run the frozen V2 official evaluation as a separate task using the verified protocol. The evaluation should consume only existing frozen outputs and must not modify V2.
