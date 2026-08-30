# Codex Context: Sensor-Lightweight Digital Twin Fidelity — Post-LOSO Analysis and Paper Completion

## 0. Purpose of this file

This file is the working context for continuing the **sensor-lightweight digital twin fidelity** research project after completion of the frozen Twin V2 full i2Nav leave-one-sequence-out study.

The next phase is **analysis, characterization, benchmarking, asset-specific instantiation, and paper integration**.

It is **not** another neural-network architecture search.

It is **not** a cybersecurity paper.

It is **not** an odometry/state-estimation paper whose main goal is simply to minimize ATE.

The object of study is the **digital twin itself**: how faithfully a computational replica of a specific mobile robot follows the physical asset, how divergence evolves over time and operating conditions, what causes loss of fidelity, how much sensing is needed, and what asset-specific information is required before a generic template can reasonably be called the twin of a particular robot.

---

# 1. Research question

The central research problem is:

> **How can a sensor-lightweight computational replica of a specific mobile robot be instantiated and maintained as a faithful digital twin, and how can its physical–virtual fidelity and divergence be quantified as operating conditions and sensor behavior change?**

The learned model is only a **fidelity-maintenance mechanism** inside the twin.

Do not reframe the project as:
- “a new GRU for odometry,”
- “a SOTA localization model,”
- “a cybersecurity detector,”
- “attack detection through twin divergence,”
- or “domain adaptation.”

Security is explicitly deferred to later work.

---

# 2. Non-negotiable scope constraints

## 2.1 Freeze Twin V2

Twin V2 is now frozen.

Do **not**:
- change the V2 architecture,
- tune loss weights,
- change context windows,
- change seeds,
- retrain to improve a weak fold,
- invent V2.1/V3/V4,
- add new model heads,
- optimize hyperparameters after inspecting the full LOSO result.

The full 10-fold × 3-seed result is now the scientific source of truth.

## 2.2 No Kaggle / GPU runs are required for the next analysis phase

The expensive training is complete.

The user will unzip the full LOSO archive locally into the repository, expected somewhere under:

```text
results/i2nav_v2_full_loso/
```

Do not assume the exact nested layout without inspecting it.

The next work should be designed to run locally using normal Python/pandas/numpy/scipy/matplotlib where possible.

GPU use should not be introduced unless a task genuinely requires model inference that cannot be reproduced from the saved artifacts.

## 2.3 Preserve the statistical hierarchy

The hierarchy is:

```text
timestamp ⊂ seed run ⊂ held-out physical sequence ⊂ dataset
```

Rules:
- timestamps are correlated and are **not independent replicates**;
- three seeds on the same held-out sequence measure algorithmic variability, not three independent physical experiments;
- for dataset-level claims, first aggregate the 3 seeds within each held-out sequence;
- the **held-out sequence is the primary statistical unit**;
- condition-dependent statistics should be computed inside each sequence/run first, then seed-aggregated, then aggregated across sequences;
- do not obtain tiny p-values by pooling thousands of timestamps.

## 2.4 Keep DT-fidelity evaluation separate from official benchmark evaluation

The DT-fidelity analysis is performed in a common initialized physical–virtual frame and must not use post-hoc alignment to hide drift.

Official i2Nav benchmark evaluation may use whatever alignment/protocol the benchmark requires, but it must be reported separately as:

```text
official benchmark comparison
```

and not substituted for the physical–virtual fidelity analysis.

---

# 3. Frozen experiment provenance

The completed full Twin V2 LOSO run used:

```text
Repository:
https://github.com/CEISCA-VT/DigitalTwinDivergence

Full-LOSO repository commit:
6540c01f90f3c1074de0d8dae9964a5276fbbc91

Twin V2 schema:
i2nav_twin_v2_slow_additive_sensor_consistency_v1

Pilot reference commit:
2e8710f405cdd63fc0fd7960950d038077696eb9

Base seeds:
42
1042
2042

Held-out sequences:
building00
building01
building02
parking00
parking01
parking02
playground00
street00
street01
street02

Total runs:
10 sequences × 3 seeds = 30
```

Runtime used for the completed Kaggle experiment:

```text
Python 3.12.13
PyTorch 2.5.1+cu118
CUDA 11.8
Tesla P100-PCIE-16GB
compute capability 6.0
```

The runtime is provenance only. Do not require this runtime for post-hoc analysis.

---

# 4. Completion audit — already established

The completed summary reports:

```text
expected_runs: 30
run_summary_count: 30
run_complete_marker_count: 30
unique_sequence_seed_keys: 30
all_30_complete: true
```

Replicates:

```text
replicate_01_base42
replicate_02_base1042
replicate_03_base2042
```

Codex should still independently audit the extracted full archive locally before analysis.

---

# 5. Current full-LOSO V1 → V2 results

The full study uses the sequence as the primary unit after averaging three seeds within each sequence.

## 5.1 Macro means

| Metric | Frozen V1 | Frozen V2 | V2 vs V1 |
|---|---:|---:|---:|
| ATE RMSE | 2.8339329 m | 2.3979394 m | **-15.3848%** |
| Heading MAE | 3.3358077 deg | 2.5689349 deg | **-22.9891%** |
| RPE 1 s | 0.0606222 m | 0.0611486 m | **+0.8683%** |
| RPE 5 s | 0.1669656 m | 0.1602911 m | **-3.9976%** |
| RPE 10 s | 0.2714466 m | 0.2532473 m | **-6.7046%** |

## 5.2 Sequence wins

```text
ATE:         V2 better on 6/10
Heading MAE: V2 better on 8/10
RPE 1 s:    V2 better on 7/10
RPE 5 s:    V2 better on 9/10
RPE 10 s:   V2 better on 8/10
```

## 5.3 Sequence-aware bootstrap 95% CI for V2 − V1

```text
ATE:
mean Δ = -0.4359936 m
95% CI = [-1.0318985, -0.0003816]

Heading MAE:
mean Δ = -0.7668728 deg
95% CI = [-1.6934204, -0.0725420]

RPE 1 s:
mean Δ = +0.0005264 m
95% CI = [-0.0040758, +0.0066614]

RPE 5 s:
mean Δ = -0.0066745 m
95% CI = [-0.0152431, +0.0049196]

RPE 10 s:
mean Δ = -0.0181993 m
95% CI = [-0.0303197, -0.0060844]
```

Do not overstate the ATE result. The large mean ATE gain is driven heavily by difficult sequences, especially parking01/parking02. Report both macro mean and distribution of sequence-level paired changes.

## 5.4 Particularly important folds

### parking01

Approximate three-seed means:

```text
V1 ATE = 5.694578 m
V2 ATE = 4.071047 m
change = -28.5101%

V1 heading MAE = 7.805087 deg
V2 heading MAE = 4.910776 deg
```

But V2 seed variability is relatively large:

```text
V2 ATE seed SD ≈ 2.1548 m
```

### parking02

```text
V1 ATE = 14.016535 m
V2 ATE = 11.350361 m
change = -19.0216%

V1 heading MAE = 20.638377 deg
V2 heading MAE = 16.719753 deg

V2 RPE 1 s ≈ 0.019314 m
V2 RPE 5 s ≈ 0.054970 m
V2 RPE 10 s ≈ 0.097139 m

V2 Dp p95 ≈ 22.344767 m
V2 Dp max ≈ 25.561698 m
V2 Dtheta p95 ≈ 30.415317 deg
V2 Dtheta max ≈ 32.836314 deg
```

This remains the key demonstration that:

> short-horizon relative motion can remain highly faithful while a small persistent orientation mismatch creates severe long-horizon physical–virtual divergence.

Do not claim parking02 has been “solved.”

### easy sequences

V2 generally preserves already-good global performance while often improving heading/RPE.

Examples:

```text
street00 ATE change ≈ +0.3947%
playground00 ATE change ≈ +0.6620%
parking00 ATE change ≈ +1.3612%
```

The small regressions are scientifically useful because V2 is not simply over-correcting every trajectory.

---

# 6. Expected artifacts in each V2 full-LOSO run directory

The authoritative runner records these artifacts:

```text
v2_slow_additive_yaw.pt
training_history.csv
v2_prediction_trace.csv
v2_evaluated_trajectory.csv
fidelity_profile.json
fidelity_timeseries.csv
run_summary.json
run_manifest.json
RUN_COMPLETE.json
```

Do not assume these exist until auditing the extracted archive.

Relevant trajectory columns expected by the architecture-independent evaluator:

```text
time_s
gt_east_m
gt_north_m
gt_heading_rad
estimate_east_m
estimate_north_m
estimate_heading_rad
```

Core V2 prediction-trace columns expected by the fidelity evaluator:

```text
time_s
true_delta_v_mps
pred_delta_v_mps
true_delta_omega_radps
pred_total_delta_omega_radps
```

The fidelity evaluator derives:

```text
ATE
heading MAE
RPEp 1/5/10 s
Dp p95/max
Dtheta p95/max
Dv metrics when prediction trace is available
Domega metrics when prediction trace is available
signed/absolute persistent yaw residual
Iomega accumulated yaw residual
```

---

# 7. Existing canonical sensor representation

The platform adapter provides canonical wheel/IMU/ODO quantities.

Important existing canonical signals include:

```text
time_s
wheel_forward_mps
wheel_lateral_mps
wheel_yaw_radps
imu_yaw_radps
odo_forward_mps
yaw_disagreement_radps
yaw_disagreement_normalized
```

The learned Twin core should remain platform-independent. Raw platform-specific wheel IDs or steering labels stay in deterministic adapters.

Existing 30 s sensor summaries include:

```text
mean/std/rms/mean_abs IMU yaw
mean/std/rms/mean_abs wheel yaw
mean/std/rms wheel–IMU yaw disagreement
mean/std normalized disagreement
mean/std/mean_abs ODO forward speed
```

These are useful for the upcoming condition-dependent analysis.

---

# 8. Clarification: UGV01 asset-specific instantiation vs new data collection

## 8.1 What “asset-specific instantiation” means

It does **not** automatically mean “collect a huge new UGV01 dataset.”

It means demonstrating the transition:

```text
generic twin template
        ↓
UGV01-specific sensor/frame/geometry adapter
        ↓
UGV01 deterministic calibration / asset parameters
        ↓
validated computational replica of this specific physical UGV01
```

Examples of asset-specific information include:

```text
wheel/vehicle geometry
encoder/odometry scale
left/right or CW/CCW turn behavior
IMU sign/alignment/bias handling
body-frame conventions
sampling/timing synchronization
native telemetry → canonical body-motion mapping
```

The scientific question is:

> How much does measured physical–virtual fidelity change when the template is bound/calibrated to the specific robot?

## 8.2 Existing documented UGV01 physical evidence

Existing project records already document a low-speed carpet AprilTag-referenced physical experiment.

The independent reference uses a fixed-camera AprilTag/ChArUco pipeline.

Documented current values include approximately:

```text
Position ATE RMSE:
asset/calibrated carpet candidate = 0.252 m
matched frozen baseline           = 0.263 m

Median position error:
candidate = 0.116 m
baseline  = 0.168 m

Position p95:
candidate = 0.554 m
baseline  = 0.586 m

1 s RPE:
candidate = 0.046 m
baseline  = 0.051 m

Heading MAE:
candidate = 21.4 deg
baseline  = 25.2 deg

Heading p95:
candidate = 46.9 deg
baseline  = 53.7 deg
```

The current deterministic low-speed carpet candidate has previously been documented approximately as:

```text
distance scale c_d = 0.95
CW effective baseline = 0.18 m
CCW effective baseline = 0.20 m
gyro blend/parameter beta_g = 0.20
gyro scale c_g = 1.0
```

Treat these as **current carpet-specific calibration evidence**, not universal UGV01 constants.

The reference system also has documented limitations:
- approximately 6.38-pixel mean fixed-reference reprojection discrepancy;
- one older validation case reused a homography because one fixed tag was not visible;
- a previously identified incorrect video/telemetry pairing must remain excluded;
- one carpet evaluation reported synchronization correlation around 0.965 and timing uncertainty around 0.025 s.

## 8.3 Important distinction from the old security project

Old project documents sometimes say synchronized **telemetry + GPS + AprilTag** is still required for final GPS-fusion localization claims.

That requirement came from the earlier security/GPS-fusion project.

The **current fidelity paper is not a GPS-security paper**.

For the current sensor-lightweight DT paper:
- the learned twin is wheel/odometry + IMU based;
- GPS is not required as a learned input;
- an independent AprilTag/video physical trajectory reference can be sufficient for physical–virtual fidelity evaluation if its synchronization and reference accuracy are defensible.

Therefore do not automatically demand a new GPS-synchronized UGV01 campaign just because an older security document says so.

## 8.4 Do we need new UGV01 condition-shift data?

Do **not** assume yes.

First inventory what UGV01 data actually exists in the repository.

The records currently available to this context clearly establish:
- low-speed carpet AprilTag-referenced motion data;
- bench telemetry / IMU / encoder / GPS infrastructure;
- calibration work;
- some repeated physical evidence.

The records available here do **not** clearly establish a completed synchronized multi-condition UGV01 matrix across:
- carpet vs tile/pavement,
- multiple speeds,
- multiple turn/trajectory regimes,
- controlled slip/traction conditions.

If those files now exist in the repository, use them.

If they do not exist, do **not** immediately tell the user to collect a large new dataset.

Instead use this claim hierarchy:

### Minimum strong-paper route

Use:
1. i2Nav for the full condition-dependent fidelity analysis across diverse trajectories/operating regimes;
2. UGV01 existing AprilTag-referenced data as the **asset-specific instantiation case study**.

This can be sufficient if the UGV01 evidence clearly shows the template-to-specific-asset calibration process and measured fidelity change.

### Stronger optional route

Only request additional UGV01 physical data if the paper wants to make a specific claim such as:

> “The instantiated UGV01 twin maintains fidelity across surface, speed, and traction changes.”

A cross-condition physical claim requires cross-condition physical data.

Do not require data for claims the paper does not intend to make.

---

# 9. Immediate local workflow

Assume repository root is the working directory.

The user plans to place the extracted full LOSO results under `results/`.

## Step 0 — inspect before modifying anything

Run an inventory such as:

```bash
git status
git rev-parse HEAD
find results/i2nav_v2_full_loso -maxdepth 4 -type f | sort
```

Do not delete or overwrite raw results.

Create a separate analysis output root, for example:

```text
results/i2nav_frozen_v2_fidelity_analysis/
```

All new analysis must write there.

Recommended subdirectories:

```text
results/i2nav_frozen_v2_fidelity_analysis/
├── audit/
├── reproducibility/
├── condition_fidelity/
├── twin_fidelity_profiles/
├── official_i2nav/
├── sensing_fidelity/
├── ugv01_instantiation/
├── figures/
├── tables/
└── manuscript_numbers/
```

---

# 10. Task A — full result audit

Create a script such as:

```text
DigitalTwin/analysis/i2nav_v2_post_loso_audit.py
```

It should:

1. recursively identify all run directories;
2. require exactly 30 unique `(test_sequence, base_seed)` keys;
3. verify:
   - `RUN_COMPLETE.json`,
   - `run_summary.json`,
   - `run_manifest.json`,
   - `v2_evaluated_trajectory.csv`,
   - `v2_prediction_trace.csv`,
   - `fidelity_profile.json`,
   - `fidelity_timeseries.csv`;
4. verify the 10 expected sequences and 3 expected base seeds;
5. verify recorded full-LOSO commit is:
   ```text
   6540c01f90f3c1074de0d8dae9964a5276fbbc91
   ```
6. verify trajectory and prediction timestamps align;
7. re-run the architecture-independent fidelity evaluator on at least all 30 V2 run artifacts;
8. compare recomputed profile values to saved profile/run summary within numerical tolerance;
9. report missing/duplicate/corrupt/non-finite artifacts.

Outputs:

```text
audit/full_loso_audit.json
audit/full_loso_audit.csv
audit/checksums.csv
```

Acceptance condition:

```text
30/30 complete
30 unique sequence × seed identities
0 missing required artifacts
0 duplicate identities
0 fidelity replay mismatches beyond tolerance
```

---

# 11. Task B — final Fixed Physics → V1 → V2 reproducibility package

Do not show only V1 → V2.

The paper needs the full progression:

```text
Fixed Physics → Twin V1 → Twin V2
```

Find the authoritative frozen Fixed Physics and V1 results already in the repository.

Do not recalculate Fixed/V1 from a different code path if canonical frozen values already exist.

For each sequence and model, report:

```text
ATE RMSE
heading MAE
RPE 1 s
RPE 5 s
RPE 10 s
Dp p95
Dp max
Dtheta p95
Dtheta max
persistent yaw residual
Iomega max absolute accumulated yaw residual
```

Statistical outputs:

1. per-run table;
2. 3-seed mean and SD within each sequence;
3. dataset macro mean across the 10 sequence means;
4. paired sequence-level V2−V1 differences;
5. paired sequence-level V1−Fixed differences;
6. sequence-level bootstrap 95% CI;
7. exact paired sign-flip/permutation tests where appropriate;
8. effect sizes, but do not oversell significance with only 10 sequence units;
9. number of sequence wins/losses;
10. worst regression and largest improvement.

Important:
- bootstrap should resample held-out sequences, not timestamps;
- seed variability should be reported separately;
- show the distribution of sequence-level differences, not just a single macro percentage.

Recommended outputs:

```text
reproducibility/fixed_v1_v2_per_run.csv
reproducibility/fixed_v1_v2_per_sequence.csv
reproducibility/fixed_v1_v2_macro.json
reproducibility/paired_tests.json

tables/table_fixed_v1_v2_main.csv
tables/table_sequence_level_results.csv

figures/fig_sequence_ate_changes.*
figures/fig_sequence_heading_changes.*
figures/fig_rpe_horizon_changes.*
figures/fig_seed_variability.*
```

---

# 12. Task C — condition-dependent fidelity analysis

This is the highest-priority next scientific analysis.

## 12.1 Goal

Answer:

> Under what measurable operating conditions does physical–virtual fidelity improve or deteriorate?

Candidate context variables:

```text
forward speed
absolute forward speed
longitudinal acceleration
absolute acceleration
yaw rate
absolute yaw rate
curvature / turning intensity
wheel yaw
wheel–IMU yaw disagreement
normalized wheel–IMU disagreement
lateral wheel-motion proxy
persistent yaw residual
time / elapsed trajectory fraction
sequence/environment family
```

Prefer sensor/adapter-derived variables for mechanistic interpretation.

Ground truth may be used for **descriptive physical analysis**, but clearly distinguish:
- variables available to the twin online,
- variables available only for offline analysis.

## 12.2 Reconstruct context without retraining

Use:
- existing public i2Nav raw data,
- existing `original.prepare_sequence`,
- existing canonical platform adapter,
- saved V2 trajectories,
- saved V2 prediction traces,
- saved fidelity timeseries.

Do not rerun training.

If V1 and Fixed per-timestep trajectories exist, analyze all three models under the same context bins.

If V1/Fixed per-timestep trajectories are unavailable:
- first inspect frozen directories for equivalent traces;
- only regenerate deterministic evaluation if it can be done exactly from frozen checkpoints/configs without retuning;
- document any regenerated path.

## 12.3 Condition binning

Avoid arbitrary post-hoc threshold hunting.

Preferred options:

### Physically meaningful bins

For example:

```text
low / medium / high speed
near-straight / moderate turn / strong turn
low / medium / high wheel–IMU disagreement
low / high acceleration
early / middle / late trajectory
```

Thresholds should be:
- defined once,
- documented,
- applied consistently across models,
- and not tuned to maximize V2 improvement.

If quantile bins are used, compute them in a transparent way and explain whether they are global or sequence-specific.

## 12.4 Statistical hierarchy for conditions

For each run and each condition bin:

1. compute condition-specific metrics within that run;
2. aggregate 3 seeds within the held-out sequence;
3. treat the sequence as the independent unit for cross-dataset inference.

Do not concatenate all timestamps from all sequences and run a t-test.

## 12.5 Condition metrics

At minimum report condition-specific:

```text
mean/median Dp
Dp p95
mean/median Dtheta
Dtheta p95
RPE where a horizon is meaningful inside the context definition
Dv / Domega when available
persistent yaw residual
Iomega growth / accumulated yaw residual
coverage count and duration
```

Also report how much fidelity degrades relative to a nominal/easier condition.

Possible form:

```text
degradation ratio = metric_hard / metric_nominal
```

or

```text
Δ metric = metric_hard - metric_nominal
```

Keep physical units visible.

## 12.6 Mechanistic analysis

Explicitly test the chain:

```text
wheel/IMU mismatch
    ↓
persistent yaw-rate residual
    ↓
accumulated Iomega
    ↓
heading-state divergence Dtheta
    ↓
global position divergence Dp / ATE
```

Useful analyses:
- within-sequence correlations;
- sequence-level summaries of correlation strength;
- lagged relationships if defensible;
- easy vs representative vs hard fold visualization;
- parking02 as a hard case, but not the only example.

Avoid presenting timestamp-level correlation significance as independent evidence.

Recommended outputs:

```text
condition_fidelity/condition_definition.json
condition_fidelity/per_run_condition_metrics.csv
condition_fidelity/per_sequence_condition_metrics.csv
condition_fidelity/condition_macro_summary.csv
condition_fidelity/mechanism_correlations.csv

figures/fig_condition_speed.*
figures/fig_condition_turning.*
figures/fig_condition_yaw_disagreement.*
figures/fig_condition_time.*
figures/fig_mechanism_chain_easy_hard.*
```

---

# 13. Task D — finalize the empirical Twin Fidelity Profile

The run-level fidelity object should remain multidimensional.

For each `(model, sequence, seed)` define the profile approximately as:

```text
Phi = [
    ATE,
    heading MAE,
    RPEp(1 s),
    RPEp(5 s),
    RPEp(10 s),
    |persistent yaw residual|,
    Dp p95,
    Dtheta p95,
    Dp max,
    Dtheta max,
    Iomega max
]
```

Do not collapse meters, degrees, and rates into one arbitrary score.

Produce:
- per-run profiles;
- per-sequence seed mean/SD profiles;
- dataset macro summaries;
- normalized visualization only for visualization, never as the scientific definition.

Possible figures:
- profile heatmap;
- easy/representative/hard radar chart only as secondary visualization;
- trajectory divergence plots.

Recommended outputs:

```text
twin_fidelity_profiles/all_run_profiles.csv
twin_fidelity_profiles/sequence_profiles.csv
twin_fidelity_profiles/macro_profiles.csv
```

---

# 14. Task E — empirical benign fidelity envelope

This paper is **not a security paper**.

The envelope is simply the empirical normal operating range of twin divergence.

For operating context `c`, characterize componentwise distributions such as:

```text
Dp | c
Dtheta | c
Dv | c
Domega | c
```

and report descriptive percentiles, especially p95.

Do not:
- call p95 an attack threshold;
- report attack detection;
- invent universal “trusted/untrusted” labels;
- define one scalar trust score unless strongly justified.

The output should answer:

> How much physical–virtual disagreement is normal for the validated twin under this operating context?

Recommended outputs:

```text
condition_fidelity/benign_envelope_per_sequence.csv
condition_fidelity/benign_envelope_macro.csv
figures/fig_benign_envelope_by_condition.*
```

---

# 15. Task F — official i2Nav benchmark evaluation

This is separate from DT fidelity.

## Goal

Export the frozen Twin V2 trajectories in the format required by the official i2Nav evaluation protocol and compute protocol-compatible benchmark metrics.

Rules:
- do not change V2;
- do not tune on benchmark results;
- preserve original frozen predictions;
- use official ground truth / alignment rules exactly;
- document any conversion and coordinate-frame transformations.

First search the repository for:
- official i2Nav trajectory files;
- official evaluation scripts;
- prior `i2nav_final_model_study*.py`;
- benchmark-format conversion utilities.

If the exact official evaluator is not in the repository, create a clearly isolated export pipeline and document what still requires external official tooling.

Outputs:

```text
official_i2nav/exported_trajectories/
official_i2nav/evaluation_manifest.json
official_i2nav/official_metrics.csv
official_i2nav/protocol_notes.md
```

Never mix official aligned ATE with common-initialized DT ATE in the same column without labeling the protocol.

---

# 16. Task G — sensing–fidelity comparison

Only do this after official i2Nav metrics are available.

The scientific question is not necessarily:

> Does ODO+IMU beat every camera/LiDAR system?

The question is:

> Where does a wheel/odometry + IMU digital twin lie on the fidelity-versus-sensing-burden frontier?

For each comparable method collect:

```text
method
sensor stack
official metric(s)
whether metric/protocol is directly comparable
compute burden if actually reported
sensor-count/type burden
localization infrastructure assumptions
```

Avoid invented power/latency claims.

Classify comparison rows:

```text
directly protocol-compatible
partially comparable
not directly comparable
```

Only use the first class for quantitative headline comparisons.

Recommended output:

```text
sensing_fidelity/sota_comparison.csv
sensing_fidelity/sensor_stack_summary.csv
figures/fig_sensing_fidelity_frontier.*
```

---

# 17. Task H — UGV01 asset-specific instantiation using existing data first

Before planning any new experiment, recursively inventory all UGV01 files in the repository.

Search for:
- telemetry CSVs;
- AprilTag/video-derived ground-truth CSVs;
- ChArUco/calibration outputs;
- route labels;
- surface labels;
- command-speed labels;
- calibration constants;
- time-alignment diagnostics;
- run manifests;
- scripts producing the 0.252 m ATE result.

Create:

```text
ugv01_instantiation/data_inventory.csv
ugv01_instantiation/data_quality_report.md
```

For every physical run record:

```text
run id
date/session
route
surface
command speed
available encoder/ODO
available IMU
available AprilTag/video ground truth
available GPS (optional for this paper)
time alignment method
time alignment uncertainty
reference reprojection error / quality
whether run is accepted or excluded
reason for exclusion
```

## Asset-instantiation comparison

If existing data supports it, compare stages such as:

```text
A. generic nominal kinematics
B. UGV01 geometry/frame adapter
C. deterministic UGV01 calibration
D. optional frozen learned-template correction if technically valid
```

For each stage compute the same fidelity profile where possible.

The claim should be:

> asset-specific information measurably changes physical–virtual fidelity.

Do not claim:
> the generic i2Nav model is automatically a full digital twin of UGV01.

If only the low-speed carpet data is defensible, label this honestly:

```text
UGV01 asset-instantiation case study under low-speed carpet operation
```

That can still be valuable.

---

# 18. UGV01 condition-shift study — conditional, not automatically required

First determine whether the repository already contains multiple physical UGV01 conditions.

If yes:
- analyze them.

If no:
- do **not** create a new collection requirement unless the paper needs a physical UGV01 cross-condition claim.

The main paper can use:

```text
i2Nav = broad condition-dependent fidelity evidence
UGV01 = specific-asset instantiation evidence
```

This division is scientifically defensible and avoids unnecessary hardware work.

Additional UGV01 data becomes high-value only if the paper wants stronger claims across:
- surfaces,
- speeds,
- turn regimes,
- slip/traction changes,
- repeated sessions.

---

# 19. TerraSentia role

TerraSentia is secondary portability evidence.

Existing evidence indicates only a small but consistent zero-shot gain.

Do not turn the paper into a domain-adaptation paper.

Use TerraSentia, if retained, to distinguish:

```text
generic template portability
```

from:

```text
asset-specific digital-twin instantiation
```

A zero-shot transferred model should not automatically be called the digital twin of the target platform.

---

# 20. Core paper story after the LOSO result

The strongest final narrative is:

```text
1. Define mobile-robot DT fidelity as a multidimensional temporal physical–virtual divergence process.

2. Show that local/finite-horizon fidelity and global synchronization are different:
   a twin can have low 1–10 s relative error and still lose long-horizon synchronization.

3. Diagnose persistent yaw mismatch as a major mechanism for accumulated divergence.

4. Introduce a sensor-lightweight fidelity-maintenance mechanism (V2) that specifically addresses the persistent component while preserving fast transient correction.

5. Demonstrate reproducibility with frozen 10-fold LOSO × 3 seeds.

6. Characterize when fidelity deteriorates as operating context changes.

7. Demonstrate how a generic template becomes a twin of a specific UGV01 through asset-specific binding/calibration.

8. Quantify where the lightweight wheel/IMU twin lies relative to heavier sensing systems under a compatible benchmark protocol.
```

The contribution is not the GRU.

---

# 21. Claims that are currently defensible

With the completed full LOSO, likely defensible statements include:

- a lightweight wheel/odometry + IMU computational replica can maintain strong finite-horizon fidelity on many i2Nav trajectories;
- the frozen V2 reduces macro heading error and long-horizon RPE relative to V1;
- V2 improves mean ATE substantially because it reduces difficult-fold divergence, although the improvement is not uniform across sequences;
- short-horizon fidelity can remain strong while persistent yaw mismatch produces severe long-horizon global divergence;
- fidelity varies substantially by physical sequence/operating regime;
- parking02 remains a significant hard condition rather than a solved case.

Do not turn these into stronger claims without the corresponding evidence.

---

# 22. Claims that are not yet justified

Do not claim:

- V2 is universally superior on all i2Nav sequences;
- parking02 is now high-fidelity;
- the twin is robust across all surfaces/speeds/slip regimes unless physical or benchmark condition analysis shows it;
- UGV01 has been validated across conditions that are not present in the data;
- zero-shot TerraSentia is automatically a target-platform digital twin;
- the lightweight twin matches SOTA heavy sensing until official compatible metrics are computed;
- the p95 benign envelope is an attack detector;
- the work is a security system;
- the network architecture itself is the primary novelty.

---

# 23. Paper-result figures to target

A strong final figure set would likely contain:

## Figure 1 — System / twin concept
Physical robot → canonical wheel/IMU adapter → nominal physics + bounded learned maintenance → virtual twin → multidimensional fidelity evaluator.

## Figure 2 — Fixed → V1 → V2 macro and sequence-level performance
Show ATE, heading, and RPE horizons.

## Figure 3 — sequence-level paired changes
Highlight that gains are concentrated on hard folds and easy-fold regressions are small.

## Figure 4 — local vs global fidelity
Compare easy/representative/hard trajectories; especially show parking02 low RPE but high accumulated divergence.

## Figure 5 — mechanistic yaw chain
Wheel–IMU disagreement / persistent yaw residual → Iomega → heading divergence → position divergence.

## Figure 6 — condition-dependent fidelity
Speed / turning / disagreement / time bins.

## Figure 7 — UGV01 asset-specific instantiation
Nominal template vs UGV01-calibrated twin against independent AprilTag/video reference.

## Figure 8 — sensing–fidelity frontier
Only after official benchmark comparison is valid.

Do not create all figures before the data analysis determines which ones are actually informative.

---

# 24. Recommended code organization

Prefer new analysis modules rather than modifying frozen training code.

Suggested files:

```text
DigitalTwin/analysis/
    i2nav_v2_post_loso_audit.py
    i2nav_v2_reproducibility_summary.py
    i2nav_v2_condition_fidelity.py
    i2nav_v2_mechanism_analysis.py
    i2nav_official_export.py
    ugv01_asset_instantiation_analysis.py
```

Tests:

```text
tests/
    test_post_loso_audit.py
    test_condition_fidelity.py
    test_official_export.py
```

Do not edit:

```text
DigitalTwin/analysis/i2nav_v2_full_loso.py
```

except for an objectively necessary compatibility bug fix, and if that ever happens, do not rerun/tune the frozen result without explicit user approval.

---

# 25. Reproducibility requirements for every new analysis

Every analysis output should record:

```text
analysis script
Git commit
input result root
source full-LOSO commit
timestamp
metric definitions
bin definitions
seed aggregation rule
sequence aggregation rule
bootstrap/permutation seed
software versions where relevant
```

Write machine-readable manifests.

Never overwrite original full-LOSO artifacts.

---

# 26. What Codex should do first

Execute the following workflow in order.

## Phase 1 — inventory and audit

1. Locate `results/i2nav_v2_full_loso`.
2. Inspect its exact structure.
3. Verify 30/30 completion.
4. Verify all expected artifacts.
5. Recompute saved fidelity profiles from trajectory/prediction-trace files.
6. Produce audit report.
7. Stop and report any inconsistency before continuing.

## Phase 2 — canonical reproducibility tables

1. Find frozen Fixed Physics evidence.
2. Find frozen V1 evidence.
3. Merge Fixed/V1/V2 per run and per sequence.
4. Produce final statistical tables/paired tests.
5. Reproduce the known V1→V2 macro values above as a sanity check.

## Phase 3 — condition-dependent fidelity

1. Reconstruct sensor/kinematic context for each sequence.
2. Align context to saved V2 fidelity time series.
3. Freeze bins/definitions.
4. Compute within-run condition summaries.
5. Seed-aggregate within sequence.
6. Aggregate across sequences.
7. Produce mechanism analysis and figures.

## Phase 4 — empirical Twin Fidelity Profiles/envelopes

1. Build complete profile database.
2. Build componentwise condition-dependent p95 envelopes.
3. Keep results descriptive and fidelity-focused.

## Phase 5 — UGV01 inventory and existing-data instantiation analysis

1. Inventory all physical UGV01 data.
2. Determine exactly which asset-specific stages can be compared with existing accepted ground truth.
3. Do the existing-data analysis first.
4. Only identify missing data if a desired paper claim is impossible without it.

## Phase 6 — official i2Nav export/evaluation

Do after the internal DT-fidelity characterization is stable.

---

# 27. Decision rule about requesting more data

Codex should never say “collect more data” merely because more data would be nice.

It should use this rule:

```text
1. State the exact desired claim.
2. Identify the minimum evidence needed for that claim.
3. Inventory existing evidence.
4. If existing evidence satisfies the claim, analyze it.
5. If not, identify the exact missing signal/condition/repetition.
6. Only then recommend additional collection.
```

Examples:

### Claim
“Asset-specific calibration improves UGV01 twin fidelity on low-speed carpet.”

Existing AprilTag-referenced carpet data may already be enough.

### Claim
“The UGV01 twin remains robust across carpet, tile, speed changes, and controlled slip.”

This requires those physical conditions to actually exist in the accepted data.

Do not conflate these two claims.

---

# 28. Definition of success for the next phase

The post-LOSO analysis phase is successful when the project has:

```text
[ ] audited 30-run frozen V2 result
[ ] final Fixed/V1/V2 tables
[ ] sequence-aware statistical tests
[ ] condition-dependent fidelity results
[ ] mechanistic divergence analysis
[ ] empirical Twin Fidelity Profile database
[ ] empirical benign fidelity envelopes
[ ] UGV01 existing-data asset-instantiation analysis
[ ] clear determination of whether any additional UGV01 data is truly necessary
[ ] official i2Nav trajectory exports
[ ] official benchmark metrics
[ ] sensing–fidelity comparison
[ ] paper-ready tables/figures
[ ] manuscript numbers updated from the frozen full LOSO
```

---

# 29. Final instruction to Codex

Be conservative with claims.

Prefer:
- reproducible analysis,
- physical interpretation,
- sequence-aware statistics,
- exact provenance,
- clear distinction between template and asset-specific twin,
- and honest failure characterization.

Do not optimize the research story around whatever metric happens to look best.

Do not reopen model design unless the user explicitly asks.

The next scientific question is no longer:

> “Can we make the model more accurate?”

It is:

> **“What do the frozen results tell us about digital-twin fidelity, when does that fidelity break, why does it break, and what information is required to instantiate and maintain a faithful twin of a specific robot?”**
