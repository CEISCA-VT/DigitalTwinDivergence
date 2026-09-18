# Existing-data robustness of service timing qualification

This is a **secondary sensitivity analysis**, not a new training or experimental campaign. It uses the thirty frozen i2Nav V2 evaluated trajectories (ten physical sequences, three seeds), the existing timing grid, and one already saved deterministic fixed-physics trajectory per sequence. Physical sequence remains the independent unit; cells, seeds, services, and source phases are nested observations. The completed primary timing outputs and manuscript were not changed.

Reproduce from repository root:

```powershell
C:\Users\shrey\miniconda3\python.exe -m DigitalTwin.analysis.service_timing_robustness
C:\Users\shrey\miniconda3\python.exe -m pytest tests/test_service_timing_robustness.py -q -p no:cacheprovider
```

`protocol_manifest.json` records settings and fixed-physics source paths. All CSVs and PDF/PNG figures in this directory are newly generated. The 11 focused tests pass.

## 1. Does rate-delay information add value beyond service identity?

Yes, at the **pooled, approximately matched** 30% operating point. The service-identity-only score is the training-nine qualification frequency for the service, constant over its 20 delivery cells. Its threshold is selected by inner training-sequence acceptance, just like the empirical transfer-frequency surface. At 30% target, identity-only accepts **200/800** conditions (25.0%) and falsely qualifies **82/200 (41.0%)**. The surface accepts **226/800** (28.25%) and falsely qualifies **10/226 (4.42%)**. The paired pooled false-qualification difference (surface minus identity) is **-36.6 percentage points**, 95% physical-sequence bootstrap interval **-45.4 to -26.8 points**. This comparison is approximate, not exactly matched: achieved acceptance differs by 3.25 points. Source: `identity_vs_surface_paired_bootstrap.csv`.

Within any one service, identity-only has a single tied score, so thresholding accepts **all 20 delivery cells or none**. At the 30% overall target it accepts all 200 local-10-s cells and none of the other three services. Therefore **no within-service matched-acceptance comparison with identity-only is available**. The pooled gain is evidence that condition-specific cells add value beyond an all-or-none service prior, but is not a within-service discrimination estimate at matched acceptance. The stronger reviewer-facing test would require a baseline with usable within-service score resolution.

| Service; 30% overall target | Empirical accepted; false/accepted | AoI-only | Kinematic staleness | Logistic | Identity-only |
|---|---:|---:|---:|---:|---:|
| Local 1 s | 90; 1/90 | 0; undefined | 28; 0/28 | 68; 12/68 | 0; undefined |
| Local 5 s | 54; 4/54 | 40; 0/40 | 28; 0/28 | 56; 16/56 | 0; undefined |
| Local 10 s | 82; 5/82 | 142; 29/142 | 39; 6/39 | 75; 12/75 | 200; 82/200 |
| Global | 0; undefined | 40; 24/40 | 146; 97/146 | 36; 26/36 | 0; undefined |

There are 200 delivery cells per service (ten sequences x 20 grid settings), **not 200 independent experiments**. Most methods have substantially different within-service acceptance. With a predeclared 5-percentage-point tolerance for achieved acceptance, only the empirical-versus-logistic comparisons for local 5 s and local 10 s are near-matched; their paired physical-sequence bootstrap intervals both cross zero. The global service provides **no empirical accepted cells** at this overall operating point. Thus the pooled advantage cannot be stated as a demonstrated win for each service. `per_service_nested_predictions.csv`, `per_service_nested_summary.csv`, and `per_service_paired_bootstrap.csv` preserve all five predeclared acceptance targets, exact denominators, intervals, and unavailable comparisons.

![Per-service held-out results](per_service_qualification.png)

![Surface versus identity-only](surface_vs_identity.png)

## 2. Contract-setting sensitivity

The original 0.80 qualification requirement is primary. The predeclared descriptive requirements 0.70 and 0.90 reuse the saved 10-Hz ZOH response cells, horizons, tolerances, and coverage gate without retuning. At requirements 0.70/0.80/0.90, the 40 sequence-service cases partition into:

| Joint requirement | Already qualified degraded | Delivery-remediable | Persistent under ideal |
|---:|---:|---:|---:|
| 0.70 | 9 | 26 | 5 |
| **0.80 primary** | **3** | **31** | **6** |
| 0.90 | 1 | 30 | 9 |

The existence of delivery-remediable **and** persistent cases survives these requirements, but counts move materially and must not be treated as invariant. `contract_sensitivity_per_sequence.csv`, `contract_sensitivity_summary.csv`, and `contract_remediability_by_requirement.csv` include every setting. These are not safety thresholds. There is no pre-frozen neighboring **physical-unit discrepancy tolerance grid** in this timing replay; changing position/heading limits after examining these results would be a new protocol, so no favorable tolerance point was selected.

## 3. Sampling-phase and represented-history aliasing

Only source instants on the saved **10-Hz** clock are observed. Inside one native 100-ms interval, **phase 0 is the only valid recorded source instant**; 10/25/50-ms sub-sample phases would require invented trajectory states. As an additional sensitivity, we enumerated every native-grid residue within each rate's delivery stride: one phase at 10 Hz, two at 5 Hz, five at 2 Hz, ten at 1 Hz. Each phase remains nested within its physical sequence.

The alias audit hashes the complete source-index history on the evaluation clock. **25-ms and 50-ms delays are identical in all 540 comparable run/rate/phase histories**. Depending on rate and phase, five nominal delays yield only **three or four unique represented-state histories**. The apparent extra delay-grid resolution must not be presented as five distinct receiver histories. `represented_history_signatures.csv` and `represented_history_alias_summary.csv` give every hash and count.

At 2 Hz/200 ms, sequence-mean ZOH joint satisfaction across its five native phases ranges approximately **0.394-0.419 global**, **0.677-0.717 local 10 s**, **0.514-0.589 local 1 s**, and **0.599-0.638 local 5 s**. Phase can matter quantitatively; it does not introduce new independent evidence. For the two phases shared by degraded/practical conditions (using the actual phase-0 10-Hz ideal reference), ZOH retains **31 delivery-remediable and six persistent** cases at both phases. Full seed, phase, service, and condition results are in `receiver_phase_per_run.csv` and `receiver_phase_per_sequence.csv`.

## 4. Causal receiver reconstruction

The V2 files contain `corrected_v_mps` and `corrected_omega_radps`. The sensitivity receiver takes the **last delivered** virtual pose and those rates at its source index, then integrates a constant-turn arc over its age. It does not read later virtual samples or physical measurements. Freshness continues to use the original represented-state timestamp; extrapolation **does not create a fresh observation**. The zero-order-hold receiver remains the primary protocol.

At 2 Hz/200 ms, mean sequence-level joint satisfaction changes from ZOH to extrapolation as follows: global **0.418 -> 0.468**, local 10 s **0.677 -> 0.768**, local 5 s **0.599 -> 0.672**, but local 1 s **0.514 -> 0.497**. Freshness is unchanged by construction. At sequence-aggregate qualification, extrapolation changes the 40 cases from **31 remediable / six persistent / three already qualified** to **26 / six / eight** for phase 0. The effect is service-dependent, not an unconditional improvement. At 5 Hz/50 ms, local-10-s qualification rises from 9/10 to 10/10, while local-1-s stays 10/10. Exact physical, freshness, joint, position, heading, duration, and denominator columns are in the receiver CSVs; `receiver_remediability.csv` preserves classifications.

![Receiver sensitivity](receiver_robustness.png)

## 5. Paired frozen-twin configuration

The alternate is the already saved deterministic `fixed_v5` physics trajectory, one per held-out sequence, previously used in the official-benchmark workflow. Clock, length, initial ENU position, and quaternion-to-yaw conventions are checked before pairing it with the same sequence's physical reference. No trajectory is trained or chosen based on timing performance. Fixed physics has one deterministic run per sequence, while V2 has three seeds averaged within sequence; these are paired **configuration variants**, not additional physical experiments. V1 was not used because an exact matching frozen trajectory set was not present.

| Configuration | Already qualified degraded | Delivery-remediable | Persistent under ideal | Global persistent |
|---|---:|---:|---:|---:|
| Fixed physics | 1 | 23 | 16 | 10/10 |
| Frozen V2 | 3 | 31 | 6 | 6/10 |

Improved delivery restores some qualification in **both** configurations. The distinction between delivery-induced and persistent error is visible in both. Averaged descriptively over the 40 sequence-service cases, ideal mean position discrepancy is **1.289 m** for fixed physics versus **0.575 m** for V2; under 2 Hz/200 ms, signed held-minus-ideal position increments are **+0.113 m** and **+0.186 m**, respectively. Lower ideal discrepancy therefore does not imply a smaller held-state increment. In fact, V2 has lower ideal position error in all 40 cases, but worse degraded held error in one (building02 global); it remains better in the other 39. This is a narrow counterexample, **not** evidence that V2 is generally less delay-tolerant. `configuration_comparison.csv`, `configuration_remediability.csv`, and `configuration_paired_discrepancy.csv` preserve the paired rows.

![Paired configuration sensitivity](configuration_robustness.png)

## Interpretation for the manuscript

**Strengthened:** Timing-cell information does add pooled value beyond a service-only prior at roughly comparable acceptance; remediable/persistent distinctions survive 0.70-0.90 qualification requirements and a second, legitimate frozen configuration. The receiver has sufficient saved rates for a genuinely causal extrapolation sensitivity.

**Narrowed:** Per-service matched superiority is mostly unavailable because scores and achieved coverage differ. Nominal 25 and 50 ms delay cells are indistinguishable at the evaluation clock. Receiver reconstruction and source phase affect counts/satisfaction, so numeric budgets depend on replay implementation. The alternate fixed model has many more persistent global failures.

**Unsupported:** A universal timing threshold, monotone delay boundary, optimal synchronization policy, per-service win over all baselines, extrapolation as universally better, or additional independent sequence evidence from seeds/phases/configurations.

**Main-paper recommendation (one or two additions only):** include the pooled surface-versus-service-identity result **with its acceptance caveat**, and a compact paired fixed-physics/V2 remediability table. Put per-service unmatched operating points, aliased delay cells, phase, reconstruction, and 0.70/0.90 sensitivity in supplementary material. Do not revise the manuscript until these limitations are accepted as part of the claim.
