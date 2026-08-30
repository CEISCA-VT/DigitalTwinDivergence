# Frozen Protocol: Service-Relative Operational Fidelity

## Research question

Can a sensor-lightweight, continuously synchronized digital twin be **valid for one operational service but invalid for another**, and can that service validity be characterized using a frozen physical–virtual evaluation protocol plus online-observable operating context?

This is deliberately narrower than claiming a new trajectory-error law. The analysis does **not** claim that local/global drift, yaw accumulation, or condition dependence are newly discovered phenomena.

## Frozen evidence first

Before any new service analysis is accepted, the script must reproduce the known parking00/parking02 frozen signature within the stated numerical tolerance. A mismatch aborts the run.

Expected three-seed means include:

- parking00: ATE ≈ 2.106805 m; RPE10 ≈ 0.453027 m; Dp p95 ≈ 3.085755 m; Dtheta p95 ≈ 2.395108 deg.
- parking02: ATE ≈ 11.350361 m; RPE10 ≈ 0.097139 m; Dp p95 ≈ 22.344767 m; Dtheta p95 ≈ 30.415317 deg.

Therefore parking02 has better 10-s local translation fidelity but much poorer global physical–virtual synchronization. This inversion is motivation; it is not itself the novelty claim.

## Services

Two service families are separated rather than collapsed:

1. **Local relative-motion service** at 1, 5, and 10 s: the twin must reproduce physical relative translation and heading change within stated tolerances.
2. **Global synchronization service**: the twin's current global position and heading must remain within stated tolerances.

Representative tolerances are illustrative only. The primary analysis sweeps a broad tolerance grid so conclusions cannot depend on one favorable cutoff.

## Ground-truth boundary

Ground truth is allowed for:

- generating the actual service-success/failure label;
- offline leave-one-sequence-out calibration of empirical error envelopes;
- final evaluation.

Ground truth is **not** an online input to the condition-aware support decision.

Online-observable context is restricted to proprioceptive/timing-like quantities derivable causally from the sensor-lightweight interface: odometry speed, IMU yaw rate, causal acceleration, wheel–IMU disagreement, curvature proxy, and elapsed time since twin/trajectory start.

## Statistical unit

The physical sequence is the independent experimental unit.

- timestamps are nested within a run;
- seeds are nested within a physical sequence;
- seed realizations are averaged at matched physical windows before cross-sequence aggregation;
- local windows are dependence-reduced by using non-overlapping horizon spacing.

## Comparators

1. **Retrospective aggregate scalar** — a deliberately strong hindsight comparator based on whole-sequence aggregate error; never called deployable.
2. **Unconditional LOSO envelope** — calibrated from the other nine physical sequences and independent of current condition.
3. **Condition-aware LOSO envelope** — same LOSO calibration, but q95 bounds are conditioned on predeclared online-observable feature bins. The held-out sequence does not set the bin cutoffs or error envelopes.

## Primary failure modes

- **False-safe:** the evaluator supports the service but held-out ground truth violates the service requirement.
- **False-reject:** the evaluator rejects the service but the requirement is actually met.
- **Support rate:** fraction of operation for which the evaluator permits the service.

The main tradeoff is false-safe reduction **without** achieving it merely by rejecting nearly all operation.

## Falsification rule

If the condition-aware approach does not robustly reduce false-safe support across the predeclared tolerance sweep, or only does so through severe support loss, the operational-monitoring claim is not considered established by i2Nav. No threshold hunting or model retraining is allowed to rescue the claim.

## Prospective validation

The final UGV01 experiment should use the service definitions and support logic frozen before ground-truth inspection. That physical run is the cleaner prospective test of the operational-validity claim.
