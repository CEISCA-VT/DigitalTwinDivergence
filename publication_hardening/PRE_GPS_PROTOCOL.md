# Pre-GPS protocol freeze

## Central claim
The Twin Fidelity Profile evaluates physical--virtual fidelity as a
multidimensional operational property, separating local motion agreement,
global synchronization, componentwise disagreement, persistent drift, tail
behavior, timing degradation, and operating-condition dependence.

## Terminology freeze
- **Physical state:** measured/reference physical trajectory.
- **Virtual state:** digital-twin trajectory.
- **Operational fidelity:** unaligned physical--virtual agreement in the common initialized frame.
- **Benchmark accuracy:** officially aligned localization performance.
- **External reference:** RTK/GNSS, AprilTag, or dataset reference.
- **Asset-specific instantiation:** platform-specific binding + empirical physical validation.
- **Framework portability:** the same evaluator applied to another platform.
- **Model portability:** the same learned maintenance model transferred to another platform.

## Claim boundaries
Do not claim:
- universal superiority over trace alignment;
- general empirical validation across all digital-twin domains;
- safety/security certification from a benign p95 envelope;
- that conditioned envelopes universally improve coverage;
- that reduced-input LWOI/YNet adaptations are exact published reproductions;
- that reconstructed Fixed Physics or the failed planar EKF equals an official baseline.

## Timing terminology
Use **controlled replay-based physical--virtual timing sensitivity**.
Distinguish timestamp offset, zero-mean jitter, sustained delay/state age,
packet loss, and actual network transport latency. The existing timing study
perturbs saved physical--virtual timing during replay; it is not a wireless
network experiment.

## GPS/RTK run freeze
Before collecting data, record:
- route and operating condition;
- number of repetitions;
- GNSS/RTK hardware and quality flags;
- reference update rate and coordinate frame;
- antenna/body-frame offset;
- time synchronization method;
- expected reference accuracy;
- whether GNSS/RTK is evaluation-only (recommended);
- predefined acceptance/rejection criteria.

Predefine comparisons:
1. twin vs independent physical reference;
2. raw wheel/IMU propagation vs independent physical reference;
3. twin vs raw wheel/IMU propagation.

Predefine metrics:
unaligned/common-initial-frame ATE, RPE1/5/10, heading error where supported,
p95/max divergence, and accumulated residuals. Keep official post-hoc aligned
benchmark scoring separate.
