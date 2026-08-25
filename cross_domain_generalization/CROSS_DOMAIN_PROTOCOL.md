# Frozen E3 cross-domain operational-fidelity contract protocol

## Scientific question

Does the same **quantity × horizon × tolerance** operational-fidelity contract remain computable and informative across independent non-robot digital-twin domains without choosing thresholds after seeing outcomes?

This experiment is a **structural/generalization audit**. It is not a contest among the underlying twin models.

## Frozen contract

For each physical quantity `q`, define a unit-free discrepancy

\[
e_q(t)=\frac{|x_T(t)-x_P(t)|}{S_q},
\qquad
S_q=P_{95}(x_P)-P_{05}(x_P),
\]

with only a small relative floor for near-constant physical channels.

For service horizon `h`, use non-overlapping service windows and compute

\[
D_{q,h}=P_{95}\{e_q(t):t\text{ lies in the service window}\}.
\]

The service contract at tolerance `tau` passes when

\[
D_{q,h}\le\tau.
\]

### Frozen horizons

- 60 s
- 300 s
- 600 s

### Frozen dimensionless tolerance grid

- 0.01
- 0.02
- 0.05
- 0.10
- 0.20
- 0.50 of the physical robust scale

The same horizon/tolerance grid is used for MAGNET, FreeTwinEV, and TU Wien SNG. No dataset gets a post-hoc threshold selected because it makes the results look favorable.

## Domain instantiations

### MAGNET heat-pipe DT

- Physical: 10 thermowells from the released experiment.
- Twin: released ML/digital-twin forecast streams.
- Dependence control: greedy non-overlapping 600 s forecast windows, matching the existing hardened MAGNET analysis.
- Service horizon means how long the forecast trajectory must remain within the normalized discrepancy contract.

### FreeTwinEV 1S4P battery module

- Physical: released ID22 battery-module experiment.
- Twin: released ID22 3D electro-thermal CFD validation simulations for cooldown and discharge.
- Quantities: thermal aggregate states that can be paired from the released files (cell/plate means, maxima, and spreads when semantically available; otherwise transparent thermal aggregates).
- The official package itself identifies the cooldown physical segment as 1305–3500 s and discharge segment as 4110–8105 s.
- No simulation recalibration is performed by this audit.

### TU Wien biomass-to-SNG DT

Two released DT subsystems are used:

1. DFB MPC/Kalman state estimates:
   - product-gas volume flow measurement ↔ estimate;
   - product-gas temperature measurement ↔ estimate.
2. Soft sensor:
   - measured product-gas composition ↔ soft-sensor prediction for H2, CO, CO2, CH4, and C2H4.

Only released physical/virtual pairs with explicit semantic correspondence are used. Operator setpoints are not treated as ground truth.

## Structural-transfer gate

A dataset passes the structural portability gate only if:

1. at least two physical/virtual quantities are paired;
2. all three frozen horizons have at least five contract units;
3. the frozen tolerance sweep produces a non-degenerate validity surface.

A PASS means **the contract abstraction transfers**. It does not mean the underlying model is high-fidelity, superior to baselines, or safe for any application.
