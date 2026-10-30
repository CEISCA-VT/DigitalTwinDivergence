# Delivery feasibility and minimum communication

All source trajectories and references are frozen. One schedule is shared across three model seeds.
Centered states include 100 ms of feature availability delay; causal states use source-time availability.
The exact optimizer chooses one of four periodic rates per 10-s block. Its minima apply to this family.
The broader bound permits any consistent held-state schedule with fixed latency and age at most 1.5 s.
It relaxes consistency between samples; a bound below 0.80 certifies infeasibility within that class.
All optimization uses reference outcomes retrospectively. It is an achievable-limit benchmark, not an online scheduler.

## causal, 0 ms added network delay

Single-service upper bounds certify 5/40 cases infeasible.

| Required services | Proven optimum | Feasible at time limit | Bound proves impossible | Restricted family infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_1s | 9 | 1 | 0 | 0 | 0 |
| local_5s | 9 | 0 | 0 | 1 | 0 |
| local_10s | 9 | 0 | 0 | 1 | 0 |
| global | 5 | 0 | 5 | 0 | 0 |
| local_1s+local_5s+local_10s | 8 | 1 | 0 | 1 | 0 |
| local_1s+local_5s+local_10s+global | 5 | 0 | 5 | 0 | 0 |

Across 47 groups with both an optimized witness and a qualifying periodic baseline, median transmission reduction is 15.6% (range 0.0--59.5%).

## causal, 200 ms added network delay

Single-service upper bounds certify 11/40 cases infeasible.

| Required services | Proven optimum | Feasible at time limit | Bound proves impossible | Restricted family infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_1s | 8 | 0 | 0 | 2 | 0 |
| local_5s | 5 | 0 | 3 | 2 | 0 |
| local_10s | 8 | 0 | 2 | 0 | 0 |
| global | 4 | 0 | 6 | 0 | 0 |
| local_1s+local_5s+local_10s | 5 | 0 | 3 | 2 | 0 |
| local_1s+local_5s+local_10s+global | 3 | 0 | 6 | 1 | 0 |

Across 33 groups with both an optimized witness and a qualifying periodic baseline, median transmission reduction is 33.4% (range 2.7--57.7%).

## centered, 0 ms added network delay

Single-service upper bounds certify 6/40 cases infeasible.

| Required services | Proven optimum | Feasible at time limit | Bound proves impossible | Restricted family infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_1s | 10 | 0 | 0 | 0 | 0 |
| local_5s | 10 | 0 | 0 | 0 | 0 |
| local_10s | 10 | 0 | 0 | 0 | 0 |
| global | 4 | 0 | 6 | 0 | 0 |
| local_1s+local_5s+local_10s | 10 | 0 | 0 | 0 | 0 |
| local_1s+local_5s+local_10s+global | 3 | 1 | 6 | 0 | 0 |

Across 48 groups with both an optimized witness and a qualifying periodic baseline, median transmission reduction is 35.0% (range 0.0--60.4%).

## centered, 200 ms added network delay

Single-service upper bounds certify 22/40 cases infeasible.

| Required services | Proven optimum | Feasible at time limit | Bound proves impossible | Restricted family infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_1s | 2 | 0 | 4 | 4 | 0 |
| local_5s | 2 | 0 | 8 | 0 | 0 |
| local_10s | 6 | 0 | 3 | 1 | 0 |
| global | 3 | 0 | 7 | 0 | 0 |
| local_1s+local_5s+local_10s | 1 | 0 | 8 | 1 | 0 |
| local_1s+local_5s+local_10s+global | 1 | 0 | 8 | 1 | 0 |

Across 15 groups with both an optimized witness and a qualifying periodic baseline, median transmission reduction is 16.3% (range 0.2--53.9%).
