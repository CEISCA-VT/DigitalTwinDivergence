# Delivery feasibility and minimum communication study

Run from the repository root:

```powershell
python -m unittest tests.test_delivery_feasibility
python -m DigitalTwin.analysis.delivery_feasibility
python -m scripts.refine_delivery_feasibility --solver-seconds 60
python -m scripts.compare_delivery_greedy
python -m scripts.plot_delivery_feasibility
python -m scripts.audit_delivery_feasibility
```

The output directory is `results/delivery_feasibility/`. The manuscript is not edited.
The refinement command spends additional solver time only on time-limited
solutions and writes a physical-recording summary with 5,000 bootstrap samples.
Use the default ten-second blocks, 1.5-second age cap, both banks, and the
0-ms/200-ms delays for this complete workflow. The greedy comparison, plotting,
audit, and report assume those settings; alternate command-line settings require
corresponding changes to those follow-up tools before their output is interpreted.

## Source evidence

Use the 30 outer-test trajectories in each of the complete centered and causal
doubly nested banks. Verify source hashes and identities, matching reference
measurements and timestamps across the three seeds, and the causal preflight.
Reproduce all 2,400 existing outer-test run/service/delivery cells per bank before
running the new experiment. Average satisfaction across seeds without averaging
their estimated trajectories. Each recording remains one physical observation.

## Schedule assumptions

The receiver holds the newest arrived timestamp without extrapolation. States
are generated on the existing 10-Hz clock. Centered features incur a 100-ms
availability lag; causal features incur zero modeled computation lag. Added
network delay is either 0 or 200 ms, with no added loss in this first study.
Packets arrive in timestamp order at a fixed latency. The packet index is shared
across seeds. Existing service thresholds, 30-s prefix, completed local horizons,
inclusive freshness boundaries and observability semantics are preserved.

All candidate schedules maintain held-state age at most 1.5 s at every
post-prefix sample, including the historical endpoint of a local interval.
The earlier endpoint is not required to meet the service's freshness limit:
the existing contract checks age only at the current evaluation instant.
All these full-reference schedules have observability one.

## Optimistic feasibility bound

At each eligible sample, consider every currently available source state within
the service age limit and the 1.5-s cap. For a local service, also consider every
earlier held state within the cap, enforcing timestamp order. Choose the pair
maximizing the number of seeds passing the physical check. Average those maxima
over samples. These choices need not form a consistent schedule, so the result is
an upper bound for every schedule in the declared bounded-age class. A value
below 0.80 certifies that no such schedule can qualify the service. A value at or
above 0.80 is only inconclusive until a consistent schedule is found.

## Exact optimization within a schedule family

Split the receiver window after the prefix into 10-s blocks. In each block,
choose 1, 2, 5 or 10 Hz, always delivering the first update at the block boundary.
Source timestamps are shifted back by the combined feature and network latency.
The decision is when to transmit a recorded state; no state is changed or
extrapolated. A 10-s maximum service horizon means a check depends on at most two
adjacent blocks' rate choices. Precompute per-seed check outcomes for each pair.

A mixed-integer flow program chooses a single path of block rates minimizing
transmissions while meeting each requested service's mean joint satisfaction
of 0.80. Solve separately for each of the four services, all three local services,
and all four services. Solver bounds, gaps and stopping reasons are retained.
An optimum applies to this family; a restricted-family infeasibility result is
not a general impossibility certificate. A time-limited incumbent is a feasible
upper bound on required transmissions, not a proved minimum.

Reconstruct every returned schedule and replay it independently. Its service
satisfaction must equal the factor-based calculation, meet each requested
contract, and lie below the optimistic feasibility bound. Compare against the
cheapest qualifying member of the same four fixed-rate configurations. Costs
count arrivals in the post-prefix receiver window; common warm-up traffic is
excluded. A state stream can be read once and shared by all requested services.

The closer baseline performs greedy rate reduction in the same block family.
It tries each qualifying fixed-rate initialization, then repeatedly lowers one
block's rate using the smallest normalized satisfaction loss per saved packet,
accepting only changes preserving all requested contracts. The best resulting
schedule is retained. Like the exact optimizer, this baseline sees reference
outcomes. Its schedules are also checked by direct receiver replay. This measures
the benefit of exact optimization beyond simply allowing time-varying rates.

## Interpretation

Optimization and feasibility bounds use recorded physical references and complete
trajectories. They characterize retrospective achievable performance and resource
requirements. They do not establish an online scheduling policy. The exact
schedule family, cap, timing assumptions and any solver limits accompany results.
No GPU work, GRU fitting, manuscript edits or deployment are part of this run.
