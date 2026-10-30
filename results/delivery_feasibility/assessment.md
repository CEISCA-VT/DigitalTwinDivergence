# Findings and manuscript implications

## What was implemented

1. An optimistic upper bound on service satisfaction for any timestamp-ordered
   packet schedule with the stated fixed latency and a maximum held-state age of
   1.5 s. Local services include both receiver-history endpoints. A bound below
   0.80 certifies that no schedule in this class can qualify the service.
2. A mixed-integer path formulation minimizing transmissions while qualifying
   individual services or groups of services. It chooses 1/2/5/10-Hz rates in
   10-s blocks. Its exact minima apply to that schedule family.
3. A closer reference-informed greedy comparator in the same family, alongside
   the best qualifying fixed rate. Every returned schedule is directly replayed.

Both complete doubly nested banks were used, selecting only their 30 outer-test
trajectories apiece. Each recording has one shared schedule across its three
seeds. Centered features incur 100 ms of availability delay; causal features use
source-time availability. Added network delays are 0 and 200 ms. Source models,
state estimates, service limits, reference data and manuscript are unchanged.

## Main numerical findings

For supporting all three local services simultaneously at zero added network
delay:

| Bank | Recordings with feasible schedules | Median fewer transmissions than best qualifying fixed rate | Median fewer transmissions than greedy |
| --- | ---: | ---: | ---: |
| Centered, including 100-ms feature availability | 10/10 | 31.4% | 4.5% |
| Causal | 9/10 | 12.2% | 1.7% |

The corresponding 95% physical-recording bootstrap intervals for the median
fixed-rate reductions are 16.4--47.1% and 6.1--46.7%. These are conditional on
recordings with feasible schedules and a qualifying fixed-rate comparison.

For centered building00, the all-local schedule uses 8,761 updates instead of
11,709 for the best qualifying fixed rate: a 25.2% reduction. All three local
contracts still qualify on direct replay. The global-pose upper bound is 0.234,
below 0.80 even after allowing optimistic state choices. This distinguishes
supportable services from a service that delivery cannot restore in the declared
class, on the same saved recording.

At zero added delay, the upper bounds certify all six centered-bank global-pose
failures and five causal-bank global-pose cases infeasible. In the causal bank,
two local cases fail the restricted block family while their upper bounds remain
above 0.80; their broader feasibility is unresolved. At 200 ms added network
delay, the bounds certify 22/40 centered and 11/40 causal sequence-service cases
infeasible. These counts refer to the newly specified timing conditions.

Across all conditions and service groups, 143 feasible schedules were found.
The solver proves 140 minima in the block family. Three remaining incumbents
have objective gaps below 0.3%. All optimized and greedy schedules are replayed
and checked against their required contracts. No optimized group in this run
became feasible when every fixed-rate baseline failed; the observed optimization
benefit is reduced communication cost.

## What this adds scientifically

The existing ledger reports outcomes under selected delivery conditions. The
new bound can rule out recovery throughout a declared class of schedules. That
is a stronger claim than failure at the tested highest-rate condition.

The optimization extends qualification into a resource-planning problem with
explicit decisions, service constraints, feasible schedules, costs and optimality
bounds. Its local-service factors preserve the coupling between both endpoints
of each motion interval. The same transmitted stream can support multiple
services, allowing their joint communication requirement to be measured.

The greedy comparison shows that much of the benefit comes from allowing
delivery to adapt while enforcing the contracts. Exact optimization provides
a benchmark and sometimes additional savings. The evidence supports feasibility
certificates and resource requirements as the main new contribution, rather than
a claim of a large universal advantage for the optimizer.

These are offline planning experiments using recorded references and complete
trajectories. They establish achievable limits and costs for those recordings.
An operational scheduler would need a rule using information available at its
decision time. Scheduling and optimization are established research areas, so
the novelty argument should focus on the specific service-contract problem,
imperfect source estimates, coupled local histories and verified feasibility
results. The use of a generic mixed-integer solver alone is not new.

## Recommended use

This is a substantive extension worth considering. Develop feasibility bounds
and minimum communication as one coherent method, with the current ledger as the
motivation and empirical reference. Use the closer greedy comparison, report
the precise schedule class, and keep unsupported cases distinct from certified
impossibility. The results do not by themselves establish a deployment policy
or a guaranteed increase in publication rating.

Files: `report.md`, `recording_summary.csv`, `greedy_summary.csv`,
`feasibility_and_cost.png`, `feasibility_and_cost_plotted.csv`, `manifest.json`,
`verification.json`, and the saved optimized/greedy schedules.
