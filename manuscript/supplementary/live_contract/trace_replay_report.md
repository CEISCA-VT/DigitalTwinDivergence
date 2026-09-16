# UGV01 Trace-Driven Communication Replay

Five measured contract-aware UGV01 traces were replayed under all four frozen policies. Every policy in a trial shares the same physical source trace. The simulation uses measured response times and payload sizes, a serial HTTP client, and the latest recorded sample available at request time. No new sensor samples are fabricated.

| Policy | Requests/s | Response bytes/s | Repeated source fraction | p95 delivery AoI (s) |
|---|---:|---:|---:|---:|
| static-low | 1.536 | 951.8 | 0.053 | 1.304 |
| static-high | 1.536 | 951.8 | 0.053 | 1.304 |
| aoi-only | 1.536 | 951.8 | 0.053 | 1.304 |
| contract-aware | 1.536 | 951.8 | 0.053 | 1.304 |

These are simulated request and response-payload costs under the recorded service time. Policy inputs use recorded contract states, so counterfactual contract satisfaction is not estimated. A request to a cached source may return a repeated sample. The paper must distinguish these replayed resource quantities from directly measured live throughput.

The low-speed GPS course limitation in the observed live experiment remains. No policy-level claim about preservation of 1/5/10-s service qualification follows from this replay.

## Capacity sensitivity

Using the measured mean response size (619.4 B), a separate fluid calculation caps each policy's observed offered demand at 1.5, 2, 5, and 10 updates/s. At the measured 1.5-update/s capacity every policy saturates and no resource separation is possible. At 10 updates/s, the nominal payload-demand reductions relative to static-high are 80.0% for static-low, 72.4% for AoI-only, and 13.9% for contract-aware. These are provisioned-capacity demand bounds, not measured savings or service-preservation results.
