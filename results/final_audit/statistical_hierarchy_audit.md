# Statistical Hierarchy Audit

| analysis | status | rationale |
|---|---|---|
| official benchmark | PASS_WITH_CAVEAT | Per-run values retained; sequence aggregation used. Fixed Physics is deterministic one-run-per-sequence; V2 has three seed replicates per sequence. Caveat: Fixed orientation convention mismatch limits rotation/RPE interpretation. |
| full LOSO V1->V2 statistics | PASS | Context records sequence-aggregated macro means, sequence wins, bootstrap over physical sequences, and sign-flip tests. |
| all-sequence mechanism analysis | PASS_WITH_CAVEAT | Summary explicitly says timestamp correlations are descriptive only; sequence-level associations are primary. |
| condition-dependent fidelity | PASS | Condition summaries are within run, then seed-aggregated within physical sequence, then sequence-level. |
| benign fidelity characterization | PASS_WITH_CAVEAT | Correctly labels p95 envelopes as descriptive and sequence-sensitive, not thresholds. |
| LOSO benign envelope validation | PASS | Holds out physical sequences and reports sequence-level coverage/sensitivity. |
| UGV01 asset-specific instantiation | PASS_WITH_CAVEAT | Asset-specific evidence exists, but claims should remain condition-limited to available AprilTag/telemetry runs. |
| Fixed Physics comparison | PASS_WITH_CAVEAT | Translation comparison can be reported carefully; orientation/RPE gap likely affected by body-frame convention mismatch. |
| V1 official comparison | PASS_WITH_CAVEAT | Not available because exact frozen V1 trajectories are absent; no questionable reconstruction performed. |
