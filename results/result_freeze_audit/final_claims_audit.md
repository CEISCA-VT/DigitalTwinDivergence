# Final Claims Audit

| claim | evidence strength | allowed wording | prohibited overclaim |
|---|---|---|---|
| V2 improves trajectory performance | moderate/strong | V2 improves official translation APE relative to the available Fixed Physics baseline and improves internal LOSO metrics relative to V1. | SOTA or universal odometry superiority. |
| local-vs-global fidelity distinction | strong | Short-horizon relative fidelity and long-horizon synchronization are distinct. | RPE alone proves twin fidelity. |
| parking01/parking02 failure mode | strong | parking02 is an extreme point in a broader hard-sequence pattern. | parking02 is solved or unique anecdote. |
| persistent yaw mismatch pathway | moderate | Persistent yaw mismatch is a measurable failure pathway. | Universal monotonic causal law. |
| condition-dependent fidelity | strong descriptive | Fidelity depends on operating condition. | One scalar condition explains everything. |
| benign fidelity characterization | moderate descriptive | Benign divergence can be characterized empirically. | Exceeding p95 means attack/failure. |
| LOSO benign-envelope behavior | strong descriptive | Envelope is partially stable; rate-domain components generalize better. | Envelope is universal stable guarantee. |
| UGV01 asset instantiation | moderate | The framework can be instantiated on UGV01 under tested conditions. | Universal UGV01 performance. |
| official i2Nav benchmark | strong | Official benchmark layer validates trajectory performance under standardized alignment. | Official APE replaces fidelity profile. |
| Fixed Physics comparison | moderate | V2 improves translation APE relative to available fixed baseline. | Huge rotation/RPE gap is entirely model superiority. |
| official alignment vs operational synchronization | strong | Official alignment changes apparent long-horizon error magnitude. | Official benchmark is wrong. |
