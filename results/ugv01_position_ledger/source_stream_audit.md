# UGV01 Position-Ledger Source Audit

## Decision

All three August 29 runs can support a common position-only delivery replay at
2 Hz and 1 Hz. Only the calibration/reference run supports 10 Hz and 5 Hz.
Consequently, 10/5 Hz results may be reported only as a run-1 diagnostic and
must not be pooled with the two holdouts.

| Run | Rows | Duration | Median rate | p95 / max gap | Supported replay rates |
|---|---:|---:|---:|---:|---|
| Run 1, 15:35:55 | 1664 | 164.64 s | 10.10 Hz | 0.099 / 0.099 s | 10, 5, 2, 1 Hz |
| Run 2, 16:01:14 | 624 | 308.39 s | 2.02 Hz | 0.495 / 0.495 s | 2, 1 Hz |
| Run 3, 16:15:16 | 635 | 315.31 s | 2.02 Hz | 0.495 / 0.990 s | 2, 1 Hz |

The run-3 reference contains one interval exceeding the frozen 0.75 s maximum
gap. Samples and relative-motion pairs crossing that interval must be excluded
from the observable denominator. The aligned exports do not expose a reliable
direct-detection-versus-reconstruction flag, so no stronger direct-decode claim
is made from `tracking_status_code`.

## Frozen Analysis Boundary

The protocol is recorded in
`docs/protocols/ugv01_position_ledger_protocol.json`. It uses position only,
because AprilTag position is the independent reference while the existing
heading and synchronization limitations remain material. The common primary
comparison is 1 Hz/200 ms degraded delivery versus 2 Hz/0 ms ideal delivery.

This is a retrospective protocol frozen after the earlier physical-fidelity
characterization but before ledger computation. It is not described as a
prospectively blinded experiment.
