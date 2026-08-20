# Paper-Scope Code Review

Review date: 2026-08-20

## Review boundary

The current manuscript is the sensor-lightweight digital-twin fidelity paper. Frozen V2 outputs were treated as read-only. Security/attack modules, rover-control tools, raw logs, checkpoints, and exploratory model studies were not used as evidence for the paper.

## Findings

| Severity | Finding | Disposition |
| --- | --- | --- |
| High | The repository mixed fidelity analysis, legacy security experiments, rover bench tools, and generated artifacts in one visible surface. | Added an explicit paper-facing boundary in `docs/paper_submission.md` and a read-only package audit. Historical code and data were preserved outside that boundary. |
| Medium | `DigitalTwin/analysis/DigitalTwin/analysis/` was an accidental duplicate namespace. | Moved `canonical_motion_features.py` and `i2nav_v2_physical_yaw_pilot.py` into `DigitalTwin/analysis/`, where their documented imports resolve. |
| Medium | The paper package had no machine-readable check that required result inputs and figures were present. | Added `scripts/audit_paper_package.py` and canonical paths in `DigitalTwin/paper/paths.py`. |
| Low | Direct execution of the audit script initially failed to resolve the repository package. | Fixed direct execution by adding the repository root to the script import path. |
| Informational | The full test suite could not be executed because `pytest` is not installed in this environment. | Python compilation and paper-facing imports were run successfully; install the project test dependencies before release. |

## Checks completed

- `python -m compileall -q DigitalTwin scripts`: passed.
- All paper-facing analysis modules imported successfully.
- `python scripts/audit_paper_package.py`: passed; no required result inputs or figures missing.
- Frozen result files were not regenerated or modified.

## Deliberately retained

The legacy security and hardware code was not deleted because it is part of the project's historical reproducibility and future scope. It is explicitly excluded from the current paper package. Raw logs and generated analysis outputs were also retained to avoid irreversible loss of evidence.

