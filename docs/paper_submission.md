# Paper Submission Surface

The current paper is the sensor-lightweight digital-twin fidelity study. The submission surface is intentionally smaller than the complete engineering repository.

## Paper-facing analysis

The frozen evidence is consumed from `results/` by these analysis modules:

- `i2nav_v2_fidelity_evaluator.py`
- `i2nav_v2_all_sequence_mechanism.py`
- `i2nav_v2_condition_fidelity.py`
- `i2nav_v2_benign_fidelity_envelope.py`
- `i2nav_v2_loso_envelope_validation.py`
- `ugv01_asset_instantiation.py`
- `i2nav_official_benchmark.py`
- `i2nav_sensing_fidelity.py`
- `final_result_freeze_audit.py`

The manuscript source is `DigitalTwin_Fidelity_Research_Draft.tex`. Its figure dependencies are in `figures/`.

Run the read-only package audit with:

```powershell
python scripts/audit_paper_package.py
```

## Explicitly outside the current paper

The following remain in the engineering repository for historical reproducibility or future work, but are not part of the current fidelity-paper claim:

- GPS/security attack campaign modules and alarm policies;
- bench bring-up and motion-command scripts;
- exploratory model bake-offs and abandoned uncertainty pilots;
- raw rover logs, downloaded public datasets, checkpoints, and generated intermediate outputs.

They should not be copied into a paper-submission archive unless a reviewer specifically requests them.

## Evidence boundary

The current paper keeps internal physical--virtual fidelity separate from official i2Nav benchmark scoring. UGV01 claims are limited to the accepted AprilTag-referenced low-speed indoor carpet condition. The benign p95 envelope is descriptive and is not an alarm threshold.

