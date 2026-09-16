# Reproducibility snapshot

- Git commit: `aee718cb54e49f92c6c4a905ccfe671e88ebf190`
- V1 CSV count: 30
- Frozen V2 evaluated trajectories: 30
- Primary statistical unit: physical sequence.
- Operational fidelity uses the common initialized physical--virtual frame without post-hoc trajectory alignment.
- Official benchmark alignment is reported separately.
- The V1/V2 protocol-equivalence validator should be retained with the release.

## Environment

- Python: `3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:49:16) [MSC v.1929 64 bit (AMD64)]`
- Platform: `Windows-11-10.0.26200-SP0`

## Release checklist

- Record interpolation and missing-data rules.
- Record quantile definition and bootstrap seed/count.
- Record Muñoz MAD/gap/LCAW settings.
- Record timing perturbation procedure.
- Release per-sequence and per-seed trajectories/results where permitted.
- Keep GPS/RTK evaluation-only if used as an independent reference.