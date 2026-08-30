# Digital Twin Divergence

This repository supports the current sensor-lightweight mobile-robot digital
twin fidelity paper. The project studies how a computational twin follows a
physical robot over time, how local and global physical-virtual divergence
behave under changing conditions, and how a generic model becomes an
asset-specific UGV01 twin through calibration and independent reference data.

Security and GPS-attack experiments from earlier project phases are not the
active paper scope. Current claims should be made from the frozen i2Nav V2
LOSO results, post-LOSO fidelity analyses, UGV01 AprilTag validation, official
i2Nav benchmark outputs, and TerraSentia/AIFARMS portability analysis.

## Repository Map

- `DigitalTwin/`: core models, adapters, evaluators, analysis scripts, and the
  live dashboard prototype.
- `public_datasets/`: small prepared public/physical datasets used by the
  paper workflows, including UGV01 physical validation exports.
- `results/`: retained paper-relevant result artifacts and figures.
- `docs/`: concise experiment and operator documentation.
- `figures/`: paper and presentation-ready figure assets.
- `tests/`: focused regression tests.

## Install

```powershell
python -m pip install -r requirements.txt
```

Some workflows also require optional computer-vision packages such as OpenCV
and AprilTag/ArUco support, depending on the local Python build.

## Main Workflows

Run the live dashboard prototype:

```powershell
python -m DigitalTwin.dashboard.server --open
```

Run the dashboard against a recorded CSV:

```powershell
python -m DigitalTwin.dashboard.server --mode replay --csv path\to\run.csv --open
```

Run tests:

```powershell
python -m pytest -q
```

The frozen V2 study should not be retrained or retuned when preparing paper
results. Use the saved result artifacts and documented post-LOSO analyses.
