# Digital Twin Fidelity Research Draft

This package contains the research-paper source and the figures referenced by it.

## Main file

- `DigitalTwin_Fidelity_Research_Draft.tex`

The bibliography is embedded in the TeX file. The `figures/` directory is required because the manuscript uses relative figure paths.

## Evidence boundary

The manuscript uses the frozen Twin V2 outputs and separates three evaluation layers:

1. Internal physical--virtual fidelity: ATE, heading, RPE, divergence, mechanism, condition, and benign-envelope analyses.
2. UGV01 asset-specific instantiation: AprilTag-referenced low-speed indoor carpet evidence.
3. Official i2Nav benchmark positioning: standardized aligned benchmark metrics, reported separately from operational twin fidelity.

Claims are scoped to the available evidence. In particular, the UGV01 result is not presented as an all-surface or all-speed claim, and the benign p95 envelope is descriptive rather than an alarm threshold.

## Figure provenance

Figures were copied from the repository's advisor-review, publication-ready, and official-benchmark result directories. The numerical claims are grounded in the corresponding CSV/JSON/Markdown artifacts under `results/`.

## Compilation

From a TeX installation, run twice:

```powershell
pdflatex -interaction=nonstopmode DigitalTwin_Fidelity_Research_Draft.tex
pdflatex -interaction=nonstopmode DigitalTwin_Fidelity_Research_Draft.tex
```

