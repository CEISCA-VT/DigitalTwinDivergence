# IoTJ Manuscript Package

Main source:

- `main.tex`

The main paper is centered on a causal, held-out qualification-and-remediability
protocol. The `figures/` directory contains vector figures for the cross-sequence
qualification surface, held-out baselines, paired discrepancy, and frozen-twin
configuration robustness. The standalone supplement is `supplementary/supplement.tex`.
The pre-rewrite source is preserved in `main.pre_final_timing_rewrite.tex` and
`supplementary/supplement.pre_final_timing_rewrite.tex`.

Regenerate the primary timing results and manuscript figures from the
repository root with:

```powershell
python -m DigitalTwin.analysis.service_timing_budget_study
python -m DigitalTwin.analysis.timing_reviewer_checks
python -m DigitalTwin.analysis.service_timing_robustness
python -m DigitalTwin.analysis.final_timing_paper_figures
python -m pytest tests/test_service_timing_robustness.py -q -p no:cacheprovider
python scripts/audit_paper_package.py
python scripts/audit_manuscript_syntax.py
```

When a LaTeX engine is available, build the paper with:

```powershell
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Build the supplement with:

```powershell
cd manuscript\supplementary
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

The supporting live experiment has two evidence layers:

1. measured hardware traces from 20 UGV01 runs;
2. trace-driven transport replay and a labeled capacity-sensitivity study.

The main narrative uses one physical-validation summary and one paired
edge-policy figure. Detailed trajectory, parameter-sensitivity, and
position-only plots remain in `results/` as auditable supporting artifacts.

The manuscript does not report the replay as measured communication savings
or as evidence of fresh 5/10-Hz sensor fidelity.

The 20 trial videos are stored in Git LFS. After cloning, run `git lfs pull`
to materialize the MP4 files. The final paper dataset is under
`public_datasets/ugv01_live_contract/trials/`; the source development logs
are not part of the submission dataset.
