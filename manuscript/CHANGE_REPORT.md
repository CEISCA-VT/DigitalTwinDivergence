# Final timing-qualification manuscript rewrite

## Central claim

The paper now presents a causal, held-out qualification-and-remediability protocol for frozen IoT digital twins. Its **cross-sequence qualification surface** estimates transfer for each service and sampled delivery cell; paired replay distinguishes held-state discrepancy from discrepancy persistent under ideal available delivery. It is not a scheduler, safety certificate, or measured resource-saving method.

## Verified headline values

- At the nominal 30% pooled acceptance target, the surface accepted 235/800 cells and falsely qualified 5/235 (2.1%); service identity alone accepted 200/800 and falsely qualified 75/200 (37.5%).
- The paired physical-sequence bootstrap false-qualification difference (surface minus identity) is -35.4 percentage points, 95% interval [-44.4, -25.1]. Achieved acceptance differs by 4.375 points. This is **approximately matched and pooled**, not a per-service superiority result.
- Secondary references: AoI-only 60/250 (24.0%); adapted kinematic staleness 97/242 (40.1%); motion logistic 56/235 (23.8%). Per-service accepted counts differ substantially. Only two surface-versus-logistic per-service comparisons are near matched, and both paired intervals cross zero.
- Degraded/ideal case accounting is 30 delivery-remediable, six persistent under ideal available delivery, four already qualified under degraded delivery. The practical 5-Hz/50-ms sampled configuration restores 27/30. These are 40 service cases on **ten physical sequences**. All remediable degraded failures are physical-discrepancy-only under the evaluated freshness requirement.
- Fixed physics has 23 remediable, 16 persistent, and one already-qualified case. Both configurations show the delivery-induced/persistent distinction; all ten fixed-physics global cases are persistent versus six for V2.
- At phase zero, 5-Hz/0-ms and 5-Hz/50-ms represented histories differ in all 30 V2 runs. The 25- and 50-ms nominal delays alias on the saved 10-Hz clock, so the paper does not claim a separately resolved 50-ms delay effect.

## Source and figure changes

- Rewrote [main.tex](main.tex) around the held-out qualification-and-remediability thesis, with explicit service identity, AoI, adapted kinematic, and logistic comparators.
- Moved phase, receiver, requirement sensitivity, per-service unmatched comparisons, geofence null, and other evidence boundaries to [supplement.tex](supplementary/supplement.tex).
- Added [final_timing_paper_figures.py](../DigitalTwin/analysis/final_timing_paper_figures.py), which reads completed result CSVs and generates the four main vector PDFs plus supplementary robustness figures in `figures/`.
- Preserved pre-rewrite source as `main.pre_final_timing_rewrite.tex` and `supplementary/supplement.pre_final_timing_rewrite.tex`. Existing bibliography entries remain in the main source.

## Claim boundaries

No per-service matched win over identity, superiority over Li et al., optimal synchronization policy, model repair, safety guarantee, MAGNET timing claim, or measured communication/energy saving is asserted. MAGNET is limited to target-aligned thermal forecast-fidelity evidence because issuance/availability times are unverified. Numerical qualifications depend on the saved 10-Hz clock and zero-order-hold receiver; alternate phase and causal extrapolation sensitivities are supplementary.

## Verification and build

Run from repository root:

```powershell
C:\Users\shrey\miniconda3\python.exe -m DigitalTwin.analysis.final_timing_paper_figures
C:\Users\shrey\miniconda3\python.exe scripts\audit_manuscript_syntax.py
C:\Users\shrey\miniconda3\python.exe -m pytest tests\test_service_timing_robustness.py -q -p no:cacheprovider
```

When a LaTeX compiler is available, run from `manuscript/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Then run from `manuscript/supplementary/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

The current machine has no LaTeX engine. Static source, citation, reference, and figure-path checks are possible, but compilation, float placement, bibliography rendering, and page-by-page PDF inspection are **not verified**.
