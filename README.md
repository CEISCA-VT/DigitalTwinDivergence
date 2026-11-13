# Digital Twin Fidelity for Networked IoT Services

This repository contains the manuscript, analysis code, and saved evidence for
the study of service-relative digital-twin fidelity at a network receiver.
The paper evaluates whether receiver-held states satisfy physical-error,
freshness, and evidence requirements, whether delivery changes restore
qualification, and what transmission rates can meet those requirements in an
offline planning study.

## Paper package

- Manuscript source: `manuscript/main.tex`
- Bibliography: `manuscript/ref.bib`
- Figures used by the manuscript: `manuscript/figures/`
- Additional reproducibility details: `manuscript/reproducibility_notes.tex`
- Paper workflow and manuscript-to-code-to-evidence map: `docs/paper_submission.md`

## Main evidence and code

- Centered doubly nested LOSO bank and evidence: `results/doubly_nested_loso_slurm/`
- Causal backward-difference bank and evidence: `results/doubly_nested_causal_loso_slurm/`
- Buffered-centered screening study: `results/buffered_centered_heldout_100ms/`
- Delivery planning and solver audit: `results/delivery_feasibility/`
- Measured Linux network replay: `results/netem_delivery/`
- Main analysis modules: `DigitalTwin/analysis/`

The analysis scripts operate on saved trajectory banks, so reproducing the
paper analyses does not require retraining the twin. Generated result tables
and raw trajectory banks are not all included in this repository. Full
training reruns require prepared i2Nav data and the CUDA environment described
in `docs/slurm_doubly_nested_loso.md` and
`docs/slurm_doubly_nested_causal_loso.md`.

## Checks

From the repository root:

```powershell
python scripts/audit_paper_package.py
python scripts/audit_manuscript_syntax.py
```

Compile the manuscript from `manuscript/` with the local LaTeX toolchain.
The package audit checks the manuscript and its figure assets; add
`--require-evidence` when the local generated results are available and should
also be checked. The syntax audit checks labels, citations, and figure
references.

Security/attack experiments, exploratory analyses, and presentation materials
remain in the repository as separate project work. They are not part of the
claims in `manuscript/main.tex` unless explicitly cited there.
