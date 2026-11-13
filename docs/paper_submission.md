# Manuscript Package and Reproduction Guide

The current paper is **Digital Twin Fidelity for Networked IoT Services**.
`manuscript/main.tex` is the source of truth for the claims, terminology, and
figures. This guide maps each major result to its implementation and gives the
commands needed to check or reproduce it.

## Paper files

- `manuscript/main.tex`: manuscript source
- `manuscript/ref.bib`: bibliography
- `manuscript/figures/`: figure assets referenced by the manuscript
- `manuscript/reproducibility_notes.tex`: extended model, evaluation, and
  optimization details

## Evidence map

| Paper result | Main implementation | Evidence location |
| --- | --- | --- |
| Receiver-side contract evaluation and centered held-out qualification | `DigitalTwin/analysis/service_evidence_from_bank.py` | `results/doubly_nested_loso_evidence/` |
| Past-only feature construction and held-out qualification | Same evaluator with `--expected-feature-derivative-mode causal_backward` | `results/doubly_nested_causal_loso_slurm/service_evidence/` |
| 100-ms buffered-centered availability comparison | `scripts/run_buffered_centered_heldout.py` | `results/buffered_centered_heldout_100ms/` |
| Reference-based rate planning, bounds, and greedy comparison | `DigitalTwin/analysis/delivery_feasibility.py` and `scripts/` planning tools | `results/delivery_feasibility/` |
| Measured Linux `tc netem` replay | `DigitalTwin/analysis/netem_delivery_replay.py`; capture instructions in `docs/linux_netem_delivery_study.md` | `results/netem_delivery/` |
| UGV01 position-only ledger | `DigitalTwin/analysis/ugv01_position_remediability_ledger.py` | `results/ugv01_position_ledger/` |
| MAGNET thermal discrepancy analysis | `DigitalTwin/analysis/` cross-domain analysis modules | `results/cross_domain_contract_generalization/` |

The centered and causal banks each contain 300 verified trajectories, with
2,400 run/service/delivery measurements per bank. The buffered study reuses
the 300 centered trajectories and verifies that its zero-buffer control
reproduces the original measurements before adding the 100-ms availability
delay. UGV01 and MAGNET apply selected framework components, as described in
the manuscript. The result paths above are local output locations; generated
result tables and raw trajectory banks are not all distributed in the GitHub
repository. The source code, manuscript figures, and available provenance
artifacts are kept separate from exploratory analyses.

## Reproducing analyses

Run commands from the repository root in the project Python environment. The
buffered study and the two-bank analyses require the completed trajectory
banks and evidence files at the paths listed above. The main qualification
evidence can be rebuilt from an available completed bank with:

```powershell
python -m DigitalTwin.analysis.service_evidence_from_bank `
  --bank-manifest results\doubly_nested_loso_slurm\merged_manifest.json `
  --output results\doubly_nested_loso_evidence
```

For the causal bank, provide its manifest and add
`--expected-feature-derivative-mode causal_backward`. Reproduce the
100-ms buffered comparison with:

```powershell
python -m scripts.run_buffered_centered_heldout
```

The reference-mask demonstration and its plotted summaries are reproduced
with `python -m scripts.build_review_revision_20261109`. It requires the
centered bank, buffered-study operating-point tables, and past-only
operating-point tables. Its focused tests are
`tests/test_review_reference_mask.py` and
`tests/test_buffered_centered_heldout.py`.

Planning assumptions and validation steps are in
`docs/delivery_feasibility_protocol.md`. Measured-network instructions are in
`docs/linux_netem_delivery_study.md`. Full GPU trajectory-bank instructions
are in `docs/slurm_doubly_nested_loso.md` and
`docs/slurm_doubly_nested_causal_loso.md`.

## Package checks

```powershell
python -m pip install -r requirements.txt
python scripts/audit_paper_package.py
python scripts/audit_manuscript_syntax.py
```

The package audit checks the manuscript and all cited figure assets. Since
generated evidence is not bundled in every checkout, use
`python scripts/audit_paper_package.py --require-evidence` when auditing a
workspace that contains the full local result set. The syntax audit checks
citations, labels, and figure references. Compile `main.tex` from the
`manuscript/` directory to verify the PDF and float placement.
