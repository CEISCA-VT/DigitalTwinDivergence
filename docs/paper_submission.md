# Manuscript Package and Reproduction Guide

The current submission is **Digital Twin Fidelity for Networked IoT
Services**, with source at `manuscript/main.tex`. The manuscript is the
authority for claims and presentation. This guide maps its analyses to the
saved evidence and reproduction commands.

## Paper files

- `manuscript/main.tex`: manuscript source
- `manuscript/ref.bib`: bibliography
- `manuscript/figures/`: figure assets referenced by the manuscript
- `manuscript/reproducibility_notes.tex`: extended model, evaluation, and
  optimization details

## Evidence map

| Manuscript analysis | Saved evidence |
| --- | --- |
| Centered doubly nested qualification and delivery ledger | `results/doubly_nested_loso_slurm/` and `results/doubly_nested_loso_evidence/` |
| Causal backward-feature doubly nested analysis | `results/doubly_nested_causal_loso_slurm/` |
| 100-ms buffered-centered availability comparison | `results/buffered_centered_heldout_100ms/` |
| Reference-based transmission planning and feasibility bounds | `results/delivery_feasibility/` |
| Measured Linux `tc netem` delivery replay | `results/netem_delivery/` |
| UGV01 and MAGNET supporting studies | `results/ugv01_position_ledger/` and `results/cross_domain_contract_generalization/` |

The centered and causal trajectory-bank manifests report 300 verified
trajectories each. Their service-evidence manifests report 2,400 evaluated
run/service/delivery rows per bank. The buffered-centered manifest records
300 source trajectories and successful reproduction of the unbuffered
control. These checks support the corresponding manuscript analyses; they
do not imply that every exploratory artifact in `results/` is part of the
paper.

## Reproducing analyses

Run commands from the repository root. The main qualification evidence can
be rebuilt from a completed trajectory bank with:

```powershell
python -m DigitalTwin.analysis.service_evidence_from_bank `
  --bank-manifest results\doubly_nested_loso_slurm\merged_manifest.json `
  --output results\doubly_nested_loso_evidence
```

The causal bank uses the same command with the causal manifest and
`--expected-feature-derivative-mode causal_backward`. The buffered study is
reproduced with:

```powershell
python -m scripts.run_buffered_centered_heldout
```

Delivery-planning instructions, assumptions, and validation steps are in
`docs/delivery_feasibility_protocol.md`. Full GPU training-run instructions
are in `docs/slurm_doubly_nested_loso.md` and
`docs/slurm_doubly_nested_causal_loso.md`.

## Package checks

```powershell
python scripts/audit_paper_package.py
python scripts/audit_manuscript_syntax.py
```

The first command checks required paper inputs and figures. The second checks
manuscript citations, labels, and figure references. Compile `main.tex` from
the `manuscript/` directory to verify the final PDF and float placement.
