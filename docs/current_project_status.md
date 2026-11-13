# Current Paper Status

Date: 2026-11-12
Scope: `manuscript/main.tex`, *Digital Twin Fidelity for Networked IoT Services*.

## Manuscript and evidence

The manuscript combines receiver-side service qualification, paired delivery
remediability, held-out screening, measured network replay, and reference-based
offline delivery planning. UGV01 and MAGNET are supporting applications of
selected parts of the evaluation framework.

The saved centered and causal doubly nested banks each contain 300 verified
trajectories across ten outer sequences and three seeds. Their evidence
manifests record complete service-evidence analysis and reproduce 2,400
run/service/delivery measurements per bank. The centered sampled-delivery
ledger is 30 remediable, six persistent under the tested ideal delivery, and
four already qualified. The causal bank ledger is 29/7/4. The buffered-centered
study uses the same 300 centered trajectories with 100-ms added availability
delay and records successful reproduction of the unbuffered control.

The delivery-planning evidence is saved under `results/delivery_feasibility/`.
It evaluates buffered-centered and past-only timing configurations, compares
optimized rate-block schedules with qualifying fixed-rate schedules, and
reports both solver outcomes and broader optimistic bounds. See
`docs/delivery_feasibility_protocol.md` for assumptions and reproduction.

## Reproducibility checks

From the repository root, run:

```powershell
python scripts/audit_paper_package.py
python scripts/audit_manuscript_syntax.py
```

The package audit checks the paper source, required evidence inputs, and
figures. The syntax audit checks manuscript references. Full trajectory-bank
training instructions are in `docs/slurm_doubly_nested_loso.md` and
`docs/slurm_doubly_nested_causal_loso.md`; saved-bank analysis does not require
retraining.

## Scope of repository contents

The repository also retains exploratory studies, historical project work,
and presentation material. Their presence does not make them part of the
manuscript. `docs/paper_submission.md` is the canonical map from the current
paper analyses to their code and evidence.
