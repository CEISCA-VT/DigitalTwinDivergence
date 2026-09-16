# IoTJ Manuscript Package

Main source:

- `main.tex`

The `figures/` directory contains only the fourteen figures referenced by the
manuscript. The `supplementary/live_contract/` directory contains the
machine-readable UGV01 dataset audit and trace-driven communication replay
used for the live experiment section.

Regenerate the live results from the repository root with:

```powershell
python -m DigitalTwin.analysis.audit_ugv01_live_contract_dataset
python -m DigitalTwin.analysis.live_contract_trace_replay
python scripts/audit_paper_package.py
```

The live experiment has two evidence layers:

1. measured hardware traces from 20 UGV01 runs;
2. trace-driven transport replay and a labeled capacity-sensitivity study.

The manuscript does not report the replay as measured communication savings
or as evidence of fresh 5/10-Hz sensor fidelity.

The 20 trial videos are stored in Git LFS. After cloning, run `git lfs pull`
to materialize the MP4 files. The final paper dataset is under
`public_datasets/ugv01_live_contract/trials/`; the source development logs
are not part of the submission dataset.
