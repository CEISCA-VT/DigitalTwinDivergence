# Corrected Trajectory Evidence Pipeline

This workflow separates expensive model fitting from local evidence analysis.
Kaggle produces audited trajectory shards. The repository merges only complete,
hash-matched shards and then recomputes service qualification, selective risk,
and delivery-remediability evidence. No manuscript file is modified by these
commands.

## 1. Import Kaggle shard archives

Place downloaded ZIP files in directories of your choice. Do not unpack them.
The importer validates ZIP integrity, repository commits, shard ledgers, V1
checkpoint and results hashes, V2 manifests, split provenance, causal preflight
where applicable, and trajectory hashes.

```powershell
python -m DigitalTwin.analysis.kaggle_trajectory_bank --study doubly_nested --input "C:\path\to\nested_shards"
python -m DigitalTwin.analysis.kaggle_trajectory_bank --study causal_derivative --input "C:\path\to\causal_shards"
```

The resulting manifests are:

- `results/doubly_nested_loso/merged_manifest.json`
- `results/causal_derivative_loso/merged_manifest.json`

Proceed only when `ready_for_evidence_analysis` is `true`. A status of
`incomplete` is expected before all shards arrive. `contaminated` means at
least one provenance, hash, commit, archive, or duplicate-shard check failed.

## 2. Recompute corrected service evidence

```powershell
python -m DigitalTwin.analysis.service_evidence_from_bank --bank-manifest results\doubly_nested_loso\merged_manifest.json --output results\doubly_nested_loso\service_evidence
python -m DigitalTwin.analysis.service_evidence_from_bank --bank-manifest results\causal_derivative_loso\merged_manifest.json --output results\causal_derivative_loso\service_evidence
```

The analysis produces response surfaces, five held-out comparators, pooled and
within-service matched-acceptance risk curves, sequence-bootstrap uncertainty,
and a complete sequence-service remediability ledger. It refuses incomplete or
contaminated banks.

## 3. Existing-bank sensitivity analyses

These run on the current frozen centered-gradient V2 trajectories and require
no training or GPU:

```powershell
python -m DigitalTwin.analysis.lookahead_delay_sensitivity
python -m DigitalTwin.analysis.deterministic_delivery_replay
python -m DigitalTwin.analysis.service_threshold_curve_ledger
```

Interpretation boundaries:

- Lookahead-as-delay is a latency sensitivity check, not a replacement for the
  causal-retraining experiment.
- Deterministic delivery replay is emulation over precomputed states, not live
  network measurement or evidence of communication savings.
- Threshold curves vary one threshold family at a time; they show robustness
  and dependence of the ledger, not application-independent tolerances.

For OS-level Linux delivery emulation over the same precomputed states, use the
isolated `tc netem` harness documented in
[`linux_netem_delivery_study.md`](linux_netem_delivery_study.md). This records
actual UDP arrival timestamps and does not modify the host's physical network
interface.

## 4. Evidence freeze

Do not freeze headline counts until the nested manifest is complete and the
bank-driven service analysis finishes. The older `5/235 versus 75/200`,
`30/6/4`, and `27/30` values remain historical until replaced by the corrected
nested-bank outputs.
