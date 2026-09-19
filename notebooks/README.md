# Kaggle notebooks

## Two-stage provenance repair

`kaggle_doubly_nested_loso.ipynb` is the authoritative expensive-compute notebook for repairing the qualification pipeline. It has two modes:

- `subset5`: parking01, parking02, building00, playground00, and street00;
- `full10`: all ten i2Nav physical sequences.

The complete full design has 300 paired V1-to-V2 pipelines (600 model fits), including the outer-test trajectories. Its default 60 shards assign five pipelines (ten model fits) to each Kaggle session. Use a common `SHARD_COUNT` and run every `SHARD_INDEX` from zero through `SHARD_COUNT - 1`. Preserve every archive. No headline timing result may be computed from an incomplete set of shards.

For outer sequence `u`, each qualification-training target `v` uses V1 and V2 checkpoints whose training, normalization, validation, and checkpoint selection exclude both `u` and `v`. The outer-test V1 and V2 checkpoints exclude `u`. This distinction is recorded in the V1 split file and V2 run manifest.

Each shard audit now verifies the exact V1 checkpoint path, V1 results CSV path, normalization-fit sequences, checkpoint-selection sequences, checkpoint hashes, trajectory hashes, completion markers, and repository commit. Completed V2 tasks are skipped only when all required artifacts are present and provenance-matched. Archives contain only the current shard's task directories and are checked with `ZipFile.testzip()`.

## Backward-derivative sensitivity

`kaggle_causal_derivative_v2_loso.ipynb` runs 30 paired V1-to-V2 LOSO pipelines using backward-only acceleration and yaw-acceleration inputs in both learned stages. Its default six shards assign five pipelines (ten model fits) to each session. It is a new sensitivity pipeline, not the frozen V2. It does not prove full online causality because interpolation availability, computation, and transport latency are outside this derivative change.

The causal notebook includes a numerical no-lookahead preflight: it perturbs future samples, verifies earlier feature rows are unchanged, checks the backward-difference columns, and records the first-sample boundary rule in `causal_no_lookahead_preflight.json`.

## Before running

1. Commit and push the runner support in `DigitalTwin/analysis/i2nav_v2_full_loso.py`.
2. Set `EXPECTED_COMMIT` in each notebook to that pushed commit hash.
3. Enable a Kaggle GPU and Internet access.
4. Choose shard settings before the first run and do not change them between sessions.
5. Download each ZIP and verify its printed SHA-256 hash.

The default runner behavior remains the original centered-feature, one-sequence LOSO protocol. The new command-line switches are opt-in.
