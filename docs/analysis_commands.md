# Twin V1 Freeze + Replay Commands

These commands assume you are in the cloned repository root.

## 0. Put the scripts in the repository

Copy:

- `freeze_i2nav_v1.py`
- `test_i2nav_v1_replay.py`

into:

```text
DigitalTwin/analysis/
```

## 1. Make sure the original 30-run result package is available

On Kaggle, your package previously appeared at:

```text
/kaggle/input/datasets/shreyasudaya/i2nav-dual-10fold-3seed-results-zip
```

The script expects exactly **30 `gru_dual.pt` files** under that directory.

Quick check:

```bash
find /kaggle/input/datasets/shreyasudaya/i2nav-dual-10fold-3seed-results-zip \
  -name "gru_dual.pt" | wc -l
```

Expected output:

```text
30
```

## 2. Freeze V1 and create canonical predictions/metrics

```bash
python -u -m DigitalTwin.analysis.freeze_i2nav_v1 \
  --root public_datasets/im2nav \
  --source-dir /kaggle/input/datasets/shreyasudaya/i2nav-dual-10fold-3seed-results-zip \
  --frozen-dir results/i2nav_v1_frozen \
  --device cuda
```

This performs **no training**.

It will create approximately:

```text
results/i2nav_v1_frozen/
├── FROZEN_MANIFEST.json
├── SOURCE_HASHES.json
├── EVIDENCE_HASHES.json
├── source_snapshot/
├── evidence/
├── checkpoints/
│   ├── replicate_00.../
│   ├── replicate_01.../
│   └── replicate_02_base1042/
├── canonical_predictions/
├── canonical_metrics_per_run.csv
├── canonical_metrics_per_fold.csv
└── canonical_metrics_summary.json
```

## 3. Inspect the canonical headline

```bash
cat results/i2nav_v1_frozen/canonical_metrics_summary.json
```

The macro numbers should be consistent with the accepted 30-run V1 study. If they are not, **do not continue to residual analysis yet**.

## 4. Run the hard frozen replay test

```bash
python -u -m DigitalTwin.analysis.test_i2nav_v1_replay \
  --root public_datasets/im2nav \
  --frozen-dir results/i2nav_v1_frozen \
  --device cuda
```

Expected final line:

```text
V1 STATUS: VERIFIED / UNCHANGED
```

The command returns non-zero if any source snapshot, checkpoint, trajectory, or metric fails replay.

## 5. Optional CPU replay

```bash
python -u -m DigitalTwin.analysis.test_i2nav_v1_replay \
  --root public_datasets/im2nav \
  --frozen-dir results/i2nav_v1_frozen \
  --device cpu
```

GPU replay is preferable for the first acceptance test because the canonical freeze was likely created on GPU.

## 6. After it passes

Treat this directory as immutable:

```text
results/i2nav_v1_frozen/
```

Do not use it as an output directory for Twin V2 or residual-analysis scripts.

The next work directory should be separate, e.g.:

```text
results/i2nav_physics_residual_diagnostics/
```
