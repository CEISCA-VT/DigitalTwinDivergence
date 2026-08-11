"""Fine-tune IBM PatchTSMixer for GPS-independent EKF covariance proxies.

This script loads ``ibm/patchtsmixer-etth1-pretrain`` directly through
Transformers and replaces/adapts the regression head for rover covariance
targets. It remains a proxy experiment: the inputs exclude GPS coordinates and
GPS residuals, while the labels are benign covariance surrogates rather than
AprilTag physical ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import pickle

import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSMixerConfig, PatchTSMixerForRegression

from .common import parse_run_name
from .train_patchtsmixer_uncertainty import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TARGET_FLOOR,
    _bounded_exp_target,
    _expand_inputs,
    _log_target,
    _paths_from_manifest,
    _extract_run_series,
)


DEFAULT_MODEL_ID = "ibm/patchtsmixer-etth1-pretrain"
WINDOW_UPDATES = 16
PATCH_LENGTH = 4
TARGET_HORIZON_UPDATES = 5


def build_sequence_examples(paths: list[Path], window_updates: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    groups: list[str] = []
    sources: list[str] = []

    for path in paths:
        meta = parse_run_name(path)
        if meta.get("attack") not in {"", "none"}:
            continue
        try:
            features, targets = _extract_run_series(path)
        except RuntimeError:
            continue
        run_id = f"{meta.get('speed', '')}_{meta.get('surface', '')}_trial-{meta.get('trial', '')}_{path.stem[-15:]}"
        for index in range(window_updates - 1, len(features) - TARGET_HORIZON_UPDATES):
            X.append(features[index + 1 - window_updates : index + 1])
            y.append(np.median(targets[index + 1 : index + 1 + TARGET_HORIZON_UPDATES], axis=0))
            groups.append(run_id)
        sources.append(str(path))

    if not X:
        raise RuntimeError("no sequence windows were extracted")

    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    low = np.quantile(y_array, 0.01, axis=0)
    high = np.quantile(y_array, 0.99, axis=0)
    y_array = np.clip(y_array, np.maximum(low, TARGET_FLOOR), np.maximum(high, TARGET_FLOOR)).astype(np.float32)
    return X_array, y_array, np.asarray(groups), sources


def make_model(model_id: str, window_updates: int, patch_length: int, num_channels: int, num_targets: int) -> PatchTSMixerForRegression:
    config = PatchTSMixerConfig.from_pretrained(model_id)
    config.context_length = window_updates
    config.patch_length = patch_length
    config.patch_stride = patch_length
    config.num_input_channels = num_channels
    config.num_targets = num_targets
    config.prediction_length = window_updates
    config.scaling = "std"
    config.loss = "mse"
    config.output_range = None
    config.num_patches = max(1, (window_updates - patch_length) // patch_length + 1)
    return PatchTSMixerForRegression.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )


def train_one(
    model: PatchTSMixerForRegression,
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_log, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(past_values=xb, target_values=yb, return_loss=True)
            loss = output.loss
            if loss is None:
                predictions = output.regression_outputs
                loss = torch.nn.functional.mse_loss(predictions, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def predict(model: PatchTSMixerForRegression, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            out = model(past_values=xb.to(device), return_loss=False)
            pred = out.regression_outputs
            outputs.append(pred.detach().cpu().numpy())
    return np.vstack(outputs)


def _render_report(metadata: dict[str, object]) -> str:
    lines = [
        "# IBM PatchTSMixer Direct Fine-Tuning Experiment",
        "",
        f"Model checkpoint: `{metadata['model_id']}`",
        "",
        "The model was fine-tuned as a GPS-independent covariance proxy estimator.",
        "Inputs exclude GPS coordinates and GPS residuals. Targets are benign proxy covariance labels,",
        "so this is not yet an AprilTag-validated physical uncertainty model.",
        "",
        "Important: this run changes the checkpoint input shape from the ETTh1 pretraining setup",
        "to the rover telemetry setup. Transformers reports several mismatched layers, so this",
        "is a direct checkpoint initialization experiment rather than a clean architecture-preserving",
        "transfer result.",
        "",
        "## Summary",
        "",
        f"- Runs/groups: `{metadata['runs']}`",
        f"- Training windows: `{metadata['windows']}`",
        f"- Window length: `{metadata['window_updates']}` updates",
        f"- Patch length: `{metadata['patch_length']}` updates",
        f"- Epochs per fold: `{metadata['epochs']}`",
        f"- Model status: `{metadata['model_status']}`",
        "",
        "## Grouped Cross-Validation",
        "",
        "| Target | Model MAE | Median Baseline MAE | Improvement | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for column in TARGET_COLUMNS:
        lines.append(
            f"| {column} | {metadata['cross_validated_mae'][column]:.6g} | "
            f"{metadata['median_baseline_mae'][column]:.6g} | "
            f"{100.0 * metadata['mae_improvement_over_median'][column]:.1f}% | "
            f"{metadata['cross_validated_r2'][column]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If this beats the median baseline, the pretrained backbone is useful for proxy uncertainty prediction.",
            "- If it does not, the bottleneck is probably the proxy label quality and limited data, not only architecture.",
            "- The model should not be activated in the EKF until bounded outputs and the evidence gate are integrated.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="benign raw T:147 CSVs or glob patterns")
    parser.add_argument("--manifest", default="DigitalTwin/datasets/analysis/real_data_study/benign_manifest.csv")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/ibm_patchtsmixer_uncertainty")
    parser.add_argument("--model-out", default="DigitalTwin/configs/ibm_patchtsmixer_uncertainty_model.pkl")
    parser.add_argument("--window-updates", type=int, default=WINDOW_UPDATES)
    parser.add_argument("--patch-length", type=int, default=PATCH_LENGTH)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.window_updates % args.patch_length:
        raise SystemExit("--window-updates must be divisible by --patch-length")

    torch.manual_seed(args.seed)
    paths = _expand_inputs(args.inputs) if args.inputs else _paths_from_manifest(Path(args.manifest))
    X, y, groups, sources = build_sequence_examples(paths, args.window_updates)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        raise RuntimeError("need at least four complete benign runs for grouped validation")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )

    fold_mae: list[np.ndarray] = []
    fold_baseline_mae: list[np.ndarray] = []
    fold_r2: list[np.ndarray] = []
    cv_folds = max(2, min(args.cv_folds, len(unique_groups)))
    splitter = GroupKFold(n_splits=cv_folds)
    for fold, (train_indices, test_indices) in enumerate(splitter.split(X, y, groups), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_indices].reshape(len(train_indices), -1)).reshape(
            len(train_indices), args.window_updates, len(FEATURE_COLUMNS)
        )
        X_test = scaler.transform(X[test_indices].reshape(len(test_indices), -1)).reshape(
            len(test_indices), args.window_updates, len(FEATURE_COLUMNS)
        )
        y_train = y[train_indices]
        y_train_log = _log_target(y_train).astype(np.float32)
        low = np.quantile(y_train, 0.01, axis=0)
        high = np.quantile(y_train, 0.99, axis=0)

        model = make_model(args.model_id, args.window_updates, args.patch_length, len(FEATURE_COLUMNS), len(TARGET_COLUMNS))
        train_one(
            model,
            X_train.astype(np.float32),
            y_train_log,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        predictions = _bounded_exp_target(
            predict(model, X_test.astype(np.float32), device, args.batch_size),
            low,
            high,
        )
        fold_mae.append(mean_absolute_error(y[test_indices], predictions, multioutput="raw_values"))
        baseline = np.repeat(np.median(y_train, axis=0, keepdims=True), len(test_indices), axis=0)
        fold_baseline_mae.append(mean_absolute_error(y[test_indices], baseline, multioutput="raw_values"))
        fold_r2.append(r2_score(y[test_indices], predictions, multioutput="raw_values"))

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X.reshape(len(X), -1)).reshape(len(X), args.window_updates, len(FEATURE_COLUMNS))
    final_low = np.quantile(y, 0.01, axis=0)
    final_high = np.quantile(y, 0.99, axis=0)
    final_model = make_model(args.model_id, args.window_updates, args.patch_length, len(FEATURE_COLUMNS), len(TARGET_COLUMNS))
    train_one(
        final_model,
        X_scaled.astype(np.float32),
        _log_target(y).astype(np.float32),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )
    final_model.to("cpu")

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model_id": args.model_id,
                "scaler": final_scaler,
                "state_dict": final_model.state_dict(),
                "config": final_model.config.to_dict(),
                "target_low": final_low,
                "target_high": final_high,
                "feature_columns": FEATURE_COLUMNS,
                "target_columns": TARGET_COLUMNS,
            },
            file,
        )

    mean_mae = np.mean(np.asarray(fold_mae), axis=0)
    mean_baseline_mae = np.mean(np.asarray(fold_baseline_mae), axis=0)
    improvements = 1.0 - mean_mae / np.maximum(mean_baseline_mae, TARGET_FLOOR)
    mean_r2 = np.mean(np.asarray(fold_r2), axis=0)
    accepted = bool(np.all(improvements > 0.0))

    metadata: dict[str, object] = {
        "schema": "ugv01_ibm_patchtsmixer_uncertainty_v1",
        "model_id": args.model_id,
        "model": "PatchTSMixerForRegression direct fine-tune",
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "target_definition": "future-window benign process-error covariance surrogate",
        "gps_coordinate_residual_inputs_allowed": False,
        "attack_rows_allowed": False,
        "window_updates": args.window_updates,
        "patch_length": args.patch_length,
        "epochs": args.epochs,
        "cv_folds": cv_folds,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "checkpoint_transfer_note": (
            "The ETTh1 checkpoint was loaded with ignore_mismatched_sizes=True. "
            "Changing context_length, patch_length, num_input_channels, and num_targets "
            "causes some patch/head layers to be reinitialized."
        ),
        "runs": int(len(unique_groups)),
        "windows": int(len(X)),
        "source_files": sources,
        "validation": "complete-run GroupKFold",
        "cross_validated_mae": {column: float(value) for column, value in zip(TARGET_COLUMNS, mean_mae)},
        "median_baseline_mae": {column: float(value) for column, value in zip(TARGET_COLUMNS, mean_baseline_mae)},
        "mae_improvement_over_median": {column: float(value) for column, value in zip(TARGET_COLUMNS, improvements)},
        "cross_validated_r2": {column: float(value) for column, value in zip(TARGET_COLUMNS, mean_r2)},
        "output_bounds": {
            column: {"low": float(low), "high": float(high)}
            for column, low, high in zip(TARGET_COLUMNS, final_low, final_high)
        },
        "model_status": "candidate_passed_proxy_cv" if accepted else "candidate_rejected_proxy_cv",
        "limitations": [
            "Targets are proxy covariance labels, not AprilTag physical error labels.",
            "Inputs exclude GPS coordinates and residuals.",
            "The model should be evidence-gated before EKF activation.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ibm_patchtsmixer_uncertainty_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "ibm_patchtsmixer_uncertainty_report.md").write_text(_render_report(metadata), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
