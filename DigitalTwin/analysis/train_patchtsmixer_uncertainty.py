"""Train a PatchTSMixer-style GPS-independent covariance proxy.

This is a lightweight, repo-native experiment inspired by PatchTSMixer. It does
not require ``torch`` or ``transformers``. The IBM PatchTSMixer checkpoint can be
used later as a backbone once those dependencies are installed, but this script
keeps the same project-safe objective: learn bounded process-uncertainty
surrogates from GPS-independent telemetry windows, not position or attack labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from DigitalTwin.kinematics import ugv01_calibrated_geometry, wrap_angle
from DigitalTwin.telemetry import gps_to_local_xy
from DigitalTwin.uncertainty import TelemetryStatisticsWindow

from .common import parse_bool, parse_float, parse_int, parse_run_name, read_rows
from .train_uncertainty import STANDARD_GRAVITY_MPS2, TARGET_COLUMNS, TARGET_FLOOR


WINDOW_UPDATES = 16
PATCH_LENGTH = 4
TARGET_HORIZON_UPDATES = 5

FEATURE_COLUMNS = (
    "cmd_left",
    "cmd_right",
    "delta_left_ticks",
    "delta_right_ticks",
    "encoder_velocity_mps",
    "encoder_yaw_rate_radps",
    "encoder_disagreement_ticks",
    "imu_vertical_std",
    "imu_yaw_std",
    "velocity_variance",
    "packet_dt_s",
    "http_latency_ms",
    "stale_packet",
    "queue_depth",
    "sequence_gap_count",
)


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    return float(parse_float(row.get(key, ""), default) or default)


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(parse_int(row.get(key, ""), default) or default)


def _sample_time_s(row: dict[str, str]) -> float:
    sample_ms = parse_float(row.get("sample_ms", ""), None)
    if sample_ms is not None:
        return float(sample_ms) / 1000.0
    return _f(row, "source_sample_time_s", _f(row, "rover_millis_s"))


def _successful_rows(path: Path) -> list[dict[str, str]]:
    return [
        row
        for row in read_rows(path)
        if parse_bool(row.get("cycle_ok", "True"))
        and parse_bool(row.get("gps_valid", "False"))
        and parse_float(row.get("lat", ""), None) is not None
        and parse_float(row.get("lon", ""), None) is not None
    ]


def _extract_run_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _successful_rows(path)
    if len(rows) <= WINDOW_UPDATES + TARGET_HORIZON_UPDATES + 2:
        raise RuntimeError(f"{path} has too few successful GPS-valid rows")

    geometry = ugv01_calibrated_geometry()
    times = np.asarray([_sample_time_s(row) for row in rows], dtype=float)
    elapsed = times - times[0]
    origin_lat = _f(rows[0], "lat")
    origin_lon = _f(rows[0], "lon")
    gps_xy = np.asarray(
        [gps_to_local_xy(_f(row, "lat"), _f(row, "lon"), origin_lat, origin_lon) for row in rows],
        dtype=float,
    )

    startup_count = min(5, len(rows))
    startup_deltas = np.diff(gps_xy[:startup_count], axis=0)
    gps_step_noise = (
        np.median(startup_deltas**2, axis=0)
        if len(startup_deltas)
        else np.full(2, TARGET_FLOOR)
    )

    stats = TelemetryStatisticsWindow()
    features: list[list[float]] = []
    encoder_distances: list[float] = []
    imu_yaws: list[float] = []
    heading_errors: list[float] = []

    prev_left = _i(rows[0], "enc_left")
    prev_right = _i(rows[0], "enc_right")
    prev_yaw = math.radians(_f(rows[0], "y"))
    previous_arrival: float | None = None
    previous_sequence = _i(rows[0], "seq")

    for index, row in enumerate(rows):
        dt_s = 0.1 if index == 0 else max(float(elapsed[index] - elapsed[index - 1]), 1e-3)
        arrival = _f(row, "edge_arrival_time_s", _f(row, "t_edge_rx_ns") / 1e9)
        arrival_dt = dt_s if previous_arrival is None else max(arrival - previous_arrival, 1e-3)
        previous_arrival = arrival

        left = _i(row, "enc_left")
        right = _i(row, "enc_right")
        d_left = left - prev_left
        d_right = right - prev_right
        velocity, encoder_yaw_rate = geometry.ticks_to_control(d_left, d_right, dt_s)
        prev_left, prev_right = left, right

        imu_yaw = math.radians(_f(row, "y"))
        imu_yaw_increment = 0.0 if index == 0 else wrap_angle(imu_yaw - prev_yaw)
        prev_yaw = imu_yaw
        encoder_distances.append(float(velocity) * dt_s)
        imu_yaws.append(imu_yaw)
        heading_errors.append(wrap_angle(imu_yaw_increment - encoder_yaw_rate * dt_s))

        stats.observe(
            dead_reckoning_residual_m=0.0,
            accel_z=_f(row, "az") * STANDARD_GRAVITY_MPS2 / 1000.0,
            gyro_z=math.radians(_f(row, "gz")),
            velocity_mps=velocity,
            packet_dt_s=arrival_dt,
        )
        rolling = stats.features(
            gps_hdop=0.0,
            gps_satellites=0,
            fallback_dt_s=arrival_dt,
        )
        seq = _i(row, "seq")
        seq_gap = max(0, seq - previous_sequence - 1) if index else 0
        previous_sequence = seq

        features.append(
            [
                _f(row, "L"),
                _f(row, "R"),
                float(d_left),
                float(d_right),
                float(velocity),
                float(encoder_yaw_rate),
                float(abs(abs(d_left) - abs(d_right))),
                rolling.imu_vertical_std,
                rolling.imu_yaw_std,
                rolling.velocity_variance,
                rolling.packet_dt_s,
                _f(row, "http_latency_ms"),
                float(parse_bool(row.get("stale_packet", "False"))),
                _f(row, "queue_depth"),
                float(seq_gap + _i(row, "sequence_gap_count")),
            ]
        )

    gps_deltas = np.vstack([np.zeros(2), np.diff(gps_xy, axis=0)])
    encoder_body = np.column_stack(
        [
            np.asarray(encoder_distances) * np.cos(np.asarray(imu_yaws)),
            np.asarray(encoder_distances) * np.sin(np.asarray(imu_yaws)),
        ]
    )
    moving = np.abs(np.asarray(encoder_distances)) >= 0.005
    if np.any(moving):
        dot = float(np.sum(encoder_body[moving] * gps_deltas[moving]))
        cross = float(
            np.sum(
                encoder_body[moving, 0] * gps_deltas[moving, 1]
                - encoder_body[moving, 1] * gps_deltas[moving, 0]
            )
        )
        alignment = math.atan2(cross, dot)
    else:
        alignment = 0.0
    rotation = np.array(
        [[math.cos(alignment), -math.sin(alignment)], [math.sin(alignment), math.cos(alignment)]],
        dtype=float,
    )
    encoder_global = encoder_body @ rotation.T
    position_errors = gps_deltas - encoder_global
    position_targets = np.maximum(position_errors**2 - gps_step_noise, TARGET_FLOOR)
    targets = np.column_stack(
        [
            position_targets[:, 0],
            position_targets[:, 1],
            np.maximum(np.asarray(heading_errors) ** 2, TARGET_FLOOR),
        ]
    )
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def _patch_mix(window: np.ndarray, patch_length: int) -> np.ndarray:
    """Convert a [time, channel] window into patch/channel mixed features."""

    patches = []
    for start in range(0, len(window), patch_length):
        patch = window[start : start + patch_length]
        if len(patch) < patch_length:
            break
        patches.append(patch)
    patch_array = np.asarray(patches, dtype=float)
    pieces = [
        window[-1],
        np.mean(window, axis=0),
        np.std(window, axis=0),
        window[-1] - window[0],
        np.mean(patch_array, axis=1).reshape(-1),
        np.std(patch_array, axis=1).reshape(-1),
        np.mean(patch_array, axis=2).reshape(-1),
    ]
    return np.concatenate(pieces)


def build_examples(paths: list[Path], window_updates: int, patch_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
            window = features[index + 1 - window_updates : index + 1]
            future = targets[index + 1 : index + 1 + TARGET_HORIZON_UPDATES]
            X.append(_patch_mix(window, patch_length))
            y.append(np.median(future, axis=0))
            groups.append(run_id)
        sources.append(str(path))

    if not X:
        raise RuntimeError("no training windows were extracted")

    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)
    low = np.quantile(y_array, 0.01, axis=0)
    high = np.quantile(y_array, 0.99, axis=0)
    y_array = np.clip(y_array, np.maximum(low, TARGET_FLOOR), np.maximum(high, TARGET_FLOOR))
    return X_array, y_array, np.asarray(groups), sources


def _paths_from_manifest(path: Path) -> list[Path]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}; run real_data_study first")
    with path.open(newline="", encoding="utf-8") as file:
        return [Path(row["source_csv"]) for row in csv.DictReader(file)]


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = sorted(Path().glob(item)) if any(character in item for character in "*?[") else [Path(item)]
        paths.extend(path for path in matches if path.exists())
    return paths


def _make_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("x_scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(96, 48),
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=700,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _log_target(y: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(y, TARGET_FLOOR))


def _bounded_exp_target(y_log: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(y_log), np.maximum(low, TARGET_FLOOR), np.maximum(high, TARGET_FLOOR))


def _render_report(metadata: dict[str, object]) -> str:
    lines = [
        "# PatchTSMixer-Style GPS-Independent Covariance Experiment",
        "",
        "This experiment trains a lightweight patch-mixer neural regressor on existing UGV01 telemetry.",
        "It is PatchTSMixer-inspired and uses no GPS coordinate residuals as inputs. Labels are still",
        "proxy covariance targets derived from benign replay consistency, so this is not a final",
        "physical-accuracy claim without AprilTag ground truth.",
        "",
        "## Summary",
        "",
        f"- Runs/groups: `{metadata['runs']}`",
        f"- Training windows: `{metadata['windows']}`",
        f"- Window length: `{metadata['window_updates']}` updates",
        f"- Patch length: `{metadata['patch_length']}` updates",
        f"- Model status: `{metadata['model_status']}`",
        "",
        "## Grouped Cross-Validation",
        "",
        "| Target | Model MAE | Median Baseline MAE | Improvement | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for column in TARGET_COLUMNS:
        lines.append(
            "| "
            + column
            + " | "
            + f"{metadata['cross_validated_mae'][column]:.6g} | "
            + f"{metadata['median_baseline_mae'][column]:.6g} | "
            + f"{100.0 * metadata['mae_improvement_over_median'][column]:.1f}% | "
            + f"{metadata['cross_validated_r2'][column]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A positive improvement means the patch-mixer features predicted the covariance proxy better than a run-fold median baseline.",
            "- This model should be treated as a candidate uncertainty estimator, not as an attack detector.",
            "- The deployment-safe role is to propose bounded `Q_k` values from protected telemetry, then let the evidence gate accept, freeze, or fall back.",
            "- AprilTag ground truth is still needed to show that the learned uncertainty corresponds to true physical trajectory error.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="benign raw T:147 CSVs or glob patterns")
    parser.add_argument("--manifest", default="DigitalTwin/datasets/analysis/real_data_study/benign_manifest.csv")
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/patchtsmixer_uncertainty")
    parser.add_argument("--model-out", default="DigitalTwin/configs/patchtsmixer_uncertainty_model.pkl")
    parser.add_argument("--window-updates", type=int, default=WINDOW_UPDATES)
    parser.add_argument("--patch-length", type=int, default=PATCH_LENGTH)
    parser.add_argument("--random-state", type=int, default=17)
    args = parser.parse_args()

    if args.window_updates % args.patch_length:
        raise SystemExit("--window-updates must be divisible by --patch-length")

    paths = _expand_inputs(args.inputs) if args.inputs else _paths_from_manifest(Path(args.manifest))
    X, y, groups, sources = build_examples(paths, args.window_updates, args.patch_length)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        raise RuntimeError("need at least four complete benign runs for grouped validation")

    fold_mae: list[np.ndarray] = []
    fold_baseline_mae: list[np.ndarray] = []
    fold_r2: list[np.ndarray] = []
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    for fold, (train_indices, test_indices) in enumerate(splitter.split(X, y, groups), start=1):
        model = _make_model(args.random_state + fold)
        y_train = y[train_indices]
        low = np.quantile(y_train, 0.01, axis=0)
        high = np.quantile(y_train, 0.99, axis=0)
        model.fit(X[train_indices], _log_target(y_train))
        predictions = _bounded_exp_target(model.predict(X[test_indices]), low, high)
        fold_mae.append(mean_absolute_error(y[test_indices], predictions, multioutput="raw_values"))
        baseline = np.repeat(np.median(y[train_indices], axis=0, keepdims=True), len(test_indices), axis=0)
        fold_baseline_mae.append(mean_absolute_error(y[test_indices], baseline, multioutput="raw_values"))
        fold_r2.append(r2_score(y[test_indices], predictions, multioutput="raw_values"))

    final_model = _make_model(args.random_state)
    final_low = np.quantile(y, 0.01, axis=0)
    final_high = np.quantile(y, 0.99, axis=0)
    final_model.fit(X, _log_target(y))
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model": final_model,
                "target_low": final_low,
                "target_high": final_high,
                "feature_columns": FEATURE_COLUMNS,
                "target_columns": TARGET_COLUMNS,
                "prediction": "exp(model(window_features)) clipped to target_low/target_high",
            },
            file,
        )

    mean_mae = np.mean(np.asarray(fold_mae), axis=0)
    mean_baseline_mae = np.mean(np.asarray(fold_baseline_mae), axis=0)
    improvements = 1.0 - mean_mae / np.maximum(mean_baseline_mae, TARGET_FLOOR)
    mean_r2 = np.mean(np.asarray(fold_r2), axis=0)
    accepted = bool(np.all(improvements > 0.0))

    metadata: dict[str, object] = {
        "schema": "ugv01_patchtsmixer_uncertainty_v1",
        "model": "PatchTSMixer-inspired patch feature mixer + MLPRegressor",
        "pretrained_start": "ibm/patchtsmixer-etth1-pretrain intended future backbone; not loaded in this dependency-light run",
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "target_definition": "future-window benign process-error covariance surrogate",
        "gps_coordinate_residual_inputs_allowed": False,
        "attack_rows_allowed": False,
        "window_updates": args.window_updates,
        "patch_length": args.patch_length,
        "runs": int(len(unique_groups)),
        "windows": int(len(X)),
        "source_files": sources,
        "validation": "complete-run GroupKFold",
        "cross_validated_mae": {column: float(value) for column, value in zip(TARGET_COLUMNS, mean_mae)},
        "median_baseline_mae": {column: float(value) for column, value in zip(TARGET_COLUMNS, mean_baseline_mae)},
        "mae_improvement_over_median": {
            column: float(value) for column, value in zip(TARGET_COLUMNS, improvements)
        },
        "cross_validated_r2": {column: float(value) for column, value in zip(TARGET_COLUMNS, mean_r2)},
        "target_median": {column: float(value) for column, value in zip(TARGET_COLUMNS, np.median(y, axis=0))},
        "target_p90": {column: float(value) for column, value in zip(TARGET_COLUMNS, np.quantile(y, 0.90, axis=0))},
        "output_bounds": {
            column: {"low": float(low), "high": float(high)}
            for column, low, high in zip(TARGET_COLUMNS, final_low, final_high)
        },
        "model_status": "candidate_passed_proxy_cv" if accepted else "candidate_rejected_proxy_cv",
        "limitations": [
            "This is a proxy covariance model trained without sufficient AprilTag ground truth.",
            "The model does not directly predict rover position or detect attacks.",
            "The IBM pretrained checkpoint is not loaded because transformers/torch are not project dependencies here.",
            "Activation in the primary EKF should require bounded outputs and the independent evidence gate.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patchtsmixer_uncertainty_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (out_dir / "patchtsmixer_uncertainty_report.md").write_text(
        _render_report(metadata),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
