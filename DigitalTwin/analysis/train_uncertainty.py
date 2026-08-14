"""Train the GPS-independent uncertainty model from benign UGV01 logs.

Features are causal and exclude GPS coordinate residuals. Labels use future
benign process-error energy: encoder/GPS travel disagreement for position and
encoder/IMU yaw-increment disagreement for heading. The label may use clean GPS
during offline training; the deployed feature vector may not.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import pickle
from pathlib import Path

import numpy as np

from DigitalTwin.kinematics import ugv01_calibrated_geometry, wrap_angle
from DigitalTwin.telemetry import gps_to_local_xy
from DigitalTwin.uncertainty import LEARNED_FEATURE_COLUMNS, TelemetryStatisticsWindow

from .common import parse_bool, parse_float, parse_int, parse_run_name, read_rows


STANDARD_GRAVITY_MPS2 = 9.80665
TARGET_COLUMNS = ("q_xx", "q_yy", "q_tt")
TARGET_FLOOR = 1e-8
TARGET_HORIZON_UPDATES = 5
MODEL_ACCEPTANCE_MIN_IMPROVEMENT = 0.0


@dataclass(slots=True)
class TrainingExamples:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    source_files: list[str]


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


def extract_run_examples(
    path: Path,
    *,
    horizon_updates: int = TARGET_HORIZON_UPDATES,
) -> tuple[np.ndarray, np.ndarray]:
    rows = _successful_rows(path)
    if len(rows) <= horizon_updates + 2:
        raise RuntimeError(f"{path} has too few successful GPS-valid rows")

    geometry = ugv01_calibrated_geometry()
    times = np.asarray([_sample_time_s(row) for row in rows], dtype=float)
    elapsed = times - times[0]
    origin_lat = _f(rows[0], "lat")
    origin_lon = _f(rows[0], "lon")
    gps_xy = np.asarray(
        [gps_to_local_xy(_f(row, "lat"), _f(row, "lon"), origin_lat, origin_lon) for row in rows]
    )

    # The collection protocol begins with a five-update stationary hold. Using
    # later points would mix commanded motion into the GPS noise estimate.
    startup_count = min(5, len(rows))
    startup_deltas = np.diff(gps_xy[:startup_count], axis=0)
    gps_step_noise = (
        np.median(startup_deltas**2, axis=0)
        if len(startup_deltas)
        else np.full(2, TARGET_FLOOR)
    )

    stats = TelemetryStatisticsWindow()
    feature_rows: list[np.ndarray] = []
    encoder_distances: list[float] = []
    imu_yaws: list[float] = []
    heading_errors: list[float] = []
    prev_left = _i(rows[0], "enc_left")
    prev_right = _i(rows[0], "enc_right")
    prev_yaw = math.radians(_f(rows[0], "y"))
    previous_arrival: float | None = None

    for index, row in enumerate(rows):
        dt_s = 0.1 if index == 0 else max(float(elapsed[index] - elapsed[index - 1]), 1e-3)
        arrival = _f(row, "edge_arrival_time_s", _f(row, "t_edge_rx_ns") / 1e9)
        arrival_dt = dt_s if previous_arrival is None else max(arrival - previous_arrival, 1e-3)
        previous_arrival = arrival
        left = _i(row, "enc_left")
        right = _i(row, "enc_right")
        velocity, encoder_yaw_rate = geometry.ticks_to_control(left - prev_left, right - prev_right, dt_s)
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
        features = stats.features(
            gps_hdop=_f(row, "hdop", 99.99),
            gps_satellites=_i(row, "sat"),
            fallback_dt_s=arrival_dt,
        )
        feature_rows.append(features.gps_independent_model_vector())

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
    cosine = math.cos(alignment)
    sine = math.sin(alignment)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    encoder_global = encoder_body @ rotation.T
    position_errors = gps_deltas - encoder_global
    position_targets = np.maximum(position_errors**2 - gps_step_noise, TARGET_FLOOR)
    raw = np.column_stack(
        [
            position_targets[:, 0],
            position_targets[:, 1],
            np.maximum(np.asarray(heading_errors) ** 2, TARGET_FLOOR),
        ]
    )
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for index in range(4, len(rows) - horizon_updates):
        future = raw[index + 1 : index + 1 + horizon_updates]
        X.append(feature_rows[index])
        y.append(np.median(future, axis=0))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def build_training_examples(paths: list[Path]) -> TrainingExamples:
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[str] = []
    source_files: list[str] = []
    for path in paths:
        meta = parse_run_name(path)
        if meta.get("attack") not in {"", "none"}:
            continue
        X_run, y_run = extract_run_examples(path)
        surface = meta.get("surface", meta.get("terrain", ""))
        run_id = f"{meta.get('speed', '')}_{surface}_trial-{meta.get('trial', '')}"
        matrices.append(X_run)
        targets.append(y_run)
        groups.extend([run_id] * len(X_run))
        source_files.append(str(path))
    if not matrices:
        raise RuntimeError("no benign UGV01 training logs were found")

    X = np.vstack(matrices)
    y = np.vstack(targets)
    low = np.quantile(y, 0.01, axis=0)
    high = np.quantile(y, 0.99, axis=0)
    y = np.clip(y, np.maximum(low, TARGET_FLOOR), np.maximum(high, TARGET_FLOOR))
    return TrainingExamples(X=X, y=y, groups=np.asarray(groups), source_files=source_files)


def _paths_from_manifest(path: Path) -> list[Path]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}; run real_data_study --manifest-only first")
    with path.open(newline="", encoding="utf-8") as file:
        return [Path(row["source_csv"]) for row in csv.DictReader(file)]


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = sorted(Path().glob(item)) if any(character in item for character in "*?[") else [Path(item)]
        paths.extend(path for path in matches if path.exists())
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="benign raw T:147 CSVs or glob patterns")
    parser.add_argument(
        "--manifest",
        default="DigitalTwin/datasets/analysis/real_data_study/benign_manifest.csv",
    )
    parser.add_argument("--out", default="DigitalTwin/configs/uncertainty_model.pkl")
    parser.add_argument("--metadata-out", default="DigitalTwin/configs/uncertainty_model.json")
    parser.add_argument(
        "--model",
        choices=("mlp", "random_forest"),
        default="mlp",
        help="candidate model to fit on all rows after grouped validation",
    )
    args = parser.parse_args()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    paths = _expand_inputs(args.inputs) if args.inputs else _paths_from_manifest(Path(args.manifest))
    examples = build_training_examples(paths)
    unique_groups = np.unique(examples.groups)
    if len(unique_groups) < 4:
        raise RuntimeError("need at least four complete benign runs for grouped validation")

    def make_model(kind: str, *, seed: int) -> object:
        if kind == "random_forest":
            return RandomForestRegressor(n_estimators=200, random_state=seed, min_samples_leaf=5, n_jobs=-1)
        if kind == "mlp":
            return Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(16, 8),
                            activation="relu",
                            alpha=3e-3,
                            learning_rate_init=1e-3,
                            max_iter=350,
                            early_stopping=True,
                            validation_fraction=0.20,
                            n_iter_no_change=25,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        raise ValueError(f"unsupported model: {kind}")

    def fit_candidate(kind: str, X: np.ndarray, y: np.ndarray, *, seed: int) -> object:
        model = make_model(kind, seed=seed)
        target = np.log(np.maximum(y, TARGET_FLOOR)) if kind == "mlp" else y
        model.fit(X, target)
        return model

    def predict_candidate(
        model: object,
        kind: str,
        X: np.ndarray,
        target_low: np.ndarray,
        target_high: np.ndarray,
    ) -> np.ndarray:
        raw_predictions = model.predict(X)
        predictions = np.exp(raw_predictions) if kind == "mlp" else raw_predictions
        return np.clip(
            predictions,
            np.maximum(target_low, TARGET_FLOOR),
            np.maximum(target_high, TARGET_FLOOR),
        )

    target_low = np.quantile(examples.y, 0.01, axis=0)
    target_high = np.quantile(examples.y, 0.99, axis=0)
    fold_mae_by_model: dict[str, list[np.ndarray]] = {"random_forest": [], "mlp": []}
    fold_baseline_mae: list[np.ndarray] = []
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(examples.X, examples.y, examples.groups)
    ):
        for kind in fold_mae_by_model:
            model = fit_candidate(kind, examples.X[train_indices], examples.y[train_indices], seed=7 + fold_index)
            predictions = predict_candidate(model, kind, examples.X[test_indices], target_low, target_high)
            fold_mae_by_model[kind].append(
                mean_absolute_error(examples.y[test_indices], predictions, multioutput="raw_values")
            )
        baseline = np.repeat(
            np.median(examples.y[train_indices], axis=0, keepdims=True),
            len(test_indices),
            axis=0,
        )
        fold_baseline_mae.append(
            mean_absolute_error(examples.y[test_indices], baseline, multioutput="raw_values")
        )

    model = fit_candidate(args.model, examples.X, examples.y, seed=7)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as file:
        pickle.dump(
            {
                "model_type": args.model,
                "model": model,
                "target_low": target_low,
                "target_high": target_high,
                "target_floor": TARGET_FLOOR,
                "feature_columns": list(LEARNED_FEATURE_COLUMNS),
                "target_columns": list(TARGET_COLUMNS),
            },
            file,
        )

    mean_mae_by_model = {
        kind: np.mean(np.asarray(values), axis=0) for kind, values in fold_mae_by_model.items()
    }
    mean_baseline_mae = np.mean(np.asarray(fold_baseline_mae), axis=0)
    improvements_by_model = {
        kind: 1.0 - mae / np.maximum(mean_baseline_mae, TARGET_FLOOR)
        for kind, mae in mean_mae_by_model.items()
    }
    mean_mae = mean_mae_by_model[args.model]
    improvements = 1.0 - mean_mae / np.maximum(mean_baseline_mae, TARGET_FLOOR)
    accepted = bool(np.all(improvements > MODEL_ACCEPTANCE_MIN_IMPROVEMENT))
    best_method_by_target = {
        column: min(
            [
                ("median_baseline", float(mean_baseline_mae[index])),
                ("random_forest", float(mean_mae_by_model["random_forest"][index])),
                ("mlp", float(mean_mae_by_model["mlp"][index])),
            ],
            key=lambda item: item[1],
        )[0]
        for index, column in enumerate(TARGET_COLUMNS)
    }
    metadata = {
        "schema": "ugv01_learned_uncertainty_v1",
        "model": "MLPRegressor" if args.model == "mlp" else "RandomForestRegressor",
        "selected_model_key": args.model,
        "feature_columns": list(LEARNED_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "target_definition": "future-window benign process-error covariance surrogate",
        "target_horizon_updates": TARGET_HORIZON_UPDATES,
        "coordinate_residual_feature_allowed": False,
        "attack_rows_allowed": False,
        "validation": "complete-run GroupKFold",
        "acceptance_rule": "MAE improvement over the training-fold median must be positive for all Q targets",
        "accepted_for_primary_campaign": accepted,
        "model_status": "accepted_grouped_cv" if accepted else "candidate_rejected_grouped_cv",
        "runs": int(len(unique_groups)),
        "rows": int(len(examples.X)),
        "source_files": examples.source_files,
        "cross_validated_mae": {column: float(value) for column, value in zip(TARGET_COLUMNS, mean_mae)},
        "candidate_cross_validated_mae": {
            kind: {column: float(value) for column, value in zip(TARGET_COLUMNS, mae)}
            for kind, mae in mean_mae_by_model.items()
        },
        "median_baseline_mae": {
            column: float(value) for column, value in zip(TARGET_COLUMNS, mean_baseline_mae)
        },
        "mae_improvement_over_median": {
            column: float(value) for column, value in zip(TARGET_COLUMNS, improvements)
        },
        "candidate_mae_improvement_over_median": {
            kind: {column: float(value) for column, value in zip(TARGET_COLUMNS, model_improvements)}
            for kind, model_improvements in improvements_by_model.items()
        },
        "best_method_by_target": best_method_by_target,
        "target_median": {
            column: float(value) for column, value in zip(TARGET_COLUMNS, np.median(examples.y, axis=0))
        },
        "target_p90": {
            column: float(value) for column, value in zip(TARGET_COLUMNS, np.quantile(examples.y, 0.90, axis=0))
        },
        "target_floor_fraction": {
            column: float(value)
            for column, value in zip(
                TARGET_COLUMNS,
                np.mean(examples.y <= TARGET_FLOOR * (1.0 + 1e-9), axis=0),
            )
        },
        "feature_importance": (
            {
                column: float(value)
                for column, value in zip(LEARNED_FEATURE_COLUMNS, model.feature_importances_)
            }
            if args.model == "random_forest"
            else {}
        ),
        "limitations": [
            "position labels contain residual benign GPS noise after startup-noise subtraction",
            "the model is frozen from the current 20-run corpus and has no independent prospective test set",
            "a rejected candidate is retained as an artifact but must not be activated in the primary campaign",
        ],
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
