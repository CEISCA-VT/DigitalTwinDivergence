"""Compare stronger GPS-independent Q_k models on aligned i2Nav data.

This script reuses the i2Nav study's exact features, targets, chronological
split, calibration, and EKF replay. It is intentionally separate from the main
study so experimental model bake-offs do not destabilize the frozen analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DigitalTwin.analysis.i2nav_uncertainty_study import (
    DEFAULT_INPUT,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TARGET_FLOOR,
    _calibration_factors,
    _coverage_metrics,
    _predict,
    _regression_metrics,
    build_features,
    calibrate_gps_bias_ekf_scales,
    estimate_motion_calibration,
    future_covariance_targets,
    load_aligned,
    process_residuals,
    replay_gps_bias_ekf,
    temporal_split_indices,
)


DEFAULT_OUTPUT = Path("DigitalTwin/datasets/analysis/i2nav_model_bakeoff")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fit: Callable[[np.ndarray, np.ndarray, np.ndarray, int], object]
    predict: Callable[[object, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    optional_dependency: str | None = None


def _clip_range(y: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.maximum(np.quantile(y[indices], 0.005, axis=0), TARGET_FLOOR)
    high = np.maximum(np.quantile(y[indices], 0.995, axis=0), low * 1.01)
    return low, high


def _fit_log_pipeline(factory: Callable[[int], object]):
    def fit(X: np.ndarray, y: np.ndarray, indices: np.ndarray, seed: int) -> object:
        low, high = _clip_range(y, indices)
        model = factory(seed)
        model.fit(X[indices], np.log(np.clip(y[indices], low, high)))
        return {"model": model, "low": low, "high": high}

    return fit


def _predict_log_model(bundle: object, X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    model = bundle["model"]
    return np.clip(np.exp(model.predict(X)), low, high)


def _fit_raw_pipeline(factory: Callable[[int], object]):
    def fit(X: np.ndarray, y: np.ndarray, indices: np.ndarray, seed: int) -> object:
        low, high = _clip_range(y, indices)
        model = factory(seed)
        model.fit(X[indices], np.clip(y[indices], low, high))
        return {"model": model, "low": low, "high": high}

    return fit


def _predict_raw_model(bundle: object, X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    model = bundle["model"]
    return np.clip(model.predict(X), low, high)


def _make_mlp(seed: int):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.002,
                    learning_rate_init=0.001,
                    max_iter=700,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=40,
                    random_state=seed,
                ),
            ),
        ]
    )


def _fit_deep_ensemble(X: np.ndarray, y: np.ndarray, indices: np.ndarray, seed: int) -> object:
    low, high = _clip_range(y, indices)
    models = []
    for offset in range(5):
        model = _make_mlp(seed + 101 * offset)
        model.fit(X[indices], np.log(np.clip(y[indices], low, high)))
        models.append(model)
    return {"models": models, "low": low, "high": high}


def _predict_deep_ensemble(bundle: object, X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    logs = np.stack([model.predict(X) for model in bundle["models"]], axis=0)
    return np.clip(np.exp(np.mean(logs, axis=0)), low, high)


def _make_random_forest(seed: int):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=350,
        min_samples_leaf=6,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )


def _make_extra_trees(seed: int):
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features=0.9,
        n_jobs=-1,
        random_state=seed,
    )


def _make_gradient_boosting(seed: int):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=8,
            subsample=0.85,
            random_state=seed,
        )
    )


def _make_hist_gradient_boosting(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=450,
            learning_rate=0.035,
            l2_regularization=0.02,
            min_samples_leaf=12,
            random_state=seed,
        )
    )


def _make_quantile_gradient_boosting(seed: int):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return MultiOutputRegressor(
        GradientBoostingRegressor(
            loss="quantile",
            alpha=0.5,
            n_estimators=350,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=8,
            random_state=seed,
        )
    )


def _make_xgboost(seed: int):
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor

    return MultiOutputRegressor(
        XGBRegressor(
            n_estimators=550,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    )


def _make_lightgbm(seed: int):
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=650,
            learning_rate=0.025,
            num_leaves=24,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    )


def _make_catboost(seed: int):
    from catboost import CatBoostRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return MultiOutputRegressor(
        CatBoostRegressor(
            iterations=650,
            depth=5,
            learning_rate=0.025,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    )


def _fit_ngboost(X: np.ndarray, y: np.ndarray, indices: np.ndarray, seed: int) -> object:
    from ngboost import NGBRegressor
    from ngboost.distns import LogNormal
    from sklearn.tree import DecisionTreeRegressor

    low, high = _clip_range(y, indices)
    models = []
    clipped = np.clip(y[indices], low, high)
    for column in range(clipped.shape[1]):
        model = NGBRegressor(
            Dist=LogNormal,
            Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=8),
            n_estimators=450,
            learning_rate=0.025,
            random_state=seed + column,
            verbose=False,
        )
        model.fit(X[indices], clipped[:, column])
        models.append(model)
    return {"models": models, "low": low, "high": high}


def _predict_ngboost(bundle: object, X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.column_stack([model.predict(X) for model in bundle["models"]])
    return np.clip(values, low, high)


def _available_optional(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def model_specs(include_optional: bool) -> list[ModelSpec]:
    specs = [
        ModelSpec("mlp_log", _fit_log_pipeline(_make_mlp), _predict_log_model),
        ModelSpec("deep_ensemble_mlp_log", _fit_deep_ensemble, _predict_deep_ensemble),
        ModelSpec("random_forest_raw", _fit_raw_pipeline(_make_random_forest), _predict_raw_model),
        ModelSpec("extra_trees_log", _fit_log_pipeline(_make_extra_trees), _predict_log_model),
        ModelSpec("gradient_boosting_log", _fit_log_pipeline(_make_gradient_boosting), _predict_log_model),
        ModelSpec("hist_gradient_boosting_log", _fit_log_pipeline(_make_hist_gradient_boosting), _predict_log_model),
        ModelSpec("quantile_gradient_boosting_p50_log", _fit_log_pipeline(_make_quantile_gradient_boosting), _predict_log_model),
    ]
    if include_optional:
        optional = [
            ModelSpec("xgboost_log", _fit_log_pipeline(_make_xgboost), _predict_log_model, "xgboost"),
            ModelSpec("lightgbm_log", _fit_log_pipeline(_make_lightgbm), _predict_log_model, "lightgbm"),
            ModelSpec("catboost_log", _fit_log_pipeline(_make_catboost), _predict_log_model, "catboost"),
            ModelSpec("ngboost_lognormal", _fit_ngboost, _predict_ngboost, "ngboost"),
        ]
        specs.extend(
            spec for spec in optional if _available_optional(str(spec.optional_dependency))
        )
    return specs


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_normalized_mae(metrics: dict[str, np.ndarray]) -> float:
    return float(np.mean(metrics["normalized_mae"]))


def _serialize_vector(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(TARGET_COLUMNS, values)}


def _plot_bakeoff_results(
    output_dir: Path,
    ranked: list[dict[str, object]],
    ekf_rows: list[dict[str, object]],
    data: dict[str, np.ndarray],
    start: int,
    stop: int,
    best_states_by_model: dict[str, np.ndarray],
) -> None:
    if ranked:
        top = ranked[: min(8, len(ranked))]
        names = [str(row["model"]) for row in top]
        values = [float(row["mean_normalized_mae"]) for row in top]
        coverage = [100.0 * float(row["joint_95_coverage"]) for row in top]
        x = np.arange(len(names))
        fig, ax1 = plt.subplots(figsize=(10.0, 5.2))
        ax1.bar(x, values, color="#4c78a8", label="Mean normalized MAE")
        ax1.set_ylabel("Mean normalized MAE (lower is better)")
        ax1.set_xticks(x, names, rotation=25, ha="right")
        ax1.grid(axis="y", alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(x, coverage, color="#f58518", marker="o", label="Joint 95% coverage")
        ax2.axhline(95.0, color="black", linestyle="--", linewidth=1.0)
        ax2.set_ylabel("Joint coverage (%)")
        ax1.set_title("Q prediction bake-off")
        fig.tight_layout()
        fig.savefig(output_dir / "model_bakeoff_ranking.png", dpi=180)
        plt.close(fig)

    if ekf_rows:
        names = [str(row["model"]) for row in ekf_rows]
        rmse = [float(row["position_rmse_m"]) for row in ekf_rows]
        nees = [float(row["nees_mean"]) for row in ekf_rows]
        x = np.arange(len(names))
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
        axes[0].bar(x, rmse, color="#54a24b")
        axes[0].set_ylabel("RMSE (m)")
        axes[0].set_title("Downstream GPS-bias EKF performance")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(x, nees, color="#e45756")
        axes[1].axhline(3.0, color="black", linestyle="--", linewidth=1.0, label="ideal mean")
        axes[1].set_ylabel("Full NEES mean")
        axes[1].set_xticks(x, names, rotation=25, ha="right")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / "model_bakeoff_ekf_performance.png", dpi=180)
        plt.close(fig)

    if best_states_by_model:
        fig, ax = plt.subplots(figsize=(8.5, 6.4))
        ax.plot(
            data["gt_east_m"][start:stop],
            data["gt_north_m"][start:stop],
            color="black",
            linewidth=2.0,
            label="Ground truth",
        )
        gps = np.flatnonzero(data["gps_available"][start:stop] > 0.5) + start
        ax.scatter(
            data["gps_east_m"][gps],
            data["gps_north_m"][gps],
            s=8,
            alpha=0.35,
            label="F9P GNSS",
        )
        for name, states in best_states_by_model.items():
            ax.plot(states[:, 0], states[:, 1], linewidth=1.3, label=name)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Best model digital-twin trajectory")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "model_bakeoff_best_trajectory.png", dpi=180)
        plt.close(fig)


def run_bakeoff(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int,
    include_optional: bool,
    ekf_top: int,
    max_rows: int | None,
) -> dict[str, object]:
    data = load_aligned(input_path)
    if max_rows is not None:
        data = {name: values[:max_rows] for name, values in data.items()}
    rate_hz = 1.0 / float(np.median(data["dt_s"]))
    history_steps = max(5, int(round(rate_hz)))
    horizon_steps = max(5, int(round(rate_hz)))
    X = build_features(data, history_steps)
    splits = temporal_split_indices(
        len(data["time_s"]), history_steps=history_steps, horizon_steps=horizon_steps
    )
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    motion_calibration = estimate_motion_calibration(data, train)
    residuals = process_residuals(data, motion_calibration)
    target = future_covariance_targets(residuals, horizon_steps)
    low, high = _clip_range(target, train)

    train_median = np.median(target[train], axis=0)
    baseline_test = np.repeat(train_median[None, :], len(test), axis=0)
    baseline_metrics = _regression_metrics(target[test], baseline_test)

    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    bundles: dict[str, object] = {}
    start_time = perf_counter()
    for spec in model_specs(include_optional):
        model_started = perf_counter()
        try:
            bundle = spec.fit(X, target, train, seed)
            prediction = spec.predict(bundle, X[test], low, high)
            validation_prediction = spec.predict(bundle, X[validation], low, high)
        except Exception as exc:
            rows.append(
                {
                    "model": spec.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        factor = _calibration_factors(
            residuals, validation_prediction, validation, horizon_steps
        )
        calibrated_prediction = prediction * factor
        metrics = _regression_metrics(target[test], prediction)
        coverage = _coverage_metrics(residuals, calibrated_prediction, test, horizon_steps)
        predictions_by_model[spec.name] = spec.predict(bundle, X, low, high) * factor
        bundles[spec.name] = bundle

        for column, target_name in enumerate(TARGET_COLUMNS):
            rows.append(
                {
                    "model": spec.name,
                    "status": "ok",
                    "target": target_name,
                    "test_mae": float(metrics["mae"][column]),
                    "test_normalized_mae": float(metrics["normalized_mae"][column]),
                    "test_r2": float(metrics["r2"][column]),
                    "improvement_over_mlp_fraction": math.nan,
                    "improvement_over_median_fraction": float(
                        1.0 - metrics["mae"][column] / baseline_metrics["mae"][column]
                    ),
                    "calibration_factor": float(factor[column]),
                    "runtime_s": float(perf_counter() - model_started),
                }
            )
        coverage_rows.append(
            {
                "model": spec.name,
                "mean_normalized_mae": _mean_normalized_mae(metrics),
                "joint_95_coverage": float(coverage["joint_95_coverage"]),
                "mean_joint_normalized_error_squared": float(
                    coverage["mean_joint_normalized_error_squared"]
                ),
                **{
                    f"{name}_marginal_95_coverage": float(value)
                    for name, value in zip(
                        TARGET_COLUMNS, coverage["marginal_95_coverage"]
                    )
                },
                **{
                    f"{name}_mean_normalized_error_squared": float(value)
                    for name, value in zip(
                        TARGET_COLUMNS, coverage["mean_normalized_error_squared"]
                    )
                },
            }
        )

    mlp_rows = [
        row
        for row in rows
        if row.get("model") == "mlp_log" and row.get("status") == "ok"
    ]
    if mlp_rows:
        mlp_mae = {row["target"]: float(row["test_mae"]) for row in mlp_rows}
        for row in rows:
            if row.get("status") == "ok" and "target" in row:
                row["improvement_over_mlp_fraction"] = float(
                    1.0 - float(row["test_mae"]) / mlp_mae[str(row["target"])]
                )

    ranked = sorted(
        coverage_rows,
        key=lambda row: (
            float(row["mean_normalized_mae"]),
            abs(float(row["joint_95_coverage"]) - 0.95),
        ),
    )
    ekf_names = [row["model"] for row in ranked[: max(0, ekf_top)]]
    if "mlp_log" in predictions_by_model and "mlp_log" not in ekf_names:
        ekf_names.append("mlp_log")

    start, stop = int(test[0]), int(test[-1] + 1)
    validation_start, validation_stop = int(validation[0]), int(validation[-1] + 1)
    ekf_rows: list[dict[str, object]] = []
    best_states_by_model: dict[str, np.ndarray] = {}
    for name in ekf_names:
        q_all = predictions_by_model[name]
        scales, validation_metrics, _ = calibrate_gps_bias_ekf_scales(
            data, q_all, validation_start, validation_stop, motion_calibration
        )
        metrics, states, _, _ = replay_gps_bias_ekf(
            data, q_all, start, stop, motion_calibration, **scales
        )
        best_states_by_model[name] = states
        ekf_rows.append(
            {
                "model": name,
                **{f"scale_{key}": value for key, value in scales.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                **metrics,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "model_bakeoff_metrics.csv", rows)
    _write_csv(output_dir / "model_bakeoff_coverage.csv", coverage_rows)
    _write_csv(output_dir / "model_bakeoff_ekf_metrics.csv", ekf_rows)
    _plot_bakeoff_results(
        output_dir, ranked, ekf_rows, data, start, stop, best_states_by_model
    )
    with (output_dir / "model_bakeoff.pkl").open("wb") as file:
        pickle.dump(
            {
                "schema": "i2nav_model_bakeoff_models_v1",
                "input": str(input_path),
                "feature_columns": FEATURE_COLUMNS,
                "target_columns": TARGET_COLUMNS,
                "bundles": bundles,
            },
            file,
        )

    summary = {
        "schema": "i2nav_model_bakeoff_v1",
        "input": str(input_path),
        "rows": int(len(data["time_s"])),
        "duration_s": float(data["elapsed_s"][-1]),
        "rate_hz": float(rate_hz),
        "split": {name: int(len(indices)) for name, indices in splits.items()},
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "motion_calibration_train_only": motion_calibration,
        "baseline_median_mae": _serialize_vector(baseline_metrics["mae"]),
        "ranked_models": ranked,
        "ekf_evaluated_models": ekf_names,
        "total_runtime_s": float(perf_counter() - start_time),
    }
    (output_dir / "model_bakeoff_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# i2Nav Model Bake-Off",
        "",
        "## Inputs And Targets",
        "",
        "The models receive GPS-independent odometry, IMU, rolling-motion, and timing features.",
        "They predict body-frame process covariance targets: forward, lateral, and heading variance.",
        "",
        "## Ranking",
        "",
        "| Rank | Model | Mean normalized MAE | Joint 95% coverage |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        report.append(
            f"| {rank} | `{row['model']}` | {row['mean_normalized_mae']:.3f} | {100.0 * row['joint_95_coverage']:.1f}% |"
        )
    if ekf_rows:
        report.extend(
            [
                "",
                "## EKF Downstream Check",
                "",
                "| Model | RMSE | NIS mean | NEES mean | Position NEES | Heading NEES |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ekf_rows:
            report.append(
                f"| `{row['model']}` | {row['position_rmse_m']:.3f} m | {row['nis_mean']:.3f} | {row['nees_mean']:.3f} | {row['position_nees_mean']:.3f} | {row['heading_nees_mean']:.3f} |"
            )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "A model only beats the MLP if it improves held-out target error and does not damage EKF consistency.",
        ]
    )
    (output_dir / "model_bakeoff_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--ekf-top", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    run_bakeoff(
        args.input,
        args.output_dir,
        seed=args.seed,
        include_optional=args.include_optional,
        ekf_top=args.ekf_top,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
