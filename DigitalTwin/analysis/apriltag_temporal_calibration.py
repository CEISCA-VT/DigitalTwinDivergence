"""Fit tracked-drive parameters on temporal prefixes and score held-out tails."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from DigitalTwin.analysis.apriltag_fidelity import (
    _activity_sync_offset,
    _integrate,
    _interpolate_states,
    _load_ground_truth,
    _load_prediction,
    _rpe,
    _rmse,
)
from DigitalTwin.kinematics import UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M, wrap_angle


OUTPUT_DIR = Path(
    "DigitalTwin/datasets/analysis/apriltag_temporal_calibration"
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tracking: Path
    telemetry: Path
    train_end_s: float


@dataclass
class PreparedDataset:
    spec: DatasetSpec
    duration_s: float
    gt_time: np.ndarray
    gt_xy: np.ndarray
    gt_heading: np.ndarray
    tracking_status: np.ndarray
    elapsed: np.ndarray
    base_controls: np.ndarray
    corrected_gyro_radps: np.ndarray
    offset_s: float
    sync_correlation: float
    sync_uncertainty_s: float
    reverse_model_axis: bool


DATASETS = (
    DatasetSpec(
        name="trapezoid",
        tracking=Path(
            "DigitalTwin/datasets/analysis/apriltag_trapezoid_metric/"
            "apriltag_still_summary.json"
        ),
        telemetry=Path(
            "raw_logs/telemetry/ugv_t147_interactive_20260805_192736.csv"
        ),
        train_end_s=-1.0,  # Replaced with 75% of the video duration.
    ),
    DatasetSpec(
        name="trial1_square_1p5",
        tracking=Path(
            "DigitalTwin/datasets/analysis/apriltag_trial1_square_1p5_tracking/"
            "apriltag_still_summary.json"
        ),
        telemetry=Path(
            "raw_logs/telemetry/ugv_t147_interactive_20260805_174551.csv"
        ),
        train_end_s=90.0,
    ),
)


def _prepare(spec: DatasetSpec) -> PreparedDataset:
    payload = json.loads(spec.tracking.read_text(encoding="utf-8"))
    duration_s = float(payload["video"]["frame_count"]) / float(
        payload["video"]["fps"]
    )
    train_end_s = 0.75 * duration_s if spec.train_end_s < 0 else spec.train_end_s
    resolved_spec = DatasetSpec(
        spec.name, spec.tracking, spec.telemetry, train_end_s
    )
    gt_time, gt_xy, gt_heading, _, tracking_status = _load_ground_truth(
        spec.tracking, ((0.0, duration_s),)
    )
    prediction = _load_prediction(
        spec.telemetry,
        effective_track_width_m=UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M,
        gyro_weight=0.0,
    )
    elapsed = np.asarray(prediction["elapsed_s"], dtype=float)
    controls = np.asarray(prediction["controls"], dtype=float)
    corrected_gyro_radps = np.asarray(
        prediction["corrected_gyro_radps"], dtype=float
    )
    train = gt_time <= train_end_s
    offset, correlation, uncertainty = _activity_sync_offset(
        gt_time[train], gt_xy[train], gt_heading[train], elapsed, controls
    )
    first_motion = np.flatnonzero(np.abs(controls[:, 0]) > 0.02)
    reverse_model_axis = bool(
        len(first_motion) and controls[int(first_motion[0]), 0] < 0.0
    )
    return PreparedDataset(
        spec=resolved_spec,
        duration_s=duration_s,
        gt_time=gt_time,
        gt_xy=gt_xy,
        gt_heading=gt_heading,
        tracking_status=tracking_status,
        elapsed=elapsed,
        base_controls=controls,
        corrected_gyro_radps=corrected_gyro_radps,
        offset_s=offset,
        sync_correlation=correlation,
        sync_uncertainty_s=uncertainty,
        reverse_model_axis=reverse_model_axis,
    )


def _candidate_controls(
    base: np.ndarray,
    corrected_gyro_radps: np.ndarray,
    distance_scale: float,
    clockwise_width_m: float,
    counterclockwise_width_m: float,
    gyro_weight: float,
    gyro_scale: float,
) -> np.ndarray:
    controls = np.asarray(base, dtype=float).copy()
    controls[:, 0] *= distance_scale
    widths = np.where(
        base[:, 1] >= 0.0,
        counterclockwise_width_m,
        clockwise_width_m,
    )
    controls[:, 1] *= (
        distance_scale * UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M / widths
    )
    controls[:, 1] = (
        (1.0 - gyro_weight) * controls[:, 1]
        + gyro_weight * gyro_scale * corrected_gyro_radps
    )
    return controls


def _segmented_path_length(times: np.ndarray, points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    valid = np.diff(times) <= 0.5
    return float(np.linalg.norm(np.diff(points, axis=0)[valid], axis=1).sum())


def _evaluate(
    dataset: PreparedDataset,
    split: str,
    distance_scale: float,
    clockwise_width_m: float,
    counterclockwise_width_m: float,
    gyro_weight: float,
    gyro_scale: float,
) -> dict[str, float | int | str]:
    if split == "train":
        selected = dataset.gt_time <= dataset.spec.train_end_s
    elif split == "validation":
        selected = dataset.gt_time > dataset.spec.train_end_s
    else:
        raise ValueError(f"unknown split {split!r}")
    times = dataset.gt_time[selected]
    truth_xy = dataset.gt_xy[selected]
    truth_heading = dataset.gt_heading[selected]
    statuses = dataset.tracking_status[selected]
    query = times - dataset.offset_s
    valid = (query >= dataset.elapsed[0]) & (query <= dataset.elapsed[-1])
    times = times[valid]
    truth_xy = truth_xy[valid]
    truth_heading = truth_heading[valid]
    statuses = statuses[valid]
    query = query[valid]
    if len(times) < 10:
        raise RuntimeError(f"{dataset.spec.name} {split} has too few aligned samples")

    controls = _candidate_controls(
        dataset.base_controls,
        dataset.corrected_gyro_radps,
        distance_scale,
        clockwise_width_m,
        counterclockwise_width_m,
        gyro_weight,
        gyro_scale,
    )
    anchor_index = int(np.argmin(np.abs(dataset.elapsed - query[0])))
    integration_elapsed = dataset.elapsed[anchor_index:]
    integration_controls = controls[anchor_index:].copy()
    integration_controls[0] = 0.0
    heading_anchor = times <= times[0] + 1.0
    physical_heading = math.atan2(
        float(np.mean(np.sin(truth_heading[heading_anchor]))),
        float(np.mean(np.cos(truth_heading[heading_anchor]))),
    )
    model_heading = wrap_angle(
        physical_heading + (math.pi if dataset.reverse_model_axis else 0.0)
    )
    states = _integrate(
        integration_elapsed,
        integration_controls,
        truth_xy[0],
        model_heading,
    )
    estimate = _interpolate_states(integration_elapsed, states, query)
    estimated_heading = estimate[:, 2] - (
        math.pi if dataset.reverse_model_axis else 0.0
    )
    position_error = np.linalg.norm(estimate[:, :2] - truth_xy, axis=1)
    heading_error = np.asarray(
        [
            abs(wrap_angle(float(predicted - observed)))
            for predicted, observed in zip(estimated_heading, truth_heading)
        ]
    )
    interval_ids = np.ones(len(times), dtype=int)
    rpe = _rpe(times, truth_xy, estimate[:, :2], interval_ids)
    truth_path = _segmented_path_length(times, truth_xy)
    estimated_path = _segmented_path_length(times, estimate[:, :2])
    return {
        "dataset": dataset.spec.name,
        "split": split,
        "samples": int(len(times)),
        "decoded_fraction": float(np.mean(statuses == "decoded")),
        "position_rmse_m": _rmse(position_error),
        "position_median_m": float(np.median(position_error)),
        "position_p95_m": float(np.quantile(position_error, 0.95)),
        "within_0p10_fraction": float(np.mean(position_error <= 0.10)),
        "within_0p25_fraction": float(np.mean(position_error <= 0.25)),
        "rpe_1s_rmse_m": _rmse(rpe),
        "heading_mae_deg": float(np.degrees(np.mean(heading_error))),
        "heading_within_30_fraction": float(
            np.mean(np.degrees(heading_error) <= 30.0)
        ),
        "truth_path_m": truth_path,
        "estimated_path_m": estimated_path,
        "path_ratio": estimated_path / truth_path if truth_path > 1e-9 else math.nan,
    }


def _aggregate(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    return {
        "position_rmse_m": mean("position_rmse_m"),
        "rpe_1s_rmse_m": mean("rpe_1s_rmse_m"),
        "heading_mae_deg": mean("heading_mae_deg"),
        "within_0p10_fraction": mean("within_0p10_fraction"),
        "within_0p25_fraction": mean("within_0p25_fraction"),
        "heading_within_30_fraction": mean("heading_within_30_fraction"),
        "path_ratio_absolute_error": mean("path_ratio") - 1.0,
        "path_agreement_fraction": float(
            np.mean([1.0 - abs(1.0 - float(row["path_ratio"])) for row in rows])
        ),
    }


def _loss(metrics: dict[str, float]) -> float:
    return float(
        0.5 * metrics["position_rmse_m"] / 0.25
        + metrics["rpe_1s_rmse_m"] / 0.05
        + metrics["heading_mae_deg"] / 30.0
        + abs(metrics["path_ratio_absolute_error"]) / 0.10
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--dataset-name", default="custom")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--include-imu-grid",
        action="store_true",
        help="Also fit a bounded bias-corrected gyro contribution and sign.",
    )
    args = parser.parse_args()
    if (args.tracking is None) != (args.telemetry is None):
        parser.error("--tracking and --telemetry must be supplied together")
    if not 0.5 <= args.train_fraction < 1.0:
        parser.error("--train-fraction must be in [0.5, 1.0)")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = DATASETS
    if args.tracking is not None:
        payload = json.loads(args.tracking.read_text(encoding="utf-8"))
        duration_s = float(payload["video"]["frame_count"]) / float(
            payload["video"]["fps"]
        )
        specs = (
            DatasetSpec(
                name=args.dataset_name,
                tracking=args.tracking,
                telemetry=args.telemetry,
                train_end_s=args.train_fraction * duration_s,
            ),
        )
    datasets = [_prepare(spec) for spec in specs]
    candidates = []
    distance_scales = np.arange(0.95, 1.0501, 0.025)
    turn_widths = np.arange(0.150, 0.2801, 0.010)
    gyro_candidates = (
        ((0.0, 1.0), (0.10, -1.0), (0.10, 1.0), (0.20, -1.0), (0.20, 1.0))
        if args.include_imu_grid
        else ((0.0, 1.0),)
    )
    for distance_scale in distance_scales:
        for clockwise_width_m in turn_widths:
            for counterclockwise_width_m in turn_widths:
                for gyro_weight, gyro_scale in gyro_candidates:
                    train_rows = [
                        _evaluate(
                            dataset,
                            "train",
                            float(distance_scale),
                            float(clockwise_width_m),
                            float(counterclockwise_width_m),
                            float(gyro_weight),
                            float(gyro_scale),
                        )
                        for dataset in datasets
                    ]
                    metrics = _aggregate(train_rows)
                    candidates.append(
                        {
                            "distance_scale": float(distance_scale),
                            "clockwise_width_m": float(clockwise_width_m),
                            "counterclockwise_width_m": float(counterclockwise_width_m),
                            "gyro_weight": float(gyro_weight),
                            "gyro_scale": float(gyro_scale),
                            "training_loss": _loss(metrics),
                            **{f"training_{key}": value for key, value in metrics.items()},
                        }
                    )
    best = min(candidates, key=lambda row: row["training_loss"])
    baseline_parameters = {
        "distance_scale": 1.0,
        "clockwise_width_m": UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M,
        "counterclockwise_width_m": UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M,
        "gyro_weight": 0.0,
        "gyro_scale": 1.0,
    }
    fitted_parameters = {
        key: float(best[key]) for key in baseline_parameters
    }

    result_rows = []
    aggregates = {}
    for model_name, parameters in (
        ("baseline", baseline_parameters),
        ("fitted", fitted_parameters),
    ):
        for split in ("train", "validation"):
            rows = [
                _evaluate(dataset, split, **parameters) for dataset in datasets
            ]
            for row in rows:
                result_rows.append({"model": model_name, **row})
            aggregates[f"{model_name}_{split}"] = _aggregate(rows)

    candidate_fields = list(candidates[0])
    with (output_dir / "calibration_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidates)
    result_fields = list(result_rows[0])
    with (output_dir / "split_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(result_rows)

    payload = {
        "schema": "ugv01_apriltag_temporal_calibration_v1",
        "split_policy": {
            dataset.spec.name: {
                "train_interval_s": [0.0, dataset.spec.train_end_s],
                "validation_interval_s": [dataset.spec.train_end_s, dataset.duration_s],
                "synchronization_offset_s_fitted_on_training": dataset.offset_s,
                "synchronization_correlation": dataset.sync_correlation,
                "synchronization_uncertainty_s": dataset.sync_uncertainty_s,
            }
            for dataset in datasets
        },
        "selection_rule": (
            "Minimum equal-run mean normalized training loss using position ATE, "
            "1-second RPE, heading MAE, and path-ratio error. Validation was not "
            "used for parameter selection."
        ),
        "baseline_parameters": baseline_parameters,
        "fitted_parameters": fitted_parameters,
        "aggregates": aggregates,
        "per_dataset_results": result_rows,
    }
    (output_dir / "temporal_calibration_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    before = aggregates["baseline_validation"]
    after = aggregates["fitted_validation"]
    report = [
        "# AprilTag Temporal Calibration",
        "",
        "Parameters were selected only on the specified training prefixes. The "
        "held-out tails were evaluated once after parameter selection.",
        "",
        "## Frozen Parameters",
        "",
        f"- Distance scale: `{fitted_parameters['distance_scale']:.3f}`",
        f"- Effective clockwise width: `{fitted_parameters['clockwise_width_m']:.3f} m`",
        f"- Effective counterclockwise width: `{fitted_parameters['counterclockwise_width_m']:.3f} m`",
        f"- Bias-corrected gyro weight: `{fitted_parameters['gyro_weight']:.2f}`",
        f"- Gyro sign/scale: `{fitted_parameters['gyro_scale']:.1f}`",
        "",
        "## Equal-Run Mean Validation Results",
        "",
        "| Metric | Baseline | Fitted |",
        "|---|---:|---:|",
        f"| Position RMSE | {before['position_rmse_m']:.3f} m | {after['position_rmse_m']:.3f} m |",
        f"| 1-second RPE RMSE | {before['rpe_1s_rmse_m']:.3f} m | {after['rpe_1s_rmse_m']:.3f} m |",
        f"| Heading MAE | {before['heading_mae_deg']:.1f} deg | {after['heading_mae_deg']:.1f} deg |",
        f"| Within 10 cm | {100*before['within_0p10_fraction']:.1f}% | {100*after['within_0p10_fraction']:.1f}% |",
        f"| Within 25 cm | {100*before['within_0p25_fraction']:.1f}% | {100*after['within_0p25_fraction']:.1f}% |",
        f"| Heading within 30 deg | {100*before['heading_within_30_fraction']:.1f}% | {100*after['heading_within_30_fraction']:.1f}% |",
        f"| Path-length agreement | {100*before['path_agreement_fraction']:.1f}% | {100*after['path_agreement_fraction']:.1f}% |",
        "",
        "## Per-Dataset Validation",
        "",
        "| Model | Dataset | Position RMSE | Heading MAE | Within 25 cm | Path ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        if row["split"] != "validation":
            continue
        report.append(
            f"| {row['model']} | {row['dataset']} | "
            f"{float(row['position_rmse_m']):.3f} m | "
            f"{float(row['heading_mae_deg']):.1f} deg | "
            f"{100*float(row['within_0p25_fraction']):.1f}% | "
            f"{float(row['path_ratio']):.3f} |"
        )
    report.extend(
        [
            "",
            "## Limitation",
            "",
            "This is a temporal holdout rather than a fully independent future run. "
            "Adjacent portions share the same surface, camera placement, and rover "
            "session, so a new untouched recording remains necessary for final validation.",
            "",
        ]
    )
    (output_dir / "temporal_calibration_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    labels = ["Position RMSE\n(m)", "Heading MAE\n(deg / 100)", "1 s RPE\n(m)"]
    baseline_values = [
        before["position_rmse_m"],
        before["heading_mae_deg"] / 100.0,
        before["rpe_1s_rmse_m"],
    ]
    fitted_values = [
        after["position_rmse_m"],
        after["heading_mae_deg"] / 100.0,
        after["rpe_1s_rmse_m"],
    ]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(x - 0.18, baseline_values, 0.36, label="Baseline", color="#d95d39")
    axis.bar(x + 0.18, fitted_values, 0.36, label="Training-selected", color="#16697a")
    axis.set_xticks(x, labels)
    axis.set_title("Held-out temporal validation: baseline vs fitted model")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_comparison.png", dpi=180)
    plt.close(figure)

    print(output_dir / "temporal_calibration_report.md")
    print(json.dumps({"fitted_parameters": fitted_parameters, "validation": after}, indent=2))


if __name__ == "__main__":
    main()
