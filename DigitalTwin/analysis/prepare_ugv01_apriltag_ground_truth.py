"""Package UGV01 AprilTag fidelity samples in an i2Nav-like aligned format.

The output mirrors the public i2Nav preparation artifacts closely enough for
inspection and downstream plotting: `aligned_samples.csv`,
`aligned_samples.npz`, and `preparation_summary.json`. Ground truth is the
AprilTag trajectory, while odometry-like features are derived from the already
aligned digital-twin prediction. GPS fields are intentionally marked
unavailable because the current pilot clips did not include synchronized GPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from DigitalTwin.kinematics import wrap_angle


DEFAULT_INPUT = Path(
    "DigitalTwin/datasets/analysis/validation_carpet_142023_candidate/"
    "aligned_fidelity_samples.csv"
)
DEFAULT_SUMMARY = Path(
    "DigitalTwin/datasets/analysis/validation_carpet_142023_candidate/"
    "fidelity_summary.json"
)
DEFAULT_OUTPUT = Path("DigitalTwin/datasets/analysis/ugv01_apriltag_carpet_142023")

OUTPUT_COLUMNS = (
    "time_s",
    "elapsed_s",
    "dt_s",
    "odo_forward_mps",
    "odo_lateral_mps",
    "wheel_speed_std_mps",
    "steering_abs_mean_rad",
    "steering_std_rad",
    "imu_yaw_rate_radps",
    "imu_yaw_rate_std_radps",
    "imu_accel_norm_std_mps2",
    "imu_accel_z_std_mps2",
    "gt_east_m",
    "gt_north_m",
    "gt_heading_rad",
    "gt_speed_mps",
    "gps_available",
    "gps_east_m",
    "gps_north_m",
    "gps_sigma_east_m",
    "gps_sigma_north_m",
    "twin_east_m",
    "twin_north_m",
    "twin_heading_rad",
    "position_error_m",
    "heading_error_rad",
    "tracking_status_code",
    "interval_id",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 3:
        raise RuntimeError(f"{path} has too few rows for aligned export")
    return rows


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def _gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, times)


def prepare(input_csv: Path, summary_json: Path | None = None) -> dict[str, np.ndarray]:
    rows = _read_rows(input_csv)
    video_time = np.asarray([_f(row, "video_time_s") for row in rows], dtype=float)
    telemetry_elapsed = np.asarray(
        [_f(row, "telemetry_elapsed_s") for row in rows], dtype=float
    )
    time_s = video_time
    elapsed_s = time_s - time_s[0]
    dt_s = np.diff(elapsed_s, prepend=elapsed_s[0])
    if len(dt_s) > 1:
        dt_s[0] = float(np.median(dt_s[1:]))
    dt_s = np.clip(dt_s, 1e-3, 1.0)

    gt_east = np.asarray([_f(row, "truth_x_m") for row in rows], dtype=float)
    gt_north = np.asarray([_f(row, "truth_y_m") for row in rows], dtype=float)
    gt_heading = np.radians(
        np.asarray([_f(row, "truth_heading_deg") for row in rows], dtype=float)
    )
    twin_east = np.asarray([_f(row, "twin_x_m") for row in rows], dtype=float)
    twin_north = np.asarray([_f(row, "twin_y_m") for row in rows], dtype=float)
    twin_heading = np.radians(
        np.asarray([_f(row, "twin_heading_deg") for row in rows], dtype=float)
    )
    position_error = np.asarray(
        [_f(row, "position_error_m") for row in rows], dtype=float
    )
    heading_error = np.radians(
        np.asarray([_f(row, "heading_error_deg") for row in rows], dtype=float)
    )
    interval_id = np.asarray([_f(row, "interval") for row in rows], dtype=float)
    status = np.asarray(
        [0.0 if row.get("tracking_status") == "decoded" else 1.0 for row in rows],
        dtype=float,
    )

    twin_dx = _gradient(twin_east, elapsed_s)
    twin_dy = _gradient(twin_north, elapsed_s)
    heading_unwrapped = np.unwrap(twin_heading)
    twin_yaw_rate = _gradient(heading_unwrapped, elapsed_s)
    cos_h = np.cos(twin_heading)
    sin_h = np.sin(twin_heading)
    odo_forward = twin_dx * cos_h + twin_dy * sin_h
    odo_lateral = -twin_dx * sin_h + twin_dy * cos_h

    gt_dx = _gradient(gt_east, elapsed_s)
    gt_dy = _gradient(gt_north, elapsed_s)
    gt_speed = np.hypot(gt_dx, gt_dy)

    columns = {
        "time_s": time_s,
        "elapsed_s": elapsed_s,
        "dt_s": dt_s,
        "odo_forward_mps": odo_forward,
        "odo_lateral_mps": odo_lateral,
        "wheel_speed_std_mps": np.zeros(len(rows)),
        "steering_abs_mean_rad": np.zeros(len(rows)),
        "steering_std_rad": np.zeros(len(rows)),
        "imu_yaw_rate_radps": twin_yaw_rate,
        "imu_yaw_rate_std_radps": np.zeros(len(rows)),
        "imu_accel_norm_std_mps2": np.zeros(len(rows)),
        "imu_accel_z_std_mps2": np.zeros(len(rows)),
        "gt_east_m": gt_east,
        "gt_north_m": gt_north,
        "gt_heading_rad": np.asarray([wrap_angle(value) for value in gt_heading]),
        "gt_speed_mps": gt_speed,
        "gps_available": np.zeros(len(rows)),
        "gps_east_m": np.full(len(rows), np.nan),
        "gps_north_m": np.full(len(rows), np.nan),
        "gps_sigma_east_m": np.full(len(rows), np.nan),
        "gps_sigma_north_m": np.full(len(rows), np.nan),
        "twin_east_m": twin_east,
        "twin_north_m": twin_north,
        "twin_heading_rad": np.asarray([wrap_angle(value) for value in twin_heading]),
        "position_error_m": position_error,
        "heading_error_rad": heading_error,
        "tracking_status_code": status,
        "interval_id": interval_id,
    }
    if summary_json and summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        columns["source_estimated_video_minus_telemetry_offset_s"] = np.full(
            len(rows), float(summary.get("estimated_video_minus_telemetry_offset_s", math.nan))
        )
        columns["source_synchronization_uncertainty_s"] = np.full(
            len(rows), float(summary.get("synchronization_uncertainty_s", math.nan))
        )
        columns["telemetry_elapsed_s"] = telemetry_elapsed
    return columns


def write_outputs(
    columns: dict[str, np.ndarray],
    output_dir: Path,
    *,
    input_csv: Path,
    summary_json: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "aligned_samples.npz", **columns)
    with (output_dir / "aligned_samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(OUTPUT_COLUMNS)
        for index in range(len(columns["time_s"])):
            writer.writerow([float(columns[column][index]) for column in OUTPUT_COLUMNS])

    duration = float(columns["elapsed_s"][-1] - columns["elapsed_s"][0])
    dt = np.diff(columns["elapsed_s"])
    summary = {
        "schema": "ugv01_apriltag_aligned_v1",
        "source_aligned_fidelity_samples": str(input_csv),
        "source_fidelity_summary": str(summary_json) if summary_json else None,
        "coordinate_frame": "AprilTag world frame treated as ENU-style local coordinates: x=east-like, y=north-like, heading counterclockwise in radians",
        "ground_truth_role": "AprilTag trajectory labels and evaluation reference",
        "gnss_role": "unavailable in this pilot export; gps_available is 0 for every row",
        "odometry_feature_role": "derived from aligned digital-twin prediction, not raw i2Nav wheel odometry",
        "rows": int(len(columns["time_s"])),
        "duration_s": duration,
        "median_rate_hz": float(1.0 / np.median(dt)) if len(dt) else None,
        "gps_updates": 0,
        "mean_position_error_m": float(np.mean(columns["position_error_m"])),
        "position_rmse_m": float(np.sqrt(np.mean(columns["position_error_m"] ** 2))),
        "heading_mae_deg": float(np.degrees(np.mean(np.abs(columns["heading_error_rad"])))),
        "limitations": [
            "pilot export only; not final synchronized GPS plus AprilTag validation",
            "camera/telemetry offset is inherited from fidelity analysis",
            "feature columns are i2Nav-like for tooling compatibility, not sensor-identical",
        ],
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# UGV01 AprilTag Aligned Ground-Truth Export",
        "",
        f"- Rows: **{summary['rows']}**",
        f"- Duration: **{summary['duration_s']:.2f} s**",
        f"- Median rate: **{summary['median_rate_hz']:.2f} Hz**",
        f"- GPS updates: **{summary['gps_updates']}**",
        f"- Position RMSE: **{summary['position_rmse_m']:.3f} m**",
        f"- Heading MAE: **{summary['heading_mae_deg']:.1f} deg**",
        "",
        "This package mirrors the i2Nav `aligned_samples` layout for convenient",
        "inspection and downstream tooling. It is not a replacement for a final",
        "synchronized UGV01 run with telemetry, GPS, AprilTag video, and a hardware",
        "or visible sync event.",
        "",
        "## Files",
        "",
        "- `aligned_samples.csv`: human-readable aligned table.",
        "- `aligned_samples.npz`: NumPy package with the same columns and extra metadata arrays.",
        "- `preparation_summary.json`: machine-readable provenance and limitations.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    columns = prepare(args.input, args.summary)
    write_outputs(
        columns,
        args.output_dir,
        input_csv=args.input,
        summary_json=args.summary,
    )


if __name__ == "__main__":
    main()
