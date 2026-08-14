"""Align an i2Nav-Robot raw sequence for digital-twin uncertainty studies.

The exported table is deliberately modest: wheel odometry and incremental IMU
data form causal model inputs, F9P GNSS is the EKF measurement, and the
navigation-grade trajectory is retained only for labels and final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from DigitalTwin.kinematics import wrap_angle
from DigitalTwin.telemetry import gps_to_local_xy


DEFAULT_INPUT = Path("public_datasets/im2nav/playground00")
DEFAULT_OUTPUT = Path("DigitalTwin/datasets/analysis/i2nav_playground00")
ODO_TO_IMU_TIME_OFFSET_S = -0.01

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
)


def _load_numeric(path: Path, expected_columns: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, dtype=float)
    if values.ndim != 2 or values.shape[1] != expected_columns:
        raise RuntimeError(
            f"{path} has shape {values.shape}; expected N x {expected_columns}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path} contains non-finite values")
    return values


def _interp_columns(query: np.ndarray, source: np.ndarray, columns: list[int]) -> np.ndarray:
    return np.column_stack(
        [np.interp(query, source[:, 0], source[:, column]) for column in columns]
    )


def _angle_interp(query: np.ndarray, time: np.ndarray, angle_rad: np.ndarray) -> np.ndarray:
    return np.asarray(
        [wrap_angle(value) for value in np.interp(query, time, np.unwrap(angle_rad))],
        dtype=float,
    )


def _window_slices(sample_times: np.ndarray, grid: np.ndarray, step_s: float):
    left = np.searchsorted(sample_times, grid - step_s, side="right")
    right = np.searchsorted(sample_times, grid, side="right")
    for start, stop in zip(left, right):
        yield slice(int(start), int(stop))


def _aggregate_imu(imu: np.ndarray, grid: np.ndarray, step_s: float) -> dict[str, np.ndarray]:
    sample_dt = np.diff(imu[:, 0], prepend=imu[0, 0] - np.median(np.diff(imu[:, 0])))
    sample_dt = np.clip(sample_dt, 1e-4, 0.1)
    gyro_z = -imu[:, 3] / sample_dt  # front-right-down yaw -> ENU heading rate
    acceleration = imu[:, 4:7] / sample_dt[:, None]

    yaw_rate: list[float] = []
    yaw_std: list[float] = []
    accel_norm_std: list[float] = []
    accel_z_std: list[float] = []
    previous = 0.0
    for window in _window_slices(imu[:, 0], grid, step_s):
        if window.stop <= window.start:
            yaw_rate.append(previous)
            yaw_std.append(0.0)
            accel_norm_std.append(0.0)
            accel_z_std.append(0.0)
            continue
        values = gyro_z[window]
        previous = float(-np.sum(imu[window, 3]) / step_s)
        yaw_rate.append(previous)
        yaw_std.append(float(np.std(values)))
        accel_norm_std.append(float(np.std(np.linalg.norm(acceleration[window], axis=1))))
        accel_z_std.append(float(np.std(acceleration[window, 2])))
    return {
        "imu_yaw_rate_radps": np.asarray(yaw_rate),
        "imu_yaw_rate_std_radps": np.asarray(yaw_std),
        "imu_accel_norm_std_mps2": np.asarray(accel_norm_std),
        "imu_accel_z_std_mps2": np.asarray(accel_z_std),
    }


def _odometry_features(odo: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    values = _interp_columns(grid, odo, list(range(1, 9)))
    wheel_speed = values[:, :4]
    wheel_angle = values[:, 4:]
    return {
        "odo_forward_mps": np.mean(wheel_speed * np.cos(wheel_angle), axis=1),
        "odo_lateral_mps": np.mean(wheel_speed * np.sin(wheel_angle), axis=1),
        "wheel_speed_std_mps": np.std(wheel_speed, axis=1),
        "steering_abs_mean_rad": np.mean(np.abs(wheel_angle), axis=1),
        "steering_std_rad": np.std(wheel_angle, axis=1),
    }


def _ground_truth(gt: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    # Source navigation coordinates are NED. The EKF uses east, north, ENU yaw.
    position_velocity = _interp_columns(grid, gt, [1, 2, 4, 5])
    yaw_ned = np.deg2rad(gt[:, 9])
    heading_enu = _angle_interp(grid, gt[:, 0], math.pi / 2.0 - yaw_ned)
    return {
        "gt_east_m": position_velocity[:, 1],
        "gt_north_m": position_velocity[:, 0],
        "gt_heading_rad": heading_enu,
        "gt_speed_mps": np.linalg.norm(position_velocity[:, 2:4], axis=1),
    }


def _gnss_measurements(
    gnss: np.ndarray,
    gt: np.ndarray,
    grid: np.ndarray,
    step_s: float,
) -> dict[str, np.ndarray]:
    origin_lat, origin_lon = float(gnss[0, 1]), float(gnss[0, 2])
    local = np.asarray(
        [gps_to_local_xy(row[1], row[2], origin_lat, origin_lon) for row in gnss],
        dtype=float,
    )
    first_gt = _interp_columns(np.asarray([gnss[0, 0]]), gt, [1, 2])[0]
    local[:, 0] += first_gt[1]
    local[:, 1] += first_gt[0]

    nearest = np.searchsorted(grid, gnss[:, 0])
    nearest = np.clip(nearest, 0, len(grid) - 1)
    previous = np.clip(nearest - 1, 0, len(grid) - 1)
    use_previous = np.abs(grid[previous] - gnss[:, 0]) < np.abs(grid[nearest] - gnss[:, 0])
    nearest[use_previous] = previous[use_previous]

    available = np.zeros(len(grid), dtype=float)
    east = np.full(len(grid), np.nan)
    north = np.full(len(grid), np.nan)
    sigma_east = np.full(len(grid), np.nan)
    sigma_north = np.full(len(grid), np.nan)
    for source_index, target_index in enumerate(nearest):
        if abs(float(grid[target_index] - gnss[source_index, 0])) > 0.55 * step_s:
            continue
        available[target_index] = 1.0
        east[target_index], north[target_index] = local[source_index]
        sigma_north[target_index] = max(float(gnss[source_index, 4]), 0.05)
        sigma_east[target_index] = max(float(gnss[source_index, 5]), 0.05)
    return {
        "gps_available": available,
        "gps_east_m": east,
        "gps_north_m": north,
        "gps_sigma_east_m": sigma_east,
        "gps_sigma_north_m": sigma_north,
    }


def prepare_sequence(
    input_dir: Path,
    *,
    rate_hz: float = 10.0,
    max_duration_s: float | None = None,
) -> dict[str, np.ndarray]:
    sequence = input_dir.name
    imu = _load_numeric(input_dir / f"{sequence}_ADIS16465_IMU.txt", 7)
    odo = _load_numeric(input_dir / f"{sequence}_RANGER_ODO.txt", 9)
    gnss = _load_numeric(input_dir / f"{sequence}_F9P_GNSS.pos", 7)
    gt = _load_numeric(input_dir / f"{sequence}_groundtruth.nav", 10)
    odo = odo.copy()
    odo[:, 0] += ODO_TO_IMU_TIME_OFFSET_S

    step_s = 1.0 / rate_hz
    start = max(imu[0, 0], odo[0, 0], gnss[0, 0], gt[0, 0])
    stop = min(imu[-1, 0], odo[-1, 0], gnss[-1, 0], gt[-1, 0])
    start = math.ceil(start / step_s) * step_s
    stop = math.floor(stop / step_s) * step_s
    if max_duration_s is not None:
        stop = min(stop, start + max_duration_s)
    if stop - start < 10.0:
        raise RuntimeError("the selected sensor overlap is shorter than 10 seconds")
    grid = np.arange(start, stop + 0.25 * step_s, step_s)

    columns: dict[str, np.ndarray] = {
        "time_s": grid,
        "elapsed_s": grid - grid[0],
        "dt_s": np.full(len(grid), step_s),
    }
    columns.update(_odometry_features(odo, grid))
    columns.update(_aggregate_imu(imu, grid, step_s))
    columns.update(_ground_truth(gt, grid))
    columns.update(_gnss_measurements(gnss, gt, grid, step_s))
    return columns


def write_outputs(columns: dict[str, np.ndarray], output_dir: Path, source: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "aligned_samples.npz", **columns)
    with (output_dir / "aligned_samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(OUTPUT_COLUMNS)
        for index in range(len(columns["time_s"])):
            writer.writerow([float(columns[column][index]) for column in OUTPUT_COLUMNS])

    gps_count = int(np.sum(columns["gps_available"]))
    duration = float(columns["elapsed_s"][-1])
    summary = {
        "schema": "i2nav_robot_aligned_v1",
        "source_directory": str(source),
        "coordinate_frame": "ENU: x=east, y=north, heading counterclockwise from east",
        "ground_truth_role": "labels and evaluation only",
        "gnss_role": "EKF measurement only",
        "rows": len(columns["time_s"]),
        "duration_s": duration,
        "rate_hz": 1.0 / float(columns["dt_s"][0]),
        "gnss_updates": gps_count,
        "gnss_rate_hz": gps_count / duration,
        "odo_to_imu_time_offset_s": ODO_TO_IMU_TIME_OFFSET_S,
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--max-duration-s", type=float)
    args = parser.parse_args()
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be positive")
    columns = prepare_sequence(
        args.input_dir,
        rate_hz=args.rate_hz,
        max_duration_s=args.max_duration_s,
    )
    write_outputs(columns, args.output_dir, args.input_dir)


if __name__ == "__main__":
    main()
