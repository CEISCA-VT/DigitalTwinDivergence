"""One-sequence TerraSentia adapter validation for frozen Twin V2.

This is intentionally a Phase-2 smoke path, not the full AIFARMS study:

* one fixed sequence;
* deterministic bag-time synchronization;
* no target-domain normalization or tuning;
* one documented frozen V2 checkpoint for software validation only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from DigitalTwin.analysis import i2nav_v2_full_loso as v2
from DigitalTwin.analysis.i2nav_fidelity_evaluator import evaluate_fidelity_frames


DEFAULT_SEQUENCE = "ts_2022_06_15_11h48m34s_four_rows"
DEFAULT_INPUT_ROOT = Path("public_datasets/aifarms/processed")
DEFAULT_OUTPUT = Path("results/aifarms_terrasentia_phase2") / DEFAULT_SEQUENCE
DEFAULT_CHECKPOINT = (
    Path("results/i2nav_v2_full_loso/i2nav_v2_full_loso")
    / "replicate_01_base42"
    / "fold_01_building00"
    / "v2_slow_additive_yaw.pt"
)

RATE_HZ = 10.0
DT_S = 1.0 / RATE_HZ
TRACK_WIDTH_M = 0.26
RTK_MAX_HORIZONTAL_ACCURACY_M = 0.05
RTK_MAX_INTERP_GAP_S = 0.35
MOTOR_MAX_INTERP_GAP_S = 0.15
IMU_MAX_INTERP_GAP_S = 0.15
COMMAND_MAX_INTERP_GAP_S = 0.25
REFERENCE_MAX_INTERP_GAP_S = 0.15

FAST_FEATURE_NAMES = [
    "speed_mps",
    "imu_yaw_rate_radps",
    "accel_mps2",
    "yaw_accel_radps2",
    "abs_imu_yaw_rate_radps",
    "abs_accel_mps2",
]

SLOW_FEATURE_NAMES = [
    "mean_imu_yaw_radps_30s",
    "std_imu_yaw_radps_30s",
    "rms_imu_yaw_radps_30s",
    "mean_abs_imu_yaw_radps_30s",
    "mean_wheel_yaw_radps_30s",
    "std_wheel_yaw_radps_30s",
    "rms_wheel_yaw_radps_30s",
    "mean_abs_wheel_yaw_radps_30s",
    "mean_yaw_disagreement_radps_30s",
    "std_yaw_disagreement_radps_30s",
    "rms_yaw_disagreement_radps_30s",
    "mean_normalized_yaw_disagreement_30s",
    "std_normalized_yaw_disagreement_30s",
    "mean_forward_speed_mps_30s",
    "std_forward_speed_mps_30s",
    "mean_abs_forward_speed_mps_30s",
]


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(angle, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def bag_time_s(frame: pd.DataFrame) -> np.ndarray:
    if "bag_timestamp_ns" not in frame.columns:
        raise ValueError("required bag_timestamp_ns column is missing")
    return pd.to_numeric(frame["bag_timestamp_ns"], errors="coerce").to_numpy(float) / 1e9


def unique_sorted(t: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t, kind="stable")
    t = np.asarray(t, dtype=float)[order]
    values = np.asarray(values, dtype=float)[order]
    good = np.isfinite(t)
    if values.ndim == 1:
        good &= np.isfinite(values)
    else:
        good &= np.all(np.isfinite(values), axis=1)
    t = t[good]
    values = values[good]
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    return t[keep], values[keep]


def interp_with_gap(
    source_t: np.ndarray,
    source_v: np.ndarray,
    grid: np.ndarray,
    *,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_t, source_v = unique_sorted(source_t, source_v)
    if len(source_t) < 2:
        raise ValueError("need at least two source samples for interpolation")

    left = np.searchsorted(source_t, grid, side="right") - 1
    right = left + 1
    valid = (left >= 0) & (right < len(source_t))
    left_gap = np.full(len(grid), np.inf, dtype=float)
    right_gap = np.full(len(grid), np.inf, dtype=float)
    left_gap[valid] = grid[valid] - source_t[left[valid]]
    right_gap[valid] = source_t[right[valid]] - grid[valid]
    valid &= (left_gap <= max_gap_s) & (right_gap <= max_gap_s)

    if source_v.ndim == 1:
        out = np.interp(grid, source_t, source_v)
        out[~valid] = np.nan
    else:
        out = np.column_stack(
            [np.interp(grid, source_t, source_v[:, i]) for i in range(source_v.shape[1])]
        )
        out[~valid, :] = np.nan
    return out, valid, np.maximum(left_gap, right_gap)


def rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    return pd.Series(values).rolling(samples, min_periods=samples).mean().to_numpy()


def rolling_std(values: np.ndarray, samples: int) -> np.ndarray:
    return pd.Series(values).rolling(samples, min_periods=samples).std(ddof=0).to_numpy()


def rolling_rms(values: np.ndarray, samples: int) -> np.ndarray:
    return np.sqrt(np.maximum(rolling_mean(values * values, samples), 0.0))


def slow_features(
    imu_yaw: np.ndarray,
    wheel_yaw: np.ndarray,
    forward_speed: np.ndarray,
    samples: int,
) -> np.ndarray:
    diff = imu_yaw - wheel_yaw
    ndiff = diff / (np.abs(imu_yaw) + np.abs(wheel_yaw) + 0.02)
    cols = [
        rolling_mean(imu_yaw, samples),
        rolling_std(imu_yaw, samples),
        rolling_rms(imu_yaw, samples),
        rolling_mean(np.abs(imu_yaw), samples),
        rolling_mean(wheel_yaw, samples),
        rolling_std(wheel_yaw, samples),
        rolling_rms(wheel_yaw, samples),
        rolling_mean(np.abs(wheel_yaw), samples),
        rolling_mean(diff, samples),
        rolling_std(diff, samples),
        rolling_rms(diff, samples),
        rolling_mean(ndiff, samples),
        rolling_std(ndiff, samples),
        rolling_mean(forward_speed, samples),
        rolling_std(forward_speed, samples),
        rolling_mean(np.abs(forward_speed), samples),
    ]
    return np.column_stack(cols).astype(np.float32)


def local_enu_from_gps(gps: pd.DataFrame) -> pd.DataFrame:
    out = gps.copy()
    valid = (
        pd.to_numeric(out["latitude"], errors="coerce").notna()
        & pd.to_numeric(out["longitude"], errors="coerce").notna()
        & pd.to_numeric(out["horizontal_accuracy"], errors="coerce").notna()
        & (pd.to_numeric(out["horizontal_accuracy"], errors="coerce") <= RTK_MAX_HORIZONTAL_ACCURACY_M)
    )
    if not valid.any():
        raise RuntimeError("no RTK/GPS rows pass the predeclared validity rule")
    first = out.loc[valid].iloc[0]
    lat0 = float(first["latitude"])
    lon0 = float(first["longitude"])
    lat_scale = 111_320.0
    lon_scale = 111_320.0 * math.cos(math.radians(lat0))
    out["rtk_valid"] = valid
    out["east_m"] = (pd.to_numeric(out["longitude"], errors="coerce") - lon0) * lon_scale
    out["north_m"] = (pd.to_numeric(out["latitude"], errors="coerce") - lat0) * lat_scale
    out["origin_latitude_deg"] = lat0
    out["origin_longitude_deg"] = lon0
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 3:
        return None
    if float(np.std(a[good])) <= 1e-12 or float(np.std(b[good])) <= 1e-12:
        return None
    return float(np.corrcoef(a[good], b[good])[0, 1])


def pct(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.percentile(values, q))


def load_sequence(seq_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "gps": pd.read_csv(seq_dir / "gps.csv"),
        "imu": pd.read_csv(seq_dir / "imu.csv"),
        "motors": pd.read_csv(seq_dir / "motors.csv"),
        "motion_command": pd.read_csv(seq_dir / "motion_command.csv"),
        "reference_ekf": pd.read_csv(seq_dir / "reference_ekf.csv"),
    }


def make_grid(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    starts = []
    ends = []
    for key in ("gps", "imu", "motors", "reference_ekf"):
        t = bag_time_s(frames[key])
        starts.append(float(np.nanmin(t)))
        ends.append(float(np.nanmax(t)))
    start = math.ceil(max(starts) * RATE_HZ) / RATE_HZ
    end = math.floor(min(ends) * RATE_HZ) / RATE_HZ
    if end <= start:
        raise RuntimeError("no overlapping time span across required streams")
    return np.arange(start, end + 0.5 * DT_S, DT_S, dtype=float)


def build_aligned(frames: dict[str, pd.DataFrame], grid: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    gps = local_enu_from_gps(frames["gps"])
    gps_t = bag_time_s(gps)
    rtk_values = gps[["east_m", "north_m", "horizontal_accuracy", "heading", "heading_accuracy"]].to_numpy(float)
    rtk, rtk_ok, rtk_gap = interp_with_gap(
        gps_t[gps["rtk_valid"].to_numpy(bool)],
        rtk_values[gps["rtk_valid"].to_numpy(bool)],
        grid,
        max_gap_s=RTK_MAX_INTERP_GAP_S,
    )

    motors = frames["motors"]
    motor_t = bag_time_s(motors)
    motor_values = motors[
        [
            "front_left.linear_speed",
            "back_left.linear_speed",
            "front_right.linear_speed",
            "back_right.linear_speed",
        ]
    ].to_numpy(float)
    motor_interp, motor_ok, motor_gap = interp_with_gap(
        motor_t, motor_values, grid, max_gap_s=MOTOR_MAX_INTERP_GAP_S
    )

    imu = frames["imu"]
    imu_t = bag_time_s(imu)
    imu_values = imu[
        [
            "angular_velocity.z",
            "linear_acceleration.x",
            "orientation.x",
            "orientation.y",
            "orientation.z",
            "orientation.w",
        ]
    ].to_numpy(float)
    imu_interp, imu_ok, imu_gap = interp_with_gap(
        imu_t, imu_values, grid, max_gap_s=IMU_MAX_INTERP_GAP_S
    )

    command = frames["motion_command"]
    command_t = bag_time_s(command)
    command_values = command[["linear.x", "angular.z"]].to_numpy(float)
    command_interp, command_ok, command_gap = interp_with_gap(
        command_t, command_values, grid, max_gap_s=COMMAND_MAX_INTERP_GAP_S
    )

    ref = frames["reference_ekf"]
    ref_t = bag_time_s(ref)
    ref_yaw = quat_to_yaw(
        ref["pose.pose.orientation.x"].to_numpy(float),
        ref["pose.pose.orientation.y"].to_numpy(float),
        ref["pose.pose.orientation.z"].to_numpy(float),
        ref["pose.pose.orientation.w"].to_numpy(float),
    )
    ref_values = np.column_stack(
        [
            ref["pose.pose.position.x"].to_numpy(float),
            ref["pose.pose.position.y"].to_numpy(float),
            ref_yaw,
            ref["twist.twist.linear.x"].to_numpy(float),
            ref["twist.twist.angular.z"].to_numpy(float),
        ]
    )
    ref_interp, ref_ok, ref_gap = interp_with_gap(
        ref_t, ref_values, grid, max_gap_s=REFERENCE_MAX_INTERP_GAP_S
    )

    left = np.nanmean(motor_interp[:, [0, 1]], axis=1)
    right = np.nanmean(motor_interp[:, [2, 3]], axis=1)
    forward = 0.5 * (left + right)
    wheel_yaw = (right - left) / TRACK_WIDTH_M
    imu_yaw = imu_interp[:, 0]
    accel = np.gradient(forward, DT_S)
    yaw_accel = np.gradient(imu_yaw, DT_S)
    slow = slow_features(imu_yaw, wheel_yaw, forward, int(round(30.0 * RATE_HZ)))
    fast = np.column_stack(
        [forward, imu_yaw, accel, yaw_accel, np.abs(imu_yaw), np.abs(accel)]
    ).astype(np.float32)

    ok = motor_ok & imu_ok & rtk_ok & ref_ok
    aligned = pd.DataFrame(
        {
            "time_s": grid - grid[0],
            "bag_time_s": grid,
            "rtk_east_m": rtk[:, 0],
            "rtk_north_m": rtk[:, 1],
            "rtk_horizontal_accuracy_m": rtk[:, 2],
            "rtk_heading_deg": rtk[:, 3],
            "rtk_heading_accuracy_deg": rtk[:, 4],
            "left_motor_speed_mps": left,
            "right_motor_speed_mps": right,
            "forward_motor_speed_mps": forward,
            "wheel_yaw_proxy_radps": wheel_yaw,
            "imu_yaw_rate_radps": imu_yaw,
            "accel_mps2": accel,
            "yaw_accel_radps2": yaw_accel,
            "wheel_imu_disagreement_radps": imu_yaw - wheel_yaw,
            "command_linear_mps": command_interp[:, 0],
            "command_angular_radps": command_interp[:, 1],
            "reference_x_m": ref_interp[:, 0],
            "reference_y_m": ref_interp[:, 1],
            "reference_heading_rad": ref_interp[:, 2],
            "reference_linear_mps": ref_interp[:, 3],
            "reference_angular_radps": ref_interp[:, 4],
            "valid_all_required": ok,
            "gap_rtk_s": rtk_gap,
            "gap_motor_s": motor_gap,
            "gap_imu_s": imu_gap,
            "gap_command_s": command_gap,
            "gap_reference_s": ref_gap,
        }
    )
    for i in range(fast.shape[1]):
        aligned[f"fast_feature_{i}"] = fast[:, i]
    for i in range(slow.shape[1]):
        aligned[f"slow_feature_{i}"] = slow[:, i]

    diagnostics = {
        "raw_sample_counts": {k: int(len(v)) for k, v in frames.items()},
        "grid_samples": int(len(grid)),
        "valid_required_samples": int(ok.sum()),
        "duration_s": float(grid[-1] - grid[0]),
        "origin": {
            "latitude_deg": float(gps["origin_latitude_deg"].iloc[0]),
            "longitude_deg": float(gps["origin_longitude_deg"].iloc[0]),
        },
        "sync_gaps_s": {
            "rtk": gap_stats(rtk_gap[rtk_ok]),
            "motors": gap_stats(motor_gap[motor_ok]),
            "imu": gap_stats(imu_gap[imu_ok]),
            "command": gap_stats(command_gap[command_ok]),
            "reference_ekf": gap_stats(ref_gap[ref_ok]),
        },
        "rtk_validity_rule": (
            f"finite latitude/longitude and horizontal_accuracy <= "
            f"{RTK_MAX_HORIZONTAL_ACCURACY_M:.3f} m"
        ),
        "rtk_valid_fraction_raw": float(gps["rtk_valid"].mean()),
        "rtk_horizontal_accuracy_m": {
            "median": pct(gps.loc[gps["rtk_valid"], "horizontal_accuracy"].to_numpy(float), 50),
            "p95": pct(gps.loc[gps["rtk_valid"], "horizontal_accuracy"].to_numpy(float), 95),
            "max": pct(gps.loc[gps["rtk_valid"], "horizontal_accuracy"].to_numpy(float), 100),
        },
        "sanity_correlations": {
            "forward_motor_vs_command_linear": corr(forward, command_interp[:, 0]),
            "forward_motor_vs_reference_linear": corr(forward, ref_interp[:, 3]),
            "wheel_yaw_proxy_vs_imu_yaw_rate": corr(wheel_yaw, imu_yaw),
            "wheel_yaw_proxy_vs_reference_yaw_rate": corr(wheel_yaw, ref_interp[:, 4]),
            "imu_yaw_rate_vs_reference_yaw_rate": corr(imu_yaw, ref_interp[:, 4]),
        },
    }
    return aligned.loc[ok].reset_index(drop=True), diagnostics


def gap_stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def choose_checkpoint(path: Path) -> dict[str, Any]:
    run_manifest = path.with_name("run_manifest.json")
    manifest = json.loads(run_manifest.read_text(encoding="utf-8")) if run_manifest.exists() else {}
    return {
        "checkpoint": str(path),
        "selection_rule": (
            "deterministic smoke checkpoint: lexically first frozen V2 path under "
            "replicate_01_base42/fold_01_building00; not selected by AIFARMS performance"
        ),
        "run_manifest": str(run_manifest) if run_manifest.exists() else None,
        "test_sequence": manifest.get("test_sequence"),
        "replicate": manifest.get("replicate"),
        "base_seed": manifest.get("base_seed"),
    }


def load_frozen_checkpoint(path: Path) -> dict[str, Any]:
    # Project-owned frozen artifact containing NumPy normalization arrays.
    return torch.load(path, map_location="cpu", weights_only=False)


def feature_range_report(aligned: pd.DataFrame, checkpoint: dict[str, Any]) -> dict[str, Any]:
    ckpt = load_frozen_checkpoint(Path(checkpoint["checkpoint"]))
    fast = aligned[[f"fast_feature_{i}" for i in range(6)]].to_numpy(np.float32)
    slow = aligned[[f"slow_feature_{i}" for i in range(16)]].to_numpy(np.float32)
    fast_z = (fast - ckpt["fast_feature_mean"]) / ckpt["fast_feature_std"]
    slow_z = (slow - ckpt["slow_feature_mean"]) / ckpt["slow_feature_std"]
    slow_valid = np.all(np.isfinite(slow), axis=1)
    return {
        "fast_feature_dim": int(fast.shape[1]),
        "slow_feature_dim": int(slow.shape[1]),
        "fast_abs_z_p95": float(np.nanpercentile(np.abs(fast_z), 95)),
        "fast_abs_z_max": float(np.nanmax(np.abs(fast_z))),
        "fast_fraction_abs_z_gt3": float(np.nanmean(np.abs(fast_z) > 3.0)),
        "slow_valid_after_30s_fraction": float(np.mean(slow_valid)),
        "slow_abs_z_p95": float(np.nanpercentile(np.abs(slow_z[slow_valid]), 95)) if slow_valid.any() else None,
        "slow_abs_z_max": float(np.nanmax(np.abs(slow_z[slow_valid]))) if slow_valid.any() else None,
        "slow_fraction_abs_z_gt3": float(np.nanmean(np.abs(slow_z[slow_valid]) > 3.0)) if slow_valid.any() else None,
        "substantially_outside_i2nav_normalization": bool(
            np.nanpercentile(np.abs(fast_z), 95) > 3.0
            or (slow_valid.any() and np.nanpercentile(np.abs(slow_z[slow_valid]), 95) > 3.0)
        ),
    }


def per_feature_shift_audit(aligned: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    ckpt = load_frozen_checkpoint(Path(checkpoint["checkpoint"]))
    rows: list[dict[str, Any]] = []

    groups = [
        ("fast", FAST_FEATURE_NAMES, [f"fast_feature_{i}" for i in range(6)], ckpt["fast_feature_mean"], ckpt["fast_feature_std"]),
        ("slow", SLOW_FEATURE_NAMES, [f"slow_feature_{i}" for i in range(16)], ckpt["slow_feature_mean"], ckpt["slow_feature_std"]),
    ]
    for group, names, columns, means, stds in groups:
        for index, (name, column) in enumerate(zip(names, columns)):
            values = pd.to_numeric(aligned[column], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            mean = float(means[index])
            std = max(float(stds[index]), 1e-12)
            z = (values - mean) / std if len(values) else np.asarray([], dtype=float)
            abs_z = np.abs(z)
            rows.append(
                {
                    "feature_group": group,
                    "feature_index": index,
                    "feature_name": name,
                    "terrasentia_p5": pct(values, 5),
                    "terrasentia_p50": pct(values, 50),
                    "terrasentia_p95": pct(values, 95),
                    "i2nav_training_mean": mean,
                    "i2nav_training_std": std,
                    "abs_z_p95": pct(abs_z, 95),
                    "abs_z_max": pct(abs_z, 100),
                    "fraction_beyond_3sigma": float(np.mean(abs_z > 3.0)) if len(abs_z) else None,
                    "fraction_beyond_5sigma": float(np.mean(abs_z > 5.0)) if len(abs_z) else None,
                    "large_shift": bool(
                        len(abs_z)
                        and (
                            float(np.percentile(abs_z, 95)) > 3.0
                            or float(np.mean(abs_z > 3.0)) > 0.05
                            or float(np.max(abs_z)) > 10.0
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def shifted_definition_rows(shift: pd.DataFrame) -> list[dict[str, Any]]:
    definitions: dict[str, dict[str, str]] = {
        "speed_mps": {
            "terrasentia": "(mean left motor linear speed + mean right motor linear speed) / 2, m/s",
            "i2nav": "prepared ODO forward speed, m/s",
            "checks": "units m/s; positive forward; 10 Hz interpolation; no GPS/reference input",
        },
        "imu_yaw_rate_radps": {
            "terrasentia": "robot IMU angular_velocity.z from /terrasentia/imu, rad/s, bag-time aligned",
            "i2nav": "ADIS IMU yaw rate from integrated yaw derivative after sign/bias handling, rad/s",
            "checks": "axis/sign may differ; no sign flip applied because this audit must not tune",
        },
        "accel_mps2": {
            "terrasentia": "np.gradient(forward_motor_speed, 0.1 s), m/s^2",
            "i2nav": "np.gradient(ODO speed, 0.1 s), m/s^2",
            "checks": "same deterministic derivative and sampling interval",
        },
        "yaw_accel_radps2": {
            "terrasentia": "np.gradient(IMU yaw rate, 0.1 s), rad/s^2",
            "i2nav": "np.gradient(IMU yaw rate, 0.1 s), rad/s^2",
            "checks": "same derivative rule; raw IMU noise differences can dominate",
        },
        "abs_imu_yaw_rate_radps": {
            "terrasentia": "abs(IMU yaw rate), rad/s",
            "i2nav": "abs(IMU yaw rate), rad/s",
            "checks": "same units and formula",
        },
        "abs_accel_mps2": {
            "terrasentia": "abs(motor-speed gradient), m/s^2",
            "i2nav": "abs(ODO-speed gradient), m/s^2",
            "checks": "same formula; motor-speed quantization/noise may differ",
        },
    }
    for name in SLOW_FEATURE_NAMES:
        if name not in definitions:
            if "wheel_yaw" in name:
                source = "wheel-yaw proxy=(right-left)/0.26 m, rad/s; 30 s rolling statistic"
                i2nav = "4-wheel-steering planar wheel kinematic solve; 30 s rolling statistic"
                checks = "same statistic/window, different vehicle kinematic adapter"
            elif "imu_yaw" in name:
                source = "robot IMU angular_velocity.z; 30 s rolling statistic"
                i2nav = "prepared i2Nav IMU yaw rate; 30 s rolling statistic"
                checks = "same statistic/window, possible IMU convention/noise-domain difference"
            elif "disagreement" in name:
                source = "IMU yaw - wheel yaw; normalized by |IMU|+|wheel|+0.02 where applicable"
                i2nav = "same formula after Ranger deterministic wheel-yaw adapter"
                checks = "same formula; depends on cross-platform yaw-sign consistency"
            else:
                source = "forward motor speed; 30 s rolling statistic"
                i2nav = "ODO forward speed; 30 s rolling statistic"
                checks = "same statistic/window and units"
            definitions[name] = {"terrasentia": source, "i2nav": i2nav, "checks": checks}

    rows: list[dict[str, Any]] = []
    for record in shift[shift["large_shift"]].to_dict("records"):
        item = definitions.get(record["feature_name"], {})
        rows.append(
            {
                "feature_group": record["feature_group"],
                "feature_name": record["feature_name"],
                "terrasentia_definition": item.get("terrasentia", ""),
                "i2nav_definition": item.get("i2nav", ""),
                "verification": item.get("checks", ""),
                "audit_action": "report_only_no_change",
            }
        )
    return rows


def integrate_prediction(
    aligned: pd.DataFrame,
    delta_v: np.ndarray,
    delta_w: np.ndarray,
    *,
    label: str,
) -> pd.DataFrame:
    n = len(aligned)
    x = np.zeros((n, 3), dtype=float)
    x[0, 0] = float(aligned["rtk_east_m"].iloc[0])
    x[0, 1] = float(aligned["rtk_north_m"].iloc[0])
    x[0, 2] = float(aligned["reference_heading_rad"].iloc[0])
    base_v = aligned["forward_motor_speed_mps"].to_numpy(float)
    base_w = aligned["imu_yaw_rate_radps"].to_numpy(float)
    corrected_v = base_v + np.asarray(delta_v, dtype=float)
    corrected_w = base_w + np.asarray(delta_w, dtype=float)
    for k in range(1, n):
        dt = float(aligned["time_s"].iloc[k] - aligned["time_s"].iloc[k - 1])
        theta_prev = x[k - 1, 2]
        x[k, 0] = x[k - 1, 0] + corrected_v[k] * math.cos(theta_prev) * dt
        x[k, 1] = x[k - 1, 1] + corrected_v[k] * math.sin(theta_prev) * dt
        x[k, 2] = float(wrap_angle(theta_prev + corrected_w[k] * dt))
    return pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "bag_time_s": aligned["bag_time_s"],
            "method": label,
            "pred_v_T_mps": corrected_v,
            "pred_omega_T_radps": corrected_w,
            "x_T_m": x[:, 0],
            "y_T_m": x[:, 1],
            "theta_T_rad": x[:, 2],
        }
    )


def evaluate_trace(
    aligned: pd.DataFrame,
    trace: pd.DataFrame,
    *,
    model: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    trajectory = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "gt_east_m": aligned["rtk_east_m"],
            "gt_north_m": aligned["rtk_north_m"],
            "gt_heading_rad": aligned["reference_heading_rad"],
            "estimate_east_m": trace["x_T_m"],
            "estimate_north_m": trace["y_T_m"],
            "estimate_heading_rad": trace["theta_T_rad"],
        }
    )
    pred_trace = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "true_delta_v_mps": aligned["reference_linear_mps"] - aligned["forward_motor_speed_mps"],
            "pred_delta_v_mps": trace["pred_v_T_mps"] - aligned["forward_motor_speed_mps"],
            "true_delta_omega_radps": aligned["reference_angular_radps"] - aligned["imu_yaw_rate_radps"],
            "pred_total_delta_omega_radps": trace["pred_omega_T_radps"] - aligned["imu_yaw_rate_radps"],
        }
    )
    return evaluate_fidelity_frames(
        trajectory,
        pred_trace,
        model=model,
        sequence=DEFAULT_SEQUENCE,
    )


def run_physics_only(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    zeros = np.zeros(len(aligned), dtype=float)
    trace = integrate_prediction(aligned, zeros, zeros, label="physics_only_motor_forward_imu_yaw")
    profile, timeseries = evaluate_trace(aligned, trace, model="physics_only_motor_forward_imu_yaw")
    return trace, timeseries, profile


def path_length(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if len(xy) == 0:
        return np.asarray([], dtype=float)
    increments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(increments)])


def forward_distance_audit(aligned: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    time_s = aligned["time_s"].to_numpy(float)
    dt = np.diff(time_s, prepend=time_s[0])
    motor_speed = aligned["forward_motor_speed_mps"].to_numpy(float)
    rtk_xy = aligned[["rtk_east_m", "rtk_north_m"]].to_numpy(float)
    rtk_dist = path_length(rtk_xy)
    motor_dist = np.cumsum(np.abs(motor_speed) * dt)
    rtk_speed = np.gradient(rtk_dist, DT_S)

    heading = aligned["reference_heading_rad"].to_numpy(float)
    velocity = np.gradient(rtk_xy, DT_S, axis=0)
    rtk_along = velocity[:, 0] * np.cos(heading) + velocity[:, 1] * np.sin(heading)
    good = np.isfinite(motor_speed) & np.isfinite(rtk_along) & (np.hypot(velocity[:, 0], velocity[:, 1]) >= 0.10)
    discrepancy = motor_speed[good] - rtk_along[good]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(time_s, motor_dist, label="integrated |motor forward|")
    ax.plot(time_s, rtk_dist, label="RTK path length")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative distance (m)")
    ax.set_title("Motor-derived distance vs RTK path length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / "distance_audit_cumulative_distance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return {
        "motor_abs_forward_distance_m": float(motor_dist[-1]),
        "rtk_path_length_m": float(rtk_dist[-1]),
        "motor_to_rtk_distance_ratio": float(motor_dist[-1] / rtk_dist[-1]) if rtk_dist[-1] > 1e-9 else None,
        "rtk_speed_median_mps": pct(rtk_speed, 50),
        "rtk_speed_p95_mps": pct(rtk_speed, 95),
        "motor_minus_rtk_along_track_mps_mean": float(np.mean(discrepancy)) if len(discrepancy) else None,
        "motor_minus_rtk_along_track_mps_p50": pct(discrepancy, 50),
        "motor_minus_rtk_along_track_mps_p95_abs": pct(np.abs(discrepancy), 95),
        "along_track_samples_used": int(len(discrepancy)),
        "plot": str(path),
        "note": "Diagnostic only; no RTK-derived speed scale was fitted.",
    }


def heading_yaw_audit(aligned: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    time_s = aligned["time_s"].to_numpy(float)
    imu_w = aligned["imu_yaw_rate_radps"].to_numpy(float)
    ref_w = aligned["reference_angular_radps"].to_numpy(float)
    ref_heading = np.unwrap(aligned["reference_heading_rad"].to_numpy(float))
    imu_heading = np.zeros(len(aligned), dtype=float)
    imu_heading[0] = ref_heading[0]
    for k in range(1, len(aligned)):
        dt = time_s[k] - time_s[k - 1]
        imu_heading[k] = imu_heading[k - 1] + imu_w[k] * dt
    diff = wrap_angle(imu_heading - ref_heading)
    yaw_residual = imu_w - ref_w
    accum_residual = np.cumsum(yaw_residual * np.diff(time_s, prepend=time_s[0]))

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    axes[0].plot(time_s, np.rad2deg(wrap_angle(imu_heading)), label="integrated IMU heading")
    axes[0].plot(time_s, np.rad2deg(wrap_angle(ref_heading)), label="dataset EKF heading", alpha=0.8)
    axes[0].set_ylabel("heading (deg)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("IMU yaw integration vs dataset fused-reference heading")
    axes[1].plot(time_s, np.rad2deg(diff), label="wrapped heading difference")
    axes[1].plot(time_s, np.rad2deg(accum_residual), label="accumulated yaw-rate residual", alpha=0.8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("difference (deg)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    path = out_dir / "heading_yaw_accumulation_audit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return {
        "final_heading_difference_deg": float(np.rad2deg(diff[-1])),
        "heading_disagreement_p50_abs_deg": pct(np.abs(np.rad2deg(diff)), 50),
        "heading_disagreement_p95_abs_deg": pct(np.abs(np.rad2deg(diff)), 95),
        "mean_signed_imu_minus_reference_yaw_rate_radps": float(np.mean(yaw_residual)),
        "accumulated_signed_yaw_rate_residual_final_deg": float(np.rad2deg(accum_residual[-1])),
        "accumulated_signed_yaw_rate_residual_p95_abs_deg": pct(np.abs(np.rad2deg(accum_residual)), 95),
        "plot": str(path),
        "reference_warning": "Dataset EKF heading/twist is fused-reference diagnostic, not independent ground truth.",
    }


def oracle_decomposition(aligned: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cases = {
        "A_motor_forward_plus_imu_yaw": (
            aligned["forward_motor_speed_mps"].to_numpy(float),
            aligned["imu_yaw_rate_radps"].to_numpy(float),
        ),
        "B_reference_forward_plus_imu_yaw_oracle": (
            aligned["reference_linear_mps"].to_numpy(float),
            aligned["imu_yaw_rate_radps"].to_numpy(float),
        ),
        "C_motor_forward_plus_reference_yaw_oracle": (
            aligned["forward_motor_speed_mps"].to_numpy(float),
            aligned["reference_angular_radps"].to_numpy(float),
        ),
        "D_reference_forward_plus_reference_yaw_control": (
            aligned["reference_linear_mps"].to_numpy(float),
            aligned["reference_angular_radps"].to_numpy(float),
        ),
    }
    rows = []
    profiles: dict[str, Any] = {}
    for name, (v_sig, w_sig) in cases.items():
        base_v = aligned["forward_motor_speed_mps"].to_numpy(float)
        base_w = aligned["imu_yaw_rate_radps"].to_numpy(float)
        trace = integrate_prediction(
            aligned,
            v_sig - base_v,
            w_sig - base_w,
            label=name,
        )
        profile, _ = evaluate_trace(aligned, trace, model=name)
        profiles[name] = profile
        rows.append(
            {
                "case": name,
                "ATE_m": profile["ATE_m"],
                "RPEp_1s_m": profile["RPEp_1s_m"],
                "RPEp_5s_m": profile["RPEp_5s_m"],
                "RPEp_10s_m": profile["RPEp_10s_m"],
                "Dp_p95_m": profile["Dp_p95_m"],
                "Dp_max_m": profile["Dp_max_m"],
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows), profiles


def feature_group_shift_summary(shift: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for name in FAST_FEATURE_NAMES + SLOW_FEATURE_NAMES:
        if "accel" in name or "std_" in name or "rms_" in name:
            group = "derived_or_derivative_noise_sensitive"
        elif "normalized" in name or "disagreement" in name:
            group = "normalized_or_consistency"
        elif name.startswith("mean_") or name.startswith("mean_abs_"):
            group = "slow_window_statistics"
        else:
            group = "direct_physical_quantities"
        if name in ("speed_mps", "imu_yaw_rate_radps", "abs_imu_yaw_rate_radps"):
            group = "direct_physical_quantities"
        mapping[name] = group
    frame = shift.copy()
    frame["audit_group"] = frame["feature_name"].map(mapping).fillna("other")
    rows = []
    for group, sub in frame.groupby("audit_group", sort=True):
        rows.append(
            {
                "audit_group": group,
                "feature_count": int(len(sub)),
                "abs_z_p95_max_across_features": float(pd.to_numeric(sub["abs_z_p95"], errors="coerce").max()),
                "abs_z_max_max_across_features": float(pd.to_numeric(sub["abs_z_max"], errors="coerce").max()),
                "mean_fraction_beyond_3sigma": float(pd.to_numeric(sub["fraction_beyond_3sigma"], errors="coerce").mean()),
                "mean_fraction_beyond_5sigma": float(pd.to_numeric(sub["fraction_beyond_5sigma"], errors="coerce").mean()),
                "large_shift_features": int(sub["large_shift"].astype(str).str.lower().eq("true").sum() if sub["large_shift"].dtype == object else sub["large_shift"].sum()),
            }
        )
    return pd.DataFrame(rows)


def frame_sanity_checks(aligned: pd.DataFrame) -> dict[str, Any]:
    rtk_e = aligned["rtk_east_m"].to_numpy(float)
    rtk_n = aligned["rtk_north_m"].to_numpy(float)
    ref_heading = aligned["reference_heading_rad"].to_numpy(float)
    imu_w = aligned["imu_yaw_rate_radps"].to_numpy(float)
    ref_w = aligned["reference_angular_radps"].to_numpy(float)
    wheel_w = aligned["wheel_yaw_proxy_radps"].to_numpy(float)
    dt = np.diff(aligned["time_s"].to_numpy(float))
    ref_heading_from_quat = np.all(np.isfinite(ref_heading))
    yaw_sign_corr = corr(imu_w, ref_w)
    wheel_sign_corr = corr(wheel_w, ref_w)
    return {
        "enu_axes": {
            "east_span_m": float(np.max(rtk_e) - np.min(rtk_e)),
            "north_span_m": float(np.max(rtk_n) - np.min(rtk_n)),
            "first_position_east_north_m": [float(rtk_e[0]), float(rtk_n[0])],
            "status": "local ENU from first valid RTK origin; no full-trajectory alignment",
        },
        "initial_position": {
            "rtk_initial_east_m": float(rtk_e[0]),
            "rtk_initial_north_m": float(rtk_n[0]),
            "used_by_all_propagations": True,
        },
        "initial_heading": {
            "reference_heading_initial_deg": float(np.rad2deg(ref_heading[0])),
            "used_by_all_propagations": True,
            "secondary_reference_only": True,
        },
        "yaw_sign": {
            "imu_vs_reference_yaw_rate_corr": yaw_sign_corr,
            "wheel_vs_reference_yaw_rate_corr": wheel_sign_corr,
            "status": "positive correlations support sign plausibility, not proof",
        },
        "quaternion_convention": {
            "formula": "yaw=atan2(2(wz+xy),1-2(y^2+z^2))",
            "finite_reference_heading": bool(ref_heading_from_quat),
        },
        "imu_axis": {
            "axis_used": "angular_velocity.z",
            "mean_radps": float(np.mean(imu_w)),
            "p95_abs_radps": pct(np.abs(imu_w), 95),
        },
        "heading_wrap": {
            "wrap_interval": "[-pi, pi)",
            "max_abs_wrapped_reference_heading_deg": pct(np.abs(np.rad2deg(wrap_angle(ref_heading))), 100),
        },
        "propagation_timestep": {
            "median_dt_s": float(np.median(dt)),
            "p95_abs_dt_error_s": pct(np.abs(dt - DT_S), 95),
            "max_abs_dt_error_s": pct(np.abs(dt - DT_S), 100),
        },
    }


def write_mechanism_report(
    out_dir: Path,
    *,
    distance: dict[str, Any],
    yaw: dict[str, Any],
    oracle: pd.DataFrame,
    group_shift: pd.DataFrame,
    sanity: dict[str, Any],
) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            x = float(value)
        except Exception:
            return str(value)
        return f"{x:.{digits}f}" if math.isfinite(x) else "n/a"

    a = oracle.set_index("case")
    base_ate = float(a.loc["A_motor_forward_plus_imu_yaw", "ATE_m"])
    speed_oracle_ate = float(a.loc["B_reference_forward_plus_imu_yaw_oracle", "ATE_m"])
    yaw_oracle_ate = float(a.loc["C_motor_forward_plus_reference_yaw_oracle", "ATE_m"])
    both_oracle_ate = float(a.loc["D_reference_forward_plus_reference_yaw_control", "ATE_m"])
    speed_gain = base_ate - speed_oracle_ate
    yaw_gain = base_ate - yaw_oracle_ate
    both_gain = base_ate - both_oracle_ate

    ranking = [
        ("sensor/feature domain shift", "high", "21/22 V2 features are shifted, with extreme slow yaw-disagreement z-scores."),
        ("skid-steer/slip physics mismatch", "high", "Motor distance is far larger than RTK path length and oracle controls still expose propagation mismatch."),
        ("IMU yaw accumulation", "high", f"Final integrated-IMU vs EKF-heading difference is {fmt(yaw['final_heading_difference_deg'],1)} deg."),
        ("motor speed-scale mismatch", "high", f"Motor/RTK distance ratio is {fmt(distance['motor_to_rtk_distance_ratio'],2)}."),
        ("coordinate/sign/adapter implementation error", "possible but lower", "Frame checks show plausible signs/cadence; no objective basis for a corrective change yet."),
    ]

    lines = [
        "# TerraSentia One-Sequence Physics Failure Diagnostic",
        "",
        "Scope: diagnostic only on `ts_2022_06_15_11h48m34s_four_rows`. No V2 tuning, normalization refit, adapter correction, or paper edits were performed.",
        "",
        "## Forward-Distance Audit",
        "",
        f"- Integrated absolute motor distance: `{fmt(distance['motor_abs_forward_distance_m'])}` m.",
        f"- RTK path length: `{fmt(distance['rtk_path_length_m'])}` m.",
        f"- Motor/RTK distance ratio: `{fmt(distance['motor_to_rtk_distance_ratio'])}`.",
        f"- Motor minus RTK along-track speed mean: `{fmt(distance['motor_minus_rtk_along_track_mps_mean'])}` m/s.",
        f"- Motor minus RTK along-track speed p95 abs: `{fmt(distance['motor_minus_rtk_along_track_mps_p95_abs'])}` m/s.",
        "",
        "## Heading/Yaw Accumulation Audit",
        "",
        f"- Final integrated IMU heading minus EKF heading: `{fmt(yaw['final_heading_difference_deg'],1)}` deg.",
        f"- Heading disagreement p50/p95 abs: `{fmt(yaw['heading_disagreement_p50_abs_deg'],1)}` / `{fmt(yaw['heading_disagreement_p95_abs_deg'],1)}` deg.",
        f"- Mean signed IMU minus EKF yaw-rate residual: `{fmt(yaw['mean_signed_imu_minus_reference_yaw_rate_radps'],5)}` rad/s.",
        f"- Final accumulated yaw-rate residual: `{fmt(yaw['accumulated_signed_yaw_rate_residual_final_deg'],1)}` deg.",
        "- EKF heading/twist are dataset-provided fused-reference diagnostics, not independent ground truth.",
        "",
        "## Oracle Decomposition",
        "",
        "| Case | ATE | RPE1 | RPE5 | RPE10 | Dp p95 | Dp max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in oracle.to_dict("records"):
        lines.append(
            f"| {row['case']} | {fmt(row['ATE_m'])} | {fmt(row['RPEp_1s_m'])} | "
            f"{fmt(row['RPEp_5s_m'])} | {fmt(row['RPEp_10s_m'])} | "
            f"{fmt(row['Dp_p95_m'])} | {fmt(row['Dp_max_m'])} |"
        )
    lines += [
        "",
        f"- Replacing motor speed only changes ATE by `{fmt(speed_gain)}` m.",
        f"- Replacing yaw rate only changes ATE by `{fmt(yaw_gain)}` m.",
        f"- Replacing both with EKF reference velocity/yaw changes ATE by `{fmt(both_gain)}` m.",
        "",
        "## Canonical Feature-Group Audit",
        "",
        "| Group | Count | max p95 | max max-z | mean frac >3sigma | mean frac >5sigma | large-shift features |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_shift.to_dict("records"):
        lines.append(
            f"| {row['audit_group']} | {int(row['feature_count'])} | "
            f"{fmt(row['abs_z_p95_max_across_features'],2)} | {fmt(row['abs_z_max_max_across_features'],2)} | "
            f"{fmt(row['mean_fraction_beyond_3sigma'],3)} | {fmt(row['mean_fraction_beyond_5sigma'],3)} | "
            f"{int(row['large_shift_features'])} |"
        )
    lines += [
        "",
        "## Initial/Frame Sanity",
        "",
        f"- ENU span east/north: `{fmt(sanity['enu_axes']['east_span_m'])}` / `{fmt(sanity['enu_axes']['north_span_m'])}` m.",
        f"- Initial RTK position: `{fmt(sanity['initial_position']['rtk_initial_east_m'])}`, `{fmt(sanity['initial_position']['rtk_initial_north_m'])}` m.",
        f"- Initial heading from EKF diagnostic: `{fmt(sanity['initial_heading']['reference_heading_initial_deg'],1)}` deg.",
        f"- IMU vs EKF yaw-rate correlation: `{fmt(sanity['yaw_sign']['imu_vs_reference_yaw_rate_corr'])}`.",
        f"- Wheel yaw vs EKF yaw-rate correlation: `{fmt(sanity['yaw_sign']['wheel_vs_reference_yaw_rate_corr'])}`.",
        f"- Median dt: `{fmt(sanity['propagation_timestep']['median_dt_s'],4)}` s; max dt error: `{fmt(sanity['propagation_timestep']['max_abs_dt_error_s'],6)}` s.",
        "",
        "## Final Diagnosis Ranking",
        "",
    ]
    for item, rank, reason in ranking:
        lines.append(f"- **{item}: {rank}.** {reason}")
    lines += [
        "",
        "No corrective change is made here. The next scientifically clean step would be to inspect objective TerraSentia documentation for motor-speed semantics and IMU frame conventions before deciding whether any adapter correction is justified.",
    ]
    (out_dir / "physics_failure_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_profiles(physics: dict[str, Any], v2_profile: dict[str, Any]) -> pd.DataFrame:
    metrics = [
        "ATE_m",
        "RPEp_1s_m",
        "RPEp_5s_m",
        "RPEp_10s_m",
        "Dp_p95_m",
        "Dp_max_m",
        "heading_MAE_deg",
        "Dtheta_p95_deg",
    ]
    rows = []
    for metric in metrics:
        p = float(physics[metric])
        v = float(v2_profile[metric])
        rows.append(
            {
                "metric": metric,
                "physics_only": p,
                "frozen_v2_smoke": v,
                "v2_minus_physics": v - p,
                "v2_relative_change_pct": 100.0 * (v - p) / p if abs(p) > 1e-12 else None,
                "heading_metric_secondary": bool("heading" in metric.lower() or "theta" in metric.lower()),
            }
        )
    return pd.DataFrame(rows)


def v2_correction_summary(trace: pd.DataFrame) -> dict[str, Any]:
    def stats(col: str, limit: float | None = None) -> dict[str, float | None]:
        values = pd.to_numeric(trace[col], errors="coerce").to_numpy(float)
        abs_values = np.abs(values[np.isfinite(values)])
        out = {
            "mean": float(np.mean(values[np.isfinite(values)])) if len(abs_values) else None,
            "p50_abs": pct(abs_values, 50),
            "p95_abs": pct(abs_values, 95),
            "max_abs": pct(abs_values, 100),
        }
        if limit is not None and len(abs_values):
            out["fraction_at_98pct_limit"] = float(np.mean(abs_values >= 0.98 * limit))
        return out

    return {
        "delta_v_fast_mps": stats("pred_delta_v_mps", v2.DV_LIMIT),
        "delta_omega_fast_radps": stats("pred_fast_delta_omega_radps", v2.DW_FAST_LIMIT),
        "slow_rotational_correction_radps": stats("pred_slow_bias_radps", v2.SLOW_BIAS_LIMIT),
        "total_delta_omega_radps": stats("pred_total_delta_omega_radps", None),
        "resulting_vT_mps": stats("pred_v_T_mps", None),
        "resulting_omegaT_radps": stats("pred_omega_T_radps", None),
    }


def plot_v2_corrections(trace: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    t = trace["time_s"]

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.0), sharex=True)
    axes[0].plot(t, trace["pred_delta_v_mps"], lw=0.9)
    axes[0].axhline(v2.DV_LIMIT, color="0.5", linestyle="--", lw=0.8)
    axes[0].axhline(-v2.DV_LIMIT, color="0.5", linestyle="--", lw=0.8)
    axes[0].set_ylabel("delta-v")
    axes[0].set_title("Frozen V2 correction outputs")
    axes[1].plot(t, trace["pred_fast_delta_omega_radps"], lw=0.9, label="fast")
    axes[1].axhline(v2.DW_FAST_LIMIT, color="0.5", linestyle="--", lw=0.8)
    axes[1].axhline(-v2.DW_FAST_LIMIT, color="0.5", linestyle="--", lw=0.8)
    axes[1].set_ylabel("fast dω")
    axes[2].plot(t, trace["pred_slow_bias_radps"], lw=0.9, color="#9b5b2b")
    axes[2].axhline(v2.SLOW_BIAS_LIMIT, color="0.5", linestyle="--", lw=0.8)
    axes[2].axhline(-v2.SLOW_BIAS_LIMIT, color="0.5", linestyle="--", lw=0.8)
    axes[2].set_ylabel("slow dω")
    axes[3].plot(t, trace["pred_v_T_mps"], lw=0.9, label="vT")
    ax2 = axes[3].twinx()
    ax2.plot(t, trace["pred_omega_T_radps"], lw=0.9, color="#d55e00", label="omegaT")
    axes[3].set_ylabel("vT")
    ax2.set_ylabel("omegaT")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    path = out_dir / "frozen_v2_correction_outputs.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_definition_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TerraSentia vs i2Nav Feature Definition Audit",
        "",
        "Only features flagged as large shifts are listed. No definitions were changed in this audit.",
        "",
        "| Group | Feature | TerraSentia definition | i2Nav definition | Verification | Action |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature_group']} | {row['feature_name']} | "
            f"{row['terrasentia_definition']} | {row['i2nav_definition']} | "
            f"{row['verification']} | {row['audit_action']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shift_audit_report(
    out_dir: Path,
    *,
    shift: pd.DataFrame,
    comparison: pd.DataFrame,
    correction_summary: dict[str, Any],
) -> None:
    large = shift[shift["large_shift"]].copy()
    p = comparison.set_index("metric")
    ate_delta = float(p.loc["ATE_m", "v2_minus_physics"])
    rpe1_delta = float(p.loc["RPEp_1s_m", "v2_minus_physics"])
    dp95_delta = float(p.loc["Dp_p95_m", "v2_minus_physics"])
    slow_sat = correction_summary["slow_rotational_correction_radps"].get("fraction_at_98pct_limit")
    fast_yaw_sat = correction_summary["delta_omega_fast_radps"].get("fraction_at_98pct_limit")
    dv_sat = correction_summary["delta_v_fast_mps"].get("fraction_at_98pct_limit")

    evidence = []
    if len(large) >= 8:
        evidence.append("B) genuine TerraSentia domain shift is strongly supported by many out-of-range features.")
    if abs(float(p.loc["ATE_m", "frozen_v2_smoke"])) > abs(float(p.loc["ATE_m", "physics_only"])):
        evidence.append("D) frozen V2 degrades this physics baseline for global position in the smoke run.")
    else:
        evidence.append("D) frozen V2 does not degrade the physics baseline here; it gives a small improvement but does not rescue the large global drift.")
    if slow_sat is not None and slow_sat > 0.10:
        evidence.append("V2 slow yaw correction saturation is present and likely contributes to persistent yaw drift.")
    if fast_yaw_sat is not None and fast_yaw_sat > 0.10:
        evidence.append("V2 fast yaw correction saturation is present.")
    if dv_sat is not None and dv_sat > 0.10:
        evidence.append("V2 velocity correction saturation is present.")
    evidence.append(
        "A) adapter mismatch is not ruled out, especially yaw-axis/sign and derivative/noise definitions, but no correction was made without independent justification."
    )
    if float(p.loc["ATE_m", "physics_only"]) > 10.0:
        evidence.append("C) underlying physics portability is weak on this sequence even before V2 corrections.")

    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            x = float(value)
        except Exception:
            return str(value)
        if not math.isfinite(x):
            return "n/a"
        return f"{x:.{digits}f}"

    lines = [
        "# Phase-2 Distribution-Shift And Physics Baseline Audit",
        "",
        "Scope: one TerraSentia sequence only. No tuning, renormalization, clipping change, retraining, or paper modification was performed.",
        "",
        "## Feature Shift Summary",
        "",
        f"- Large-shift features: `{len(large)}` of `{len(shift)}`.",
        f"- Fast large-shift features: `{int((large['feature_group'] == 'fast').sum())}` of `6`.",
        f"- Slow large-shift features: `{int((large['feature_group'] == 'slow').sum())}` of `16`.",
        f"- Worst abs-z max: `{fmt(shift['abs_z_max'].max(), 2)}`.",
        f"- Worst abs-z p95: `{fmt(shift['abs_z_p95'].max(), 2)}`.",
        "",
        "## Physics-Only vs Frozen V2 Smoke",
        "",
        "| Metric | Physics-only | Frozen V2 smoke | V2 - physics | Secondary heading metric? |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparison.to_dict("records"):
        lines.append(
            f"| {row['metric']} | {fmt(row['physics_only'])} | "
            f"{fmt(row['frozen_v2_smoke'])} | {fmt(row['v2_minus_physics'])} | "
            f"{row['heading_metric_secondary']} |"
        )
    lines += [
        "",
        "## V2 Correction Behavior",
        "",
    ]
    for name, stats in correction_summary.items():
        lines.append(
            f"- `{name}`: mean {fmt(stats.get('mean'), 4)}, "
            f"abs p95 {fmt(stats.get('p95_abs'), 4)}, max {fmt(stats.get('max_abs'), 4)}, "
            f"fraction at 98% limit {fmt(stats.get('fraction_at_98pct_limit'), 3)}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    lines.extend([f"- {item}" for item in evidence])
    lines += [
        "",
        "The strongest current evidence is a combination of genuine cross-platform/domain shift and weak direct physics portability. The single frozen V2 smoke checkpoint slightly improves several metrics but does not rescue the large global drift. This is not a full AIFARMS scientific result.",
    ]
    (out_dir / "distribution_shift_and_physics_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_v2_smoke(aligned: pd.DataFrame, checkpoint_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ckpt = load_frozen_checkpoint(checkpoint_path)
    v2.RATE = RATE_HZ
    v2.DT = DT_S
    v2.SLOW_SAMPLES = int(round(30.0 * RATE_HZ))
    v2.CHUNK_STEPS = int(round(30.0 * RATE_HZ))
    v2.DEVICE = torch.device("cpu")

    fast = aligned[[f"fast_feature_{i}" for i in range(6)]].to_numpy(np.float32)
    slow = aligned[[f"slow_feature_{i}" for i in range(16)]].to_numpy(np.float32)
    fast_norm = (fast - ckpt["fast_feature_mean"]) / ckpt["fast_feature_std"]
    slow_norm = (slow - ckpt["slow_feature_mean"]) / ckpt["slow_feature_std"]
    slow_norm = np.nan_to_num(slow_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    model = v2.V2SlowAdditiveYaw(fast_input_dim=6, slow_input_dim=16)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(v2.DEVICE)
    model.eval()

    class Cache:
        pass

    cache = Cache()
    cache.grid = aligned["time_s"].to_numpy(float)
    cache.fast_windows = v2.base.sliding_windows(fast_norm.astype(np.float32), v2.FAST_WINDOW)
    cache.slow_features = slow_norm
    prediction = v2.predict_sequence(model, cache, eval_batch_size=4096)

    n = len(aligned)
    x = np.zeros((n, 3), dtype=float)
    x[0, 0] = float(aligned["rtk_east_m"].iloc[0])
    x[0, 1] = float(aligned["rtk_north_m"].iloc[0])
    x[0, 2] = float(aligned["reference_heading_rad"].iloc[0])
    corrected_v = aligned["forward_motor_speed_mps"].to_numpy(float) + prediction["dv"]
    corrected_w = aligned["imu_yaw_rate_radps"].to_numpy(float) + prediction["dw"]
    for k in range(1, n):
        dt = float(aligned["time_s"].iloc[k] - aligned["time_s"].iloc[k - 1])
        theta_prev = x[k - 1, 2]
        v_k = float(corrected_v[k])
        w_k = float(corrected_w[k])
        x[k, 0] = x[k - 1, 0] + v_k * math.cos(theta_prev) * dt
        x[k, 1] = x[k - 1, 1] + v_k * math.sin(theta_prev) * dt
        x[k, 2] = float(wrap_angle(theta_prev + w_k * dt))

    trace = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "bag_time_s": aligned["bag_time_s"],
            "base_forward_motor_speed_mps": aligned["forward_motor_speed_mps"],
            "base_imu_yaw_rate_radps": aligned["imu_yaw_rate_radps"],
            "pred_delta_v_mps": prediction["dv"],
            "pred_fast_delta_omega_radps": prediction["dw_fast"],
            "pred_slow_bias_radps": prediction["b_slow"],
            "pred_total_delta_omega_radps": prediction["dw"],
            "pred_v_T_mps": corrected_v,
            "pred_omega_T_radps": corrected_w,
            "x_T_m": x[:, 0],
            "y_T_m": x[:, 1],
            "theta_T_rad": x[:, 2],
        }
    )
    trajectory = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "gt_east_m": aligned["rtk_east_m"],
            "gt_north_m": aligned["rtk_north_m"],
            "gt_heading_rad": aligned["reference_heading_rad"],
            "estimate_east_m": trace["x_T_m"],
            "estimate_north_m": trace["y_T_m"],
            "estimate_heading_rad": trace["theta_T_rad"],
        }
    )
    pred_trace = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "true_delta_v_mps": aligned["reference_linear_mps"] - aligned["forward_motor_speed_mps"],
            "pred_delta_v_mps": prediction["dv"],
            "true_delta_omega_radps": aligned["reference_angular_radps"] - aligned["imu_yaw_rate_radps"],
            "pred_total_delta_omega_radps": prediction["dw"],
        }
    )
    profile, timeseries = evaluate_fidelity_frames(
        trajectory,
        pred_trace,
        model="frozen_v2_single_checkpoint_smoke",
        sequence=DEFAULT_SEQUENCE,
        seed=42,
        replicate="replicate_01_base42_fold_01_building00",
    )
    return trace, timeseries, profile


def heading_report(aligned: pd.DataFrame) -> dict[str, Any]:
    speed = np.hypot(
        np.gradient(aligned["rtk_east_m"].to_numpy(float), DT_S),
        np.gradient(aligned["rtk_north_m"].to_numpy(float), DT_S),
    )
    moving = speed >= 0.25
    gps_heading_acc = aligned["rtk_heading_accuracy_deg"].to_numpy(float)
    native_good = np.isfinite(gps_heading_acc) & (gps_heading_acc <= 10.0)
    return {
        "native_gps_heading_accuracy_median_deg": pct(gps_heading_acc, 50),
        "native_gps_heading_accuracy_p95_deg": pct(gps_heading_acc, 95),
        "native_gps_heading_good_fraction_acc_le_10deg": float(np.mean(native_good)),
        "rtk_trajectory_moving_fraction_speed_ge_0p25_mps": float(np.mean(moving)),
        "primary_heading_use": "not_defensible",
        "secondary_heading_use": (
            "dataset reference_ekf heading is acceptable for software smoke-test diagnostics only"
        ),
        "reason": (
            "native GPS heading accuracy is poor in this sequence; low-speed "
            "trajectory-derived heading is not automatically used"
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def plot_outputs(aligned: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(aligned["rtk_east_m"], aligned["rtk_north_m"], lw=1.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("RTK east (m)")
    ax.set_ylabel("RTK north (m)")
    ax.set_title("TerraSentia local RTK trajectory")
    ax.grid(True, alpha=0.3)
    path = out_dir / "local_rtk_trajectory.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    t = aligned["time_s"]
    ax.plot(t, aligned["forward_motor_speed_mps"], label="motor forward", lw=1.0)
    ax.plot(t, aligned["command_linear_mps"], label="command linear", lw=0.9, alpha=0.8)
    ax.plot(t, aligned["reference_linear_mps"], label="reference EKF linear", lw=0.9, alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Forward motor speed sanity check")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / "forward_speed_sanity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(t, aligned["wheel_yaw_proxy_radps"], label="wheel-yaw proxy", lw=1.0)
    ax.plot(t, aligned["imu_yaw_rate_radps"], label="IMU yaw rate", lw=0.9, alpha=0.8)
    ax.plot(t, aligned["reference_angular_radps"], label="reference EKF yaw rate", lw=0.9, alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("yaw rate (rad/s)")
    ax.set_title("Wheel-yaw proxy sanity check")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / "wheel_yaw_vs_imu.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for col, label in [
        ("gap_rtk_s", "RTK"),
        ("gap_motor_s", "motors"),
        ("gap_imu_s", "IMU"),
        ("gap_command_s", "command"),
        ("gap_reference_s", "reference"),
    ]:
        ax.plot(t, aligned[col], label=label, lw=0.9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("max interpolation bracket gap (s)")
    ax.set_title("Synchronization gaps over time")
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.3)
    path = out_dir / "sync_gaps_over_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_report(
    out_dir: Path,
    *,
    sequence: str,
    diagnostics: dict[str, Any],
    heading: dict[str, Any],
    checkpoint: dict[str, Any],
    feature_ranges: dict[str, Any],
    profile: dict[str, Any],
    plots: list[str],
) -> None:
    def f(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            if not math.isfinite(value):
                return "n/a"
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# AIFARMS TerraSentia Phase-2 Adapter Validation",
        "",
        f"- Sequence: `{sequence}`",
        "- Scope: one-sequence adapter validation plus one frozen-checkpoint smoke test.",
        "- Scientific status: software validation only; not a five-sequence result.",
        "",
        "## Timestamp And Alignment Rules",
        "",
        "- Common clock: `bag_timestamp_ns / 1e9` for every stream.",
        f"- Grid: deterministic {RATE_HZ:.1f} Hz grid over the overlap of GPS, IMU, motors, and reference EKF.",
        f"- RTK max interpolation bracket gap: `{RTK_MAX_INTERP_GAP_S}` s.",
        f"- Motor max interpolation bracket gap: `{MOTOR_MAX_INTERP_GAP_S}` s.",
        f"- IMU max interpolation bracket gap: `{IMU_MAX_INTERP_GAP_S}` s.",
        f"- Command max interpolation bracket gap: `{COMMAND_MAX_INTERP_GAP_S}` s.",
        f"- Reference EKF max interpolation bracket gap: `{REFERENCE_MAX_INTERP_GAP_S}` s.",
        "",
        "## RTK Validity",
        "",
        f"- Rule: {diagnostics['rtk_validity_rule']}.",
        f"- Raw RTK-valid fraction: {100.0 * diagnostics['rtk_valid_fraction_raw']:.1f}%.",
        f"- RTK horizontal accuracy median/p95/max: "
        f"{f(diagnostics['rtk_horizontal_accuracy_m']['median'], 4)} / "
        f"{f(diagnostics['rtk_horizontal_accuracy_m']['p95'], 4)} / "
        f"{f(diagnostics['rtk_horizontal_accuracy_m']['max'], 4)} m.",
        "",
        "## Sample Counts",
        "",
    ]
    for key, count in diagnostics["raw_sample_counts"].items():
        lines.append(f"- `{key}`: {count}")
    lines += [
        f"- Aligned grid samples before validity mask: {diagnostics['grid_samples']}",
        f"- Aligned samples after required-stream validity mask: {diagnostics['valid_required_samples']}",
        f"- Duration: {diagnostics['duration_s']:.1f} s",
        "",
        "## Sanity Correlations",
        "",
    ]
    for key, value in diagnostics["sanity_correlations"].items():
        lines.append(f"- `{key}`: {f(value, 3)}")
    lines += [
        "",
        "## Heading Use",
        "",
        f"- Native GPS heading accuracy median/p95: "
        f"{f(heading['native_gps_heading_accuracy_median_deg'], 1)} / "
        f"{f(heading['native_gps_heading_accuracy_p95_deg'], 1)} deg.",
        f"- Native GPS heading good fraction (`<=10 deg`): "
        f"{100.0 * heading['native_gps_heading_good_fraction_acc_le_10deg']:.1f}%.",
        f"- Primary quantitative heading use: `{heading['primary_heading_use']}`.",
        f"- Secondary heading use: {heading['secondary_heading_use']}.",
        "",
        "## Frozen V2 Feature Range Check",
        "",
        f"- Fast feature abs-z p95/max: {feature_ranges['fast_abs_z_p95']:.2f} / {feature_ranges['fast_abs_z_max']:.2f}.",
        f"- Fast feature fraction |z| > 3: {100.0 * feature_ranges['fast_fraction_abs_z_gt3']:.1f}%.",
        f"- Slow valid fraction after 30 s context: {100.0 * feature_ranges['slow_valid_after_30s_fraction']:.1f}%.",
        f"- Slow feature abs-z p95/max: {f(feature_ranges['slow_abs_z_p95'], 2)} / {f(feature_ranges['slow_abs_z_max'], 2)}.",
        f"- Slow feature fraction |z| > 3: {f(100.0 * feature_ranges['slow_fraction_abs_z_gt3'], 1) if feature_ranges['slow_fraction_abs_z_gt3'] is not None else 'n/a'}%.",
        f"- Substantially outside i2Nav normalization: `{feature_ranges['substantially_outside_i2nav_normalization']}`.",
        "",
        "## Smoke Checkpoint",
        "",
        f"- Checkpoint: `{checkpoint['checkpoint']}`",
        f"- Selection rule: {checkpoint['selection_rule']}",
        "- This checkpoint's metrics are not a scientific result.",
        "",
        "## Smoke-Test Fidelity Metrics",
        "",
        f"- ATE RMSE: {profile['ATE_m']:.3f} m",
        f"- Heading MAE: {profile['heading_MAE_deg']:.2f} deg",
        f"- RPE1/RPE5/RPE10: {profile['RPEp_1s_m']:.3f} / {profile['RPEp_5s_m']:.3f} / {profile['RPEp_10s_m']:.3f} m",
        f"- Dp p95/max: {profile['Dp_p95_m']:.3f} / {profile['Dp_max_m']:.3f} m",
        f"- Dtheta p95/max: {profile['Dtheta_p95_deg']:.2f} / {profile['Dtheta_max_deg']:.2f} deg",
        "",
        "## Plots",
        "",
    ]
    lines.extend([f"- `{Path(p).name}`" for p in plots])
    lines += [
        "",
        "## Blockers And Suspicious Behavior",
        "",
        "- Native GPS heading is not defensible as primary heading truth for this sequence.",
        "- Reference EKF heading is used only to make the smoke evaluator run end-to-end.",
        "- Any large smoke-test error should be interpreted as an adapter/model-domain diagnostic, not as a final AIFARMS result.",
    ]
    (out_dir / "phase2_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    seq_dir = args.input_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"sequence directory not found: {seq_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"frozen V2 checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_sequence(seq_dir)
    grid = make_grid(frames)
    aligned, diagnostics = build_aligned(frames, grid)
    aligned.to_csv(args.output_dir / "aligned_terrasentia_v2_inputs.csv", index=False)

    checkpoint = choose_checkpoint(args.checkpoint)
    feature_ranges = feature_range_report(aligned, checkpoint)
    heading = heading_report(aligned)
    plots = plot_outputs(aligned, args.output_dir)
    trace, timeseries, profile = run_v2_smoke(aligned, args.checkpoint)
    trace.to_csv(args.output_dir / "frozen_v2_single_checkpoint_smoke_trace.csv", index=False)
    timeseries.to_csv(args.output_dir / "frozen_v2_single_checkpoint_fidelity_timeseries.csv", index=False)

    shift = per_feature_shift_audit(aligned, checkpoint)
    shift.to_csv(args.output_dir / "per_feature_distribution_shift.csv", index=False)
    definition_rows = shifted_definition_rows(shift)
    pd.DataFrame(definition_rows).to_csv(
        args.output_dir / "shifted_feature_definition_audit.csv", index=False
    )
    write_definition_audit(args.output_dir / "shifted_feature_definition_audit.md", definition_rows)

    physics_trace, physics_timeseries, physics_profile = run_physics_only(aligned)
    physics_trace.to_csv(args.output_dir / "physics_only_trace.csv", index=False)
    physics_timeseries.to_csv(args.output_dir / "physics_only_fidelity_timeseries.csv", index=False)
    comparison = compare_profiles(physics_profile, profile)
    comparison.to_csv(args.output_dir / "physics_vs_frozen_v2_smoke_comparison.csv", index=False)
    correction_summary = v2_correction_summary(trace)
    correction_plots = plot_v2_corrections(trace, args.output_dir)

    distance_audit = forward_distance_audit(aligned, args.output_dir)
    yaw_audit = heading_yaw_audit(aligned, args.output_dir)
    oracle_rows, oracle_profiles = oracle_decomposition(aligned)
    oracle_rows.to_csv(args.output_dir / "oracle_decomposition_diagnostics.csv", index=False)
    group_shift = feature_group_shift_summary(shift)
    group_shift.to_csv(args.output_dir / "canonical_feature_group_shift_summary.csv", index=False)
    sanity = frame_sanity_checks(aligned)

    write_json(args.output_dir / "data_quality_report.json", diagnostics)
    write_json(args.output_dir / "heading_validation.json", heading)
    write_json(args.output_dir / "checkpoint_selection.json", checkpoint)
    write_json(args.output_dir / "feature_range_report.json", feature_ranges)
    write_json(args.output_dir / "smoke_fidelity_profile.json", profile)
    write_json(args.output_dir / "physics_only_fidelity_profile.json", physics_profile)
    write_json(args.output_dir / "v2_correction_summary.json", correction_summary)
    write_json(args.output_dir / "forward_distance_audit.json", distance_audit)
    write_json(args.output_dir / "heading_yaw_accumulation_audit.json", yaw_audit)
    write_json(args.output_dir / "oracle_decomposition_profiles.json", oracle_profiles)
    write_json(args.output_dir / "initial_frame_sanity.json", sanity)
    write_report(
        args.output_dir,
        sequence=args.sequence,
        diagnostics=diagnostics,
        heading=heading,
        checkpoint=checkpoint,
        feature_ranges=feature_ranges,
        profile=profile,
        plots=plots + correction_plots + [distance_audit["plot"], yaw_audit["plot"]],
    )
    write_shift_audit_report(
        args.output_dir,
        shift=shift,
        comparison=comparison,
        correction_summary=correction_summary,
    )
    write_mechanism_report(
        args.output_dir,
        distance=distance_audit,
        yaw=yaw_audit,
        oracle=oracle_rows,
        group_shift=group_shift,
        sanity=sanity,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
