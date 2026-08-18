"""Model-independent digital-twin fidelity evaluator for i2Nav trajectories.

This module operationalizes the frozen Step-2 fidelity framework.  It does not
train a model and does not depend on Twin V1/V2 internals.  The required input
is an evaluated trajectory CSV containing the physical/reference and twin
state.  A V2 prediction-trace CSV may be supplied to additionally compute
velocity/yaw-rate divergence and accumulated yaw-rate residual.

Primary run-level profile (Phi_{s,r,m}):
    ATE, heading MAE, translational RPE at 1/5/10 s,
    absolute persistent yaw-rate residual (when prediction trace is supplied),
    p95/max position divergence, p95/max heading divergence,
    and max absolute accumulated yaw-rate residual.

Important terminology:
    Dtheta is computed from the actual physical and twin heading states.
    Iomega is the time-integral of the signed yaw-rate residual.  Iomega is a
    diagnostic and is NOT used as a substitute for heading-state divergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TRAJECTORY_COLUMNS = (
    "time_s",
    "gt_east_m",
    "gt_north_m",
    "gt_heading_rad",
    "estimate_east_m",
    "estimate_north_m",
    "estimate_heading_rad",
)

PREDICTION_TRACE_COLUMNS = (
    "time_s",
    "true_delta_v_mps",
    "pred_delta_v_mps",
    "true_delta_omega_radps",
    "pred_total_delta_omega_radps",
)

DEFAULT_HORIZONS_S = (1.0, 5.0, 10.0)


def wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    angle = np.asarray(angle_rad, dtype=np.float64)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _as_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for name in columns:
        values = frame[name].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} column {name!r} contains non-finite values")


def _sampling_period(time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        raise ValueError("At least two trajectory samples are required")
    diffs = np.diff(time_s)
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing")
    dt = float(np.median(diffs))
    if dt <= 0.0:
        raise ValueError("Invalid sampling period")
    return dt


def _relative_translation(
    xy: np.ndarray,
    heading_rad: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    """Return SE(2) relative translation expressed in each start pose frame."""
    dp = xy[horizon_steps:] - xy[:-horizon_steps]
    theta0 = heading_rad[:-horizon_steps]
    c = np.cos(theta0)
    s = np.sin(theta0)

    # R(theta0)^T * (p1 - p0)
    rel_x = c * dp[:, 0] + s * dp[:, 1]
    rel_y = -s * dp[:, 0] + c * dp[:, 1]
    return np.column_stack((rel_x, rel_y))


def _relative_heading(heading_rad: np.ndarray, horizon_steps: int) -> np.ndarray:
    return wrap_angle(heading_rad[horizon_steps:] - heading_rad[:-horizon_steps])


def rpe_metrics(
    gt_xy: np.ndarray,
    gt_heading_rad: np.ndarray,
    twin_xy: np.ndarray,
    twin_heading_rad: np.ndarray,
    dt_s: float,
    horizons_s: Iterable[float] = DEFAULT_HORIZONS_S,
) -> dict[str, float]:
    """Compute the same SE(2) translational RPE used by the i2Nav evaluator.

    Translation is compared in the local frame of each trajectory's start pose
    over the requested horizon.  Rotational relative-pose MAE is also reported
    as a diagnostic.
    """
    result: dict[str, float] = {}
    n = len(gt_xy)

    for horizon_s in horizons_s:
        steps = int(round(float(horizon_s) / float(dt_s)))
        if steps <= 0:
            raise ValueError(f"Invalid RPE horizon: {horizon_s}")
        if steps >= n:
            raise ValueError(
                f"RPE horizon {horizon_s:g}s requires {steps + 1} samples; only {n} available"
            )

        gt_rel = _relative_translation(gt_xy, gt_heading_rad, steps)
        twin_rel = _relative_translation(twin_xy, twin_heading_rad, steps)
        trans_error = twin_rel - gt_rel
        trans_rmse = float(np.sqrt(np.mean(np.sum(trans_error * trans_error, axis=1))))

        gt_dtheta = _relative_heading(gt_heading_rad, steps)
        twin_dtheta = _relative_heading(twin_heading_rad, steps)
        rot_error = np.abs(wrap_angle(twin_dtheta - gt_dtheta))
        rot_mae_deg = float(np.degrees(np.mean(rot_error)))

        label = f"{int(horizon_s) if float(horizon_s).is_integer() else horizon_s:g}s"
        result[f"RPEp_{label}_m"] = trans_rmse
        result[f"RPEtheta_{label}_MAE_deg"] = rot_mae_deg

    return result


def evaluate_fidelity_frames(
    trajectory: pd.DataFrame,
    prediction_trace: pd.DataFrame | None = None,
    *,
    model: str = "unknown",
    sequence: str = "unknown",
    seed: int | None = None,
    replicate: str | None = None,
    horizons_s: Iterable[float] = DEFAULT_HORIZONS_S,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one physical/reference vs twin trajectory pair.

    Returns
    -------
    profile:
        Flat, JSON-serializable run-level fidelity record.
    timeseries:
        Per-timestamp divergence series. Dv/Domega/Iomega are NaN when no
        prediction trace is supplied.
    """
    _require_columns(trajectory, TRAJECTORY_COLUMNS, "trajectory")
    _as_finite(trajectory, TRAJECTORY_COLUMNS, "trajectory")

    time_s = trajectory["time_s"].to_numpy(dtype=np.float64)
    dt_s = _sampling_period(time_s)

    gt_xy = trajectory[["gt_east_m", "gt_north_m"]].to_numpy(dtype=np.float64)
    twin_xy = trajectory[["estimate_east_m", "estimate_north_m"]].to_numpy(dtype=np.float64)
    gt_heading = trajectory["gt_heading_rad"].to_numpy(dtype=np.float64)
    twin_heading = trajectory["estimate_heading_rad"].to_numpy(dtype=np.float64)

    position_error = np.linalg.norm(twin_xy - gt_xy, axis=1)
    signed_heading_error = wrap_angle(twin_heading - gt_heading)
    heading_error = np.abs(signed_heading_error)

    profile: dict[str, Any] = {
        "schema": "i2nav_twin_fidelity_profile_v1",
        "model": str(model),
        "sequence": str(sequence),
        "seed": None if seed is None else int(seed),
        "replicate": None if replicate is None else str(replicate),
        "n_samples": int(len(trajectory)),
        "duration_s": float(time_s[-1] - time_s[0]),
        "rate_hz": float(1.0 / dt_s),
        "ATE_m": float(np.sqrt(np.mean(position_error ** 2))),
        "heading_MAE_deg": float(np.degrees(np.mean(heading_error))),
        "Dp_p95_m": float(np.percentile(position_error, 95.0)),
        "Dp_max_m": float(np.max(position_error)),
        "Dtheta_p95_deg": float(np.degrees(np.percentile(heading_error, 95.0))),
        "Dtheta_max_deg": float(np.degrees(np.max(heading_error))),
    }

    profile.update(
        rpe_metrics(
            gt_xy,
            gt_heading,
            twin_xy,
            twin_heading,
            dt_s,
            horizons_s=horizons_s,
        )
    )

    # Default to unavailable rather than silently inventing dynamics metrics.
    dynamics_fields = {
        "yaw_bias_signed_radps": None,
        "abs_yaw_bias_radps": None,
        "abs_yaw_bias_deg_per_min": None,
        "Dv_RMSE_mps": None,
        "Dv_p95_mps": None,
        "Dv_max_mps": None,
        "Domega_RMSE_radps": None,
        "Domega_p95_radps": None,
        "Domega_max_radps": None,
        "Iomega_final_deg": None,
        "Iomega_p95_abs_deg": None,
        "Iomega_max_abs_deg": None,
    }
    profile.update(dynamics_fields)

    d_v = np.full(len(trajectory), np.nan, dtype=np.float64)
    d_omega = np.full(len(trajectory), np.nan, dtype=np.float64)
    r_v = np.full(len(trajectory), np.nan, dtype=np.float64)
    r_omega = np.full(len(trajectory), np.nan, dtype=np.float64)
    i_omega_deg = np.full(len(trajectory), np.nan, dtype=np.float64)

    if prediction_trace is not None:
        _require_columns(prediction_trace, PREDICTION_TRACE_COLUMNS, "prediction trace")
        _as_finite(prediction_trace, PREDICTION_TRACE_COLUMNS, "prediction trace")

        if len(prediction_trace) != len(trajectory):
            raise ValueError(
                "Prediction trace and evaluated trajectory have different row counts: "
                f"{len(prediction_trace)} vs {len(trajectory)}"
            )

        pred_time = prediction_trace["time_s"].to_numpy(dtype=np.float64)
        if not np.allclose(pred_time, time_s, rtol=0.0, atol=max(1e-7, dt_s * 1e-5)):
            raise ValueError("Prediction trace time_s is not aligned with evaluated trajectory")

        # Because both physical and corrected signals share the same raw ODO/IMU
        # baseline, the dynamics residual is exactly true correction - predicted
        # correction. This avoids estimating derivatives from pose.
        r_v = (
            prediction_trace["true_delta_v_mps"].to_numpy(dtype=np.float64)
            - prediction_trace["pred_delta_v_mps"].to_numpy(dtype=np.float64)
        )
        r_omega = (
            prediction_trace["true_delta_omega_radps"].to_numpy(dtype=np.float64)
            - prediction_trace["pred_total_delta_omega_radps"].to_numpy(dtype=np.float64)
        )
        d_v = np.abs(r_v)
        d_omega = np.abs(r_omega)
        i_omega_rad = np.cumsum(r_omega * dt_s)
        i_omega_deg = np.degrees(i_omega_rad)

        yaw_bias_signed = float(np.mean(r_omega))
        deg_per_min = 180.0 / math.pi * 60.0

        profile.update(
            {
                "yaw_bias_signed_radps": yaw_bias_signed,
                "abs_yaw_bias_radps": abs(yaw_bias_signed),
                "abs_yaw_bias_deg_per_min": abs(yaw_bias_signed) * deg_per_min,
                "Dv_RMSE_mps": float(np.sqrt(np.mean(r_v ** 2))),
                "Dv_p95_mps": float(np.percentile(d_v, 95.0)),
                "Dv_max_mps": float(np.max(d_v)),
                "Domega_RMSE_radps": float(np.sqrt(np.mean(r_omega ** 2))),
                "Domega_p95_radps": float(np.percentile(d_omega, 95.0)),
                "Domega_max_radps": float(np.max(d_omega)),
                "Iomega_final_deg": float(i_omega_deg[-1]),
                "Iomega_p95_abs_deg": float(np.percentile(np.abs(i_omega_deg), 95.0)),
                "Iomega_max_abs_deg": float(np.max(np.abs(i_omega_deg))),
            }
        )

    timeseries = pd.DataFrame(
        {
            "time_s": time_s,
            "Dp_m": position_error,
            "signed_heading_error_rad": signed_heading_error,
            "Dtheta_rad": heading_error,
            "Dtheta_deg": np.degrees(heading_error),
            "rv_mps": r_v,
            "Dv_mps": d_v,
            "romega_radps": r_omega,
            "Domega_radps": d_omega,
            "Iomega_deg": i_omega_deg,
        }
    )

    return profile, timeseries


def evaluate_trajectory_files(
    trajectory_path: Path | str,
    prediction_trace_path: Path | str | None = None,
    *,
    model: str = "unknown",
    sequence: str = "unknown",
    seed: int | None = None,
    replicate: str | None = None,
    horizons_s: Iterable[float] = DEFAULT_HORIZONS_S,
) -> tuple[dict[str, Any], pd.DataFrame]:
    trajectory = pd.read_csv(Path(trajectory_path))
    prediction_trace = (
        None if prediction_trace_path is None else pd.read_csv(Path(prediction_trace_path))
    )
    return evaluate_fidelity_frames(
        trajectory,
        prediction_trace,
        model=model,
        sequence=sequence,
        seed=seed,
        replicate=replicate,
        horizons_s=horizons_s,
    )


def write_json(path: Path | str, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_timeseries(path: Path | str, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the frozen digital-twin fidelity profile for one evaluated trajectory."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--prediction-trace", type=Path, default=None)
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--sequence", default="unknown")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--replicate", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-timeseries", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profile, timeseries = evaluate_trajectory_files(
        args.trajectory,
        args.prediction_trace,
        model=args.model,
        sequence=args.sequence,
        seed=args.seed,
        replicate=args.replicate,
    )
    write_json(args.output_json, profile)
    if args.output_timeseries is not None:
        write_timeseries(args.output_timeseries, timeseries)

    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
