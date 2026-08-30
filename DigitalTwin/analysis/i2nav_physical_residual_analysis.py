#!/usr/bin/env python3

"""
i2Nav physical residual analysis for frozen Twin V1.

PURPOSE
-------
Diagnose the physical causes of remaining Twin V1 trajectory divergence
before designing Twin V2.

This script is deliberately tied to the ORIGINAL V1 PreparedSequence:

    name
    files
    grid
    gt_x
    gt_y
    gt_heading
    gt_forward_speed
    gt_yaw_rate
    odo_speed
    imu_yaw_rate
    features
    target_corrections
    gnss
    odo_source

The authoritative V1 physical residual targets are therefore:

    delta_v_true =
        prepared_sequence.target_corrections[:, 0]

    delta_omega_true =
        prepared_sequence.target_corrections[:, 1]

which are exactly:

    gt_forward_speed - odo_speed

and:

    gt_yaw_rate - imu_yaw_rate

No training is performed.

No files inside results/i2nav_v1_frozen are modified.

Outputs are written separately to:

    results/i2nav_physics_residual_diagnostics/

The analysis includes:

1. Exact V1 residual targets.
2. GT body-frame forward/lateral motion.
3. Slip-angle diagnostics.
4. 1/5/10/30-second rolling residual biases.
5. Integrated yaw mismatch.
6. One-effect-at-a-time diagnostic oracle trajectories.
7. Frozen V1 learned correction vs true physical residual comparison.
8. parking02-focused decomposition.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# Constants
# =============================================================================

SEQUENCES = (
    "building00",
    "building01",
    "building02",
    "parking00",
    "parking01",
    "parking02",
    "playground00",
    "street00",
    "street01",
    "street02",
)

ROLLING_WINDOWS_SECONDS = (
    1.0,
    5.0,
    10.0,
    30.0,
)


# =============================================================================
# Generic utilities
# =============================================================================

def wrap_angle(
    angle: np.ndarray | float,
) -> np.ndarray:
    """
    Wrap radians to [-pi, pi).
    """

    angle = np.asarray(
        angle,
        dtype=float,
    )

    return (
        angle
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def rmse(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                finite**2
            )
        )
    )


def mae(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return float("nan")

    return float(
        np.mean(
            np.abs(
                finite
            )
        )
    )


def safe_mean(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return float("nan")

    return float(
        np.mean(
            finite
        )
    )


def safe_max_abs(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return float("nan")

    return float(
        np.max(
            np.abs(
                finite
            )
        )
    )


def safe_percentile_abs(
    values: np.ndarray,
    percentile: float,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return float("nan")

    return float(
        np.percentile(
            np.abs(
                finite
            ),
            percentile,
        )
    )


def safe_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    mask = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    if np.count_nonzero(
        mask
    ) < 3:

        return float("nan")

    aa = a[mask]
    bb = b[mask]

    if (
        np.std(aa) < 1e-12
        or np.std(bb) < 1e-12
    ):

        return float("nan")

    return float(
        np.corrcoef(
            aa,
            bb,
        )[0, 1]
    )


def write_json(
    path: Path,
    obj: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:

        path.write_text(
            "",
            encoding="utf-8",
        )

        return

    fieldnames = []
    seen = set()

    for row in rows:

        for key in row:

            if key not in seen:

                fieldnames.append(
                    key
                )

                seen.add(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def read_csv_dicts(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
        errors="ignore",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# =============================================================================
# Rolling statistics
# =============================================================================

def rolling_mean(
    values: np.ndarray,
    samples: int,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    samples = max(
        1,
        int(samples),
    )

    if samples == 1:

        return values.copy()

    finite = np.isfinite(
        values
    )

    numerator = np.convolve(

        np.where(
            finite,
            values,
            0.0,
        ),

        np.ones(
            samples,
            dtype=float,
        ),

        mode="same",
    )

    denominator = np.convolve(

        finite.astype(
            float
        ),

        np.ones(
            samples,
            dtype=float,
        ),

        mode="same",
    )

    output = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    valid = (
        denominator > 0
    )

    output[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return output


# =============================================================================
# Get exact defaults from original V1
# =============================================================================

def original_default_args(
    original_module,
):

    old_argv = sys.argv[:]

    try:

        sys.argv = [
            "i2nav_loso_ablation.py"
        ]

        return (
            original_module.parse_args()
        )

    finally:

        sys.argv = old_argv


# =============================================================================
# Validate actual V1 PreparedSequence structure
# =============================================================================

def validate_prepared_sequence(
    sequence,
) -> None:

    required = (

        "name",

        "files",

        "grid",

        "gt_x",

        "gt_y",

        "gt_heading",

        "gt_forward_speed",

        "gt_yaw_rate",

        "odo_speed",

        "imu_yaw_rate",

        "target_corrections",
    )

    missing = [

        name

        for name
        in required

        if not hasattr(
            sequence,
            name,
        )
    ]

    if missing:

        raise RuntimeError(
            "\nPreparedSequence does not match "
            "the frozen V1 structure.\n"
            f"Missing fields: {missing}\n"
            f"Actual fields: "
            f"{list(vars(sequence).keys())}\n"
        )


# =============================================================================
# Reconstruct physical motion
# =============================================================================

def reconstruct_physical_motion(
    prepared_sequence,
) -> dict[str, np.ndarray]:
    """
    Reconstruct physical quantities using ONLY the exact V1 PreparedSequence.

    Important distinction
    ---------------------

    AUTHORITATIVE V1 residuals:

        target_corrections[:, 0]
            = gt_forward_speed - odo_speed

        target_corrections[:, 1]
            = gt_yaw_rate - imu_yaw_rate

    Lateral velocity is not stored by V1. We estimate it from derivatives of
    the already aligned GT x/y trajectory.

    This avoids assumptions about raw groundtruth.nav column layout.
    """

    validate_prepared_sequence(
        prepared_sequence
    )

    # =========================================================================
    # Exact V1 data
    # =========================================================================

    grid = np.asarray(
        prepared_sequence.grid,
        dtype=float,
    )

    gt_x = np.asarray(
        prepared_sequence.gt_x,
        dtype=float,
    )

    gt_y = np.asarray(
        prepared_sequence.gt_y,
        dtype=float,
    )

    gt_heading = np.asarray(
        prepared_sequence.gt_heading,
        dtype=float,
    )

    gt_forward_speed = np.asarray(
        prepared_sequence.gt_forward_speed,
        dtype=float,
    )

    gt_yaw_rate = np.asarray(
        prepared_sequence.gt_yaw_rate,
        dtype=float,
    )

    odo_speed = np.asarray(
        prepared_sequence.odo_speed,
        dtype=float,
    )

    imu_yaw_rate = np.asarray(
        prepared_sequence.imu_yaw_rate,
        dtype=float,
    )

    target_corrections = np.asarray(
        prepared_sequence.target_corrections,
        dtype=float,
    )

    n = len(
        grid
    )

    # =========================================================================
    # Shape validation
    # =========================================================================

    arrays = {

        "gt_x":
            gt_x,

        "gt_y":
            gt_y,

        "gt_heading":
            gt_heading,

        "gt_forward_speed":
            gt_forward_speed,

        "gt_yaw_rate":
            gt_yaw_rate,

        "odo_speed":
            odo_speed,

        "imu_yaw_rate":
            imu_yaw_rate,
    }

    for name, values in (
        arrays.items()
    ):

        if len(values) != n:

            raise RuntimeError(
                f"{prepared_sequence.name}: "
                f"{name} has length "
                f"{len(values)}, expected {n}."
            )

    if n < 3:

        raise RuntimeError(
            f"{prepared_sequence.name}: "
            "not enough samples."
        )

    if (
        target_corrections.ndim != 2

        or target_corrections.shape[0]
        != n

        or target_corrections.shape[1]
        < 2
    ):

        raise RuntimeError(
            f"{prepared_sequence.name}: "
            "unexpected target_corrections shape: "
            f"{target_corrections.shape}"
        )

    # =========================================================================
    # Time
    # =========================================================================

    dt_values = np.diff(
        grid
    )

    dt_median = float(
        np.median(
            dt_values
        )
    )

    if (
        not np.isfinite(
            dt_median
        )

        or dt_median <= 0
    ):

        raise RuntimeError(
            f"{prepared_sequence.name}: "
            "invalid time grid."
        )

    hz = (
        1.0
        / dt_median
    )

    # =========================================================================
    # Exact V1 physical residuals
    # =========================================================================

    velocity_residual = (
        target_corrections[:, 0]
    )

    yaw_residual = (
        target_corrections[:, 1]
    )

    # =========================================================================
    # Verify target definition
    # =========================================================================

    expected_dv = (
        gt_forward_speed
        - odo_speed
    )

    expected_dw = (
        gt_yaw_rate
        - imu_yaw_rate
    )

    dv_definition_error = (
        velocity_residual
        - expected_dv
    )

    dw_definition_error = (
        yaw_residual
        - expected_dw
    )

    max_dv_definition_error = (
        safe_max_abs(
            dv_definition_error
        )
    )

    max_dw_definition_error = (
        safe_max_abs(
            dw_definition_error
        )
    )

    if (
        max_dv_definition_error
        > 1e-5

        or max_dw_definition_error
        > 1e-5
    ):

        raise RuntimeError(
            f"{prepared_sequence.name}: "
            "V1 target_corrections do not match "
            "the source-code definition.\n"
            f"dv max mismatch="
            f"{max_dv_definition_error:.6e}\n"
            f"dw max mismatch="
            f"{max_dw_definition_error:.6e}"
        )

    # =========================================================================
    # Ground-truth world velocity from aligned position trajectory
    # =========================================================================

    gt_world_vx = np.gradient(
        gt_x,
        grid,
    )

    gt_world_vy = np.gradient(
        gt_y,
        grid,
    )

    # =========================================================================
    # World -> body frame
    #
    # If psi is the world-frame heading:
    #
    # forward:
    #
    #   vx_body =
    #       vx_world cos(psi)
    #       + vy_world sin(psi)
    #
    # lateral-left:
    #
    #   vy_body =
    #       -vx_world sin(psi)
    #       + vy_world cos(psi)
    # =========================================================================

    pose_derived_forward = (

        gt_world_vx
        * np.cos(
            gt_heading
        )

        +

        gt_world_vy
        * np.sin(
            gt_heading
        )
    )

    gt_lateral_speed = (

        -gt_world_vx
        * np.sin(
            gt_heading
        )

        +

        gt_world_vy
        * np.cos(
            gt_heading
        )
    )

    # =========================================================================
    # Forward-motion consistency diagnostic
    #
    # This is NOT expected to be numerically identical because:
    #
    # - gt_forward_speed comes from V1 load_gt_forward_motion()
    # - pose_derived_forward comes from differentiation of interpolated GT x/y
    #
    # But gross disagreement would signal a frame problem.
    # =========================================================================

    forward_consistency_error = (

        pose_derived_forward
        - gt_forward_speed
    )

    forward_consistency_rmse = rmse(
        forward_consistency_error
    )

    forward_consistency_mae = mae(
        forward_consistency_error
    )

    forward_consistency_corr = (
        safe_corr(
            pose_derived_forward,
            gt_forward_speed,
        )
    )

    # =========================================================================
    # Measured accelerations
    # =========================================================================

    odo_acceleration = np.gradient(
        odo_speed,
        grid,
    )

    imu_yaw_acceleration = np.gradient(
        imu_yaw_rate,
        grid,
    )

    # =========================================================================
    # Lateral slip diagnostics
    # =========================================================================

    pose_speed_magnitude = np.hypot(
        pose_derived_forward,
        gt_lateral_speed,
    )

    slip_angle_rad = np.arctan2(
        gt_lateral_speed,
        pose_derived_forward,
    )

    # Undefined / unstable at tiny speeds.
    near_stationary = (
        pose_speed_magnitude
        < 0.05
    )

    slip_angle_rad[
        near_stationary
    ] = np.nan

    slip_angle_deg = np.rad2deg(
        slip_angle_rad
    )

    # =========================================================================
    # Curvature
    # =========================================================================

    curvature = np.full(
        n,
        np.nan,
        dtype=float,
    )

    moving = (
        np.abs(
            gt_forward_speed
        )
        >= 0.05
    )

    curvature[moving] = (

        gt_yaw_rate[moving]

        /

        gt_forward_speed[moving]
    )

    # =========================================================================
    # Integrated residuals
    # =========================================================================

    integrated_yaw_residual = np.zeros(
        n,
        dtype=float,
    )

    integrated_velocity_residual = np.zeros(
        n,
        dtype=float,
    )

    # Trapezoidal integration.
    integrated_yaw_residual[1:] = np.cumsum(

        0.5

        * (
            yaw_residual[:-1]
            + yaw_residual[1:]
        )

        * dt_values
    )

    integrated_velocity_residual[1:] = np.cumsum(

        0.5

        * (
            velocity_residual[:-1]
            + velocity_residual[1:]
        )

        * dt_values
    )

    # =========================================================================
    # Build result
    # =========================================================================

    result = {

        "t":
            grid,

        "time_from_start_s":
            grid
            - grid[0],

        # GT pose
        "gt_x":
            gt_x,

        "gt_y":
            gt_y,

        "gt_heading":
            gt_heading,

        "gt_heading_deg":
            np.rad2deg(
                gt_heading
            ),

        # Exact V1 GT motion
        "gt_forward_speed":
            gt_forward_speed,

        "gt_yaw_rate":
            gt_yaw_rate,

        # Measured V1 physical inputs
        "odo_speed":
            odo_speed,

        "imu_yaw_rate":
            imu_yaw_rate,

        # Exact V1 targets
        "velocity_residual":
            velocity_residual,

        "yaw_residual":
            yaw_residual,

        "yaw_residual_degps":
            np.rad2deg(
                yaw_residual
            ),

        # GT pose-derived planar motion
        "gt_world_vx_from_pose":
            gt_world_vx,

        "gt_world_vy_from_pose":
            gt_world_vy,

        "gt_forward_from_pose":
            pose_derived_forward,

        "gt_lateral_from_pose":
            gt_lateral_speed,

        "gt_pose_speed_magnitude":
            pose_speed_magnitude,

        # Forward velocity sanity diagnostic
        "forward_consistency_error":
            forward_consistency_error,

        # Motion derivatives
        "odo_acceleration":
            odo_acceleration,

        "imu_yaw_acceleration":
            imu_yaw_acceleration,

        # Slip / curvature
        "slip_angle_rad":
            slip_angle_rad,

        "slip_angle_deg":
            slip_angle_deg,

        "curvature_1pm":
            curvature,

        # Integrated bias
        "integrated_yaw_residual_rad":
            integrated_yaw_residual,

        "integrated_yaw_residual_deg":
            np.rad2deg(
                integrated_yaw_residual
            ),

        "integrated_velocity_residual_m":
            integrated_velocity_residual,

        # Target-definition checks
        "target_dv_definition_error":
            dv_definition_error,

        "target_dw_definition_error":
            dw_definition_error,
    }

    # =========================================================================
    # Rolling biases
    # =========================================================================

    for seconds in (
        ROLLING_WINDOWS_SECONDS
    ):

        samples = max(
            1,
            int(
                round(
                    seconds
                    * hz
                )
            ),
        )

        label = int(
            seconds
        )

        result[
            f"rolling_velocity_bias_{label}s"
        ] = rolling_mean(
            velocity_residual,
            samples,
        )

        result[
            f"rolling_yaw_bias_{label}s"
        ] = rolling_mean(
            yaw_residual,
            samples,
        )

    # Store scalar consistency diagnostics separately as constant arrays only
    # if needed by caller? Better attach via reserved private keys.
    result[
        "_forward_consistency_rmse"
    ] = np.asarray(
        [
            forward_consistency_rmse
        ]
    )

    result[
        "_forward_consistency_mae"
    ] = np.asarray(
        [
            forward_consistency_mae
        ]
    )

    result[
        "_forward_consistency_corr"
    ] = np.asarray(
        [
            forward_consistency_corr
        ]
    )

    result[
        "_hz"
    ] = np.asarray(
        [
            hz
        ]
    )

    return result


# =============================================================================
# Save physical residual time series
# =============================================================================

def save_residual_timeseries(
    path: Path,
    data: dict[str, np.ndarray],
) -> None:

    # Private scalar metadata entries begin with "_".
    fields = [

        key

        for key
        in data.keys()

        if not key.startswith(
            "_"
        )
    ]

    n = len(
        data["t"]
    )

    rows = []

    for i in range(n):

        row = {}

        for field in fields:

            value = data[
                field
            ][i]

            if np.isfinite(
                value
            ):

                row[field] = float(
                    value
                )

            else:

                row[field] = ""

        rows.append(
            row
        )

    write_csv(
        path,
        rows,
    )


# =============================================================================
# Sequence summary
# =============================================================================

def summarize_physical_sequence(
    sequence_name: str,
    data: dict[str, np.ndarray],
) -> dict[str, Any]:

    t = data["t"]

    dv = data[
        "velocity_residual"
    ]

    dw = data[
        "yaw_residual"
    ]

    slip_deg = data[
        "slip_angle_deg"
    ]

    curvature = data[
        "curvature_1pm"
    ]

    return {

        "sequence":
            sequence_name,

        "duration_s":
            float(
                t[-1]
                - t[0]
            ),

        "sample_count":
            len(t),

        "analysis_hz":
            float(
                data["_hz"][0]
            ),

        # -------------------------------------------------------------
        # Longitudinal residual
        # -------------------------------------------------------------

        "velocity_residual_mean_mps":
            safe_mean(
                dv
            ),

        "velocity_residual_mae_mps":
            mae(
                dv
            ),

        "velocity_residual_rmse_mps":
            rmse(
                dv
            ),

        "velocity_residual_max_abs_mps":
            safe_max_abs(
                dv
            ),

        "integrated_velocity_residual_final_m":
            float(
                data[
                    "integrated_velocity_residual_m"
                ][-1]
            ),

        # -------------------------------------------------------------
        # Yaw residual
        # -------------------------------------------------------------

        "yaw_residual_mean_radps":
            safe_mean(
                dw
            ),

        "yaw_residual_mean_degps":
            math.degrees(
                safe_mean(
                    dw
                )
            ),

        "yaw_residual_mae_radps":
            mae(
                dw
            ),

        "yaw_residual_rmse_radps":
            rmse(
                dw
            ),

        "yaw_residual_max_abs_radps":
            safe_max_abs(
                dw
            ),

        "integrated_yaw_residual_final_deg":
            float(
                data[
                    "integrated_yaw_residual_deg"
                ][-1]
            ),

        "integrated_yaw_residual_max_abs_deg":
            safe_max_abs(
                data[
                    "integrated_yaw_residual_deg"
                ]
            ),

        # -------------------------------------------------------------
        # Persistent rolling yaw bias
        # -------------------------------------------------------------

        "rolling_yaw_bias_1s_max_abs_radps":
            safe_max_abs(
                data[
                    "rolling_yaw_bias_1s"
                ]
            ),

        "rolling_yaw_bias_5s_max_abs_radps":
            safe_max_abs(
                data[
                    "rolling_yaw_bias_5s"
                ]
            ),

        "rolling_yaw_bias_10s_max_abs_radps":
            safe_max_abs(
                data[
                    "rolling_yaw_bias_10s"
                ]
            ),

        "rolling_yaw_bias_30s_max_abs_radps":
            safe_max_abs(
                data[
                    "rolling_yaw_bias_30s"
                ]
            ),

        # -------------------------------------------------------------
        # Lateral motion
        # -------------------------------------------------------------

        "gt_lateral_velocity_mae_mps":
            mae(
                data[
                    "gt_lateral_from_pose"
                ]
            ),

        "gt_lateral_velocity_rmse_mps":
            rmse(
                data[
                    "gt_lateral_from_pose"
                ]
            ),

        "slip_angle_mean_abs_deg":
            mae(
                slip_deg
            ),

        "slip_angle_p95_abs_deg":
            safe_percentile_abs(
                slip_deg,
                95.0,
            ),

        # -------------------------------------------------------------
        # Motion relationships
        # -------------------------------------------------------------

        "corr_velocity_residual_odo_speed":
            safe_corr(
                dv,
                data[
                    "odo_speed"
                ],
            ),

        "corr_velocity_residual_acceleration":
            safe_corr(
                dv,
                data[
                    "odo_acceleration"
                ],
            ),

        "corr_yaw_residual_imu_yaw_rate":
            safe_corr(
                dw,
                data[
                    "imu_yaw_rate"
                ],
            ),

        "corr_yaw_residual_abs_curvature":
            safe_corr(
                dw,
                np.abs(
                    curvature
                ),
            ),

        "corr_abs_slip_abs_curvature":
            safe_corr(
                np.abs(
                    slip_deg
                ),
                np.abs(
                    curvature
                ),
            ),

        # -------------------------------------------------------------
        # Pose-velocity sanity diagnostic
        # -------------------------------------------------------------

        "pose_vs_v1_forward_rmse_mps":
            float(
                data[
                    "_forward_consistency_rmse"
                ][0]
            ),

        "pose_vs_v1_forward_mae_mps":
            float(
                data[
                    "_forward_consistency_mae"
                ][0]
            ),

        "pose_vs_v1_forward_correlation":
            float(
                data[
                    "_forward_consistency_corr"
                ][0]
            ),
    }


# =============================================================================
# Trajectory integration
# =============================================================================

def integrate_planar_motion(
    t: np.ndarray,
    vx_body: np.ndarray,
    vy_body: np.ndarray,
    yaw_rate: np.ndarray,
    initial_x: float,
    initial_y: float,
    initial_heading: float,
) -> np.ndarray:

    t = np.asarray(
        t,
        dtype=float,
    )

    vx_body = np.asarray(
        vx_body,
        dtype=float,
    )

    vy_body = np.asarray(
        vy_body,
        dtype=float,
    )

    yaw_rate = np.asarray(
        yaw_rate,
        dtype=float,
    )

    n = len(t)

    states = np.zeros(
        (n, 3),
        dtype=float,
    )

    states[0] = [

        float(
            initial_x
        ),

        float(
            initial_y
        ),

        float(
            initial_heading
        ),
    ]

    for k in range(
        1,
        n,
    ):

        dt = float(
            t[k]
            - t[k - 1]
        )

        if (
            not np.isfinite(
                dt
            )

            or dt <= 0
        ):

            states[k] = (
                states[k - 1]
            )

            continue

        # Average body velocity and yaw rate across interval.
        vx = 0.5 * (
            vx_body[k - 1]
            + vx_body[k]
        )

        vy = 0.5 * (
            vy_body[k - 1]
            + vy_body[k]
        )

        omega = 0.5 * (
            yaw_rate[k - 1]
            + yaw_rate[k]
        )

        psi0 = states[
            k - 1,
            2,
        ]

        psi_mid = (
            psi0
            + 0.5
            * omega
            * dt
        )

        world_vx = (

            vx
            * math.cos(
                psi_mid
            )

            -

            vy
            * math.sin(
                psi_mid
            )
        )

        world_vy = (

            vx
            * math.sin(
                psi_mid
            )

            +

            vy
            * math.cos(
                psi_mid
            )
        )

        states[
            k,
            0,
        ] = (

            states[
                k - 1,
                0,
            ]

            +

            world_vx
            * dt
        )

        states[
            k,
            1,
        ] = (

            states[
                k - 1,
                1,
            ]

            +

            world_vy
            * dt
        )

        states[
            k,
            2,
        ] = (

            psi0
            + omega
            * dt
        )

    states[
        :,
        2,
    ] = wrap_angle(
        states[
            :,
            2,
        ]
    )

    return states


# =============================================================================
# Diagnostic metrics
# =============================================================================

def diagnostic_trajectory_metrics(
    states: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    gt_heading: np.ndarray,
    hz: float,
) -> dict[str, float]:
    """
    Internal diagnostic metrics.

    These are NOT claimed to be the official i2Nav evaluator.
    """

    gt_xy = np.column_stack(
        [
            gt_x,
            gt_y,
        ]
    )

    estimate_xy = states[
        :,
        :2,
    ]

    estimate_heading = states[
        :,
        2,
    ]

    position_error = np.linalg.norm(
        estimate_xy
        - gt_xy,
        axis=1,
    )

    heading_error = wrap_angle(
        estimate_heading
        - gt_heading
    )

    def rpe_translation(
        seconds: float,
    ) -> float:

        lag = max(
            1,
            int(
                round(
                    seconds
                    * hz
                )
            ),
        )

        if (
            len(states)
            <= lag
        ):

            return float("nan")

        errors = []

        for i in range(
            len(states)
            - lag
        ):

            j = (
                i + lag
            )

            gt_delta_world = (
                gt_xy[j]
                - gt_xy[i]
            )

            estimate_delta_world = (
                estimate_xy[j]
                - estimate_xy[i]
            )

            # Express both displacements relative to their own starting
            # heading, matching a finite-horizon relative-motion concept.
            c_gt = math.cos(
                -gt_heading[i]
            )

            s_gt = math.sin(
                -gt_heading[i]
            )

            c_est = math.cos(
                -estimate_heading[i]
            )

            s_est = math.sin(
                -estimate_heading[i]
            )

            R_gt = np.array(
                [
                    [
                        c_gt,
                        -s_gt,
                    ],
                    [
                        s_gt,
                        c_gt,
                    ],
                ]
            )

            R_est = np.array(
                [
                    [
                        c_est,
                        -s_est,
                    ],
                    [
                        s_est,
                        c_est,
                    ],
                ]
            )

            local_gt = (
                R_gt
                @ gt_delta_world
            )

            local_estimate = (
                R_est
                @ estimate_delta_world
            )

            errors.append(
                np.linalg.norm(
                    local_estimate
                    - local_gt
                )
            )

        return rmse(
            np.asarray(
                errors,
                dtype=float,
            )
        )

    return {

        "ate_rmse_m":
            rmse(
                position_error
            ),

        "position_mae_m":
            mae(
                position_error
            ),

        "position_p95_m":
            float(
                np.percentile(
                    position_error,
                    95.0,
                )
            ),

        "final_position_error_m":
            float(
                position_error[-1]
            ),

        "heading_mae_deg":
            math.degrees(
                mae(
                    heading_error
                )
            ),

        "heading_final_error_deg":
            math.degrees(
                float(
                    abs(
                        heading_error[-1]
                    )
                )
            ),

        "rpe_1s_trans_rmse_m":
            rpe_translation(
                1.0
            ),

        "rpe_5s_trans_rmse_m":
            rpe_translation(
                5.0
            ),

        "rpe_10s_trans_rmse_m":
            rpe_translation(
                10.0
            ),
    }


# =============================================================================
# Oracle decomposition
# =============================================================================

def run_oracle_experiments(
    sequence_name: str,
    data: dict[str, np.ndarray],
) -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:

    t = data[
        "t"
    ]

    hz = float(
        data[
            "_hz"
        ][0]
    )

    n = len(t)

    zero_lateral = np.zeros(
        n,
        dtype=float,
    )

    # =========================================================================
    # Diagnostic experiments
    # =========================================================================

    experiments = {

        # Actual lightweight physical input baseline.
        "fixed_measured":

            (
                data[
                    "odo_speed"
                ],

                zero_lateral,

                data[
                    "imu_yaw_rate"
                ],
            ),

        # Perfect angular rate, measured ODO speed.
        "oracle_gt_yaw":

            (
                data[
                    "odo_speed"
                ],

                zero_lateral,

                data[
                    "gt_yaw_rate"
                ],
            ),

        # Perfect forward speed, measured angular rate.
        "oracle_gt_forward_speed":

            (
                data[
                    "gt_forward_speed"
                ],

                zero_lateral,

                data[
                    "imu_yaw_rate"
                ],
            ),

        # Perfect forward speed + perfect yaw rate, still nonholonomic.
        "oracle_gt_speed_and_yaw":

            (
                data[
                    "gt_forward_speed"
                ],

                zero_lateral,

                data[
                    "gt_yaw_rate"
                ],
            ),

        # Pose-derived full planar velocity + GT yaw rate.
        #
        # This tests how much error remains after allowing lateral motion.
        "oracle_full_planar_pose_motion":

            (
                data[
                    "gt_forward_from_pose"
                ],

                data[
                    "gt_lateral_from_pose"
                ],

                data[
                    "gt_yaw_rate"
                ],
            ),
    }

    trajectories = {}
    metric_rows = []

    for experiment_name, (
        vx,
        vy,
        omega,
    ) in experiments.items():

        states = integrate_planar_motion(

            t=t,

            vx_body=vx,

            vy_body=vy,

            yaw_rate=omega,

            initial_x=float(
                data[
                    "gt_x"
                ][0]
            ),

            initial_y=float(
                data[
                    "gt_y"
                ][0]
            ),

            initial_heading=float(
                data[
                    "gt_heading"
                ][0]
            ),
        )

        trajectories[
            experiment_name
        ] = states

        metrics = (
            diagnostic_trajectory_metrics(

                states=states,

                gt_x=data[
                    "gt_x"
                ],

                gt_y=data[
                    "gt_y"
                ],

                gt_heading=data[
                    "gt_heading"
                ],

                hz=hz,
            )
        )

        metric_rows.append(
            {
                "sequence":
                    sequence_name,

                "experiment":
                    experiment_name,

                **metrics,
            }
        )

    fixed = next(

        row

        for row
        in metric_rows

        if row[
            "experiment"
        ] == "fixed_measured"
    )

    fixed_ate = float(
        fixed[
            "ate_rmse_m"
        ]
    )

    for row in metric_rows:

        current = float(
            row[
                "ate_rmse_m"
            ]
        )

        if (
            np.isfinite(
                fixed_ate
            )

            and fixed_ate > 0
        ):

            row[
                "ate_reduction_vs_fixed_pct"
            ] = (

                100.0

                * (
                    fixed_ate
                    - current
                )

                / fixed_ate
            )

        else:

            row[
                "ate_reduction_vs_fixed_pct"
            ] = float("nan")

    return (
        metric_rows,
        trajectories,
    )


# =============================================================================
# Save oracle trajectory table
# =============================================================================

def save_oracle_trajectories(
    path: Path,
    data: dict[str, np.ndarray],
    trajectories: dict[str, np.ndarray],
) -> None:

    n = len(
        data[
            "t"
        ]
    )

    rows = []

    for i in range(n):

        row = {

            "time_from_start_s":
                float(
                    data[
                        "time_from_start_s"
                    ][i]
                ),

            "gt_x":
                float(
                    data[
                        "gt_x"
                    ][i]
                ),

            "gt_y":
                float(
                    data[
                        "gt_y"
                    ][i]
                ),

            "gt_heading":
                float(
                    data[
                        "gt_heading"
                    ][i]
                ),
        }

        for (
            experiment_name,
            states,
        ) in trajectories.items():

            row[
                f"{experiment_name}_x"
            ] = float(
                states[
                    i,
                    0,
                ]
            )

            row[
                f"{experiment_name}_y"
            ] = float(
                states[
                    i,
                    1,
                ]
            )

            row[
                f"{experiment_name}_heading"
            ] = float(
                states[
                    i,
                    2,
                ]
            )

        rows.append(
            row
        )

    write_csv(
        path,
        rows,
    )


# =============================================================================
# Frozen V1 metrics
# =============================================================================

def load_v1_fold_metrics(
    frozen_dir: Path,
) -> dict[str, dict[str, float]]:

    path = (
        frozen_dir
        / "canonical_metrics_per_fold.csv"
    )

    if not path.exists():

        return {}

    rows = read_csv_dicts(
        path
    )

    result = {}

    for row in rows:

        sequence = row.get(
            "test_sequence",
            "",
        )

        if not sequence:
            continue

        converted = {}

        for (
            key,
            value,
        ) in row.items():

            if (
                key
                == "test_sequence"

                or value in (
                    None,
                    "",
                )
            ):

                continue

            try:

                converted[
                    key
                ] = float(
                    value
                )

            except Exception:

                continue

        result[
            sequence
        ] = converted

    return result


def attach_v1_fold_metrics(
    physical_rows: list[dict[str, Any]],
    v1_metrics: dict[
        str,
        dict[str, float],
    ],
) -> None:

    for row in physical_rows:

        sequence = row[
            "sequence"
        ]

        metrics = v1_metrics.get(
            sequence
        )

        if metrics is None:
            continue

        for key, value in (
            metrics.items()
        ):

            row[
                f"v1_{key}"
            ] = value


# =============================================================================
# Compare V1 learned corrections against actual V1 targets
# =============================================================================

def compare_v1_corrections(
    frozen_dir: Path,
    manifest: dict[str, Any],
    physical_by_sequence: dict[
        str,
        dict[str, np.ndarray],
    ],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    per_run = []
    timeseries = []

    for run in manifest.get(
        "runs",
        [],
    ):

        sequence_name = run[
            "test_sequence"
        ]

        replicate = run[
            "replicate"
        ]

        fold = int(
            run[
                "fold"
            ]
        )

        canonical_path = (

            frozen_dir

            / run[
                "canonical_trajectory"
            ]
        )

        if not canonical_path.exists():

            print(
                f"[warning] Missing canonical trajectory: "
                f"{canonical_path}"
            )

            continue

        rows = read_csv_dicts(
            canonical_path
        )

        if not rows:
            continue

        required_columns = (

            "time_s",

            "delta_v_mps",

            "delta_omega_radps",
        )

        missing = [

            column

            for column
            in required_columns

            if column
            not in rows[0]
        ]

        if missing:

            print(
                f"[warning] {canonical_path.name}: "
                f"missing columns {missing}"
            )

            continue

        v1_time = np.asarray(
            [
                float(
                    row[
                        "time_s"
                    ]
                )
                for row
                in rows
            ],
            dtype=float,
        )

        v1_dv = np.asarray(
            [
                float(
                    row[
                        "delta_v_mps"
                    ]
                )
                for row
                in rows
            ],
            dtype=float,
        )

        v1_dw = np.asarray(
            [
                float(
                    row[
                        "delta_omega_radps"
                    ]
                )
                for row
                in rows
            ],
            dtype=float,
        )

        physical = (
            physical_by_sequence[
                sequence_name
            ]
        )

        true_dv_full = physical[
            "velocity_residual"
        ]

        true_dw_full = physical[
            "yaw_residual"
        ]

        physical_time = physical[
            "time_from_start_s"
        ]

        # Canonical V1 trajectories normally have the same length as the
        # prepared sequence. Prefer exact index alignment in that case.
        if (
            len(v1_dv)
            == len(
                true_dv_full
            )
        ):

            true_dv = (
                true_dv_full.copy()
            )

            true_dw = (
                true_dw_full.copy()
            )

            aligned_time = (
                physical_time.copy()
            )

        else:

            # Fallback: normalize canonical timestamps and interpolate.
            normalized_v1_time = (
                v1_time
                - v1_time[0]
            )

            true_dv = np.interp(
                normalized_v1_time,
                physical_time,
                true_dv_full,
            )

            true_dw = np.interp(
                normalized_v1_time,
                physical_time,
                true_dw_full,
            )

            aligned_time = (
                normalized_v1_time
            )

        dv_error = (
            v1_dv
            - true_dv
        )

        dw_error = (
            v1_dw
            - true_dw
        )

        remaining_yaw_residual = (
            true_dw
            - v1_dw
        )

        remaining_velocity_residual = (
            true_dv
            - v1_dv
        )

        per_run.append(
            {

                "replicate":
                    replicate,

                "fold":
                    fold,

                "sequence":
                    sequence_name,

                "n_samples":
                    len(
                        v1_dv
                    ),

                # Delta-v
                "true_delta_v_mean_mps":
                    safe_mean(
                        true_dv
                    ),

                "v1_delta_v_mean_mps":
                    safe_mean(
                        v1_dv
                    ),

                "delta_v_prediction_rmse_mps":
                    rmse(
                        dv_error
                    ),

                "delta_v_prediction_mae_mps":
                    mae(
                        dv_error
                    ),

                "delta_v_prediction_correlation":
                    safe_corr(
                        true_dv,
                        v1_dv,
                    ),

                "remaining_mean_velocity_residual_mps":
                    safe_mean(
                        remaining_velocity_residual
                    ),

                # Delta omega
                "true_delta_omega_mean_radps":
                    safe_mean(
                        true_dw
                    ),

                "v1_delta_omega_mean_radps":
                    safe_mean(
                        v1_dw
                    ),

                "delta_omega_prediction_rmse_radps":
                    rmse(
                        dw_error
                    ),

                "delta_omega_prediction_mae_radps":
                    mae(
                        dw_error
                    ),

                "delta_omega_prediction_correlation":
                    safe_corr(
                        true_dw,
                        v1_dw,
                    ),

                # Most important persistent-bias diagnostic.
                "remaining_mean_yaw_residual_radps":
                    safe_mean(
                        remaining_yaw_residual
                    ),

                "remaining_mean_yaw_residual_deg_per_min":
                    (

                        safe_mean(
                            remaining_yaw_residual
                        )

                        * 180.0

                        / math.pi

                        * 60.0
                    ),
            }
        )

        for i in range(
            len(
                v1_dv
            )
        ):

            timeseries.append(
                {

                    "replicate":
                        replicate,

                    "fold":
                        fold,

                    "sequence":
                        sequence_name,

                    "time_from_start_s":
                        float(
                            aligned_time[
                                i
                            ]
                        ),

                    "true_delta_v_mps":
                        float(
                            true_dv[
                                i
                            ]
                        ),

                    "v1_delta_v_mps":
                        float(
                            v1_dv[
                                i
                            ]
                        ),

                    "remaining_velocity_residual_mps":
                        float(
                            remaining_velocity_residual[
                                i
                            ]
                        ),

                    "true_delta_omega_radps":
                        float(
                            true_dw[
                                i
                            ]
                        ),

                    "v1_delta_omega_radps":
                        float(
                            v1_dw[
                                i
                            ]
                        ),

                    "remaining_yaw_residual_radps":
                        float(
                            remaining_yaw_residual[
                                i
                            ]
                        ),
                }
            )

    return (
        per_run,
        timeseries,
    )


# =============================================================================
# Sequence-level relationship to frozen V1 performance
# =============================================================================

def analyze_relationship_to_v1(
    physical_rows: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    usable = [

        row

        for row
        in physical_rows

        if (
            "v1_ate_rmse_m_mean"
            in row

            and np.isfinite(
                float(
                    row[
                        "v1_ate_rmse_m_mean"
                    ]
                )
            )
        )
    ]

    if len(usable) < 3:

        return {

            "available":
                False,

            "reason":
                "Fewer than three sequences had frozen V1 fold metrics.",
        }

    v1_ate = np.asarray(
        [
            float(
                row[
                    "v1_ate_rmse_m_mean"
                ]
            )

            for row
            in usable
        ],
        dtype=float,
    )

    candidate_explanations = {

        "abs_mean_velocity_residual":

            np.asarray(
                [
                    abs(
                        row[
                            "velocity_residual_mean_mps"
                        ]
                    )

                    for row
                    in usable
                ]
            ),

        "velocity_residual_rmse":

            np.asarray(
                [
                    row[
                        "velocity_residual_rmse_mps"
                    ]

                    for row
                    in usable
                ]
            ),

        "abs_mean_yaw_residual":

            np.asarray(
                [
                    abs(
                        row[
                            "yaw_residual_mean_radps"
                        ]
                    )

                    for row
                    in usable
                ]
            ),

        "yaw_residual_rmse":

            np.asarray(
                [
                    row[
                        "yaw_residual_rmse_radps"
                    ]

                    for row
                    in usable
                ]
            ),

        "integrated_yaw_max_abs":

            np.asarray(
                [
                    row[
                        "integrated_yaw_residual_max_abs_deg"
                    ]

                    for row
                    in usable
                ]
            ),

        "rolling_30s_yaw_bias_max_abs":

            np.asarray(
                [
                    row[
                        "rolling_yaw_bias_30s_max_abs_radps"
                    ]

                    for row
                    in usable
                ]
            ),

        "lateral_velocity_rmse":

            np.asarray(
                [
                    row[
                        "gt_lateral_velocity_rmse_mps"
                    ]

                    for row
                    in usable
                ]
            ),

        "slip_angle_mean_abs":

            np.asarray(
                [
                    row[
                        "slip_angle_mean_abs_deg"
                    ]

                    for row
                    in usable
                ]
            ),
    }

    correlations = {

        name:
            safe_corr(
                values,
                v1_ate,
            )

        for (
            name,
            values,
        )
        in candidate_explanations.items()
    }

    return {

        "available":
            True,

        "n_sequences":
            len(
                usable
            ),

        "warning":
            (
                "Only 10 held-out sequences exist. "
                "These sequence-level correlations are exploratory diagnostics, "
                "not inferential evidence."
            ),

        "correlation_with_frozen_v1_fold_mean_ate":
            correlations,
    }


# =============================================================================
# Plots
# =============================================================================

def generate_plots(
    output_dir: Path,
    physical_by_sequence: dict[
        str,
        dict[str, np.ndarray],
    ],
) -> None:

    try:

        import matplotlib.pyplot as plt

    except Exception as exc:

        print(
            "[warning] matplotlib unavailable; "
            f"plots skipped: {exc}"
        )

        return

    plot_dir = (
        output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        sequence_name,
        data,
    ) in physical_by_sequence.items():

        t = data[
            "time_from_start_s"
        ]

        # ---------------------------------------------------------------------
        # Yaw-rate residual
        # ---------------------------------------------------------------------

        plt.figure(
            figsize=(
                10,
                4,
            )
        )

        plt.plot(
            t,
            data[
                "yaw_residual"
            ],
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "GT yaw rate - IMU yaw rate (rad/s)"
        )

        plt.title(
            f"{sequence_name}: yaw-rate physical residual"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir
            / f"{sequence_name}_yaw_residual.png",
            dpi=160,
        )

        plt.close()

        # ---------------------------------------------------------------------
        # Integrated yaw mismatch
        # ---------------------------------------------------------------------

        plt.figure(
            figsize=(
                10,
                4,
            )
        )

        plt.plot(
            t,
            data[
                "integrated_yaw_residual_deg"
            ],
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "Integrated yaw residual (deg)"
        )

        plt.title(
            f"{sequence_name}: accumulated yaw mismatch"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir
            / f"{sequence_name}_integrated_yaw_residual.png",
            dpi=160,
        )

        plt.close()

        # ---------------------------------------------------------------------
        # Forward-velocity residual
        # ---------------------------------------------------------------------

        plt.figure(
            figsize=(
                10,
                4,
            )
        )

        plt.plot(
            t,
            data[
                "velocity_residual"
            ],
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "GT forward speed - ODO speed (m/s)"
        )

        plt.title(
            f"{sequence_name}: forward-velocity residual"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir
            / f"{sequence_name}_velocity_residual.png",
            dpi=160,
        )

        plt.close()

        # ---------------------------------------------------------------------
        # Pose-derived lateral motion
        # ---------------------------------------------------------------------

        plt.figure(
            figsize=(
                10,
                4,
            )
        )

        plt.plot(
            t,
            data[
                "gt_lateral_from_pose"
            ],
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "Pose-derived GT lateral velocity (m/s)"
        )

        plt.title(
            f"{sequence_name}: lateral body motion"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir
            / f"{sequence_name}_lateral_velocity.png",
            dpi=160,
        )

        plt.close()

        # ---------------------------------------------------------------------
        # Slip-angle diagnostic
        # ---------------------------------------------------------------------

        plt.figure(
            figsize=(
                10,
                4,
            )
        )

        plt.plot(
            t,
            data[
                "slip_angle_deg"
            ],
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "Pose-derived slip angle beta (deg)"
        )

        plt.title(
            f"{sequence_name}: lateral slip-angle diagnostic"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir
            / f"{sequence_name}_slip_angle.png",
            dpi=160,
        )

        plt.close()

    # =========================================================================
    # 30-second persistent yaw bias comparison
    # =========================================================================

    plt.figure(
        figsize=(
            11,
            5,
        )
    )

    for (
        sequence_name,
        data,
    ) in physical_by_sequence.items():

        t = data[
            "time_from_start_s"
        ]

        if sequence_name == "parking02":

            line_width = 2.5
            opacity = 1.0

        elif sequence_name == "parking01":

            line_width = 1.8
            opacity = 0.8

        else:

            line_width = 0.8
            opacity = 0.30

        plt.plot(

            t,

            data[
                "rolling_yaw_bias_30s"
            ],

            linewidth=line_width,

            alpha=opacity,

            label=(
                sequence_name

                if sequence_name
                in (
                    "parking01",
                    "parking02",
                )

                else None
            ),
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "30-second rolling yaw residual (rad/s)"
    )

    plt.title(
        "Persistent yaw-rate residual: parking02 versus other sequences"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plot_dir
        / "parking02_vs_all_30s_yaw_bias.png",
        dpi=180,
    )

    plt.close()


# =============================================================================
# Print helpers
# =============================================================================

def print_sequence_summary(
    row: dict[str, Any],
) -> None:

    print()
    print(
        row[
            "sequence"
        ]
    )

    print(
        "  duration                  : "
        f"{row['duration_s']:.1f} s"
    )

    print(
        "  velocity residual mean    : "
        f"{row['velocity_residual_mean_mps']:+.6f} m/s"
    )

    print(
        "  velocity residual RMSE    : "
        f"{row['velocity_residual_rmse_mps']:.6f} m/s"
    )

    print(
        "  yaw residual mean         : "
        f"{row['yaw_residual_mean_radps']:+.6e} rad/s "
        f"({row['yaw_residual_mean_degps']:+.6f} deg/s)"
    )

    print(
        "  yaw residual RMSE         : "
        f"{row['yaw_residual_rmse_radps']:.6f} rad/s"
    )

    print(
        "  final integrated yaw      : "
        f"{row['integrated_yaw_residual_final_deg']:+.3f} deg"
    )

    print(
        "  max |integrated yaw|      : "
        f"{row['integrated_yaw_residual_max_abs_deg']:.3f} deg"
    )

    print(
        "  max |30 s yaw bias|       : "
        f"{row['rolling_yaw_bias_30s_max_abs_radps']:.6e} rad/s"
    )

    print(
        "  lateral velocity RMSE     : "
        f"{row['gt_lateral_velocity_rmse_mps']:.6f} m/s"
    )

    print(
        "  mean |slip angle|         : "
        f"{row['slip_angle_mean_abs_deg']:.3f} deg"
    )

    print(
        "  pose-vs-V1 forward corr   : "
        f"{row['pose_vs_v1_forward_correlation']:.6f}"
    )

    if (
        "v1_ate_rmse_m_mean"
        in row
    ):

        print(
            "  frozen V1 fold ATE        : "
            f"{row['v1_ate_rmse_m_mean']:.6f} m"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "public_datasets/im2nav"
        ),
    )

    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path(
            "results/i2nav_v1_frozen"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/i2nav_physics_residual_diagnostics"
        ),
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
    )

    args = parser.parse_args()

    data_root = (
        args.root.resolve()
    )

    frozen_dir = (
        args.frozen_dir.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    # =========================================================================
    # Validate paths
    # =========================================================================

    if not data_root.exists():

        raise FileNotFoundError(
            f"Dataset root not found:\n"
            f"{data_root}"
        )

    if not frozen_dir.exists():

        raise FileNotFoundError(
            f"Frozen V1 directory not found:\n"
            f"{frozen_dir}"
        )

    if output_dir == frozen_dir:

        raise RuntimeError(
            "Physical residual output directory may not be "
            "the frozen V1 directory."
        )

    manifest_path = (
        frozen_dir
        / "FROZEN_MANIFEST.json"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            "Frozen V1 manifest is missing:\n"
            f"{manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get(
            "status"
        )
        != "FROZEN"
    ):

        raise RuntimeError(
            "FROZEN_MANIFEST.json does not report status=FROZEN."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # Load EXACT original V1 module
    # =========================================================================

    original = importlib.import_module(
        "DigitalTwin.analysis.i2nav_loso_ablation"
    )

    defaults = (
        original_default_args(
            original
        )
    )

    hz = float(
        defaults.rate_hz
    )

    print()
    print("=" * 90)
    print("i2Nav PHYSICAL RESIDUAL ANALYSIS — FROZEN V1")
    print("=" * 90)
    print()

    print(
        f"Dataset root : {data_root}"
    )

    print(
        f"Frozen V1    : {frozen_dir}"
    )

    print(
        f"Output       : {output_dir}"
    )

    print()
    print(
        f"Analysis rate: {hz:.3f} Hz"
    )

    # =========================================================================
    # Discover dataset using exact V1 discovery
    # =========================================================================

    discovered_list = (
        original.discover_files(
            data_root
        )
    )

    discovered = {

        item.name:
            item

        for item
        in discovered_list
    }

    missing_sequences = [

        sequence

        for sequence
        in SEQUENCES

        if sequence
        not in discovered
    ]

    if missing_sequences:

        raise RuntimeError(
            "V1 dataset discovery did not find:\n"
            f"{missing_sequences}"
        )

    # =========================================================================
    # Physical residual reconstruction
    # =========================================================================

    print()
    print(
        "Reconstructing physical residuals..."
    )
    print()

    physical_by_sequence = {}

    physical_summary_rows = []

    oracle_rows = []

    for sequence_index, sequence_name in enumerate(
        SEQUENCES,
        start=1,
    ):

        print(
            f"[{sequence_index:02d}/10] "
            f"{sequence_name}"
        )

        files = discovered[
            sequence_name
        ]

        prepared = (
            original.prepare_sequence(

                files,

                hz=
                    defaults.rate_hz,

                imu_yaw_sign=
                    defaults.imu_yaw_sign,

                gnss_sigma_max_m=
                    defaults.gnss_sigma_max_m,

                gnss_anchor_count=
                    defaults.gnss_anchor_count,
            )
        )

        validate_prepared_sequence(
            prepared
        )

        physical = (
            reconstruct_physical_motion(
                prepared
            )
        )

        physical_by_sequence[
            sequence_name
        ] = physical

        print(
            "    target definition check: PASS"
        )

        print(
            "    pose-derived forward vs V1 forward: "
            f"RMSE="
            f"{physical['_forward_consistency_rmse'][0]:.6f} m/s, "
            f"corr="
            f"{physical['_forward_consistency_corr'][0]:.6f}"
        )

        # ---------------------------------------------------------------------
        # Save raw residual table
        # ---------------------------------------------------------------------

        save_residual_timeseries(

            output_dir
            / "residual_timeseries"
            / (
                f"{sequence_name}_"
                "physical_residuals.csv"
            ),

            physical,
        )

        # ---------------------------------------------------------------------
        # Sequence summary
        # ---------------------------------------------------------------------

        physical_summary_rows.append(
            summarize_physical_sequence(
                sequence_name,
                physical,
            )
        )

        # ---------------------------------------------------------------------
        # Diagnostic oracle experiments
        # ---------------------------------------------------------------------

        (
            sequence_oracle_rows,
            oracle_trajectories,
        ) = run_oracle_experiments(
            sequence_name,
            physical,
        )

        oracle_rows.extend(
            sequence_oracle_rows
        )

        save_oracle_trajectories(

            output_dir
            / "oracle_trajectories"
            / (
                f"{sequence_name}_"
                "oracle_trajectories.csv"
            ),

            physical,

            oracle_trajectories,
        )

    # =========================================================================
    # Attach frozen V1 fold metrics
    # =========================================================================

    v1_fold_metrics = (
        load_v1_fold_metrics(
            frozen_dir
        )
    )

    attach_v1_fold_metrics(
        physical_summary_rows,
        v1_fold_metrics,
    )

    # =========================================================================
    # Save physical summaries
    # =========================================================================

    write_csv(

        output_dir
        / "physical_residual_summary.csv",

        physical_summary_rows,
    )

    write_csv(

        output_dir
        / "oracle_experiment_results.csv",

        oracle_rows,
    )

    # =========================================================================
    # Compare frozen V1 learned corrections against true V1 targets
    # =========================================================================

    print()
    print(
        "Comparing frozen V1 learned corrections "
        "against exact V1 physical residual targets..."
    )

    (
        correction_summary_rows,
        correction_timeseries_rows,
    ) = compare_v1_corrections(

        frozen_dir,

        manifest,

        physical_by_sequence,
    )

    if correction_summary_rows:

        write_csv(

            output_dir
            / "v1_vs_physical_residual_per_run.csv",

            correction_summary_rows,
        )

        write_csv(

            output_dir
            / "v1_vs_physical_residual_timeseries.csv",

            correction_timeseries_rows,
        )

        print(
            f"  compared "
            f"{len(correction_summary_rows)} "
            "frozen V1 runs"
        )

    else:

        print(
            "  [warning] no V1 correction comparisons generated"
        )

    # =========================================================================
    # Sequence-level physical factors vs frozen V1 ATE
    # =========================================================================

    relationship_summary = (
        analyze_relationship_to_v1(
            physical_summary_rows
        )
    )

    write_json(

        output_dir
        / "physical_residual_vs_v1_performance.json",

        relationship_summary,
    )

    # =========================================================================
    # Plots
    # =========================================================================

    if not args.no_plots:

        print()
        print(
            "Generating diagnostic plots..."
        )

        generate_plots(
            output_dir,
            physical_by_sequence,
        )

    # =========================================================================
    # Print all sequence summaries
    # =========================================================================

    print()
    print("=" * 90)
    print("PHYSICAL RESIDUAL SUMMARY")
    print("=" * 90)

    for row in (
        physical_summary_rows
    ):

        print_sequence_summary(
            row
        )

    # =========================================================================
    # parking02 oracle decomposition
    # =========================================================================

    parking02_oracles = [

        row

        for row
        in oracle_rows

        if row[
            "sequence"
        ] == "parking02"
    ]

    print()
    print("=" * 90)
    print("PARKING02 ORACLE DECOMPOSITION")
    print("=" * 90)
    print()

    for row in (
        parking02_oracles
    ):

        print(

            f"{row['experiment']:<34} "

            f"ATE="
            f"{row['ate_rmse_m']:.6f} m   "

            f"heading="
            f"{row['heading_mae_deg']:.3f} deg   "

            f"RPE10="
            f"{row['rpe_10s_trans_rmse_m']:.6f} m   "

            f"ATE reduction="
            f"{row['ate_reduction_vs_fixed_pct']:+.1f}%"
        )

    # =========================================================================
    # parking02 V1 correction behavior
    # =========================================================================

    parking02_corrections = [

        row

        for row
        in correction_summary_rows

        if row[
            "sequence"
        ] == "parking02"
    ]

    if parking02_corrections:

        print()
        print("=" * 90)
        print("PARKING02 V1 CORRECTION DIAGNOSTIC")
        print("=" * 90)
        print()

        for row in (
            parking02_corrections
        ):

            print(
                f"{row['replicate']}:"
            )

            print(
                "  delta-v correlation             : "
                f"{row['delta_v_prediction_correlation']:.4f}"
            )

            print(
                "  delta-omega correlation         : "
                f"{row['delta_omega_prediction_correlation']:.4f}"
            )

            print(
                "  remaining mean yaw residual     : "
                f"{row['remaining_mean_yaw_residual_radps']:+.6e} rad/s"
            )

            print(
                "  remaining yaw bias              : "
                f"{row['remaining_mean_yaw_residual_deg_per_min']:+.3f} deg/min"
            )

            print()

    # =========================================================================
    # Completion
    # =========================================================================

    print()
    print("=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)
    print()

    print(
        "Generated:"
    )

    print(
        "  physical_residual_summary.csv"
    )

    print(
        "  oracle_experiment_results.csv"
    )

    print(
        "  physical_residual_vs_v1_performance.json"
    )

    if correction_summary_rows:

        print(
            "  v1_vs_physical_residual_per_run.csv"
        )

        print(
            "  v1_vs_physical_residual_timeseries.csv"
        )

    print(
        "  residual_timeseries/"
    )

    print(
        "  oracle_trajectories/"
    )

    if not args.no_plots:

        print(
            "  plots/"
        )

    print()
    print(
        "Frozen V1 was not modified."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
