#!/usr/bin/env python3
"""
i2Nav-Robot Trusted Adaptive-Q Fidelity Baseline.

Version 6
---------
Builds directly on the frozen V5 robust EKF baseline.

V5 retained:
    * OEM719 RTK preferred.
    * F9P SPP fallback.
    * ODO_SPEED preferred.
    * ADIS16465 IMU.
    * Hard GNSS NIS gate.
    * Safe GNSS reacquisition.
    * Rejected GNSS never changes Q.
    * Rejected GNSS never directly updates the state.

V6 change:
    Replace fixed process covariance Q with a TRUSTED heuristic
    motion-adaptive Q.

Adaptive-Q inputs:
    |v|
    |omega|
    |dv/dt|
    |domega/dt|

IMPORTANT SECURITY CONSTRAINT:
    GNSS position, innovation, residual, NIS, and GNSS covariance
    are NOT used to compute Q.

Therefore:

    d Q_k
    ----- = 0
    d z_GNSS

The adaptive multipliers are:

    alpha_xy =
        1
        + c_v     * |v|
        + c_w     * |omega|
        + c_dv    * |dv/dt|

    alpha_heading =
        1
        + c_hw    * |omega|
        + c_dw    * |domega/dt|

Both multipliers are bounded.

Run
---
python -m DigitalTwin.analysis.i2nav_adaptive_q_baseline \
    --root public_datasets/im2nav

Outputs
-------
results/i2nav_adaptive_q/baseline_fidelity.csv
results/i2nav_adaptive_q/baseline_summary.json
results/i2nav_adaptive_q/<sequence>_trajectory.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

try:
    from DigitalTwin.ekf import RoverEKF
    from DigitalTwin.kinematics import wrap_angle
except ImportError:
    project_root = Path(__file__).resolve().parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from DigitalTwin.ekf import RoverEKF
    from DigitalTwin.kinematics import wrap_angle


EARTH_RADIUS_M = 6_378_137.0


KNOWN_SEQUENCES = (
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SequenceFiles:
    name: str

    groundtruth: Path
    imu: Path

    odo_speed: Path | None
    ranger_odo: Path | None

    gnss: Path | None
    gnss_source: str


@dataclass
class FidelityResult:
    sequence: str
    status: str

    mode: str
    initialization: str

    samples: int
    duration_s: float
    gt_path_length_m: float

    # --------------------------------------------------------
    # Absolute trajectory error
    # --------------------------------------------------------

    ate_rmse_m: float
    ate_median_m: float
    ate_p95_m: float
    ate_max_m: float

    # --------------------------------------------------------
    # Rigid SE(2)-aligned trajectory error
    # --------------------------------------------------------

    ate_se2_rmse_m: float
    ate_se2_median_m: float
    ate_se2_p95_m: float

    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------

    heading_mae_deg: float
    heading_p95_deg: float

    heading_se2_mae_deg: float
    heading_se2_p95_deg: float

    # --------------------------------------------------------
    # Relative pose errors
    # --------------------------------------------------------

    rpe_1s_trans_rmse_m: float
    rpe_1s_rot_rmse_deg: float

    rpe_5s_trans_rmse_m: float
    rpe_5s_rot_rmse_deg: float

    rpe_10s_trans_rmse_m: float
    rpe_10s_rot_rmse_deg: float

    # --------------------------------------------------------
    # Drift
    # --------------------------------------------------------

    final_error_m: float
    final_error_se2_m: float
    final_drift_per_m: float
    path_length_ratio: float

    # --------------------------------------------------------
    # GNSS
    # --------------------------------------------------------

    gnss_source: str
    gnss_file_present: bool

    gnss_updates_seen: int
    gnss_updates_normal: int
    gnss_updates_reacquired: int
    gnss_updates_rejected_nis: int
    gnss_updates_skipped_quality: int
    gnss_updates_used: int

    gnss_rejection_rate_pct: float

    gnss_update_rate_hz: float
    gnss_expected_1hz_coverage_pct: float

    gnss_max_gap_s: float
    gnss_max_coast_s: float

    gnss_median_sigma_m: float

    # --------------------------------------------------------
    # NIS
    # --------------------------------------------------------

    nis_median: float
    nis_p95: float
    nis_max: float

    # --------------------------------------------------------
    # Safe reacquisition
    # --------------------------------------------------------

    reacq_candidates: int
    reacq_events: int
    reacq_max_extra_sigma_m: float

    # --------------------------------------------------------
    # Adaptive-Q diagnostics
    # --------------------------------------------------------

    alpha_xy_mean: float
    alpha_xy_p95: float
    alpha_xy_max: float

    alpha_heading_mean: float
    alpha_heading_p95: float
    alpha_heading_max: float

    q_xy_sigma_effective_mean_mps: float
    q_heading_sigma_effective_mean_radps: float

    # --------------------------------------------------------
    # Motion diagnostics
    # --------------------------------------------------------

    mean_speed_mps: float
    p95_abs_yaw_rate_radps: float

    p95_abs_accel_mps2: float
    p95_abs_yaw_accel_radps2: float

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    odo_source: str
    imu_yaw_sign: float

    q_xy_sigma_mps: float
    q_heading_sigma_radps: float

    alpha_min: float
    alpha_max: float

    alpha_xy_speed_coeff: float
    alpha_xy_turn_coeff: float
    alpha_xy_accel_coeff: float

    alpha_heading_turn_coeff: float
    alpha_heading_yaw_accel_coeff: float

    gnss_sigma_max_m: float

    nis_gate_threshold: float

    reacq_start_s: float
    reacq_sigma_growth_mps: float
    reacq_sigma_max_m: float
    reacq_consecutive_required: int

    error: str = ""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _nan() -> float:
    return float("nan")


def read_numeric_table(
    path: Path,
    min_cols: int,
) -> np.ndarray:

    rows: list[list[float]] = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for raw in f:
            line = raw.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith("%"):
                continue

            line = line.replace(
                ",",
                " ",
            )

            tokens = line.split()

            if len(tokens) < min_cols:
                continue

            try:
                values = [
                    float(token)
                    for token in tokens
                ]
            except ValueError:
                continue

            if (
                len(values) >= min_cols
                and np.all(
                    np.isfinite(
                        values[:min_cols]
                    )
                )
            ):
                rows.append(values)

    if not rows:
        raise ValueError(
            f"No usable numeric rows in {path}"
        )

    width = min(
        len(row)
        for row in rows
    )

    return np.asarray(
        [
            row[:width]
            for row in rows
        ],
        dtype=float,
    )


def sorted_unique_by_time(
    values: np.ndarray,
) -> np.ndarray:

    order = np.argsort(
        values[:, 0]
    )

    values = values[order]

    _, reverse_indices = np.unique(
        values[::-1, 0],
        return_index=True,
    )

    keep = (
        len(values)
        - 1
        - reverse_indices
    )

    keep.sort()

    return values[keep]


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_files(
    root: Path,
) -> list[SequenceFiles]:

    groundtruth_files = list(
        root.rglob(
            "*_groundtruth.nav"
        )
    )

    discovered: list[SequenceFiles] = []

    def find_one(
        pattern: str,
    ) -> Path | None:

        matches = list(
            root.rglob(
                pattern
            )
        )

        if not matches:
            return None

        matches.sort(
            key=lambda path: (
                len(path.parts),
                len(str(path)),
            )
        )

        return matches[0]

    for gt in sorted(
        groundtruth_files
    ):

        name = gt.name[
            :-len("_groundtruth.nav")
        ]

        imu = find_one(
            f"{name}_ADIS16465_IMU.txt"
        )

        odo_speed = find_one(
            f"{name}_ODO_SPEED.txt"
        )

        ranger_odo = find_one(
            f"{name}_RANGER_ODO.txt"
        )

        # ----------------------------------------------------
        # GNSS priority:
        #
        # 1. OEM719 RTK
        # 2. F9P SPP
        # 3. no GNSS
        # ----------------------------------------------------

        oem = find_one(
            f"{name}_OEM7_GNSS.pos"
        )

        if oem is None:
            oem = find_one(
                f"{name}_OEM*_GNSS.pos"
            )

        f9p = find_one(
            f"{name}_F9P_GNSS.pos"
        )

        if oem is not None:

            gnss = oem
            gnss_source = "OEM719_RTK"

        elif f9p is not None:

            gnss = f9p
            gnss_source = "F9P_SPP"

        else:

            gnss = None
            gnss_source = "NONE"

        if imu is None:

            print(
                f"[skip] {name}: missing IMU"
            )

            continue

        if (
            odo_speed is None
            and ranger_odo is None
        ):

            print(
                f"[skip] {name}: missing odometry"
            )

            continue

        discovered.append(
            SequenceFiles(
                name=name,
                groundtruth=gt,
                imu=imu,
                odo_speed=odo_speed,
                ranger_odo=ranger_odo,
                gnss=gnss,
                gnss_source=gnss_source,
            )
        )

    rank = {
        name: index
        for index, name
        in enumerate(
            KNOWN_SEQUENCES
        )
    }

    discovered.sort(
        key=lambda sequence: (
            rank.get(
                sequence.name,
                999,
            ),
            sequence.name,
        )
    )

    return discovered


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def wrap_array(
    values: np.ndarray,
) -> np.ndarray:

    return (
        values
        + np.pi
    ) % (
        2.0
        * np.pi
    ) - np.pi


def geodetic_to_local_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    latitude = np.deg2rad(
        lat_deg
    )

    longitude = np.deg2rad(
        lon_deg
    )

    latitude_0 = math.radians(
        lat0_deg
    )

    longitude_0 = math.radians(
        lon0_deg
    )

    east = (
        EARTH_RADIUS_M
        * math.cos(
            latitude_0
        )
        * (
            longitude
            - longitude_0
        )
    )

    north = (
        EARTH_RADIUS_M
        * (
            latitude
            - latitude_0
        )
    )

    return east, north


def interp_angle(
    source_time: np.ndarray,
    source_angle: np.ndarray,
    destination_time: np.ndarray,
) -> np.ndarray:

    unwrapped = np.unwrap(
        source_angle
    )

    interpolated = np.interp(
        destination_time,
        source_time,
        unwrapped,
    )

    return wrap_array(
        interpolated
    )


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_groundtruth(
    path: Path,
) -> dict[str, np.ndarray]:

    data = sorted_unique_by_time(
        read_numeric_table(
            path,
            min_cols=10,
        )
    )

    timestamp = data[:, 0]

    north = data[:, 1]
    east = data[:, 2]

    velocity_north = data[:, 4]
    velocity_east = data[:, 5]

    yaw_ned = np.deg2rad(
        data[:, 9]
    )

    # NED:
    # 0 = north, clockwise positive
    #
    # ENU:
    # 0 = east, counter-clockwise positive

    heading_enu = wrap_array(
        np.pi / 2.0
        - yaw_ned
    )

    return {
        "t":
            timestamp,

        "x":
            east,

        "y":
            north,

        "heading":
            heading_enu,

        "speed":
            np.hypot(
                velocity_east,
                velocity_north,
            ),
    }


# ---------------------------------------------------------------------------
# Odometry
# ---------------------------------------------------------------------------

def load_odo(
    files: SequenceFiles,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
]:

    # Prefer official derived ODO_SPEED.

    if files.odo_speed is not None:

        data = sorted_unique_by_time(
            read_numeric_table(
                files.odo_speed,
                min_cols=2,
            )
        )

        return (
            data[:, 0],
            data[:, 1],
            "ODO_SPEED",
        )

    assert (
        files.ranger_odo
        is not None
    )

    data = sorted_unique_by_time(
        read_numeric_table(
            files.ranger_odo,
            min_cols=9,
        )
    )

    wheel_speed = (
        data[:, 1:5]
    )

    wheel_angle = (
        data[:, 5:9]
    )

    forward_speed = np.mean(
        wheel_speed
        * np.cos(
            wheel_angle
        ),
        axis=1,
    )

    return (
        data[:, 0],
        forward_speed,
        "RANGER_ODO_forward_component",
    )


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------

def load_imu_yaw(
    path: Path,
    yaw_sign: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    data = sorted_unique_by_time(
        read_numeric_table(
            path,
            min_cols=7,
        )
    )

    timestamp = data[:, 0]

    dtheta_z = data[:, 3]

    cumulative_yaw = np.cumsum(
        yaw_sign
        * dtheta_z
    )

    cumulative_yaw -= (
        cumulative_yaw[0]
    )

    return (
        timestamp,
        cumulative_yaw,
    )


# ---------------------------------------------------------------------------
# Common time grid
# ---------------------------------------------------------------------------

def make_grid(
    gt_time: np.ndarray,
    odo_time: np.ndarray,
    imu_time: np.ndarray,
    hz: float,
) -> np.ndarray:

    start = max(
        float(
            gt_time[0]
        ),
        float(
            odo_time[0]
        ),
        float(
            imu_time[0]
        ),
    )

    end = min(
        float(
            gt_time[-1]
        ),
        float(
            odo_time[-1]
        ),
        float(
            imu_time[-1]
        ),
    )

    if end <= start:
        raise ValueError(
            "No common GT/ODO/IMU time interval"
        )

    dt = (
        1.0 / hz
    )

    start = (
        math.ceil(
            start / dt
        )
        * dt
    )

    end = (
        math.floor(
            end / dt
        )
        * dt
    )

    count = (
        int(
            round(
                (
                    end
                    - start
                )
                / dt
            )
        )
        + 1
    )

    return (
        start
        + np.arange(
            count,
            dtype=float,
        )
        * dt
    )


def interpolate_gt(
    gt: dict[str, np.ndarray],
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    x = np.interp(
        grid,
        gt["t"],
        gt["x"],
    )

    y = np.interp(
        grid,
        gt["t"],
        gt["y"],
    )

    heading = interp_angle(
        gt["t"],
        gt["heading"],
        grid,
    )

    return (
        x,
        y,
        heading,
    )


def sample_yaw_rate(
    imu_time: np.ndarray,
    imu_cumulative_yaw: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:

    cumulative = np.interp(
        grid,
        imu_time,
        imu_cumulative_yaw,
    )

    omega = np.zeros_like(
        cumulative
    )

    if len(grid) > 1:

        dt = np.diff(
            grid
        )

        omega[1:] = (
            np.diff(
                cumulative
            )
            / dt
        )

        omega[0] = (
            omega[1]
        )

    return omega


def stationary_gyro_bias(
    grid: np.ndarray,
    speed: np.ndarray,
    omega: np.ndarray,
    max_seconds: float = 5.0,
) -> float:

    if len(grid) < 2:
        return 0.0

    stationary = (
        (
            grid
            - grid[0]
        )
        <= max_seconds
    ) & (
        np.abs(
            speed
        )
        < 0.05
    )

    if (
        np.count_nonzero(
            stationary
        )
        >= 5
    ):

        return float(
            np.median(
                omega[
                    stationary
                ]
            )
        )

    return 0.0


# ---------------------------------------------------------------------------
# GNSS
# ---------------------------------------------------------------------------

def load_gnss(
    path: Path | None,
    gt: dict[str, np.ndarray],
    sigma_max_m: float,
    anchor_count: int,
) -> dict[str, np.ndarray] | None:

    if (
        path is None
        or not path.exists()
    ):
        return None

    data = sorted_unique_by_time(
        read_numeric_table(
            path,
            min_cols=7,
        )
    )

    timestamp = data[:, 0]

    latitude = data[:, 1]
    longitude = data[:, 2]

    sigma_north = np.abs(
        data[:, 4]
    )

    sigma_east = np.abs(
        data[:, 5]
    )

    sigma_horizontal = np.hypot(
        sigma_north,
        sigma_east,
    )

    valid = (
        np.isfinite(
            timestamp
        )
        & np.isfinite(
            latitude
        )
        & np.isfinite(
            longitude
        )
        & np.isfinite(
            sigma_north
        )
        & np.isfinite(
            sigma_east
        )
    )

    if not np.any(
        valid
    ):
        return None

    timestamp = timestamp[
        valid
    ]

    latitude = latitude[
        valid
    ]

    longitude = longitude[
        valid
    ]

    sigma_north = sigma_north[
        valid
    ]

    sigma_east = sigma_east[
        valid
    ]

    sigma_horizontal = sigma_horizontal[
        valid
    ]

    east_relative, north_relative = (
        geodetic_to_local_enu(
            latitude,
            longitude,
            latitude[0],
            longitude[0],
        )
    )

    anchor_valid = (
        sigma_horizontal
        <= sigma_max_m
    )

    anchor_indices = np.flatnonzero(
        anchor_valid
    )

    if not len(
        anchor_indices
    ):
        return None

    anchor_indices = (
        anchor_indices[
            :max(
                1,
                anchor_count,
            )
        ]
    )

    gt_east = np.interp(
        timestamp[
            anchor_indices
        ],
        gt["t"],
        gt["x"],
    )

    gt_north = np.interp(
        timestamp[
            anchor_indices
        ],
        gt["t"],
        gt["y"],
    )

    offset_east = float(
        np.median(
            gt_east
            - east_relative[
                anchor_indices
            ]
        )
    )

    offset_north = float(
        np.median(
            gt_north
            - north_relative[
                anchor_indices
            ]
        )
    )

    return {
        "t":
            timestamp,

        "x":
            east_relative
            + offset_east,

        "y":
            north_relative
            + offset_north,

        "sigma_n":
            sigma_north,

        "sigma_e":
            sigma_east,

        "sigma_h":
            sigma_horizontal,
    }


# ---------------------------------------------------------------------------
# TRUSTED adaptive Q
# ---------------------------------------------------------------------------

def compute_trusted_adaptive_q(
    speed_now: float,
    speed_previous: float,
    omega_now: float,
    omega_previous: float,
    dt: float,
    *,
    base_xy_sigma_mps: float,
    base_heading_sigma_radps: float,
    alpha_min: float,
    alpha_max: float,
    xy_speed_coeff: float,
    xy_turn_coeff: float,
    xy_accel_coeff: float,
    heading_turn_coeff: float,
    heading_yaw_accel_coeff: float,
) -> tuple[
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    """
    Compute process covariance using only trusted motion signals.

    NO GNSS input is accepted by this function.

    Returns
    -------
    Q
    alpha_xy
    alpha_heading
    abs_accel
    abs_yaw_accel
    """

    safe_dt = max(
        float(dt),
        1e-6,
    )

    abs_speed = abs(
        float(
            speed_now
        )
    )

    abs_omega = abs(
        float(
            omega_now
        )
    )

    abs_accel = abs(
        float(
            speed_now
            - speed_previous
        )
        / safe_dt
    )

    abs_yaw_accel = abs(
        float(
            omega_now
            - omega_previous
        )
        / safe_dt
    )

    # --------------------------------------------------------
    # Position uncertainty multiplier
    #
    # More uncertainty when:
    #   * travelling faster
    #   * turning harder
    #   * changing speed quickly
    # --------------------------------------------------------

    alpha_xy = (
        1.0
        + xy_speed_coeff
        * abs_speed
        + xy_turn_coeff
        * abs_omega
        + xy_accel_coeff
        * abs_accel
    )

    # --------------------------------------------------------
    # Heading uncertainty multiplier
    #
    # More uncertainty when:
    #   * turning harder
    #   * angular rate changes rapidly
    # --------------------------------------------------------

    alpha_heading = (
        1.0
        + heading_turn_coeff
        * abs_omega
        + heading_yaw_accel_coeff
        * abs_yaw_accel
    )

    alpha_xy = float(
        np.clip(
            alpha_xy,
            alpha_min,
            alpha_max,
        )
    )

    alpha_heading = float(
        np.clip(
            alpha_heading,
            alpha_min,
            alpha_max,
        )
    )

    effective_xy_sigma = (
        base_xy_sigma_mps
        * alpha_xy
    )

    effective_heading_sigma = (
        base_heading_sigma_radps
        * alpha_heading
    )

    Q = np.diag(
        [
            (
                effective_xy_sigma
                * safe_dt
            ) ** 2,

            (
                effective_xy_sigma
                * safe_dt
            ) ** 2,

            (
                effective_heading_sigma
                * safe_dt
            ) ** 2,
        ]
    )

    return (
        Q,
        alpha_xy,
        alpha_heading,
        abs_accel,
        abs_yaw_accel,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def path_length(
    xy: np.ndarray,
) -> float:

    if len(
        xy
    ) < 2:
        return 0.0

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(
                    xy,
                    axis=0,
                ),
                axis=1,
            )
        )
    )


def se2_align(
    estimate_xy: np.ndarray,
    truth_xy: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    estimate_mean = (
        estimate_xy.mean(
            axis=0
        )
    )

    truth_mean = (
        truth_xy.mean(
            axis=0
        )
    )

    estimate_centered = (
        estimate_xy
        - estimate_mean
    )

    truth_centered = (
        truth_xy
        - truth_mean
    )

    covariance = (
        estimate_centered.T
        @ truth_centered
    )

    u, _, vt = np.linalg.svd(
        covariance
    )

    rotation = (
        vt.T
        @ u.T
    )

    if (
        np.linalg.det(
            rotation
        )
        < 0
    ):

        vt[-1, :] *= -1.0

        rotation = (
            vt.T
            @ u.T
        )

    translation = (
        truth_mean
        - rotation
        @ estimate_mean
    )

    aligned = (
        rotation
        @ estimate_xy.T
    ).T + translation

    return (
        aligned,
        rotation,
        translation,
    )


def summarize_errors(
    estimate_xy: np.ndarray,
    estimate_heading: np.ndarray,
    truth_xy: np.ndarray,
    truth_heading: np.ndarray,
    hz: float,
) -> dict[str, float]:

    raw_error = np.linalg.norm(
        estimate_xy
        - truth_xy,
        axis=1,
    )

    aligned_xy, rotation, _ = (
        se2_align(
            estimate_xy,
            truth_xy,
        )
    )

    alignment_angle = math.atan2(
        float(
            rotation[
                1,
                0,
            ]
        ),
        float(
            rotation[
                0,
                0,
            ]
        ),
    )

    aligned_heading = wrap_array(
        estimate_heading
        + alignment_angle
    )

    aligned_error = np.linalg.norm(
        aligned_xy
        - truth_xy,
        axis=1,
    )

    heading_error = np.abs(
        wrap_array(
            estimate_heading
            - truth_heading
        )
    )

    aligned_heading_error = np.abs(
        wrap_array(
            aligned_heading
            - truth_heading
        )
    )

    def rpe(
        horizon_s: float,
    ) -> tuple[
        float,
        float,
    ]:

        lag = max(
            1,
            int(
                round(
                    horizon_s
                    * hz
                )
            ),
        )

        if (
            len(
                truth_xy
            )
            <= lag
        ):

            return (
                _nan(),
                _nan(),
            )

        translation_errors = []
        rotation_errors = []

        for i in range(
            len(truth_xy)
            - lag
        ):

            j = (
                i
                + lag
            )

            delta_truth_world = (
                truth_xy[j]
                - truth_xy[i]
            )

            delta_estimate_world = (
                estimate_xy[j]
                - estimate_xy[i]
            )

            ct = math.cos(
                -truth_heading[i]
            )

            st = math.sin(
                -truth_heading[i]
            )

            ce = math.cos(
                -estimate_heading[i]
            )

            se = math.sin(
                -estimate_heading[i]
            )

            rotation_truth = np.array(
                [
                    [
                        ct,
                        -st,
                    ],
                    [
                        st,
                        ct,
                    ],
                ]
            )

            rotation_estimate = np.array(
                [
                    [
                        ce,
                        -se,
                    ],
                    [
                        se,
                        ce,
                    ],
                ]
            )

            delta_truth_body = (
                rotation_truth
                @ delta_truth_world
            )

            delta_estimate_body = (
                rotation_estimate
                @ delta_estimate_world
            )

            translation_errors.append(
                float(
                    np.linalg.norm(
                        delta_estimate_body
                        - delta_truth_body
                    )
                )
            )

            delta_heading_truth = (
                wrap_angle(
                    float(
                        truth_heading[j]
                        - truth_heading[i]
                    )
                )
            )

            delta_heading_estimate = (
                wrap_angle(
                    float(
                        estimate_heading[j]
                        - estimate_heading[i]
                    )
                )
            )

            rotation_errors.append(
                abs(
                    wrap_angle(
                        delta_heading_estimate
                        - delta_heading_truth
                    )
                )
            )

        translation_errors = np.asarray(
            translation_errors,
            dtype=float,
        )

        rotation_errors = np.asarray(
            rotation_errors,
            dtype=float,
        )

        translation_rmse = float(
            np.sqrt(
                np.mean(
                    translation_errors**2
                )
            )
        )

        rotation_rmse = math.degrees(
            float(
                np.sqrt(
                    np.mean(
                        rotation_errors**2
                    )
                )
            )
        )

        return (
            translation_rmse,
            rotation_rmse,
        )

    rpe1_t, rpe1_r = rpe(
        1.0
    )

    rpe5_t, rpe5_r = rpe(
        5.0
    )

    rpe10_t, rpe10_r = rpe(
        10.0
    )

    gt_length = path_length(
        truth_xy
    )

    estimate_length = path_length(
        estimate_xy
    )

    final_error = float(
        np.linalg.norm(
            estimate_xy[-1]
            - truth_xy[-1]
        )
    )

    final_aligned_error = float(
        np.linalg.norm(
            aligned_xy[-1]
            - truth_xy[-1]
        )
    )

    return {
        "ate_rmse_m":
            float(
                np.sqrt(
                    np.mean(
                        raw_error**2
                    )
                )
            ),

        "ate_median_m":
            float(
                np.median(
                    raw_error
                )
            ),

        "ate_p95_m":
            float(
                np.percentile(
                    raw_error,
                    95,
                )
            ),

        "ate_max_m":
            float(
                np.max(
                    raw_error
                )
            ),

        "ate_se2_rmse_m":
            float(
                np.sqrt(
                    np.mean(
                        aligned_error**2
                    )
                )
            ),

        "ate_se2_median_m":
            float(
                np.median(
                    aligned_error
                )
            ),

        "ate_se2_p95_m":
            float(
                np.percentile(
                    aligned_error,
                    95,
                )
            ),

        "heading_mae_deg":
            math.degrees(
                float(
                    np.mean(
                        heading_error
                    )
                )
            ),

        "heading_p95_deg":
            math.degrees(
                float(
                    np.percentile(
                        heading_error,
                        95,
                    )
                )
            ),

        "heading_se2_mae_deg":
            math.degrees(
                float(
                    np.mean(
                        aligned_heading_error
                    )
                )
            ),

        "heading_se2_p95_deg":
            math.degrees(
                float(
                    np.percentile(
                        aligned_heading_error,
                        95,
                    )
                )
            ),

        "rpe_1s_trans_rmse_m":
            rpe1_t,

        "rpe_1s_rot_rmse_deg":
            rpe1_r,

        "rpe_5s_trans_rmse_m":
            rpe5_t,

        "rpe_5s_rot_rmse_deg":
            rpe5_r,

        "rpe_10s_trans_rmse_m":
            rpe10_t,

        "rpe_10s_rot_rmse_deg":
            rpe10_r,

        "final_error_m":
            final_error,

        "final_error_se2_m":
            final_aligned_error,

        "final_drift_per_m": (
            final_aligned_error
            / gt_length
            if gt_length > 0
            else _nan()
        ),

        "path_length_ratio": (
            estimate_length
            / gt_length
            if gt_length > 0
            else _nan()
        ),

        "gt_path_length_m":
            gt_length,
    }


# ---------------------------------------------------------------------------
# Run one sequence
# ---------------------------------------------------------------------------

def run_sequence(
    files: SequenceFiles,
    *,
    hz: float,

    q_xy_sigma_mps: float,
    q_heading_sigma_radps: float,

    alpha_min: float,
    alpha_max: float,

    alpha_xy_speed_coeff: float,
    alpha_xy_turn_coeff: float,
    alpha_xy_accel_coeff: float,

    alpha_heading_turn_coeff: float,
    alpha_heading_yaw_accel_coeff: float,

    gnss_sigma_max_m: float,
    gnss_sigma_floor_m: float,
    gnss_anchor_count: int,

    gnss_nis_gate: float,

    reacq_start_s: float,
    reacq_sigma_growth_mps: float,
    reacq_sigma_max_m: float,
    reacq_consecutive_required: int,

    imu_yaw_sign: float,

    output_dir: Path,
) -> FidelityResult:

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    gt = load_groundtruth(
        files.groundtruth
    )

    odo_time, odo_speed, odo_source = (
        load_odo(
            files
        )
    )

    imu_time, imu_cumulative_yaw = (
        load_imu_yaw(
            files.imu,
            imu_yaw_sign,
        )
    )

    grid = make_grid(
        gt["t"],
        odo_time,
        imu_time,
        hz,
    )

    dt_nominal = (
        1.0
        / hz
    )

    gt_x, gt_y, gt_heading = (
        interpolate_gt(
            gt,
            grid,
        )
    )

    truth_xy = np.column_stack(
        [
            gt_x,
            gt_y,
        ]
    )

    # -----------------------------------------------------------------------
    # Trusted physical signals
    # -----------------------------------------------------------------------

    speed = np.interp(
        grid,
        odo_time,
        odo_speed,
    )

    omega = sample_yaw_rate(
        imu_time,
        imu_cumulative_yaw,
        grid,
    )

    gyro_bias = (
        stationary_gyro_bias(
            grid,
            speed,
            omega,
        )
    )

    omega = (
        omega
        - gyro_bias
    )

    # -----------------------------------------------------------------------
    # GNSS
    # -----------------------------------------------------------------------

    gnss = load_gnss(
        files.gnss,
        gt,
        sigma_max_m=(
            gnss_sigma_max_m
        ),
        anchor_count=(
            gnss_anchor_count
        ),
    )

    # -----------------------------------------------------------------------
    # EKF initialization
    # -----------------------------------------------------------------------

    initial_state = np.array(
        [
            gt_x[0],
            gt_y[0],
            gt_heading[0],
        ],
        dtype=float,
    )

    initial_covariance = np.diag(
        [
            0.25**2,
            0.25**2,
            math.radians(
                5.0
            ) ** 2,
        ]
    )

    ekf = RoverEKF(
        initial_state=(
            initial_state
        ),
        initial_covariance=(
            initial_covariance
        ),
    )

    estimates = np.zeros(
        (
            len(grid),
            3,
        ),
        dtype=float,
    )

    estimates[0] = (
        ekf.state.x
    )

    # -----------------------------------------------------------------------
    # Adaptive-Q arrays
    # -----------------------------------------------------------------------

    alpha_xy_values = np.ones(
        len(grid),
        dtype=float,
    )

    alpha_heading_values = np.ones(
        len(grid),
        dtype=float,
    )

    accel_values = np.zeros(
        len(grid),
        dtype=float,
    )

    yaw_accel_values = np.zeros(
        len(grid),
        dtype=float,
    )

    # -----------------------------------------------------------------------
    # GNSS measurement model
    # -----------------------------------------------------------------------

    H = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    identity_2 = np.eye(
        2,
        dtype=float,
    )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    gnss_index = 0

    gnss_seen = 0
    gnss_normal = 0
    gnss_reacquired = 0
    gnss_rejected = 0
    gnss_skipped = 0

    reacq_candidate_streak = 0
    reacq_candidates = 0
    reacq_events = 0

    max_reacq_extra_sigma = (
        0.0
    )

    nis_values: list[float] = []

    used_gnss_times: list[float] = []
    used_gnss_sigmas: list[float] = []

    last_accepted_gnss_time = float(
        grid[0]
    )

    max_coast_s = (
        0.0
    )

    if gnss is not None:

        while (
            gnss_index
            < len(
                gnss["t"]
            )
            and gnss[
                "t"
            ][gnss_index]
            < grid[0]
        ):
            gnss_index += 1

    # -----------------------------------------------------------------------
    # Replay
    # -----------------------------------------------------------------------

    for k in range(
        1,
        len(grid),
    ):

        dt = float(
            grid[k]
            - grid[
                k - 1
            ]
        )

        if (
            not np.isfinite(
                dt
            )
            or dt <= 0
        ):

            dt = (
                dt_nominal
            )

        # -------------------------------------------------------------------
        # TRUSTED ADAPTIVE-Q
        #
        # Inputs:
        #     odometry speed
        #     IMU yaw rate
        #
        # GNSS is NOT passed here.
        # -------------------------------------------------------------------

        (
            Q,
            alpha_xy,
            alpha_heading,
            abs_accel,
            abs_yaw_accel,
        ) = compute_trusted_adaptive_q(

            speed_now=float(
                speed[k]
            ),

            speed_previous=float(
                speed[
                    k - 1
                ]
            ),

            omega_now=float(
                omega[k]
            ),

            omega_previous=float(
                omega[
                    k - 1
                ]
            ),

            dt=dt,

            base_xy_sigma_mps=(
                q_xy_sigma_mps
            ),

            base_heading_sigma_radps=(
                q_heading_sigma_radps
            ),

            alpha_min=(
                alpha_min
            ),

            alpha_max=(
                alpha_max
            ),

            xy_speed_coeff=(
                alpha_xy_speed_coeff
            ),

            xy_turn_coeff=(
                alpha_xy_turn_coeff
            ),

            xy_accel_coeff=(
                alpha_xy_accel_coeff
            ),

            heading_turn_coeff=(
                alpha_heading_turn_coeff
            ),

            heading_yaw_accel_coeff=(
                alpha_heading_yaw_accel_coeff
            ),
        )

        alpha_xy_values[k] = (
            alpha_xy
        )

        alpha_heading_values[k] = (
            alpha_heading
        )

        accel_values[k] = (
            abs_accel
        )

        yaw_accel_values[k] = (
            abs_yaw_accel
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        ekf.predict(
            float(
                speed[k]
            ),
            float(
                omega[k]
            ),
            dt,
            Q,
        )

        # ----------------------------------------------------
        # GNSS measurement processing
        # ----------------------------------------------------

        if gnss is not None:

            latest = None

            while (
                gnss_index
                < len(
                    gnss["t"]
                )
                and gnss[
                    "t"
                ][gnss_index]
                <= grid[k]
                + 1e-9
            ):

                latest = (
                    gnss_index
                )

                gnss_index += 1

            if latest is not None:

                sigma_h = float(
                    gnss[
                        "sigma_h"
                    ][latest]
                )

                if (
                    np.isfinite(
                        sigma_h
                    )
                    and sigma_h
                    <= gnss_sigma_max_m
                ):

                    gnss_seen += 1

                    sigma_e = max(
                        float(
                            gnss[
                                "sigma_e"
                            ][latest]
                        ),
                        gnss_sigma_floor_m,
                    )

                    sigma_n = max(
                        float(
                            gnss[
                                "sigma_n"
                            ][latest]
                        ),
                        gnss_sigma_floor_m,
                    )

                    R = np.diag(
                        [
                            sigma_e**2,
                            sigma_n**2,
                        ]
                    )

                    z = np.array(
                        [
                            gnss[
                                "x"
                            ][latest],

                            gnss[
                                "y"
                            ][latest],
                        ],
                        dtype=float,
                    )

                    # -------------------------------------------------------
                    # Normal GNSS innovation
                    # -------------------------------------------------------

                    innovation = (
                        z
                        - H
                        @ ekf.state.x
                    )

                    S = (
                        H
                        @ ekf.state.P
                        @ H.T
                        + R
                    )

                    try:

                        nis = float(
                            innovation.T
                            @ np.linalg.solve(
                                S,
                                innovation,
                            )
                        )

                    except np.linalg.LinAlgError:

                        nis = float(
                            "inf"
                        )

                    nis_values.append(
                        nis
                    )

                    accepted = (
                        False
                    )

                    # -------------------------------------------------------
                    # Normal GNSS gate
                    # -------------------------------------------------------

                    if (
                        np.isfinite(
                            nis
                        )
                        and nis
                        <= gnss_nis_gate
                    ):

                        ekf.update_gps(
                            z,
                            R,
                        )

                        accepted = (
                            True
                        )

                        gnss_normal += 1

                        reacq_candidate_streak = (
                            0
                        )

                    # -------------------------------------------------------
                    # Hard rejection + safe reacquisition
                    # -------------------------------------------------------

                    else:

                        coast_s = max(
                            0.0,
                            float(
                                grid[k]
                                - last_accepted_gnss_time
                            ),
                        )

                        max_coast_s = max(
                            max_coast_s,
                            coast_s,
                        )

                        candidate = (
                            False
                        )

                        extra_sigma = (
                            0.0
                        )

                        if (
                            coast_s
                            >= reacq_start_s
                        ):

                            extra_sigma = min(
                                reacq_sigma_max_m,
                                reacq_sigma_growth_mps
                                * coast_s,
                            )

                            max_reacq_extra_sigma = max(
                                max_reacq_extra_sigma,
                                extra_sigma,
                            )

                            extra_covariance = (
                                identity_2
                                * extra_sigma**2
                            )

                            S_reacq = (
                                S
                                + extra_covariance
                            )

                            try:

                                reacq_nis = float(
                                    innovation.T
                                    @ np.linalg.solve(
                                        S_reacq,
                                        innovation,
                                    )
                                )

                            except np.linalg.LinAlgError:

                                reacq_nis = float(
                                    "inf"
                                )

                            if (
                                np.isfinite(
                                    reacq_nis
                                )
                                and reacq_nis
                                <= gnss_nis_gate
                            ):

                                candidate = (
                                    True
                                )

                        if candidate:

                            reacq_candidate_streak += 1

                            reacq_candidates += 1

                        else:

                            reacq_candidate_streak = (
                                0
                            )

                        if (
                            candidate
                            and reacq_candidate_streak
                            >= reacq_consecutive_required
                        ):

                            R_reacq = (
                                R
                                + identity_2
                                * extra_sigma**2
                            )

                            ekf.update_gps(
                                z,
                                R_reacq,
                            )

                            accepted = (
                                True
                            )

                            gnss_reacquired += 1

                            reacq_events += 1

                            reacq_candidate_streak = (
                                0
                            )

                        else:

                            gnss_rejected += 1

                    # -------------------------------------------------------
                    # Accepted GNSS bookkeeping
                    # -------------------------------------------------------

                    if accepted:

                        last_accepted_gnss_time = (
                            float(
                                gnss[
                                    "t"
                                ][latest]
                            )
                        )

                        used_gnss_times.append(
                            last_accepted_gnss_time
                        )

                        used_gnss_sigmas.append(
                            sigma_h
                        )

                else:

                    gnss_skipped += 1

        # ----------------------------------------------------
        # Coast duration
        # ----------------------------------------------------

        if files.gnss is not None:

            current_coast_s = max(
                0.0,
                float(
                    grid[k]
                    - last_accepted_gnss_time
                ),
            )

            max_coast_s = max(
                max_coast_s,
                current_coast_s,
            )

        estimates[k] = (
            ekf.state.x
        )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    metrics = summarize_errors(
        estimates[
            :,
            :2,
        ],
        estimates[
            :,
            2,
        ],
        truth_xy,
        gt_heading,
        hz,
    )

    duration_s = (
        float(
            grid[-1]
            - grid[0]
        )
        if len(grid) > 1
        else 0.0
    )

    gnss_used = (
        gnss_normal
        + gnss_reacquired
    )

    rejection_rate_pct = (
        100.0
        * gnss_rejected
        / gnss_seen
        if gnss_seen > 0
        else 0.0
    )

    # -----------------------------------------------------------------------
    # NIS statistics
    # -----------------------------------------------------------------------

    finite_nis = np.asarray(
        [
            value
            for value
            in nis_values
            if np.isfinite(
                value
            )
        ],
        dtype=float,
    )

    if len(
        finite_nis
    ):

        nis_median = float(
            np.median(
                finite_nis
            )
        )

        nis_p95 = float(
            np.percentile(
                finite_nis,
                95,
            )
        )

        nis_max = float(
            np.max(
                finite_nis
            )
        )

    else:

        nis_median = (
            _nan()
        )

        nis_p95 = (
            _nan()
        )

        nis_max = (
            _nan()
        )

    # -----------------------------------------------------------------------
    # GNSS gaps
    # -----------------------------------------------------------------------

    if used_gnss_times:

        gap_points = np.asarray(
            [
                float(
                    grid[0]
                ),
                *used_gnss_times,
                float(
                    grid[-1]
                ),
            ],
            dtype=float,
        )

        gnss_max_gap_s = float(
            np.max(
                np.diff(
                    gap_points
                )
            )
        )

    else:

        gnss_max_gap_s = (
            duration_s
            if files.gnss
            is not None
            else _nan()
        )

    expected_1hz_updates = max(
        1,
        int(
            math.floor(
                duration_s
            )
        )
        + 1,
    )

    coverage_pct = (
        min(
            100.0,
            100.0
            * gnss_used
            / expected_1hz_updates,
        )
        if files.gnss
        is not None
        else 0.0
    )

    # -----------------------------------------------------------------------
    # Adaptive-Q statistics
    # -----------------------------------------------------------------------

    alpha_xy_mean = float(
        np.mean(
            alpha_xy_values
        )
    )

    alpha_xy_p95 = float(
        np.percentile(
            alpha_xy_values,
            95,
        )
    )

    alpha_xy_max_value = float(
        np.max(
            alpha_xy_values
        )
    )

    alpha_heading_mean = float(
        np.mean(
            alpha_heading_values
        )
    )

    alpha_heading_p95 = float(
        np.percentile(
            alpha_heading_values,
            95,
        )
    )

    alpha_heading_max_value = float(
        np.max(
            alpha_heading_values
        )
    )

    # -----------------------------------------------------------------------
    # Save detailed trajectory
    # -----------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_path = (
        output_dir
        / f"{files.name}_trajectory.csv"
    )

    position_error = np.linalg.norm(
        estimates[
            :,
            :2,
        ]
        - truth_xy,
        axis=1,
    )

    with trajectory_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "time_s",

                "gt_east_m",
                "gt_north_m",
                "gt_heading_rad",

                "ekf_east_m",
                "ekf_north_m",
                "ekf_heading_rad",

                "odo_speed_mps",
                "imu_yaw_rate_radps",

                "abs_accel_mps2",
                "abs_yaw_accel_radps2",

                "alpha_xy",
                "alpha_heading",

                "q_xy_effective_sigma_mps",
                "q_heading_effective_sigma_radps",

                "position_error_m",

                "gnss_source",
            ]
        )

        for i in range(
            len(grid)
        ):

            writer.writerow(
                [
                    grid[i],

                    gt_x[i],
                    gt_y[i],
                    gt_heading[i],

                    estimates[
                        i,
                        0,
                    ],

                    estimates[
                        i,
                        1,
                    ],

                    estimates[
                        i,
                        2,
                    ],

                    speed[i],
                    omega[i],

                    accel_values[i],
                    yaw_accel_values[i],

                    alpha_xy_values[i],
                    alpha_heading_values[i],

                    q_xy_sigma_mps
                    * alpha_xy_values[i],

                    q_heading_sigma_radps
                    * alpha_heading_values[i],

                    position_error[i],

                    files.gnss_source,
                ]
            )

    mode = (
        f"{files.gnss_source}"
        "+ODO+IMU"
        "+TRUSTED_ADAPTIVE_Q"
        "+HARD_NIS"
        "+SAFE_REACQ"
        if gnss_seen > 0
        else (
            "ODO+IMU"
            "+TRUSTED_ADAPTIVE_Q"
        )
    )

    return FidelityResult(
        sequence=(
            files.name
        ),

        status="ok",

        mode=(
            mode
        ),

        initialization=(
            "GT initial pose; "
            "GNSS translation-only registration"
        ),

        samples=(
            len(grid)
        ),

        duration_s=(
            duration_s
        ),

        gt_path_length_m=(
            metrics[
                "gt_path_length_m"
            ]
        ),

        ate_rmse_m=(
            metrics[
                "ate_rmse_m"
            ]
        ),

        ate_median_m=(
            metrics[
                "ate_median_m"
            ]
        ),

        ate_p95_m=(
            metrics[
                "ate_p95_m"
            ]
        ),

        ate_max_m=(
            metrics[
                "ate_max_m"
            ]
        ),

        ate_se2_rmse_m=(
            metrics[
                "ate_se2_rmse_m"
            ]
        ),

        ate_se2_median_m=(
            metrics[
                "ate_se2_median_m"
            ]
        ),

        ate_se2_p95_m=(
            metrics[
                "ate_se2_p95_m"
            ]
        ),

        heading_mae_deg=(
            metrics[
                "heading_mae_deg"
            ]
        ),

        heading_p95_deg=(
            metrics[
                "heading_p95_deg"
            ]
        ),

        heading_se2_mae_deg=(
            metrics[
                "heading_se2_mae_deg"
            ]
        ),

        heading_se2_p95_deg=(
            metrics[
                "heading_se2_p95_deg"
            ]
        ),

        rpe_1s_trans_rmse_m=(
            metrics[
                "rpe_1s_trans_rmse_m"
            ]
        ),

        rpe_1s_rot_rmse_deg=(
            metrics[
                "rpe_1s_rot_rmse_deg"
            ]
        ),

        rpe_5s_trans_rmse_m=(
            metrics[
                "rpe_5s_trans_rmse_m"
            ]
        ),

        rpe_5s_rot_rmse_deg=(
            metrics[
                "rpe_5s_rot_rmse_deg"
            ]
        ),

        rpe_10s_trans_rmse_m=(
            metrics[
                "rpe_10s_trans_rmse_m"
            ]
        ),

        rpe_10s_rot_rmse_deg=(
            metrics[
                "rpe_10s_rot_rmse_deg"
            ]
        ),

        final_error_m=(
            metrics[
                "final_error_m"
            ]
        ),

        final_error_se2_m=(
            metrics[
                "final_error_se2_m"
            ]
        ),

        final_drift_per_m=(
            metrics[
                "final_drift_per_m"
            ]
        ),

        path_length_ratio=(
            metrics[
                "path_length_ratio"
            ]
        ),

        gnss_source=(
            files.gnss_source
        ),

        gnss_file_present=(
            files.gnss
            is not None
        ),

        gnss_updates_seen=(
            gnss_seen
        ),

        gnss_updates_normal=(
            gnss_normal
        ),

        gnss_updates_reacquired=(
            gnss_reacquired
        ),

        gnss_updates_rejected_nis=(
            gnss_rejected
        ),

        gnss_updates_skipped_quality=(
            gnss_skipped
        ),

        gnss_updates_used=(
            gnss_used
        ),

        gnss_rejection_rate_pct=(
            rejection_rate_pct
        ),

        gnss_update_rate_hz=(
            gnss_used
            / max(
                duration_s,
                1e-9,
            )
        ),

        gnss_expected_1hz_coverage_pct=(
            coverage_pct
        ),

        gnss_max_gap_s=(
            gnss_max_gap_s
        ),

        gnss_max_coast_s=(
            max_coast_s
        ),

        gnss_median_sigma_m=(
            float(
                np.median(
                    used_gnss_sigmas
                )
            )
            if used_gnss_sigmas
            else _nan()
        ),

        nis_median=(
            nis_median
        ),

        nis_p95=(
            nis_p95
        ),

        nis_max=(
            nis_max
        ),

        reacq_candidates=(
            reacq_candidates
        ),

        reacq_events=(
            reacq_events
        ),

        reacq_max_extra_sigma_m=(
            max_reacq_extra_sigma
        ),

        alpha_xy_mean=(
            alpha_xy_mean
        ),

        alpha_xy_p95=(
            alpha_xy_p95
        ),

        alpha_xy_max=(
            alpha_xy_max_value
        ),

        alpha_heading_mean=(
            alpha_heading_mean
        ),

        alpha_heading_p95=(
            alpha_heading_p95
        ),

        alpha_heading_max=(
            alpha_heading_max_value
        ),

        q_xy_sigma_effective_mean_mps=(
            q_xy_sigma_mps
            * alpha_xy_mean
        ),

        q_heading_sigma_effective_mean_radps=(
            q_heading_sigma_radps
            * alpha_heading_mean
        ),

        mean_speed_mps=(
            float(
                np.mean(
                    np.abs(
                        speed
                    )
                )
            )
        ),

        p95_abs_yaw_rate_radps=(
            float(
                np.percentile(
                    np.abs(
                        omega
                    ),
                    95,
                )
            )
        ),

        p95_abs_accel_mps2=(
            float(
                np.percentile(
                    accel_values,
                    95,
                )
            )
        ),

        p95_abs_yaw_accel_radps2=(
            float(
                np.percentile(
                    yaw_accel_values,
                    95,
                )
            )
        ),

        odo_source=(
            odo_source
        ),

        imu_yaw_sign=(
            imu_yaw_sign
        ),

        q_xy_sigma_mps=(
            q_xy_sigma_mps
        ),

        q_heading_sigma_radps=(
            q_heading_sigma_radps
        ),

        alpha_min=(
            alpha_min
        ),

        alpha_max=(
            alpha_max
        ),

        alpha_xy_speed_coeff=(
            alpha_xy_speed_coeff
        ),

        alpha_xy_turn_coeff=(
            alpha_xy_turn_coeff
        ),

        alpha_xy_accel_coeff=(
            alpha_xy_accel_coeff
        ),

        alpha_heading_turn_coeff=(
            alpha_heading_turn_coeff
        ),

        alpha_heading_yaw_accel_coeff=(
            alpha_heading_yaw_accel_coeff
        ),

        gnss_sigma_max_m=(
            gnss_sigma_max_m
        ),

        nis_gate_threshold=(
            gnss_nis_gate
        ),

        reacq_start_s=(
            reacq_start_s
        ),

        reacq_sigma_growth_mps=(
            reacq_sigma_growth_mps
        ),

        reacq_sigma_max_m=(
            reacq_sigma_max_m
        ),

        reacq_consecutive_required=(
            reacq_consecutive_required
        ),
    )


# ---------------------------------------------------------------------------
# Failed result
# ---------------------------------------------------------------------------

def failed_result(
    name: str,
    exc: Exception,
    args: argparse.Namespace,
) -> FidelityResult:

    nan = (
        _nan()
    )

    return FidelityResult(
        sequence=name,
        status="failed",

        mode="",
        initialization="",

        samples=0,

        duration_s=nan,
        gt_path_length_m=nan,

        ate_rmse_m=nan,
        ate_median_m=nan,
        ate_p95_m=nan,
        ate_max_m=nan,

        ate_se2_rmse_m=nan,
        ate_se2_median_m=nan,
        ate_se2_p95_m=nan,

        heading_mae_deg=nan,
        heading_p95_deg=nan,

        heading_se2_mae_deg=nan,
        heading_se2_p95_deg=nan,

        rpe_1s_trans_rmse_m=nan,
        rpe_1s_rot_rmse_deg=nan,

        rpe_5s_trans_rmse_m=nan,
        rpe_5s_rot_rmse_deg=nan,

        rpe_10s_trans_rmse_m=nan,
        rpe_10s_rot_rmse_deg=nan,

        final_error_m=nan,
        final_error_se2_m=nan,
        final_drift_per_m=nan,
        path_length_ratio=nan,

        gnss_source="NONE",
        gnss_file_present=False,

        gnss_updates_seen=0,
        gnss_updates_normal=0,
        gnss_updates_reacquired=0,
        gnss_updates_rejected_nis=0,
        gnss_updates_skipped_quality=0,
        gnss_updates_used=0,

        gnss_rejection_rate_pct=nan,

        gnss_update_rate_hz=nan,
        gnss_expected_1hz_coverage_pct=nan,

        gnss_max_gap_s=nan,
        gnss_max_coast_s=nan,

        gnss_median_sigma_m=nan,

        nis_median=nan,
        nis_p95=nan,
        nis_max=nan,

        reacq_candidates=0,
        reacq_events=0,
        reacq_max_extra_sigma_m=nan,

        alpha_xy_mean=nan,
        alpha_xy_p95=nan,
        alpha_xy_max=nan,

        alpha_heading_mean=nan,
        alpha_heading_p95=nan,
        alpha_heading_max=nan,

        q_xy_sigma_effective_mean_mps=nan,
        q_heading_sigma_effective_mean_radps=nan,

        mean_speed_mps=nan,
        p95_abs_yaw_rate_radps=nan,

        p95_abs_accel_mps2=nan,
        p95_abs_yaw_accel_radps2=nan,

        odo_source="",

        imu_yaw_sign=(
            args.imu_yaw_sign
        ),

        q_xy_sigma_mps=(
            args.q_xy_sigma_mps
        ),

        q_heading_sigma_radps=(
            args.q_heading_sigma_radps
        ),

        alpha_min=(
            args.alpha_min
        ),

        alpha_max=(
            args.alpha_max
        ),

        alpha_xy_speed_coeff=(
            args.alpha_xy_speed_coeff
        ),

        alpha_xy_turn_coeff=(
            args.alpha_xy_turn_coeff
        ),

        alpha_xy_accel_coeff=(
            args.alpha_xy_accel_coeff
        ),

        alpha_heading_turn_coeff=(
            args.alpha_heading_turn_coeff
        ),

        alpha_heading_yaw_accel_coeff=(
            args.alpha_heading_yaw_accel_coeff
        ),

        gnss_sigma_max_m=(
            args.gnss_sigma_max_m
        ),

        nis_gate_threshold=(
            args.gnss_nis_gate
        ),

        reacq_start_s=(
            args.reacq_start_s
        ),

        reacq_sigma_growth_mps=(
            args.reacq_sigma_growth_mps
        ),

        reacq_sigma_max_m=(
            args.reacq_sigma_max_m
        ),

        reacq_consecutive_required=(
            args.reacq_consecutive
        ),

        error=(
            f"{type(exc).__name__}: {exc}"
        ),
    )


# ---------------------------------------------------------------------------
# CSV / summary
# ---------------------------------------------------------------------------

def write_csv(
    path: Path,
    results: list[FidelityResult],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        FidelityResult
        .__annotations__
        .keys()
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

        for result in results:

            writer.writerow(
                asdict(
                    result
                )
            )


def finite_values(
    results: Iterable[FidelityResult],
    field: str,
) -> np.ndarray:

    values = []

    for result in results:

        if (
            result.status
            != "ok"
        ):
            continue

        value = float(
            getattr(
                result,
                field,
            )
        )

        if np.isfinite(
            value
        ):

            values.append(
                value
            )

    return np.asarray(
        values,
        dtype=float,
    )


def write_summary(
    path: Path,
    root: Path,
    results: list[FidelityResult],
    args: argparse.Namespace,
) -> None:

    successful = [
        result
        for result
        in results
        if result.status
        == "ok"
    ]

    fused = [
        result
        for result
        in successful
        if result.gnss_updates_seen
        > 0
    ]

    dead_reckoning = [
        result
        for result
        in successful
        if result.gnss_source
        == "NONE"
    ]

    def mean_field(
        subset: list[FidelityResult],
        field: str,
    ) -> float | None:

        values = finite_values(
            subset,
            field,
        )

        if not len(
            values
        ):
            return None

        return float(
            np.mean(
                values
            )
        )

    def rms_field(
        subset: list[FidelityResult],
        field: str,
    ) -> float | None:

        values = finite_values(
            subset,
            field,
        )

        if not len(
            values
        ):
            return None

        return float(
            np.sqrt(
                np.mean(
                    values**2
                )
            )
        )

    summary = {
        "schema":
            "i2nav_fidelity_v6_trusted_adaptive_q",

        "dataset_root":
            str(
                root
            ),

        "successful_sequences": [
            result.sequence
            for result
            in successful
        ],

        "failed_sequences": [
            {
                "sequence":
                    result.sequence,

                "error":
                    result.error,
            }
            for result
            in results
            if result.status
            != "ok"
        ],

        "counts": {
            "total":
                len(
                    results
                ),

            "successful":
                len(
                    successful
                ),

            "fused":
                len(
                    fused
                ),

            "odo_imu_only":
                len(
                    dead_reckoning
                ),

            "gnss_seen":
                int(
                    sum(
                        result.gnss_updates_seen
                        for result
                        in fused
                    )
                ),

            "gnss_normal":
                int(
                    sum(
                        result.gnss_updates_normal
                        for result
                        in fused
                    )
                ),

            "gnss_reacquired":
                int(
                    sum(
                        result.gnss_updates_reacquired
                        for result
                        in fused
                    )
                ),

            "gnss_rejected":
                int(
                    sum(
                        result.gnss_updates_rejected_nis
                        for result
                        in fused
                    )
                ),
        },

        "macro_means": {
            "fused_ate_mean_m":
                mean_field(
                    fused,
                    "ate_rmse_m",
                ),

            "fused_sequence_ate_rms_m":
                rms_field(
                    fused,
                    "ate_rmse_m",
                ),

            "fused_ate_se2_mean_m":
                mean_field(
                    fused,
                    "ate_se2_rmse_m",
                ),

            "all_rpe_1s_mean_m":
                mean_field(
                    successful,
                    "rpe_1s_trans_rmse_m",
                ),

            "all_heading_mae_mean_deg":
                mean_field(
                    successful,
                    "heading_mae_deg",
                ),

            "gnss_rejection_rate_mean_pct":
                mean_field(
                    fused,
                    "gnss_rejection_rate_pct",
                ),

            "alpha_xy_mean":
                mean_field(
                    successful,
                    "alpha_xy_mean",
                ),

            "alpha_xy_p95_mean":
                mean_field(
                    successful,
                    "alpha_xy_p95",
                ),

            "alpha_heading_mean":
                mean_field(
                    successful,
                    "alpha_heading_mean",
                ),

            "alpha_heading_p95_mean":
                mean_field(
                    successful,
                    "alpha_heading_p95",
                ),
        },

        "configuration": {
            "rate_hz":
                args.rate_hz,

            "base_q_xy_sigma_mps":
                args.q_xy_sigma_mps,

            "base_q_heading_sigma_radps":
                args.q_heading_sigma_radps,

            "alpha_min":
                args.alpha_min,

            "alpha_max":
                args.alpha_max,

            "alpha_xy_speed_coeff":
                args.alpha_xy_speed_coeff,

            "alpha_xy_turn_coeff":
                args.alpha_xy_turn_coeff,

            "alpha_xy_accel_coeff":
                args.alpha_xy_accel_coeff,

            "alpha_heading_turn_coeff":
                args.alpha_heading_turn_coeff,

            "alpha_heading_yaw_accel_coeff":
                args.alpha_heading_yaw_accel_coeff,

            "gnss_nis_gate":
                args.gnss_nis_gate,

            "reacq_start_s":
                args.reacq_start_s,

            "reacq_sigma_growth_mps":
                args.reacq_sigma_growth_mps,

            "reacq_sigma_max_m":
                args.reacq_sigma_max_m,

            "reacq_consecutive":
                args.reacq_consecutive,

            "imu_yaw_sign":
                args.imu_yaw_sign,
        },

        "security_constraint": (
            "Adaptive process covariance Q uses only trusted "
            "ODO/IMU motion quantities: speed, yaw rate, "
            "speed derivative, and yaw-rate derivative. "
            "GNSS position, GNSS residual, innovation, NIS, "
            "and reported GNSS uncertainty do not influence Q."
        ),

        "comparison_note": (
            "Compare this V6 result directly against the frozen "
            "V5 fixed-Q robust EKF. GNSS gating, safe reacquisition, "
            "sensor selection, R handling, initialization, and evaluation "
            "protocol should otherwise remain unchanged."
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def print_table(
    results: list[FidelityResult],
) -> None:

    print()

    print(
        f"{'sequence':<14} "
        f"{'ATE':>7} "
        f"{'RPE1':>7} "
        f"{'head':>6} "
        f"{'aXY':>7} "
        f"{'aXY95':>7} "
        f"{'aH':>7} "
        f"{'aH95':>7} "
        f"{'GNSSrej%':>9} "
        f"{'reacq':>6}"
    )

    print(
        "-"
        * 94
    )

    for result in results:

        if (
            result.status
            != "ok"
        ):

            print(
                f"{result.sequence:<14} "
                f"FAILED: {result.error}"
            )

            continue

        print(
            f"{result.sequence:<14} "
            f"{result.ate_rmse_m:7.3f} "
            f"{result.rpe_1s_trans_rmse_m:7.3f} "
            f"{result.heading_mae_deg:6.2f} "
            f"{result.alpha_xy_mean:7.2f} "
            f"{result.alpha_xy_p95:7.2f} "
            f"{result.alpha_heading_mean:7.2f} "
            f"{result.alpha_heading_p95:7.2f} "
            f"{result.gnss_rejection_rate_pct:9.1f} "
            f"{result.gnss_updates_reacquired:6d}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "i2Nav trusted motion-adaptive-Q EKF baseline."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "public_datasets/im2nav"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/i2nav_adaptive_q"
        ),
    )

    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
    )

    # -----------------------------------------------------------------------
    # Base Q
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--q-xy-sigma-mps",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--q-heading-sigma-radps",
        type=float,
        default=0.01,
    )

    # -----------------------------------------------------------------------
    # Adaptive-Q bounds
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--alpha-min",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=10.0,
    )

    # -----------------------------------------------------------------------
    # XY adaptive-Q coefficients
    #
    # Initial heuristic:
    #
    # alpha_xy =
    #   1
    #   + 0.5 |v|
    #   + 1.5 |omega|
    #   + 0.5 |dv/dt|
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--alpha-xy-speed-coeff",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--alpha-xy-turn-coeff",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--alpha-xy-accel-coeff",
        type=float,
        default=0.5,
    )

    # -----------------------------------------------------------------------
    # Heading adaptive-Q coefficients
    #
    # alpha_heading =
    #   1
    #   + 2.0 |omega|
    #   + 0.5 |domega/dt|
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--alpha-heading-turn-coeff",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--alpha-heading-yaw-accel-coeff",
        type=float,
        default=0.5,
    )

    # -----------------------------------------------------------------------
    # GNSS
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--gnss-sigma-max-m",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--gnss-sigma-floor-m",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--gnss-anchor-count",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gnss-nis-gate",
        type=float,
        default=9.21,
    )

    # -----------------------------------------------------------------------
    # Safe reacquisition
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--reacq-start-s",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--reacq-sigma-growth-mps",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--reacq-sigma-max-m",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--reacq-consecutive",
        type=int,
        default=3,
    )

    # -----------------------------------------------------------------------
    # IMU
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--imu-yaw-sign",
        type=float,
        choices=(
            -1.0,
            1.0,
        ),
        default=-1.0,
    )

    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    args = parse_args()

    root = (
        args.root
        .resolve()
    )

    output_dir = (
        args.output_dir
        .resolve()
    )

    if not root.exists():

        print(
            f"ERROR: dataset root not found: {root}"
        )

        return 2

    files = discover_files(
        root
    )

    if args.sequences:

        requested = set(
            args.sequences
        )

        files = [
            item
            for item
            in files
            if item.name
            in requested
        ]

    if not files:

        print(
            "ERROR: no usable i2Nav sequences found"
        )

        return 2

    print(
        f"Dataset root: {root}"
    )

    print(
        f"Sequences: {len(files)}"
    )

    print()

    print(
        "V6 trusted adaptive-Q policy:"
    )

    print(
        "  Q inputs = ODO speed + IMU yaw rate + their derivatives"
    )

    print(
        "  GNSS does NOT influence adaptive Q"
    )

    print(
        f"  alpha range = "
        f"[{args.alpha_min:.2f}, {args.alpha_max:.2f}]"
    )

    print(
        "  alpha_xy = "
        f"1 + {args.alpha_xy_speed_coeff}|v| "
        f"+ {args.alpha_xy_turn_coeff}|w| "
        f"+ {args.alpha_xy_accel_coeff}|dv/dt|"
    )

    print(
        "  alpha_heading = "
        f"1 + {args.alpha_heading_turn_coeff}|w| "
        f"+ {args.alpha_heading_yaw_accel_coeff}|dw/dt|"
    )

    print()

    print(
        "GNSS policy retained from V5:"
    )

    print(
        f"  hard NIS gate = {args.gnss_nis_gate}"
    )

    print(
        f"  safe reacquisition starts after "
        f"{args.reacq_start_s:.1f} s"
    )

    print()

    results: list[FidelityResult] = []

    for sequence_files in files:

        odo_label = (
            "ODO_SPEED"
            if sequence_files.odo_speed
            is not None
            else "RANGER"
        )

        print(
            f"[run] {sequence_files.name} "
            f"| GNSS={sequence_files.gnss_source} "
            f"| ODO={odo_label}"
        )

        try:

            result = run_sequence(
                sequence_files,

                hz=(
                    args.rate_hz
                ),

                q_xy_sigma_mps=(
                    args.q_xy_sigma_mps
                ),

                q_heading_sigma_radps=(
                    args.q_heading_sigma_radps
                ),

                alpha_min=(
                    args.alpha_min
                ),

                alpha_max=(
                    args.alpha_max
                ),

                alpha_xy_speed_coeff=(
                    args.alpha_xy_speed_coeff
                ),

                alpha_xy_turn_coeff=(
                    args.alpha_xy_turn_coeff
                ),

                alpha_xy_accel_coeff=(
                    args.alpha_xy_accel_coeff
                ),

                alpha_heading_turn_coeff=(
                    args.alpha_heading_turn_coeff
                ),

                alpha_heading_yaw_accel_coeff=(
                    args.alpha_heading_yaw_accel_coeff
                ),

                gnss_sigma_max_m=(
                    args.gnss_sigma_max_m
                ),

                gnss_sigma_floor_m=(
                    args.gnss_sigma_floor_m
                ),

                gnss_anchor_count=(
                    args.gnss_anchor_count
                ),

                gnss_nis_gate=(
                    args.gnss_nis_gate
                ),

                reacq_start_s=(
                    args.reacq_start_s
                ),

                reacq_sigma_growth_mps=(
                    args.reacq_sigma_growth_mps
                ),

                reacq_sigma_max_m=(
                    args.reacq_sigma_max_m
                ),

                reacq_consecutive_required=(
                    args.reacq_consecutive
                ),

                imu_yaw_sign=(
                    args.imu_yaw_sign
                ),

                output_dir=(
                    output_dir
                ),
            )

        except Exception as exc:

            result = failed_result(
                sequence_files.name,
                exc,
                args,
            )

        results.append(
            result
        )

    csv_path = (
        output_dir
        / "baseline_fidelity.csv"
    )

    summary_path = (
        output_dir
        / "baseline_summary.json"
    )

    write_csv(
        csv_path,
        results,
    )

    write_summary(
        summary_path,
        root,
        results,
        args,
    )

    print_table(
        results
    )

    print()

    print(
        f"Wrote: {csv_path}"
    )

    print(
        f"Wrote: {summary_path}"
    )

    print(
        f"Wrote trajectories under: {output_dir}"
    )

    return (
        0
        if any(
            result.status
            == "ok"
            for result
            in results
        )
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )