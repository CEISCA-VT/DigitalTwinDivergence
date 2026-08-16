#!/usr/bin/env python3
"""
i2Nav-Robot fidelity baseline for the DigitalTwin EKF.

Version 5: Hard NIS Gate + Safe GNSS Reacquisition
---------------------------------------------------
1. Prefer OEM719 RTK GNSS.
2. Fall back to F9P SPP GNSS.
3. Prefer official ODO_SPEED; fall back to RANGER_ODO.
4. Use a strict 2-D NIS gate:
      NIS <= 9.21 -> normal GNSS update
      NIS >  9.21 -> reject GNSS completely
5. NO forced periodic GNSS recovery.
6. NO ordinary soft-GNSS updates.
7. Safe reacquisition after GNSS rejection:
      - estimator continues ODO+IMU prediction,
      - rejected GNSS does not modify the state,
      - a separate reacquisition-gate uncertainty grows with coast time,
      - several consecutive GNSS fixes must pass the relaxed gate,
      - reacquisition uses an appropriately inflated R so the state cannot snap
        strongly to a potentially bad fix.
8. Core process Q is NOT inflated by rejected GNSS.

Run:
python -m DigitalTwin.analysis.i2nav_fidelity_baseline --root public_datasets/im2nav
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
# Data classes
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

    # Absolute trajectory error
    ate_rmse_m: float
    ate_median_m: float
    ate_p95_m: float
    ate_max_m: float

    # SE(2)-aligned trajectory error
    ate_se2_rmse_m: float
    ate_se2_median_m: float
    ate_se2_p95_m: float

    # Heading
    heading_mae_deg: float
    heading_p95_deg: float

    heading_se2_mae_deg: float
    heading_se2_p95_deg: float

    # Relative pose error
    rpe_1s_trans_rmse_m: float
    rpe_1s_rot_rmse_deg: float

    rpe_5s_trans_rmse_m: float
    rpe_5s_rot_rmse_deg: float

    rpe_10s_trans_rmse_m: float
    rpe_10s_rot_rmse_deg: float

    # Drift
    final_error_m: float
    final_error_se2_m: float
    final_drift_per_m: float
    path_length_ratio: float

    # GNSS source
    gnss_source: str
    gnss_file_present: bool

    # GNSS counts
    gnss_updates_seen: int
    gnss_updates_normal: int
    gnss_updates_reacquired: int
    gnss_updates_rejected_nis: int
    gnss_updates_skipped_quality: int

    gnss_updates_used: int

    gnss_rejection_rate_pct: float

    # GNSS timing
    gnss_update_rate_hz: float
    gnss_expected_1hz_coverage_pct: float
    gnss_max_gap_s: float
    gnss_max_coast_s: float

    gnss_median_sigma_m: float

    # NIS statistics
    nis_median: float
    nis_p95: float
    nis_max: float

    # Reacquisition diagnostics
    reacq_candidates: int
    reacq_events: int
    reacq_max_extra_sigma_m: float

    # Thresholds
    nis_gate_threshold: float
    reacq_start_s: float
    reacq_sigma_growth_mps: float
    reacq_sigma_max_m: float
    reacq_consecutive_required: int

    # Motion diagnostics
    mean_speed_mps: float
    p95_abs_yaw_rate_radps: float

    # Input/configuration
    odo_source: str
    imu_yaw_sign: float

    q_xy_sigma_mps: float
    q_heading_sigma_radps: float

    gnss_sigma_max_m: float

    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
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

            if line.startswith("#") or line.startswith("%"):
                continue

            line = line.replace(",", " ")
            tokens = line.split()

            if len(tokens) < min_cols:
                continue

            try:
                values = [
                    float(x)
                    for x in tokens
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
        len(r)
        for r in rows
    )

    return np.asarray(
        [
            r[:width]
            for r in rows
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
            root.rglob(pattern)
        )

        if not matches:
            return None

        matches.sort(
            key=lambda p: (
                len(p.parts),
                len(str(p)),
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
        # 3. None
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
        key=lambda x: (
            rank.get(
                x.name,
                999,
            ),
            x.name,
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

    lat0 = math.radians(
        lat0_deg
    )

    lon0 = math.radians(
        lon0_deg
    )

    east = (
        EARTH_RADIUS_M
        * math.cos(lat0)
        * (
            longitude
            - lon0
        )
    )

    north = (
        EARTH_RADIUS_M
        * (
            latitude
            - lat0
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
) -> dict[
    str,
    np.ndarray,
]:

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

    # NED yaw -> ENU heading
    heading_enu = wrap_array(
        np.pi / 2.0
        - yaw_ned
    )

    return {
        "t": timestamp,
        "x": east,
        "y": north,
        "heading": heading_enu,
        "speed": np.hypot(
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

    wheel_speeds = (
        data[:, 1:5]
    )

    wheel_angles = (
        data[:, 5:9]
    )

    forward_speed = np.mean(
        wheel_speeds
        * np.cos(
            wheel_angles
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
# Time grid
# ---------------------------------------------------------------------------

def make_grid(
    gt_time: np.ndarray,
    odo_time: np.ndarray,
    imu_time: np.ndarray,
    hz: float,
) -> np.ndarray:

    start = max(
        float(gt_time[0]),
        float(odo_time[0]),
        float(imu_time[0]),
    )

    end = min(
        float(gt_time[-1]),
        float(odo_time[-1]),
        float(imu_time[-1]),
    )

    if end <= start:
        raise ValueError(
            "No common GT/ODO/IMU time interval"
        )

    dt = 1.0 / hz

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
    gt: dict[
        str,
        np.ndarray,
    ],
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

        omega[0] = omega[1]

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
        np.abs(speed)
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
    gt: dict[
        str,
        np.ndarray,
    ],
    sigma_max_m: float,
    anchor_count: int,
) -> dict[
    str,
    np.ndarray,
] | None:

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
        np.isfinite(timestamp)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & np.isfinite(sigma_north)
        & np.isfinite(sigma_east)
    )

    if not np.any(valid):
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

    sigma_horizontal = (
        sigma_horizontal[
            valid
        ]
    )

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

    east_offset = float(
        np.median(
            gt_east
            - east_relative[
                anchor_indices
            ]
        )
    )

    north_offset = float(
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
            + east_offset,

        "y":
            north_relative
            + north_offset,

        "sigma_n":
            sigma_north,

        "sigma_e":
            sigma_east,

        "sigma_h":
            sigma_horizontal,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def path_length(
    xy: np.ndarray,
) -> float:

    if len(xy) < 2:
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
) -> dict[
    str,
    float,
]:

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
            rotation[1, 0]
        ),
        float(
            rotation[0, 0]
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

    aligned_heading_error = (
        np.abs(
            wrap_array(
                aligned_heading
                - truth_heading
            )
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
            len(truth_xy)
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

            j = i + lag

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
                    [ct, -st],
                    [st, ct],
                ]
            )

            rotation_estimate = np.array(
                [
                    [ce, -se],
                    [se, ce],
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
# Sequence execution
# ---------------------------------------------------------------------------

def run_sequence(
    files: SequenceFiles,
    *,
    hz: float,
    q_xy_sigma_mps: float,
    q_heading_sigma_radps: float,
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
    # Load input data
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
        1.0 / hz
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

    # GNSS observation matrix.
    #
    # State:
    # [east, north, heading]
    #
    # Measurement:
    # [east, north]

    H = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    identity_2 = np.eye(
        2,
        dtype=float,
    )

    # -----------------------------------------------------------------------
    # GNSS statistics
    # -----------------------------------------------------------------------

    gnss_index = 0

    gnss_seen = 0
    gnss_normal = 0
    gnss_reacquired = 0
    gnss_rejected = 0
    gnss_skipped = 0

    reacq_candidates = 0
    reacq_events = 0
    reacq_candidate_streak = 0

    max_reacq_extra_sigma = 0.0

    nis_values: list[float] = []

    used_gnss_times: list[float] = []
    used_gnss_sigmas: list[float] = []

    # Last accepted GNSS time.
    #
    # Use grid start so time-without-GNSS is
    # meaningful before the first accepted update.

    last_accepted_gnss_time = float(
        grid[0]
    )

    max_coast_s = 0.0

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
    # EKF replay
    # -----------------------------------------------------------------------

    for k in range(
        1,
        len(grid),
    ):

        dt = float(
            grid[k]
            - grid[k - 1]
        )

        if (
            not np.isfinite(dt)
            or dt <= 0
        ):
            dt = dt_nominal

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Q remains fixed here.
        #
        # Rejected GNSS measurements DO NOT inflate Q.
        # ----------------------------------------------------

        q = np.diag(
            [
                (
                    q_xy_sigma_mps
                    * dt
                ) ** 2,

                (
                    q_xy_sigma_mps
                    * dt
                ) ** 2,

                (
                    q_heading_sigma_radps
                    * dt
                ) ** 2,
            ]
        )

        ekf.predict(
            float(
                speed[k]
            ),
            float(
                omega[k]
            ),
            dt,
            q,
        )

        # -------------------------------------------------------------------
        # GNSS
        # -------------------------------------------------------------------

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

                # -----------------------------------------------------------
                # Reported-GNSS quality check
                # -----------------------------------------------------------

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
                    # Normal innovation
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

                    accepted = False

                    # -------------------------------------------------------
                    # Region 1:
                    # Normal hard-NIS gate
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

                        accepted = True

                        gnss_normal += 1

                        reacq_candidate_streak = 0

                    # -------------------------------------------------------
                    # Region 2:
                    # Normal gate rejected the GNSS fix.
                    #
                    # Do NOT update the EKF state.
                    #
                    # Test only whether a safe reacquisition candidate
                    # exists after sufficient coasting time.
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

                        reacquisition_candidate = False
                        extra_sigma = 0.0

                        if (
                            coast_s
                            >= reacq_start_s
                        ):

                            # ------------------------------------------------
                            # Reacquisition uncertainty.
                            #
                            # This is used ONLY in the reacquisition gate and
                            # potential reacquisition measurement update.
                            #
                            # It does NOT modify:
                            #   Q
                            #   P during normal prediction
                            #   state during rejected fixes
                            # ------------------------------------------------

                            extra_sigma = min(
                                reacq_sigma_max_m,
                                reacq_sigma_growth_mps
                                * coast_s,
                            )

                            max_reacq_extra_sigma = max(
                                max_reacq_extra_sigma,
                                extra_sigma,
                            )

                            gate_extra_covariance = (
                                identity_2
                                * extra_sigma**2
                            )

                            S_reacq = (
                                S
                                + gate_extra_covariance
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

                                reacquisition_candidate = True

                        # ----------------------------------------------------
                        # Require several consecutive candidate fixes.
                        #
                        # A single GNSS fix can never force reacquisition.
                        # ----------------------------------------------------

                        if reacquisition_candidate:

                            reacq_candidate_streak += 1

                            reacq_candidates += 1

                        else:

                            reacq_candidate_streak = 0

                        # ----------------------------------------------------
                        # Safe reacquisition
                        # ----------------------------------------------------

                        if (
                            reacquisition_candidate
                            and reacq_candidate_streak
                            >= reacq_consecutive_required
                        ):

                            # Use the same extra uncertainty that allowed
                            # this fix to pass the relaxed consistency gate.
                            #
                            # Therefore even a reacquisition fix has bounded
                            # influence on the state.

                            R_reacq = (
                                R
                                + identity_2
                                * extra_sigma**2
                            )

                            ekf.update_gps(
                                z,
                                R_reacq,
                            )

                            accepted = True

                            gnss_reacquired += 1

                            reacq_events += 1

                            reacq_candidate_streak = 0

                        else:

                            gnss_rejected += 1

                    # -------------------------------------------------------
                    # Accepted GNSS bookkeeping
                    # -------------------------------------------------------

                    if accepted:

                        last_accepted_gnss_time = float(
                            gnss[
                                "t"
                            ][latest]
                        )

                        used_gnss_times.append(
                            last_accepted_gnss_time
                        )

                        used_gnss_sigmas.append(
                            sigma_h
                        )

                else:

                    gnss_skipped += 1

        # Track coast duration even when no GNSS sample
        # occurred exactly at this EKF timestep.

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
    # Evaluation
    # -----------------------------------------------------------------------

    metrics = summarize_errors(
        estimates[:, :2],
        estimates[:, 2],
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

        nis_median = _nan()
        nis_p95 = _nan()
        nis_max = _nan()

    # -----------------------------------------------------------------------
    # GNSS accepted-update gap
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
    # Save trajectory
    # -----------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_path = (
        output_dir
        / f"{files.name}_trajectory.csv"
    )

    position_error = (
        np.linalg.norm(
            estimates[:, :2]
            - truth_xy,
            axis=1,
        )
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

                    estimates[i, 0],
                    estimates[i, 1],
                    estimates[i, 2],

                    speed[i],
                    omega[i],

                    position_error[i],

                    files.gnss_source,
                ]
            )

    mode = (
        f"{files.gnss_source}"
        "+ODO+IMU"
        "+HARD_NIS"
        "+SAFE_REACQ"
        if gnss_seen > 0
        else "ODO+IMU"
    )

    return FidelityResult(
        sequence=(
            files.name
        ),

        status="ok",

        mode=mode,

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

        gnss_sigma_max_m=(
            gnss_sigma_max_m
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

    nan = _nan()

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

        mean_speed_mps=nan,
        p95_abs_yaw_rate_radps=nan,

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

        gnss_sigma_max_m=(
            args.gnss_sigma_max_m
        ),

        error=(
            f"{type(exc).__name__}: {exc}"
        ),
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(
    path: Path,
    results: list[
        FidelityResult
    ],
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
    results: Iterable[
        FidelityResult
    ],
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
    results: list[
        FidelityResult
    ],
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

    oem = [
        result
        for result
        in fused
        if result.gnss_source
        == "OEM719_RTK"
    ]

    dead_reckoning = [
        result
        for result
        in successful
        if result.gnss_source
        == "NONE"
    ]

    def mean_field(
        subset: list[
            FidelityResult
        ],
        field: str,
    ) -> float | None:

        values = finite_values(
            subset,
            field,
        )

        return (
            float(
                np.mean(
                    values
                )
            )
            if len(values)
            else None
        )

    def rms_field(
        subset: list[
            FidelityResult
        ],
        field: str,
    ) -> float | None:

        values = finite_values(
            subset,
            field,
        )

        return (
            float(
                np.sqrt(
                    np.mean(
                        values**2
                    )
                )
            )
            if len(values)
            else None
        )

    summary = {
        "schema":
            "i2nav_fidelity_baseline_v5_hard_gate_safe_reacq",

        "dataset_root":
            str(root),

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
                len(results),

            "successful":
                len(successful),

            "fused":
                len(fused),

            "oem719_rtk":
                len(oem),

            "odo_imu_only":
                len(dead_reckoning),

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

            "reacquisition_events":
                int(
                    sum(
                        result.reacq_events
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

            "rpe_1s_mean_m":
                mean_field(
                    successful,
                    "rpe_1s_trans_rmse_m",
                ),

            "heading_mae_mean_deg":
                mean_field(
                    successful,
                    "heading_mae_deg",
                ),

            "gnss_rejection_rate_mean_pct":
                mean_field(
                    fused,
                    "gnss_rejection_rate_pct",
                ),

            "nis_median_mean":
                mean_field(
                    fused,
                    "nis_median",
                ),

            "nis_p95_mean":
                mean_field(
                    fused,
                    "nis_p95",
                ),

            "gnss_max_coast_mean_s":
                mean_field(
                    fused,
                    "gnss_max_coast_s",
                ),
        },

        "configuration": {
            "rate_hz":
                args.rate_hz,

            "q_xy_sigma_mps":
                args.q_xy_sigma_mps,

            "q_heading_sigma_radps":
                args.q_heading_sigma_radps,

            "gnss_sigma_max_m":
                args.gnss_sigma_max_m,

            "gnss_sigma_floor_m":
                args.gnss_sigma_floor_m,

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

        "security_note": (
            "Rejected GNSS does not directly update the state and does not "
            "inflate the core process covariance Q. Reacquisition uses a "
            "separate bounded gate uncertainty and requires consecutive "
            "consistent fixes."
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
    results: list[
        FidelityResult
    ],
) -> None:

    print()

    print(
        f"{'sequence':<14} "
        f"{'GNSS':<11} "
        f"{'ATE':>7} "
        f"{'RPE1':>7} "
        f"{'head':>6} "
        f"{'normal':>7} "
        f"{'reacq':>6} "
        f"{'reject':>7} "
        f"{'rej%':>6} "
        f"{'gap':>7} "
        f"{'coast':>7} "
        f"{'NIS50':>8}"
    )

    print(
        "-"
        * 112
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
            f"{result.gnss_source:<11} "
            f"{result.ate_rmse_m:7.3f} "
            f"{result.rpe_1s_trans_rmse_m:7.3f} "
            f"{result.heading_mae_deg:6.2f} "
            f"{result.gnss_updates_normal:7d} "
            f"{result.gnss_updates_reacquired:6d} "
            f"{result.gnss_updates_rejected_nis:7d} "
            f"{result.gnss_rejection_rate_pct:6.1f} "
            f"{result.gnss_max_gap_s:7.1f} "
            f"{result.gnss_max_coast_s:7.1f} "
            f"{result.nis_median:8.2f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "i2Nav DigitalTwin baseline with "
            "hard GNSS NIS gating and safe reacquisition."
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
            "results/i2nav"
        ),
    )

    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
    )

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

    # 99% chi-square threshold,
    # 2 degrees of freedom.

    parser.add_argument(
        "--gnss-nis-gate",
        type=float,
        default=9.21,
    )

    # --------------------------------------------------------
    # Safe reacquisition
    # --------------------------------------------------------

    parser.add_argument(
        "--reacq-start-s",
        type=float,
        default=10.0,
        help=(
            "Minimum time without accepted GNSS before "
            "safe reacquisition logic is enabled."
        ),
    )

    parser.add_argument(
        "--reacq-sigma-growth-mps",
        type=float,
        default=0.05,
        help=(
            "Growth rate of the separate reacquisition-gate "
            "position sigma in m/s of GNSS coast time."
        ),
    )

    parser.add_argument(
        "--reacq-sigma-max-m",
        type=float,
        default=5.0,
        help=(
            "Maximum extra position sigma used only by the "
            "safe reacquisition gate."
        ),
    )

    parser.add_argument(
        "--reacq-consecutive",
        type=int,
        default=3,
        help=(
            "Number of consecutive relaxed-gate-consistent GNSS "
            "measurements required before a reacquisition update."
        ),
    )

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
        args.root.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
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
        "GNSS policy:"
    )

    print(
        f"  normal NIS gate <= {args.gnss_nis_gate}"
    )

    print(
        "  rejected fixes do NOT update state"
    )

    print(
        "  rejected fixes do NOT inflate Q"
    )

    print(
        f"  reacquisition starts after "
        f"{args.reacq_start_s:.1f} s"
    )

    print(
        f"  reacquisition sigma growth = "
        f"{args.reacq_sigma_growth_mps:.3f} m/s"
    )

    print(
        f"  reacquisition sigma cap = "
        f"{args.reacq_sigma_max_m:.2f} m"
    )

    print(
        f"  consecutive fixes required = "
        f"{args.reacq_consecutive}"
    )

    print()

    results: list[
        FidelityResult
    ] = []

    for files_for_sequence in files:

        odo_label = (
            "ODO_SPEED"
            if files_for_sequence.odo_speed
            is not None
            else "RANGER"
        )

        print(
            f"[run] {files_for_sequence.name} "
            f"| GNSS={files_for_sequence.gnss_source} "
            f"| ODO={odo_label}"
        )

        try:

            result = run_sequence(
                files_for_sequence,

                hz=(
                    args.rate_hz
                ),

                q_xy_sigma_mps=(
                    args.q_xy_sigma_mps
                ),

                q_heading_sigma_radps=(
                    args.q_heading_sigma_radps
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
                files_for_sequence.name,
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