#!/usr/bin/env python3
"""
i2nav_zero_shot_observable_audit.py
===================================

Post-hoc, NO-TRAINING audit for physically meaningful ODO+IMU observables that
could improve Twin V2 without sacrificing zero-shot transfer.

Core design rule
----------------
The neural model should not consume Ranger-specific wheel identities/steering
angles directly. Instead, a deterministic platform adapter converts raw wheel
telemetry into canonical body-motion quantities that are also constructible on
other encoder+IMU robots:

    v_wheel          forward wheel/encoder velocity
    omega_wheel      wheel-kinematic yaw rate
    omega_imu        IMU yaw rate
    yaw_disagreement omega_imu - omega_wheel
    yaw_disagreement_normalized

This script evaluates whether those canonical signals make parking01 and
parking02 easier to distinguish than the original V1 six-feature history.

It does NOT modify V1, does NOT train V2, and does NOT use GNSS/camera/LiDAR
as model inputs.

i2Nav Ranger MINI 3.0 geometry defaults:
    wheelbase = 0.494 m
    track     = 0.370 m

Expected Ranger raw text format:
    t, speed1, speed2, speed3, speed4, angle1, angle2, angle3, angle4

Official i2Nav raw-text wheel order:
    1 = right front
    2 = left front
    3 = right back
    4 = left back

IMPORTANT
---------
The steering-angle sign convention must be correct. The script never silently
chooses a sign from test performance. It reports wheel-yaw-vs-IMU correlation.
If the correlation is strongly negative, rerun with --angle-sign -1.

Example (PowerShell)
--------------------
python -u -m DigitalTwin.analysis.i2nav_zero_shot_observable_audit `
    --root ./public_datasets/im2nav `
    --output-dir ./results/i2nav_zero_shot_observable_audit

If *_RANGER_ODO.txt files are not present locally, the script will tell you
exactly which sequences are missing them.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RADPS_TO_DEG_PER_MIN = 180.0 / math.pi * 60.0
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit zero-shot-safe wheel/IMU physical observables."
    )

    p.add_argument(
        "--root",
        type=Path,
        default=Path("public_datasets/im2nav"),
        help="i2Nav dataset root.",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/i2nav_zero_shot_observable_audit"),
        help="Output directory.",
    )

    p.add_argument(
        "--wheelbase-m",
        type=float,
        default=0.494,
        help="Ranger MINI 3.0 wheelbase in meters.",
    )

    p.add_argument(
        "--track-m",
        type=float,
        default=0.370,
        help="Ranger MINI 3.0 track width in meters.",
    )

    p.add_argument(
        "--angle-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Multiply all raw Ranger steering angles by this sign.",
    )

    p.add_argument(
        "--contexts-s",
        type=str,
        default="2,30",
        help="History lengths to compare for ambiguity, default 2,30.",
    )

    p.add_argument(
        "--target-s",
        type=float,
        default=30.0,
        help="Causal persistent yaw target window.",
    )

    p.add_argument(
        "--step-s",
        type=float,
        default=1.0,
        help="History endpoint spacing for NN ambiguity analysis.",
    )

    p.add_argument(
        "--query-chunk",
        type=int,
        default=256,
        help="Chunk size for exact cross-sequence nearest-neighbor search.",
    )

    return p.parse_args()


def original_default_args(original):
    old_argv = sys.argv[:]

    try:
        sys.argv = ["i2nav_loso_ablation.py"]
        return original.parse_args()

    finally:
        sys.argv = old_argv


def load_prepared_sequences(root: Path):
    original = importlib.import_module(
        "DigitalTwin.analysis.i2nav_loso_ablation"
    )

    defaults = original_default_args(original)

    files = original.discover_files(root)

    prepared = {}

    for item in files:
        prepared[item.name] = original.prepare_sequence(
            item,
            hz=defaults.rate_hz,
            imu_yaw_sign=defaults.imu_yaw_sign,
            gnss_sigma_max_m=defaults.gnss_sigma_max_m,
            gnss_anchor_count=defaults.gnss_anchor_count,
        )

    return prepared


def find_ranger_files(root: Path) -> dict[str, Path]:
    found = {}

    for p in root.rglob("*_RANGER_ODO.txt"):
        name = p.name[: -len("_RANGER_ODO.txt")]
        found[name] = p

    return found


def read_numeric_text(
    path: Path,
    min_cols: int = 9,
) -> np.ndarray:
    """
    Robust reader for whitespace/comma separated numeric raw text.

    Ignores:
      * blank lines
      * comment lines
      * nonnumeric header lines
    """

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:
            s = line.strip()

            if not s:
                continue

            if s.startswith("#") or s.startswith("%"):
                continue

            s = s.replace(",", " ")

            arr = np.fromstring(
                s,
                sep=" ",
            )

            if (
                len(arr) >= min_cols
                and np.all(
                    np.isfinite(
                        arr[:min_cols]
                    )
                )
            ):
                rows.append(
                    arr[:min_cols]
                )

    if not rows:
        raise RuntimeError(
            f"No numeric rows with >= {min_cols} columns found in {path}"
        )

    out = np.asarray(
        rows,
        dtype=np.float64,
    )

    order = np.argsort(
        out[:, 0],
        kind="stable",
    )

    return out[order]


def ranger_wheel_positions(
    wheelbase_m: float,
    track_m: float,
) -> np.ndarray:
    """
    Coordinates:
        x positive front
        y positive right

    Wheel order:
        RF
        LF
        RB
        LB
    """

    x = wheelbase_m / 2.0
    y = track_m / 2.0

    return np.asarray(
        [
            [x, y],      # right front
            [x, -y],     # left front
            [-x, y],     # right back
            [-x, -y],    # left back
        ],
        dtype=np.float64,
    )


def solve_planar_twist_batch(
    speeds: np.ndarray,
    angles: np.ndarray,
    wheel_positions: np.ndarray,
) -> np.ndarray:
    """
    Solve wheel rolling constraints for body twist:

        [vx, vy, omega]

    For wheel i:

        d_i^T (
            [vx, vy]
            +
            omega * [-y_i, x_i]
        )
        =
        speed_i

    where:

        d_i = [cos(delta_i), sin(delta_i)]

    Four wheels give an overdetermined 4x3 system.

    Each timestamp is solved independently using regularized least squares.

    IMPORTANT NUMPY 2.x FIX
    -----------------------
    ATA has shape:

        (N, 3, 3)

    ATb has shape:

        (N, 3)

    NumPy 2.x no longer necessarily interprets (N,3) as N independent
    right-hand-side vectors in np.linalg.solve.

    We therefore explicitly reshape ATb to:

        (N, 3, 1)

    solve, then squeeze back to:

        (N, 3)
    """

    speeds = np.asarray(
        speeds,
        dtype=np.float64,
    )

    angles = np.asarray(
        angles,
        dtype=np.float64,
    )

    pos = np.asarray(
        wheel_positions,
        dtype=np.float64,
    )

    if speeds.ndim != 2:
        raise ValueError(
            f"speeds must have shape (N,4), got {speeds.shape}"
        )

    if angles.ndim != 2:
        raise ValueError(
            f"angles must have shape (N,4), got {angles.shape}"
        )

    if speeds.shape != angles.shape:
        raise ValueError(
            "speeds and angles must have identical shapes: "
            f"{speeds.shape} vs {angles.shape}"
        )

    if speeds.shape[1] != 4:
        raise ValueError(
            f"Expected four wheels, got shape {speeds.shape}"
        )

    if pos.shape != (4, 2):
        raise ValueError(
            "wheel_positions must have shape (4,2), "
            f"got {pos.shape}"
        )

    c = np.cos(angles)
    s = np.sin(angles)

    n = len(speeds)

    # A shape:
    #   (N timestamps, 4 wheels, 3 unknowns)
    #
    # Unknowns are:
    #   vx
    #   vy
    #   omega
    A = np.empty(
        (n, 4, 3),
        dtype=np.float64,
    )

    A[:, :, 0] = c
    A[:, :, 1] = s

    A[:, :, 2] = (
        -c * pos[None, :, 1]
        +
        s * pos[None, :, 0]
    )

    # Batched normal equations.
    #
    # ATA:
    #   (N,3,3)
    #
    # ATb:
    #   (N,3)
    ATA = np.einsum(
        "nij,nik->njk",
        A,
        A,
    )

    ATb = np.einsum(
        "nij,ni->nj",
        A,
        speeds,
    )

    # Small ridge regularization to avoid numerical singularity.
    ATA[
        :,
        np.arange(3),
        np.arange(3),
    ] += 1e-8

    # ------------------------------------------------------------
    # FIX:
    #
    # NumPy 2.x can interpret ATb=(N,3) incorrectly for stacked
    # solves.
    #
    # Explicitly create:
    #
    #     (N,3,1)
    #
    # solve:
    #
    #     (N,3,3) @ (N,3,1) = (N,3,1)
    #
    # then remove final dimension.
    # ------------------------------------------------------------
    twist = np.linalg.solve(
        ATA,
        ATb[..., None],
    )[..., 0]

    if twist.shape != (n, 3):
        raise RuntimeError(
            "Unexpected batched twist shape: "
            f"{twist.shape}; expected {(n, 3)}"
        )

    if not np.all(np.isfinite(twist)):
        raise RuntimeError(
            "Non-finite values produced by wheel kinematic solver."
        )

    return twist


def interp_to_grid(
    source_t: np.ndarray,
    source_v: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:

    source_t = np.asarray(
        source_t,
        dtype=float,
    )

    source_v = np.asarray(
        source_v,
        dtype=float,
    )

    grid = np.asarray(
        grid,
        dtype=float,
    )

    if source_v.ndim == 1:
        return np.interp(
            grid,
            source_t,
            source_v,
        )

    cols = [
        np.interp(
            grid,
            source_t,
            source_v[:, j],
        )
        for j in range(
            source_v.shape[1]
        )
    ]

    return np.column_stack(
        cols
    )


def pearson(
    x,
    y,
) -> float:

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return float("nan")

    if np.std(x) < EPS:
        return float("nan")

    if np.std(y) < EPS:
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def rmse(x) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if not len(x):
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                x * x
            )
        )
    )


def rolling_mean_samples(
    values: np.ndarray,
    samples: int,
) -> np.ndarray:

    samples = max(
        1,
        int(samples),
    )

    return (
        pd.Series(
            np.asarray(
                values,
                dtype=float,
            )
        )
        .rolling(
            window=samples,
            min_periods=samples,
        )
        .mean()
        .to_numpy()
    )


def build_sequence_candidates(
    seq,
    ranger_path: Path,
    wheelbase_m: float,
    track_m: float,
    angle_sign: float,
) -> tuple[pd.DataFrame, dict]:

    raw = read_numeric_text(
        ranger_path,
        min_cols=9,
    )

    t_raw = raw[:, 0]

    speed_raw = raw[
        :,
        1:5,
    ]

    angle_raw = (
        angle_sign
        *
        raw[
            :,
            5:9,
        ]
    )

    positions = ranger_wheel_positions(
        wheelbase_m,
        track_m,
    )

    twist_raw = solve_planar_twist_batch(
        speed_raw,
        angle_raw,
        positions,
    )

    twist = interp_to_grid(
        t_raw,
        twist_raw,
        seq.grid,
    )

    wheel_vx = twist[:, 0]
    wheel_vy = twist[:, 1]
    wheel_omega = -twist[:, 2]

    imu_omega = np.asarray(
        seq.imu_yaw_rate,
        dtype=float,
    )

    odo_speed = np.asarray(
        seq.odo_speed,
        dtype=float,
    )

    yaw_disagreement = (
        imu_omega
        -
        wheel_omega
    )

    yaw_disagreement_norm = (
        yaw_disagreement
        /
        (
            np.abs(imu_omega)
            +
            np.abs(wheel_omega)
            +
            0.02
        )
    )

    # Diagnostic-only candidate.
    #
    # Do not automatically make this a learned V2 feature unless
    # the target platforms can construct the same canonical quantity.
    wheel_lateral_ratio = (
        wheel_vy
        /
        (
            np.abs(wheel_vx)
            +
            0.20
        )
    )

    df = pd.DataFrame(
        {
            "time_s":
                np.asarray(
                    seq.grid,
                    dtype=float,
                ),

            "odo_speed_mps":
                odo_speed,

            "imu_yaw_rate_radps":
                imu_omega,

            "wheel_vx_mps":
                wheel_vx,

            "wheel_vy_mps":
                wheel_vy,

            "wheel_yaw_rate_radps":
                wheel_omega,

            "yaw_disagreement_radps":
                yaw_disagreement,

            "yaw_disagreement_normalized":
                yaw_disagreement_norm,

            "wheel_lateral_ratio":
                wheel_lateral_ratio,

            "true_yaw_residual_radps":
                np.asarray(
                    seq.target_corrections[:, 1],
                    dtype=float,
                ),
        }
    )

    summary = {
        "test_sequence":
            seq.name,

        "ranger_file":
            str(ranger_path),

        "raw_ranger_rows":
            int(len(raw)),

        "prepared_samples":
            int(len(seq.grid)),

        "wheel_vx_vs_v1_odo_corr":
            pearson(
                wheel_vx,
                odo_speed,
            ),

        "wheel_vx_minus_v1_odo_rmse_mps":
            rmse(
                wheel_vx
                -
                odo_speed
            ),

        "wheel_yaw_vs_imu_corr":
            pearson(
                wheel_omega,
                imu_omega,
            ),

        "wheel_yaw_std_radps":
            float(
                np.std(
                    wheel_omega
                )
            ),

        "imu_yaw_std_radps":
            float(
                np.std(
                    imu_omega
                )
            ),

        "yaw_disagreement_mean_deg_per_min":
            float(
                np.mean(
                    yaw_disagreement
                )
                *
                RADPS_TO_DEG_PER_MIN
            ),

        "yaw_disagreement_std_radps":
            float(
                np.std(
                    yaw_disagreement
                )
            ),
    }

    return (
        df,
        summary,
    )


def feature_sets(
    seq,
    candidate_df: pd.DataFrame,
) -> dict[str, np.ndarray]:

    baseline = np.asarray(
        seq.features,
        dtype=np.float32,
    )

    wheel_omega = (
        candidate_df[
            "wheel_yaw_rate_radps"
        ]
        .to_numpy(
            dtype=np.float32
        )
        [:, None]
    )

    disagreement = (
        candidate_df[
            "yaw_disagreement_radps"
        ]
        .to_numpy(
            dtype=np.float32
        )
        [:, None]
    )

    disagreement_norm = (
        candidate_df[
            "yaw_disagreement_normalized"
        ]
        .to_numpy(
            dtype=np.float32
        )
        [:, None]
    )

    return {
        "v1_baseline_6":
            baseline,

        "baseline_plus_wheel_yaw":
            np.column_stack(
                [
                    baseline,
                    wheel_omega,
                ]
            ).astype(
                np.float32
            ),

        "baseline_plus_yaw_disagreement":
            np.column_stack(
                [
                    baseline,
                    disagreement,
                ]
            ).astype(
                np.float32
            ),

        "baseline_plus_wheel_consistency_pack":
            np.column_stack(
                [
                    baseline,
                    wheel_omega,
                    disagreement,
                    disagreement_norm,
                ]
            ).astype(
                np.float32
            ),
    }


def history_matrix(
    features: np.ndarray,
    context_samples: int,
    indices: np.ndarray,
) -> np.ndarray:

    rows = []

    for idx in indices:
        idx = int(idx)

        start = (
            idx
            -
            context_samples
            +
            1
        )

        rows.append(
            features[
                start:
                idx + 1
            ].reshape(-1)
        )

    return np.asarray(
        rows,
        dtype=np.float32,
    )


def nearest_cross_sequence(
    query: np.ndarray,
    reference: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:

    q = np.asarray(
        query,
        dtype=np.float32,
    )

    r = np.asarray(
        reference,
        dtype=np.float32,
    )

    if q.ndim != 2:
        raise ValueError(
            f"query must be 2D, got {q.shape}"
        )

    if r.ndim != 2:
        raise ValueError(
            f"reference must be 2D, got {r.shape}"
        )

    if q.shape[1] != r.shape[1]:
        raise ValueError(
            "query/reference dimensionality mismatch: "
            f"{q.shape} vs {r.shape}"
        )

    r_norm = np.sum(
        r * r,
        axis=1,
    )

    best_d2 = np.full(
        len(q),
        np.inf,
        dtype=np.float64,
    )

    best_idx = np.full(
        len(q),
        -1,
        dtype=np.int64,
    )

    chunk = max(
        1,
        int(chunk),
    )

    for start in range(
        0,
        len(q),
        chunk,
    ):
        stop = min(
            start + chunk,
            len(q),
        )

        qc = q[
            start:stop
        ]

        q_norm = np.sum(
            qc * qc,
            axis=1,
        )[:, None]

        d2 = (
            q_norm
            +
            r_norm[None, :]
            -
            2.0
            *
            (
                qc
                @
                r.T
            )
        )

        d2 = np.maximum(
            d2,
            0.0,
        )

        idx = np.argmin(
            d2,
            axis=1,
        )

        vals = d2[
            np.arange(
                len(idx)
            ),
            idx,
        ]

        best_d2[
            start:stop
        ] = vals

        best_idx[
            start:stop
        ] = idx

    rms_distance = np.sqrt(
        best_d2
        /
        max(
            q.shape[1],
            1,
        )
    )

    return (
        rms_distance,
        best_idx,
    )


def build_ambiguity_payload(
    seq,
    features: np.ndarray,
    context_s: float,
    target_s: float,
    step_s: float,
) -> dict:

    t = np.asarray(
        seq.grid,
        dtype=float,
    )

    if len(t) < 2:
        raise RuntimeError(
            f"{seq.name}: time grid too short."
        )

    dt = float(
        np.median(
            np.diff(t)
        )
    )

    if (
        not np.isfinite(dt)
        or
        dt <= 0
    ):
        raise RuntimeError(
            f"{seq.name}: invalid sample interval."
        )

    hz = 1.0 / dt

    context_samples = max(
        1,
        int(
            round(
                context_s
                *
                hz
            )
        ),
    )

    target_samples = max(
        1,
        int(
            round(
                target_s
                *
                hz
            )
        ),
    )

    step_samples = max(
        1,
        int(
            round(
                step_s
                *
                hz
            )
        ),
    )

    true_dw = np.asarray(
        seq.target_corrections[:, 1],
        dtype=float,
    )

    target = rolling_mean_samples(
        true_dw,
        target_samples,
    )

    first = max(
        context_samples - 1,
        target_samples - 1,
    )

    idx = np.arange(
        first,
        len(t),
        step_samples,
        dtype=int,
    )

    idx = idx[
        np.isfinite(
            target[idx]
        )
    ]

    if len(idx) < 20:
        raise RuntimeError(
            f"{seq.name}: only {len(idx)} usable histories."
        )

    X = history_matrix(
        features,
        context_samples,
        idx,
    )

    return {
        "X":
            X,

        "y":
            target[idx],

        "time_s":
            t[idx],
    }


def ambiguity_for_feature_set(
    prepared: dict,
    candidate_frames: dict,
    feature_set_name: str,
    context_s: float,
    target_s: float,
    step_s: float,
    query_chunk: int,
) -> list[dict]:

    payload = {}

    for name in (
        "parking01",
        "parking02",
    ):

        fsets = feature_sets(
            prepared[name],
            candidate_frames[name],
        )

        payload[name] = build_ambiguity_payload(
            prepared[name],
            fsets[
                feature_set_name
            ],
            context_s,
            target_s,
            step_s,
        )

    combined = np.vstack(
        [
            payload[
                "parking01"
            ]["X"],

            payload[
                "parking02"
            ]["X"],
        ]
    )

    mean = np.mean(
        combined,
        axis=0,
    )

    std = np.maximum(
        np.std(
            combined,
            axis=0,
        ),
        1e-5,
    )

    for name in payload:
        payload[name]["Xz"] = (
            payload[name]["X"]
            -
            mean
        ) / std

    rows = []

    for (
        q_name,
        r_name,
    ) in (
        (
            "parking02",
            "parking01",
        ),
        (
            "parking01",
            "parking02",
        ),
    ):

        q = payload[
            q_name
        ]

        r = payload[
            r_name
        ]

        dist, nn = nearest_cross_sequence(
            q["Xz"],
            r["Xz"],
            query_chunk,
        )

        gap = (
            q["y"]
            -
            r["y"][nn]
        ) * RADPS_TO_DEG_PER_MIN

        abs_gap = np.abs(
            gap
        )

        med = float(
            np.median(
                dist
            )
        )

        q25 = float(
            np.quantile(
                dist,
                0.25,
            )
        )

        close50 = (
            dist
            <=
            med
        )

        close25 = (
            dist
            <=
            q25
        )

        rows.append(
            {
                "context_s":
                    float(
                        context_s
                    ),

                "feature_set":
                    feature_set_name,

                "query_sequence":
                    q_name,

                "reference_sequence":
                    r_name,

                "n_histories":
                    int(
                        len(dist)
                    ),

                "history_dimensions":
                    int(
                        q["X"].shape[1]
                    ),

                "median_nearest_distance":
                    med,

                "close50_median_abs_target_gap_deg_per_min":
                    float(
                        np.median(
                            abs_gap[
                                close50
                            ]
                        )
                    ),

                "close25_median_abs_target_gap_deg_per_min":
                    float(
                        np.median(
                            abs_gap[
                                close25
                            ]
                        )
                    ),

                "close50_fraction_gap_gt_1degmin":
                    float(
                        np.mean(
                            abs_gap[
                                close50
                            ]
                            >
                            1.0
                        )
                    ),

                "close25_fraction_gap_gt_1degmin":
                    float(
                        np.mean(
                            abs_gap[
                                close25
                            ]
                            >
                            1.0
                        )
                    ),
            }
        )

    return rows


def candidate_target_correlations(
    seq,
    candidate_df: pd.DataFrame,
    target_s: float,
) -> dict:

    t = np.asarray(
        seq.grid,
        dtype=float,
    )

    dt = float(
        np.median(
            np.diff(t)
        )
    )

    samples = max(
        1,
        int(
            round(
                target_s
                /
                dt
            )
        ),
    )

    true_target = rolling_mean_samples(
        np.asarray(
            seq.target_corrections[:, 1],
            dtype=float,
        ),
        samples,
    )

    out = {
        "test_sequence":
            seq.name
    }

    candidates = {
        "wheel_yaw":
            candidate_df[
                "wheel_yaw_rate_radps"
            ].to_numpy(
                dtype=float
            ),

        "yaw_disagreement":
            candidate_df[
                "yaw_disagreement_radps"
            ].to_numpy(
                dtype=float
            ),

        "yaw_disagreement_normalized":
            candidate_df[
                "yaw_disagreement_normalized"
            ].to_numpy(
                dtype=float
            ),

        "wheel_lateral_ratio":
            candidate_df[
                "wheel_lateral_ratio"
            ].to_numpy(
                dtype=float
            ),
    }

    mask_target = np.isfinite(
        true_target
    )

    for (
        key,
        values,
    ) in candidates.items():

        out[
            f"{key}_vs_true_{int(target_s)}s_corr"
        ] = pearson(
            values[
                mask_target
            ],
            true_target[
                mask_target
            ],
        )

    return out


def write_findings(
    output_dir: Path,
    inventory: pd.DataFrame,
    ambiguity: pd.DataFrame,
    candidate_corr: pd.DataFrame,
    angle_sign: float,
) -> None:

    lines = []

    lines.append(
        "Zero-shot physical-observable audit"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        "Goal: add only canonical encoder+IMU quantities that can be "
        "reconstructed on other robots, rather than Ranger-specific raw channels."
    )

    lines.append("")

    lines.append(
        f"Ranger steering angle sign used: {angle_sign:+.0f}"
    )

    lines.append("")

    lines.append(
        "Wheel-kinematic sanity checks"
    )

    for _, r in inventory.iterrows():

        lines.append(
            f"  {r['test_sequence']}: "
            f"wheel vx vs V1 odo corr="
            f"{r['wheel_vx_vs_v1_odo_corr']:+.3f}, "
            f"RMSE="
            f"{r['wheel_vx_minus_v1_odo_rmse_mps']:.4f} m/s, "
            f"wheel yaw vs IMU corr="
            f"{r['wheel_yaw_vs_imu_corr']:+.3f}"
        )

    lines.append("")

    yaw_corr = pd.to_numeric(
        inventory[
            "wheel_yaw_vs_imu_corr"
        ],
        errors="coerce",
    )

    if (
        yaw_corr.notna().any()
        and
        yaw_corr.median() < -0.3
    ):

        lines.append(
            "WARNING: median wheel-yaw vs IMU correlation is strongly negative. "
            "The steering-angle sign convention is probably reversed. "
            "Rerun with --angle-sign -1 before interpreting ambiguity results."
        )

        lines.append("")

    lines.append(
        "Candidate correlation with true persistent yaw target"
    )

    for _, r in candidate_corr.iterrows():

        pieces = [
            f"  {r['test_sequence']}:"
        ]

        for c in candidate_corr.columns:

            if c == "test_sequence":
                continue

            value = r[c]

            if pd.isna(value):
                pieces.append(
                    f"{c}=nan"
                )
            else:
                pieces.append(
                    f"{c}={float(value):+.3f}"
                )

        lines.append(
            " ".join(
                pieces
            )
        )

    lines.append("")

    lines.append(
        "Cross-sequence ambiguity (mean of both directions)"
    )

    grouped = (
        ambiguity
        .groupby(
            [
                "context_s",
                "feature_set",
            ],
            sort=True,
        )
        .agg(
            close50_gt1=(
                "close50_fraction_gap_gt_1degmin",
                "mean",
            ),

            close25_gt1=(
                "close25_fraction_gap_gt_1degmin",
                "mean",
            ),

            close50_gap=(
                "close50_median_abs_target_gap_deg_per_min",
                "mean",
            ),

            close25_gap=(
                "close25_median_abs_target_gap_deg_per_min",
                "mean",
            ),
        )
        .reset_index()
    )

    for _, r in grouped.iterrows():

        lines.append(
            f"  {r['context_s']:.0f}s "
            f"{r['feature_set']}: "
            f"close50 >1deg/min="
            f"{100*r['close50_gt1']:.1f}%, "
            f"close50 median gap="
            f"{r['close50_gap']:.3f} deg/min"
        )

    lines.append("")

    lines.append(
        "Decision rule"
    )

    lines.append(
        "  Prefer a candidate only if it lowers cross-sequence ambiguity "
        "materially while remaining computable from encoder+IMU telemetry on "
        "i2Nav, TerraSentia, and UGV01. Do not select a feature merely because "
        "it improves i2Nav training performance."
    )

    lines.append(
        "  Raw Ranger wheel IDs/steering angles should remain inside the "
        "deterministic platform adapter; the learned Twin core should receive "
        "canonical quantities such as wheel yaw and wheel-vs-IMU disagreement."
    )

    (
        output_dir
        /
        "zero_shot_observable_findings.txt"
    ).write_text(
        "\n".join(
            lines
        )
        +
        "\n",
        encoding="utf-8",
    )


def plot_ambiguity(
    ambiguity: pd.DataFrame,
    output_dir: Path,
) -> None:

    grouped = (
        ambiguity
        .groupby(
            [
                "context_s",
                "feature_set",
            ],
            sort=True,
        )[
            "close50_fraction_gap_gt_1degmin"
        ]
        .mean()
        .reset_index()
    )

    fig = plt.figure(
        figsize=(
            10,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    for (
        name,
        g,
    ) in grouped.groupby(
        "feature_set",
        sort=False,
    ):

        g = g.sort_values(
            "context_s"
        )

        ax.plot(
            g[
                "context_s"
            ],
            100.0
            *
            g[
                "close50_fraction_gap_gt_1degmin"
            ],
            marker="o",
            label=name,
        )

    ax.set_xlabel(
        "Causal context length (s)"
    )

    ax.set_ylabel(
        "Closest-half matches with >1 deg/min target gap (%)"
    )

    ax.set_title(
        "Does canonical wheel/IMU consistency reduce parking ambiguity?"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "feature_set_ambiguity_comparison.png",
        dpi=170,
    )

    plt.close(
        fig
    )


def main() -> None:

    args = parse_args()

    root = (
        args.root.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    contexts = sorted(
        {
            float(
                x.strip()
            )
            for x
            in args.contexts_s.split(",")
            if x.strip()
        }
    )

    if not contexts:
        raise ValueError(
            "No context lengths supplied."
        )

    if any(
        x <= 0
        for x in contexts
    ):
        raise ValueError(
            "All context lengths must be > 0."
        )

    print(
        "=" * 90
    )

    print(
        "I2NAV ZERO-SHOT PHYSICAL OBSERVABLE AUDIT"
    )

    print(
        "=" * 90
    )

    print(
        "Dataset root :",
        root,
    )

    print(
        "Output       :",
        output_dir,
    )

    print(
        "Wheelbase    :",
        args.wheelbase_m,
        "m",
    )

    print(
        "Track        :",
        args.track_m,
        "m",
    )

    print(
        "Angle sign   :",
        args.angle_sign,
    )

    print(
        "Contexts     :",
        contexts,
    )

    print()

    prepared = load_prepared_sequences(
        root
    )

    ranger_files = find_ranger_files(
        root
    )

    inventory_rows = []

    for name in sorted(
        prepared
    ):

        inventory_rows.append(
            {
                "test_sequence":
                    name,

                "prepared_present":
                    True,

                "ranger_odo_present":
                    name
                    in
                    ranger_files,

                "ranger_odo_path":
                    str(
                        ranger_files[
                            name
                        ]
                    )
                    if
                    name
                    in
                    ranger_files
                    else
                    "",
            }
        )

    inventory_presence = pd.DataFrame(
        inventory_rows
    )

    inventory_presence.to_csv(
        output_dir
        /
        "zero_shot_sensor_inventory.csv",
        index=False,
    )

    missing = [
        name
        for name
        in prepared
        if name
        not in
        ranger_files
    ]

    if missing:

        print(
            "Missing *_RANGER_ODO.txt for:"
        )

        for name in missing:
            print(
                " ",
                name,
            )

        print()

        print(
            "The official i2Nav raw text data contains these files. "
            "Download only the raw Ranger odometer text files; ROS bags, "
            "camera, LiDAR, radar, and GNSS are NOT required for this audit."
        )

        raise SystemExit(
            2
        )

    candidate_frames = {}

    detailed_inventory = []

    corr_rows = []

    for name in sorted(
        prepared
    ):

        print(
            f"Processing {name} ..."
        )

        frame, summary = build_sequence_candidates(
            prepared[
                name
            ],
            ranger_files[
                name
            ],
            args.wheelbase_m,
            args.track_m,
            args.angle_sign,
        )

        candidate_frames[
            name
        ] = frame

        detailed_inventory.append(
            summary
        )

        corr_rows.append(
            candidate_target_correlations(
                prepared[
                    name
                ],
                frame,
                args.target_s,
            )
        )

        frame.to_csv(
            output_dir
            /
            f"{name}_canonical_wheel_imu_features.csv",
            index=False,
        )

    detailed_inventory_df = pd.DataFrame(
        detailed_inventory
    )

    detailed_inventory_df.to_csv(
        output_dir
        /
        "wheel_kinematic_sanity_checks.csv",
        index=False,
    )

    candidate_corr_df = pd.DataFrame(
        corr_rows
    )

    candidate_corr_df.to_csv(
        output_dir
        /
        "candidate_vs_persistent_target.csv",
        index=False,
    )

    # We only compare ambiguity using parking01 and parking02 because
    # these give us the most useful matched success/failure pair from
    # the current V2 pilot.
    if (
        "parking01"
        not in
        prepared
        or
        "parking02"
        not in
        prepared
    ):
        raise RuntimeError(
            "parking01 and parking02 are required for ambiguity analysis."
        )

    feature_names = list(
        feature_sets(
            prepared[
                "parking01"
            ],
            candidate_frames[
                "parking01"
            ],
        ).keys()
    )

    ambiguity_rows = []

    for context_s in contexts:

        print()

        print(
            f"Ambiguity analysis, context={context_s:g}s"
        )

        for feature_name in feature_names:

            rows = ambiguity_for_feature_set(
                prepared,
                candidate_frames,
                feature_name,
                context_s,
                args.target_s,
                args.step_s,
                args.query_chunk,
            )

            ambiguity_rows.extend(
                rows
            )

            mean_rate = np.mean(
                [
                    r[
                        "close50_fraction_gap_gt_1degmin"
                    ]
                    for r
                    in rows
                ]
            )

            mean_gap = np.mean(
                [
                    r[
                        "close50_median_abs_target_gap_deg_per_min"
                    ]
                    for r
                    in rows
                ]
            )

            print(
                f"  {feature_name:38s} "
                f">1deg/min="
                f"{100*mean_rate:5.1f}%  "
                f"median gap="
                f"{mean_gap:.3f} deg/min"
            )

    ambiguity_df = pd.DataFrame(
        ambiguity_rows
    )

    ambiguity_df.to_csv(
        output_dir
        /
        "zero_shot_feature_ambiguity_summary.csv",
        index=False,
    )

    plot_ambiguity(
        ambiguity_df,
        output_dir,
    )

    write_findings(
        output_dir,
        detailed_inventory_df,
        ambiguity_df,
        candidate_corr_df,
        args.angle_sign,
    )

    config = {
        "wheelbase_m":
            args.wheelbase_m,

        "track_m":
            args.track_m,

        "angle_sign":
            args.angle_sign,

        "contexts_s":
            contexts,

        "target_s":
            args.target_s,

        "step_s":
            args.step_s,

        "learned_inputs_policy":
            (
                "canonical encoder+IMU features only; "
                "no Ranger-specific raw wheel identity/"
                "steering-angle channels in learned Twin core"
            ),
    }

    (
        output_dir
        /
        "zero_shot_observable_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "=" * 90
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "Primary outputs:"
    )

    print(
        " ",
        output_dir
        /
        "wheel_kinematic_sanity_checks.csv",
    )

    print(
        " ",
        output_dir
        /
        "candidate_vs_persistent_target.csv",
    )

    print(
        " ",
        output_dir
        /
        "zero_shot_feature_ambiguity_summary.csv",
    )

    print(
        " ",
        output_dir
        /
        "zero_shot_observable_findings.txt",
    )

    print(
        " ",
        output_dir
        /
        "feature_set_ambiguity_comparison.png",
    )


if __name__ == "__main__":
    main()