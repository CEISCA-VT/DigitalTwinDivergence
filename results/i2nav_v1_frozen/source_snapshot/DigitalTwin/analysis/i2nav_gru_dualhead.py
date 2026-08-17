#!/usr/bin/env python3
"""
i2Nav trusted dual-head GRU for digital-twin fidelity.

The model uses ONLY trusted motion history:
    ODO speed, IMU yaw rate, and their time derivatives.

It predicts:
    1) dynamics corrections: delta_v, delta_omega
    2) bounded process-uncertainty multipliers: alpha_xy, alpha_heading

GNSS is never an input to the network. GNSS is used only by the EKF measurement
update/gating path during evaluation. Ground truth is used only for supervised
training targets, initialization, and evaluation.

Recommended location:
    DigitalTwin/analysis/i2nav_gru_dualhead.py

Requires:
    DigitalTwin/analysis/i2nav_adaptive_q_baseline.py

Run:
    python -m DigitalTwin.analysis.i2nav_gru_dualhead --root public_datasets/im2nav

Default whole-sequence split:
    Train:
        building00
        building01
        parking00
        parking01
        playground00
        street00

    Validation:
        building02
        street01

    Test:
        parking02
        street02

No windows from validation/test sequences are used during training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# ===========================================================================
# PyTorch
# ===========================================================================

try:
    import torch
    import torch.nn as nn

    from torch.utils.data import (
        DataLoader,
        Dataset,
    )

except ImportError as exc:

    raise SystemExit(
        "PyTorch is required.\n"
        "Install with:\n"
        "    pip install torch"
    ) from exc


# ===========================================================================
# Project imports
# ===========================================================================

try:

    from DigitalTwin.ekf import RoverEKF

    from DigitalTwin.analysis.i2nav_adaptive_q_baseline import (
        SequenceFiles,
        discover_files,
        load_odo,
        load_imu_yaw,
        load_groundtruth,
        make_grid,
        interpolate_gt,
        sample_yaw_rate,
        stationary_gyro_bias,
        load_gnss,
        summarize_errors,
        read_numeric_table,
        sorted_unique_by_time,
        wrap_array,
    )

except ImportError:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    if str(project_root) not in sys.path:

        sys.path.insert(
            0,
            str(project_root),
        )

    from DigitalTwin.ekf import RoverEKF

    from DigitalTwin.analysis.i2nav_adaptive_q_baseline import (
        SequenceFiles,
        discover_files,
        load_odo,
        load_imu_yaw,
        load_groundtruth,
        make_grid,
        interpolate_gt,
        sample_yaw_rate,
        stationary_gyro_bias,
        load_gnss,
        summarize_errors,
        read_numeric_table,
        sorted_unique_by_time,
        wrap_array,
    )


# ===========================================================================
# Configuration
# ===========================================================================

FEATURE_NAMES = (
    "odo_speed_mps",
    "imu_yaw_rate_radps",
    "odo_accel_mps2",
    "imu_yaw_accel_radps2",
    "abs_yaw_rate_radps",
    "abs_odo_accel_mps2",
)


DEFAULT_TRAIN = (
    "building00",
    "building01",
    "parking00",
    "parking01",
    "playground00",
    "street00",
)


DEFAULT_VAL = (
    "building02",
    "street01",
)


DEFAULT_TEST = (
    "parking02",
    "street02",
)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class PreparedSequence:

    name: str

    files: SequenceFiles

    grid: np.ndarray

    gt_x: np.ndarray
    gt_y: np.ndarray
    gt_heading: np.ndarray

    gt_forward_speed: np.ndarray
    gt_yaw_rate: np.ndarray

    odo_speed: np.ndarray
    imu_yaw_rate: np.ndarray

    features: np.ndarray

    target_corrections: np.ndarray

    gnss: dict[str, np.ndarray] | None

    odo_source: str


@dataclass
class EvalResult:

    sequence: str
    split: str
    status: str

    samples: int
    duration_s: float

    gnss_source: str
    odo_source: str

    # -------------------------------------------------------
    # Fidelity
    # -------------------------------------------------------

    ate_rmse_m: float
    ate_median_m: float
    ate_p95_m: float

    ate_se2_rmse_m: float

    heading_mae_deg: float

    rpe_1s_trans_rmse_m: float
    rpe_5s_trans_rmse_m: float
    rpe_10s_trans_rmse_m: float

    final_error_m: float
    final_error_se2_m: float

    final_drift_per_m: float
    path_length_ratio: float

    # -------------------------------------------------------
    # GNSS
    # -------------------------------------------------------

    gnss_seen: int
    gnss_normal: int
    gnss_reacquired: int
    gnss_rejected: int

    gnss_rejection_rate_pct: float
    gnss_max_coast_s: float

    # -------------------------------------------------------
    # Learned dynamics
    # -------------------------------------------------------

    mean_abs_delta_v_mps: float
    p95_abs_delta_v_mps: float

    mean_abs_delta_omega_radps: float
    p95_abs_delta_omega_radps: float

    # -------------------------------------------------------
    # Learned Q
    # -------------------------------------------------------

    alpha_xy_mean: float
    alpha_xy_p95: float
    alpha_xy_max: float

    alpha_heading_mean: float
    alpha_heading_p95: float
    alpha_heading_max: float

    error: str = ""


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ===========================================================================
# Ground-truth dynamics targets
# ===========================================================================

def load_gt_forward_motion(
    path: Path,
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Compute ground-truth body-forward velocity and ENU yaw rate.

    These values are used ONLY as supervised training targets
    and for evaluation.

    They are NOT model inputs.
    """

    data = sorted_unique_by_time(
        read_numeric_table(
            path,
            min_cols=10,
        )
    )

    timestamp = (
        data[:, 0]
    )

    velocity_north = (
        data[:, 4]
    )

    velocity_east = (
        data[:, 5]
    )

    yaw_ned = np.deg2rad(
        data[:, 9]
    )

    # -------------------------------------------------------
    # NED yaw -> ENU heading
    # -------------------------------------------------------

    heading_enu = wrap_array(
        np.pi / 2.0
        - yaw_ned
    )

    # -------------------------------------------------------
    # Project ENU velocity onto robot forward axis
    # -------------------------------------------------------

    forward_velocity = (
        np.cos(
            heading_enu
        )
        * velocity_east

        +

        np.sin(
            heading_enu
        )
        * velocity_north
    )

    forward_velocity_grid = np.interp(
        grid,
        timestamp,
        forward_velocity,
    )

    # -------------------------------------------------------
    # Ground-truth angular velocity
    # -------------------------------------------------------

    heading_unwrapped = np.unwrap(
        heading_enu
    )

    heading_grid = np.interp(
        grid,
        timestamp,
        heading_unwrapped,
    )

    yaw_rate_grid = np.gradient(
        heading_grid,
        grid,
    )

    return (
        forward_velocity_grid,
        yaw_rate_grid,
    )


# ===========================================================================
# Trusted feature generation
# ===========================================================================

def build_features(
    speed: np.ndarray,
    omega: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Build network features using trusted ODO/IMU data only.

    NO GNSS INFORMATION APPEARS HERE.
    """

    acceleration = np.gradient(
        speed,
        grid,
    )

    yaw_acceleration = np.gradient(
        omega,
        grid,
    )

    features = np.column_stack(
        [
            speed,

            omega,

            acceleration,

            yaw_acceleration,

            np.abs(
                omega
            ),

            np.abs(
                acceleration
            ),
        ]
    )

    return features.astype(
        np.float32
    )


# ===========================================================================
# Sequence preprocessing
# ===========================================================================

def prepare_sequence(
    files: SequenceFiles,
    hz: float,
    imu_yaw_sign: float,
    gnss_sigma_max_m: float,
    gnss_anchor_count: int,
) -> PreparedSequence:

    gt = load_groundtruth(
        files.groundtruth
    )

    (
        odo_time,
        odo_speed_raw,
        odo_source,
    ) = load_odo(
        files
    )

    (
        imu_time,
        imu_cumulative_yaw,
    ) = load_imu_yaw(
        files.imu,
        imu_yaw_sign,
    )

    # -------------------------------------------------------
    # Common timeline
    # -------------------------------------------------------

    grid = make_grid(
        gt["t"],
        odo_time,
        imu_time,
        hz,
    )

    (
        gt_x,
        gt_y,
        gt_heading,
    ) = interpolate_gt(
        gt,
        grid,
    )

    # -------------------------------------------------------
    # Trusted physical inputs
    # -------------------------------------------------------

    speed = np.interp(
        grid,
        odo_time,
        odo_speed_raw,
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

    # -------------------------------------------------------
    # GT targets
    # -------------------------------------------------------

    (
        gt_forward_speed,
        gt_yaw_rate,
    ) = load_gt_forward_motion(
        files.groundtruth,
        grid,
    )

    # -------------------------------------------------------
    # Trusted features
    # -------------------------------------------------------

    features = build_features(
        speed,
        omega,
        grid,
    )

    # -------------------------------------------------------
    # Residual dynamics targets
    #
    # GRU learns:
    #
    #   delta_v =
    #       v_GT - v_ODO
    #
    #   delta_omega =
    #       omega_GT - omega_IMU
    # -------------------------------------------------------

    target_corrections = (
        np.column_stack(
            [
                gt_forward_speed
                - speed,

                gt_yaw_rate
                - omega,
            ]
        )
        .astype(
            np.float32
        )
    )

    # -------------------------------------------------------
    # GNSS used only later by EKF measurement path.
    # -------------------------------------------------------

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

    return PreparedSequence(
        name=(
            files.name
        ),

        files=(
            files
        ),

        grid=(
            grid
        ),

        gt_x=(
            gt_x
        ),

        gt_y=(
            gt_y
        ),

        gt_heading=(
            gt_heading
        ),

        gt_forward_speed=(
            gt_forward_speed
        ),

        gt_yaw_rate=(
            gt_yaw_rate
        ),

        odo_speed=(
            speed
        ),

        imu_yaw_rate=(
            omega
        ),

        features=(
            features
        ),

        target_corrections=(
            target_corrections
        ),

        gnss=(
            gnss
        ),

        odo_source=(
            odo_source
        ),
    )


# ===========================================================================
# Sequence-window dataset
# ===========================================================================

class WindowDataset(
    Dataset
):

    def __init__(
        self,
        sequences: list[
            PreparedSequence
        ],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        window: int,
        stride: int,
    ) -> None:

        self.sequences = (
            sequences
        )

        self.feature_mean = (
            feature_mean.astype(
                np.float32
            )
        )

        self.feature_std = (
            feature_std.astype(
                np.float32
            )
        )

        self.window = (
            window
        )

        self.index: list[
            tuple[
                int,
                int,
            ]
        ] = []

        # ---------------------------------------------------
        # Windows are created only inside each sequence.
        #
        # No window crosses sequence boundaries.
        # ---------------------------------------------------

        for sequence_index, sequence in enumerate(
            sequences
        ):

            for end_index in range(
                window - 1,
                len(
                    sequence.grid
                ),
                stride,
            ):

                self.index.append(
                    (
                        sequence_index,
                        end_index,
                    )
                )

    def __len__(
        self,
    ) -> int:

        return len(
            self.index
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        (
            sequence_index,
            end_index,
        ) = self.index[
            index
        ]

        sequence = (
            self.sequences[
                sequence_index
            ]
        )

        start_index = (
            end_index
            - self.window
            + 1
        )

        window = (
            sequence.features[
                start_index
                :
                end_index + 1
            ]
        )

        # ---------------------------------------------------
        # Normalization parameters come from TRAIN only.
        # ---------------------------------------------------

        window = (
            window
            - self.feature_mean
        ) / self.feature_std

        target = (
            sequence
            .target_corrections[
                end_index
            ]
        )

        return (
            torch.from_numpy(
                window
            ),

            torch.from_numpy(
                target
            ),
        )


# ===========================================================================
# Dual-head GRU
# ===========================================================================

class DualHeadGRU(
    nn.Module
):
    """
    Physics-guided residual GRU.

    Inputs:
        trusted motion history only

    Outputs:
        delta_v
        delta_omega

        alpha_xy
        alpha_heading
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,

        dv_limit: float,
        domega_limit: float,

        alpha_min: float,
        alpha_max: float,
    ) -> None:

        super().__init__()

        self.dv_limit = float(
            dv_limit
        )

        self.domega_limit = float(
            domega_limit
        )

        self.alpha_min = float(
            alpha_min
        )

        self.alpha_max = float(
            alpha_max
        )

        # ---------------------------------------------------
        # Temporal encoder
        # ---------------------------------------------------

        self.gru = nn.GRU(
            input_size=(
                input_dim
            ),

            hidden_size=(
                hidden_size
            ),

            num_layers=(
                num_layers
            ),

            batch_first=True,

            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # ---------------------------------------------------
        # Shared representation
        # ---------------------------------------------------

        self.trunk = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),
        )

        # ---------------------------------------------------
        # Dynamics correction head
        # ---------------------------------------------------

        self.dynamics_head = nn.Linear(
            hidden_size,
            2,
        )

        # ---------------------------------------------------
        # Q multiplier head
        # ---------------------------------------------------

        self.q_head = nn.Linear(
            hidden_size,
            2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        (
            _,
            hidden,
        ) = self.gru(
            x
        )

        representation = (
            self.trunk(
                hidden[-1]
            )
        )

        # ---------------------------------------------------
        # Dynamics
        #
        # Bounded corrections prevent pathological outputs.
        # ---------------------------------------------------

        raw_dynamics = (
            self.dynamics_head(
                representation
            )
        )

        delta_v = (
            self.dv_limit
            * torch.tanh(
                raw_dynamics[
                    :,
                    0,
                ]
            )
        )

        delta_omega = (
            self.domega_limit
            * torch.tanh(
                raw_dynamics[
                    :,
                    1,
                ]
            )
        )

        corrections = torch.stack(
            (
                delta_v,
                delta_omega,
            ),
            dim=1,
        )

        # ---------------------------------------------------
        # Uncertainty
        #
        # Bounded:
        #
        # alpha_min <= alpha <= alpha_max
        # ---------------------------------------------------

        q_normalized = torch.sigmoid(
            self.q_head(
                representation
            )
        )

        alphas = (
            self.alpha_min

            +

            (
                self.alpha_max
                - self.alpha_min
            )
            * q_normalized
        )

        return (
            corrections,
            alphas,
        )


# ===========================================================================
# Training statistics
# ===========================================================================

def compute_feature_stats(
    train_sequences: list[
        PreparedSequence
    ],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    # -------------------------------------------------------
    # TRAIN ONLY.
    #
    # Validation/test data never contribute to normalization.
    # -------------------------------------------------------

    all_features = np.concatenate(
        [
            sequence.features
            for sequence
            in train_sequences
        ],
        axis=0,
    )

    mean = np.mean(
        all_features,
        axis=0,
    ).astype(
        np.float32
    )

    std = np.std(
        all_features,
        axis=0,
    ).astype(
        np.float32
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    ).astype(
        np.float32
    )

    return (
        mean,
        std,
    )


def derive_correction_limits(
    train_sequences: list[
        PreparedSequence
    ],
) -> tuple[
    float,
    float,
]:

    targets = np.concatenate(
        [
            sequence
            .target_corrections

            for sequence
            in train_sequences
        ],
        axis=0,
    )

    abs_delta_v = np.abs(
        targets[
            :,
            0,
        ]
    )

    abs_delta_omega = np.abs(
        targets[
            :,
            1,
        ]
    )

    # -------------------------------------------------------
    # Limits derived only from TRAIN.
    # -------------------------------------------------------

    dv_limit = max(
        0.25,

        float(
            np.percentile(
                abs_delta_v,
                99.5,
            )
        )
        * 1.20,
    )

    domega_limit = max(
        0.20,

        float(
            np.percentile(
                abs_delta_omega,
                99.5,
            )
        )
        * 1.20,
    )

    return (
        dv_limit,
        domega_limit,
    )


def derive_target_scales(
    train_sequences: list[
        PreparedSequence
    ],
) -> np.ndarray:

    targets = np.concatenate(
        [
            sequence
            .target_corrections

            for sequence
            in train_sequences
        ],
        axis=0,
    )

    scale = np.std(
        targets,
        axis=0,
    ).astype(
        np.float32
    )

    scale = np.where(
        scale < 1e-3,
        1.0,
        scale,
    ).astype(
        np.float32
    )

    return (
        scale
    )


# ===========================================================================
# Loss / epoch
# ===========================================================================

def run_epoch(
    model: DualHeadGRU,

    loader: DataLoader,

    optimizer:
        torch.optim.Optimizer
        | None,

    device: torch.device,

    target_scale: torch.Tensor,

    base_q_xy_sigma: float,
    base_q_heading_sigma: float,

    nll_weight: float,
    alpha_reg_weight: float,

    mean_only: bool,
) -> dict[
    str,
    float,
]:

    training = (
        optimizer is not None
    )

    model.train(
        training
    )

    total_loss = (
        0.0
    )

    total_mean_loss = (
        0.0
    )

    total_nll = (
        0.0
    )

    total_alpha_reg = (
        0.0
    )

    total_count = (
        0
    )

    for (
        features,
        target,
    ) in loader:

        features = features.to(
            device=device,
            dtype=torch.float32,
        )

        target = target.to(
            device=device,
            dtype=torch.float32,
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        (
            corrections,
            alphas,
        ) = model(
            features
        )

        error = (
            target
            - corrections
        )

        # ---------------------------------------------------
        # Dynamics correction loss
        #
        # Scale dv/domega so one output does not dominate.
        # ---------------------------------------------------

        normalized_error = (
            error
            / target_scale
        )

        mean_loss = (
            torch.nn.functional
            .smooth_l1_loss(
                normalized_error,

                torch.zeros_like(
                    normalized_error
                ),

                beta=0.5,
            )
        )

        # ---------------------------------------------------
        # Learned Q supervision via heteroscedastic likelihood.
        #
        # alpha determines uncertainty of the remaining
        # dynamics residual.
        # ---------------------------------------------------

        sigma_v = torch.clamp(
            base_q_xy_sigma
            * alphas[
                :,
                0,
            ],

            min=1e-4,
        )

        sigma_omega = torch.clamp(
            base_q_heading_sigma
            * alphas[
                :,
                1,
            ],

            min=1e-5,
        )

        nll_v = (
            0.5

            * (
                error[
                    :,
                    0,
                ]
                / sigma_v
            ) ** 2

            +

            torch.log(
                sigma_v
            )
        )

        nll_omega = (
            0.5

            * (
                error[
                    :,
                    1,
                ]
                / sigma_omega
            ) ** 2

            +

            torch.log(
                sigma_omega
            )
        )

        nll = torch.mean(
            nll_v
            + nll_omega
        )

        # ---------------------------------------------------
        # Discourage gratuitous uncertainty inflation.
        # ---------------------------------------------------

        alpha_span = max(
            model.alpha_max
            - model.alpha_min,

            1e-6,
        )

        alpha_reg = torch.mean(
            (
                (
                    alphas
                    - model.alpha_min
                )
                / alpha_span
            ) ** 2
        )

        # ---------------------------------------------------
        # Warmup:
        #
        # First learn dynamics mean.
        # Then jointly learn calibrated uncertainty.
        # ---------------------------------------------------

        if mean_only:

            loss = (
                mean_loss

                +

                alpha_reg_weight
                * alpha_reg
            )

        else:

            loss = (
                mean_loss

                +

                nll_weight
                * nll

                +

                alpha_reg_weight
                * alpha_reg
            )

        if training:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        batch_size = (
            features.shape[0]
        )

        total_count += (
            batch_size
        )

        total_loss += (
            float(
                loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_mean_loss += (
            float(
                mean_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_nll += (
            float(
                nll
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_alpha_reg += (
            float(
                alpha_reg
                .detach()
                .cpu()
            )
            * batch_size
        )

    denominator = max(
        total_count,
        1,
    )

    return {
        "loss":
            total_loss
            / denominator,

        "mean_loss":
            total_mean_loss
            / denominator,

        "nll":
            total_nll
            / denominator,

        "alpha_reg":
            total_alpha_reg
            / denominator,
    }


# ===========================================================================
# Model training
# ===========================================================================

def train_model(
    model: DualHeadGRU,

    train_loader: DataLoader,

    validation_loader: DataLoader,

    args: argparse.Namespace,

    device: torch.device,

    target_scale_numpy: np.ndarray,

    output_dir: Path,
) -> tuple[
    DualHeadGRU,
    list[
        dict[
            str,
            float,
        ]
    ],
]:

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=(
            args.lr
        ),

        weight_decay=(
            args.weight_decay
        ),
    )

    target_scale = torch.tensor(
        target_scale_numpy,

        dtype=torch.float32,

        device=device,
    )

    best_validation_loss = float(
        "inf"
    )

    best_state = (
        None
    )

    patience_remaining = (
        args.patience
    )

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        mean_only = (
            epoch
            <= args.warmup_epochs
        )

        train_statistics = run_epoch(
            model=model,

            loader=(
                train_loader
            ),

            optimizer=(
                optimizer
            ),

            device=(
                device
            ),

            target_scale=(
                target_scale
            ),

            base_q_xy_sigma=(
                args.q_xy_sigma_mps
            ),

            base_q_heading_sigma=(
                args.q_heading_sigma_radps
            ),

            nll_weight=(
                args.nll_weight
            ),

            alpha_reg_weight=(
                args.alpha_reg_weight
            ),

            mean_only=(
                mean_only
            ),
        )

        with torch.no_grad():

            validation_statistics = run_epoch(
                model=model,

                loader=(
                    validation_loader
                ),

                optimizer=None,

                device=(
                    device
                ),

                target_scale=(
                    target_scale
                ),

                base_q_xy_sigma=(
                    args.q_xy_sigma_mps
                ),

                base_q_heading_sigma=(
                    args.q_heading_sigma_radps
                ),

                nll_weight=(
                    args.nll_weight
                ),

                alpha_reg_weight=(
                    args.alpha_reg_weight
                ),

                mean_only=False,
            )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_statistics[
                    "loss"
                ],

            "train_mean_loss":
                train_statistics[
                    "mean_loss"
                ],

            "train_nll":
                train_statistics[
                    "nll"
                ],

            "validation_loss":
                validation_statistics[
                    "loss"
                ],

            "validation_mean_loss":
                validation_statistics[
                    "mean_loss"
                ],

            "validation_nll":
                validation_statistics[
                    "nll"
                ],
        }

        history.append(
            row
        )

        print(
            f"epoch={epoch:03d} "
            f"train={row['train_loss']:.5f} "
            f"val={row['validation_loss']:.5f} "
            f"mean={row['validation_mean_loss']:.5f}"
        )

        # ---------------------------------------------------
        # Early stopping uses validation sequence windows only.
        # ---------------------------------------------------

        if (
            row[
                "validation_loss"
            ]
            <
            best_validation_loss
            - args.min_delta
        ):

            best_validation_loss = (
                row[
                    "validation_loss"
                ]
            )

            best_state = {
                name:
                    parameter
                    .detach()
                    .cpu()
                    .clone()

                for (
                    name,
                    parameter,
                )

                in model
                .state_dict()
                .items()
            }

            patience_remaining = (
                args.patience
            )

        else:

            patience_remaining -= 1

            if (
                patience_remaining
                <= 0
            ):

                print(
                    "early stopping "
                    f"at epoch {epoch}; "
                    "best validation loss="
                    f"{best_validation_loss:.6f}"
                )

                break

    if best_state is None:

        raise RuntimeError(
            "Training failed to produce a checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    # -------------------------------------------------------
    # Training history
    # -------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        output_dir
        / "training_history.csv"
    )

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,

            fieldnames=list(
                history[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    return (
        model,
        history,
    )


# ===========================================================================
# Batched GRU inference
# ===========================================================================

@torch.no_grad()
def predict_sequence(
    model: DualHeadGRU,

    sequence: PreparedSequence,

    feature_mean: np.ndarray,
    feature_std: np.ndarray,

    window: int,

    device: torch.device,

    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    count = len(
        sequence.grid
    )

    corrections = np.zeros(
        (
            count,
            2,
        ),
        dtype=np.float32,
    )

    # -------------------------------------------------------
    # Before enough history exists:
    #
    # no dynamics correction
    # alpha = 1
    # -------------------------------------------------------

    alphas = np.ones(
        (
            count,
            2,
        ),
        dtype=np.float32,
    )

    model.eval()

    indices = list(
        range(
            window - 1,
            count,
        )
    )

    for batch_start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            batch_start
            :
            batch_start
            + batch_size
        ]

        windows = []

        for end_index in batch_indices:

            start_index = (
                end_index
                - window
                + 1
            )

            values = (
                sequence.features[
                    start_index
                    :
                    end_index + 1
                ]
            )

            values = (
                values
                - feature_mean
            ) / feature_std

            windows.append(
                values
            )

        input_tensor = torch.from_numpy(
            np.stack(
                windows
            )
            .astype(
                np.float32
            )
        ).to(
            device
        )

        (
            prediction_corrections,
            prediction_alphas,
        ) = model(
            input_tensor
        )

        prediction_corrections = (
            prediction_corrections
            .cpu()
            .numpy()
        )

        prediction_alphas = (
            prediction_alphas
            .cpu()
            .numpy()
        )

        for local_index, end_index in enumerate(
            batch_indices
        ):

            corrections[
                end_index
            ] = (
                prediction_corrections[
                    local_index
                ]
            )

            alphas[
                end_index
            ] = (
                prediction_alphas[
                    local_index
                ]
            )

    return (
        corrections,
        alphas,
    )


# ===========================================================================
# EKF evaluation with V5 GNSS policy
# ===========================================================================

def evaluate_sequence(
    sequence: PreparedSequence,

    split: str,

    model: DualHeadGRU,

    feature_mean: np.ndarray,
    feature_std: np.ndarray,

    args: argparse.Namespace,

    device: torch.device,

    output_dir: Path,
) -> EvalResult:

    (
        corrections,
        alphas,
    ) = predict_sequence(
        model=model,

        sequence=(
            sequence
        ),

        feature_mean=(
            feature_mean
        ),

        feature_std=(
            feature_std
        ),

        window=(
            args.window
        ),

        device=(
            device
        ),

        batch_size=(
            args.eval_batch_size
        ),
    )

    grid = (
        sequence.grid
    )

    truth_xy = np.column_stack(
        [
            sequence.gt_x,
            sequence.gt_y,
        ]
    )

    # -------------------------------------------------------
    # Same initialization as V5/V6
    # -------------------------------------------------------

    initial_state = np.array(
        [
            sequence.gt_x[0],
            sequence.gt_y[0],
            sequence.gt_heading[0],
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

    # -------------------------------------------------------
    # GNSS model
    # -------------------------------------------------------

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

    gnss = (
        sequence.gnss
    )

    gnss_index = (
        0
    )

    if gnss is not None:

        while (
            gnss_index
            < len(
                gnss["t"]
            )

            and

            gnss["t"][
                gnss_index
            ]
            < grid[0]
        ):

            gnss_index += 1

    # -------------------------------------------------------
    # GNSS diagnostics
    # -------------------------------------------------------

    gnss_seen = (
        0
    )

    gnss_normal = (
        0
    )

    gnss_reacquired = (
        0
    )

    gnss_rejected = (
        0
    )

    reacquisition_streak = (
        0
    )

    last_accepted_gnss_time = float(
        grid[0]
    )

    max_coast_s = (
        0.0
    )

    # =========================================================================
    # Replay
    # =========================================================================

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
                1.0
                / args.rate_hz
            )

        # ---------------------------------------------------
        # GRU dynamics correction
        # ---------------------------------------------------

        corrected_speed = float(
            sequence.odo_speed[k]

            +

            corrections[
                k,
                0,
            ]
        )

        corrected_omega = float(
            sequence.imu_yaw_rate[k]

            +

            corrections[
                k,
                1,
            ]
        )

        # ---------------------------------------------------
        # GRU learned trusted Q
        # ---------------------------------------------------

        alpha_xy = float(
            alphas[
                k,
                0,
            ]
        )

        alpha_heading = float(
            alphas[
                k,
                1,
            ]
        )

        Q = np.diag(
            [
                (
                    args.q_xy_sigma_mps
                    * alpha_xy
                    * dt
                ) ** 2,

                (
                    args.q_xy_sigma_mps
                    * alpha_xy
                    * dt
                ) ** 2,

                (
                    args.q_heading_sigma_radps
                    * alpha_heading
                    * dt
                ) ** 2,
            ]
        )

        # ---------------------------------------------------
        # Prediction
        # ---------------------------------------------------

        ekf.predict(
            corrected_speed,
            corrected_omega,
            dt,
            Q,
        )

        # ---------------------------------------------------
        # GNSS:
        #
        # Same hard NIS + safe reacquisition structure as V5.
        # ---------------------------------------------------

        if gnss is not None:

            latest = (
                None
            )

            while (
                gnss_index
                < len(
                    gnss["t"]
                )

                and

                gnss["t"][
                    gnss_index
                ]
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

                    and

                    sigma_h
                    <= args.gnss_sigma_max_m
                ):

                    gnss_seen += 1

                    sigma_e = max(
                        float(
                            gnss[
                                "sigma_e"
                            ][latest]
                        ),

                        args.gnss_sigma_floor_m,
                    )

                    sigma_n = max(
                        float(
                            gnss[
                                "sigma_n"
                            ][latest]
                        ),

                        args.gnss_sigma_floor_m,
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

                    # ---------------------------------------
                    # NIS
                    # ---------------------------------------

                    innovation = (
                        z

                        -

                        H
                        @ ekf.state.x
                    )

                    S = (
                        H
                        @ ekf.state.P
                        @ H.T

                        +

                        R
                    )

                    try:

                        nis = float(
                            innovation.T

                            @

                            np.linalg.solve(
                                S,
                                innovation,
                            )
                        )

                    except np.linalg.LinAlgError:

                        nis = float(
                            "inf"
                        )

                    accepted = (
                        False
                    )

                    # ---------------------------------------
                    # Normal hard gate
                    # ---------------------------------------

                    if (
                        np.isfinite(
                            nis
                        )

                        and

                        nis
                        <= args.gnss_nis_gate
                    ):

                        ekf.update_gps(
                            z,
                            R,
                        )

                        gnss_normal += 1

                        reacquisition_streak = (
                            0
                        )

                        accepted = (
                            True
                        )

                    # ---------------------------------------
                    # Safe reacquisition
                    # ---------------------------------------

                    else:

                        coast_s = max(
                            0.0,

                            float(
                                grid[k]
                                - last_accepted_gnss_time
                            ),
                        )

                        candidate = (
                            False
                        )

                        extra_sigma = (
                            0.0
                        )

                        if (
                            coast_s
                            >= args.reacq_start_s
                        ):

                            extra_sigma = min(
                                args.reacq_sigma_max_m,

                                args.reacq_sigma_growth_mps
                                * coast_s,
                            )

                            S_reacquisition = (
                                S

                                +

                                identity_2
                                * extra_sigma**2
                            )

                            try:

                                reacquisition_nis = float(
                                    innovation.T

                                    @

                                    np.linalg.solve(
                                        S_reacquisition,
                                        innovation,
                                    )
                                )

                            except np.linalg.LinAlgError:

                                reacquisition_nis = float(
                                    "inf"
                                )

                            candidate = (
                                np.isfinite(
                                    reacquisition_nis
                                )

                                and

                                reacquisition_nis
                                <= args.gnss_nis_gate
                            )

                        if candidate:

                            reacquisition_streak += 1

                        else:

                            reacquisition_streak = (
                                0
                            )

                        if (
                            candidate

                            and

                            reacquisition_streak
                            >= args.reacq_consecutive
                        ):

                            R_reacquisition = (
                                R

                                +

                                identity_2
                                * extra_sigma**2
                            )

                            ekf.update_gps(
                                z,
                                R_reacquisition,
                            )

                            gnss_reacquired += 1

                            reacquisition_streak = (
                                0
                            )

                            accepted = (
                                True
                            )

                        else:

                            gnss_rejected += 1

                    if accepted:

                        last_accepted_gnss_time = float(
                            gnss[
                                "t"
                            ][latest]
                        )

        # ---------------------------------------------------
        # Coast duration
        # ---------------------------------------------------

        if (
            sequence.files.gnss
            is not None
        ):

            max_coast_s = max(
                max_coast_s,

                max(
                    0.0,

                    float(
                        grid[k]
                        - last_accepted_gnss_time
                    ),
                ),
            )

        estimates[k] = (
            ekf.state.x
        )

    # =========================================================================
    # Metrics
    # =========================================================================

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

        sequence.gt_heading,

        args.rate_hz,
    )

    duration_s = (
        float(
            grid[-1]
            - grid[0]
        )

        if len(grid) > 1

        else 0.0
    )

    rejection_rate = (
        100.0
        * gnss_rejected
        / gnss_seen

        if gnss_seen

        else 0.0
    )

    # =========================================================================
    # Save trajectory
    # =========================================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_path = (
        output_dir
        / (
            f"{sequence.name}"
            "_gru_trajectory.csv"
        )
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

                "delta_v_mps",
                "delta_omega_radps",

                "corrected_v_mps",
                "corrected_omega_radps",

                "alpha_xy",
                "alpha_heading",

                "position_error_m",
            ]
        )

        for i in range(
            len(grid)
        ):

            writer.writerow(
                [
                    grid[i],

                    sequence.gt_x[i],
                    sequence.gt_y[i],
                    sequence.gt_heading[i],

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

                    sequence.odo_speed[i],

                    sequence.imu_yaw_rate[i],

                    corrections[
                        i,
                        0,
                    ],

                    corrections[
                        i,
                        1,
                    ],

                    sequence.odo_speed[i]
                    + corrections[
                        i,
                        0,
                    ],

                    sequence.imu_yaw_rate[i]
                    + corrections[
                        i,
                        1,
                    ],

                    alphas[
                        i,
                        0,
                    ],

                    alphas[
                        i,
                        1,
                    ],

                    position_error[i],
                ]
            )

    return EvalResult(
        sequence=(
            sequence.name
        ),

        split=(
            split
        ),

        status="ok",

        samples=(
            len(grid)
        ),

        duration_s=(
            duration_s
        ),

        gnss_source=(
            sequence
            .files
            .gnss_source
        ),

        odo_source=(
            sequence.odo_source
        ),

        ate_rmse_m=(
            float(
                metrics[
                    "ate_rmse_m"
                ]
            )
        ),

        ate_median_m=(
            float(
                metrics[
                    "ate_median_m"
                ]
            )
        ),

        ate_p95_m=(
            float(
                metrics[
                    "ate_p95_m"
                ]
            )
        ),

        ate_se2_rmse_m=(
            float(
                metrics[
                    "ate_se2_rmse_m"
                ]
            )
        ),

        heading_mae_deg=(
            float(
                metrics[
                    "heading_mae_deg"
                ]
            )
        ),

        rpe_1s_trans_rmse_m=(
            float(
                metrics[
                    "rpe_1s_trans_rmse_m"
                ]
            )
        ),

        rpe_5s_trans_rmse_m=(
            float(
                metrics[
                    "rpe_5s_trans_rmse_m"
                ]
            )
        ),

        rpe_10s_trans_rmse_m=(
            float(
                metrics[
                    "rpe_10s_trans_rmse_m"
                ]
            )
        ),

        final_error_m=(
            float(
                metrics[
                    "final_error_m"
                ]
            )
        ),

        final_error_se2_m=(
            float(
                metrics[
                    "final_error_se2_m"
                ]
            )
        ),

        final_drift_per_m=(
            float(
                metrics[
                    "final_drift_per_m"
                ]
            )
        ),

        path_length_ratio=(
            float(
                metrics[
                    "path_length_ratio"
                ]
            )
        ),

        gnss_seen=(
            gnss_seen
        ),

        gnss_normal=(
            gnss_normal
        ),

        gnss_reacquired=(
            gnss_reacquired
        ),

        gnss_rejected=(
            gnss_rejected
        ),

        gnss_rejection_rate_pct=(
            rejection_rate
        ),

        gnss_max_coast_s=(
            max_coast_s
        ),

        mean_abs_delta_v_mps=(
            float(
                np.mean(
                    np.abs(
                        corrections[
                            :,
                            0,
                        ]
                    )
                )
            )
        ),

        p95_abs_delta_v_mps=(
            float(
                np.percentile(
                    np.abs(
                        corrections[
                            :,
                            0,
                        ]
                    ),
                    95,
                )
            )
        ),

        mean_abs_delta_omega_radps=(
            float(
                np.mean(
                    np.abs(
                        corrections[
                            :,
                            1,
                        ]
                    )
                )
            )
        ),

        p95_abs_delta_omega_radps=(
            float(
                np.percentile(
                    np.abs(
                        corrections[
                            :,
                            1,
                        ]
                    ),
                    95,
                )
            )
        ),

        alpha_xy_mean=(
            float(
                np.mean(
                    alphas[
                        :,
                        0,
                    ]
                )
            )
        ),

        alpha_xy_p95=(
            float(
                np.percentile(
                    alphas[
                        :,
                        0,
                    ],
                    95,
                )
            )
        ),

        alpha_xy_max=(
            float(
                np.max(
                    alphas[
                        :,
                        0,
                    ]
                )
            )
        ),

        alpha_heading_mean=(
            float(
                np.mean(
                    alphas[
                        :,
                        1,
                    ]
                )
            )
        ),

        alpha_heading_p95=(
            float(
                np.percentile(
                    alphas[
                        :,
                        1,
                    ],
                    95,
                )
            )
        ),

        alpha_heading_max=(
            float(
                np.max(
                    alphas[
                        :,
                        1,
                    ]
                )
            )
        ),
    )


# ===========================================================================
# Split helper
# ===========================================================================

def parse_sequence_argument(
    values: list[str] | None,

    default: tuple[
        str,
        ...,
    ],
) -> tuple[
    str,
    ...,
]:

    if values:

        return tuple(
            values
        )

    return (
        default
    )


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the trusted "
            "dual-head GRU on i2Nav."
        )
    )

    # -------------------------------------------------------
    # Paths
    # -------------------------------------------------------

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
            "results/i2nav_gru_dualhead"
        ),
    )

    # -------------------------------------------------------
    # Whole-sequence splits
    # -------------------------------------------------------

    parser.add_argument(
        "--train-sequences",

        nargs="*",

        default=None,
    )

    parser.add_argument(
        "--val-sequences",

        nargs="*",

        default=None,
    )

    parser.add_argument(
        "--test-sequences",

        nargs="*",

        default=None,
    )

    # -------------------------------------------------------
    # Temporal model
    # -------------------------------------------------------

    parser.add_argument(
        "--rate-hz",

        type=float,

        default=10.0,
    )

    parser.add_argument(
        "--window",

        type=int,

        default=20,

        help=(
            "History length. "
            "20 samples at 10 Hz = 2 seconds."
        ),
    )

    parser.add_argument(
        "--stride",

        type=int,

        default=1,
    )

    parser.add_argument(
        "--hidden-size",

        type=int,

        default=64,
    )

    parser.add_argument(
        "--num-layers",

        type=int,

        default=2,
    )

    parser.add_argument(
        "--dropout",

        type=float,

        default=0.10,
    )

    # -------------------------------------------------------
    # Q bounds
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------

    parser.add_argument(
        "--epochs",

        type=int,

        default=60,
    )

    parser.add_argument(
        "--warmup-epochs",

        type=int,

        default=5,
    )

    parser.add_argument(
        "--batch-size",

        type=int,

        default=512,
    )

    parser.add_argument(
        "--eval-batch-size",

        type=int,

        default=1024,
    )

    parser.add_argument(
        "--lr",

        type=float,

        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",

        type=float,

        default=1e-4,
    )

    parser.add_argument(
        "--nll-weight",

        type=float,

        default=0.02,
    )

    parser.add_argument(
        "--alpha-reg-weight",

        type=float,

        default=0.01,
    )

    parser.add_argument(
        "--patience",

        type=int,

        default=10,
    )

    parser.add_argument(
        "--min-delta",

        type=float,

        default=1e-4,
    )

    parser.add_argument(
        "--seed",

        type=int,

        default=42,
    )

    parser.add_argument(
        "--num-workers",

        type=int,

        default=0,

        help=(
            "Keep 0 on Windows unless you specifically "
            "want multiprocessing DataLoader workers."
        ),
    )

    # -------------------------------------------------------
    # Base process uncertainty
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # GNSS
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # Safe reacquisition
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # IMU
    # -------------------------------------------------------

    parser.add_argument(
        "--imu-yaw-sign",

        type=float,

        choices=(
            -1.0,
            1.0,
        ),

        default=-1.0,
    )

    return (
        parser.parse_args()
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:

    args = parse_args()

    set_seed(
        args.seed
    )

    root = (
        args.root
        .resolve()
    )

    output_dir = (
        args.output_dir
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------
    # Whole-sequence split
    # -------------------------------------------------------

    train_names = parse_sequence_argument(
        args.train_sequences,
        DEFAULT_TRAIN,
    )

    validation_names = parse_sequence_argument(
        args.val_sequences,
        DEFAULT_VAL,
    )

    test_names = parse_sequence_argument(
        args.test_sequences,
        DEFAULT_TEST,
    )

    train_set = set(
        train_names
    )

    validation_set = set(
        validation_names
    )

    test_set = set(
        test_names
    )

    # -------------------------------------------------------
    # Prevent sequence leakage
    # -------------------------------------------------------

    if (
        train_set
        & validation_set

        or

        train_set
        & test_set

        or

        validation_set
        & test_set
    ):

        raise SystemExit(
            "ERROR: train/validation/test sequence "
            "lists must be completely disjoint."
        )

    discovered_files = {
        files.name:
            files

        for files
        in discover_files(
            root
        )
    }

    required_names = (
        train_set
        | validation_set
        | test_set
    )

    missing = sorted(
        required_names
        - set(
            discovered_files
        )
    )

    if missing:

        raise SystemExit(
            "Missing sequences: "
            f"{missing}"
        )

    print(
        "Whole-sequence split:"
    )

    print(
        "  train: "
        + ", ".join(
            train_names
        )
    )

    print(
        "  val:   "
        + ", ".join(
            validation_names
        )
    )

    print(
        "  test:  "
        + ", ".join(
            test_names
        )
    )

    print()

    # -------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------

    prepared: dict[
        str,
        PreparedSequence,
    ] = {}

    print(
        "Preparing sequences..."
    )

    for name in sorted(
        required_names
    ):

        print(
            f"  {name}"
        )

        prepared[
            name
        ] = prepare_sequence(
            discovered_files[
                name
            ],

            hz=(
                args.rate_hz
            ),

            imu_yaw_sign=(
                args.imu_yaw_sign
            ),

            gnss_sigma_max_m=(
                args.gnss_sigma_max_m
            ),

            gnss_anchor_count=(
                args.gnss_anchor_count
            ),
        )

    train_sequences = [
        prepared[
            name
        ]

        for name
        in train_names
    ]

    validation_sequences = [
        prepared[
            name
        ]

        for name
        in validation_names
    ]

    test_sequences = [
        prepared[
            name
        ]

        for name
        in test_names
    ]

    # -------------------------------------------------------
    # Statistics from TRAIN ONLY
    # -------------------------------------------------------

    (
        feature_mean,
        feature_std,
    ) = compute_feature_stats(
        train_sequences
    )

    target_scale = derive_target_scales(
        train_sequences
    )

    (
        delta_v_limit,
        delta_omega_limit,
    ) = derive_correction_limits(
        train_sequences
    )

    # -------------------------------------------------------
    # Save split / preprocessing description
    # -------------------------------------------------------

    split_payload = {
        "train":
            list(
                train_names
            ),

        "validation":
            list(
                validation_names
            ),

        "test":
            list(
                test_names
            ),

        "feature_names":
            list(
                FEATURE_NAMES
            ),

        "window":
            args.window,

        "rate_hz":
            args.rate_hz,

        "security_constraint": (
            "GRU inputs contain only ODO/IMU-derived "
            "motion quantities. GNSS position, innovation, "
            "NIS, covariance, residuals and reported GNSS "
            "uncertainty are excluded."
        ),
    }

    (
        output_dir
        / "split.json"
    ).write_text(
        json.dumps(
            split_payload,
            indent=2,
        ),

        encoding="utf-8",
    )

    # -------------------------------------------------------
    # Window datasets
    # -------------------------------------------------------

    train_dataset = WindowDataset(
        sequences=(
            train_sequences
        ),

        feature_mean=(
            feature_mean
        ),

        feature_std=(
            feature_std
        ),

        window=(
            args.window
        ),

        stride=(
            args.stride
        ),
    )

    validation_dataset = WindowDataset(
        sequences=(
            validation_sequences
        ),

        feature_mean=(
            feature_mean
        ),

        feature_std=(
            feature_std
        ),

        window=(
            args.window
        ),

        stride=(
            args.stride
        ),
    )

    if (
        len(
            train_dataset
        )
        == 0

        or

        len(
            validation_dataset
        )
        == 0
    ):

        raise SystemExit(
            "Train or validation dataset has zero windows."
        )

    train_loader = DataLoader(
        train_dataset,

        batch_size=(
            args.batch_size
        ),

        shuffle=True,

        num_workers=(
            args.num_workers
        ),

        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,

        batch_size=(
            args.batch_size
        ),

        shuffle=False,

        num_workers=(
            args.num_workers
        ),

        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    # -------------------------------------------------------
    # Device
    # -------------------------------------------------------

    device = torch.device(
        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    print(
        f"device = {device}"
    )

    print(
        "train windows = "
        f"{len(train_dataset):,}"
    )

    print(
        "validation windows = "
        f"{len(validation_dataset):,}"
    )

    print(
        "delta_v limit = "
        f"{delta_v_limit:.4f} m/s"
    )

    print(
        "delta_omega limit = "
        f"{delta_omega_limit:.4f} rad/s"
    )

    print()

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    model = DualHeadGRU(
        input_dim=(
            len(
                FEATURE_NAMES
            )
        ),

        hidden_size=(
            args.hidden_size
        ),

        num_layers=(
            args.num_layers
        ),

        dropout=(
            args.dropout
        ),

        dv_limit=(
            delta_v_limit
        ),

        domega_limit=(
            delta_omega_limit
        ),

        alpha_min=(
            args.alpha_min
        ),

        alpha_max=(
            args.alpha_max
        ),
    ).to(
        device
    )

    # -------------------------------------------------------
    # Train
    # -------------------------------------------------------

    (
        model,
        history,
    ) = train_model(
        model=model,

        train_loader=(
            train_loader
        ),

        validation_loader=(
            validation_loader
        ),

        args=(
            args
        ),

        device=(
            device
        ),

        target_scale_numpy=(
            target_scale
        ),

        output_dir=(
            output_dir
        ),
    )

    # -------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------

    checkpoint = {
        "state_dict":
            model.state_dict(),

        "feature_mean":
            feature_mean,

        "feature_std":
            feature_std,

        "target_scale":
            target_scale,

        "feature_names":
            FEATURE_NAMES,

        "window":
            args.window,

        "delta_v_limit":
            delta_v_limit,

        "delta_omega_limit":
            delta_omega_limit,

        "alpha_min":
            args.alpha_min,

        "alpha_max":
            args.alpha_max,

        "hidden_size":
            args.hidden_size,

        "num_layers":
            args.num_layers,

        "dropout":
            args.dropout,

        "train_sequences":
            train_names,

        "validation_sequences":
            validation_names,

        "test_sequences":
            test_names,
    }

    torch.save(
        checkpoint,

        output_dir
        / "gru_dualhead.pt",
    )

    # -------------------------------------------------------
    # Freeze best model and evaluate every sequence
    # -------------------------------------------------------

    results: list[
        EvalResult
    ] = []

    split_lookup = {
        name:
            "train"

        for name
        in train_names
    }

    split_lookup.update(
        {
            name:
                "validation"

            for name
            in validation_names
        }
    )

    split_lookup.update(
        {
            name:
                "test"

            for name
            in test_names
        }
    )

    print()

    print(
        "Evaluating frozen best checkpoint..."
    )

    evaluation_order = (
        list(
            train_names
        )

        +

        list(
            validation_names
        )

        +

        list(
            test_names
        )
    )

    for name in evaluation_order:

        print(
            f"  {name} "
            f"[{split_lookup[name]}]"
        )

        try:

            result = evaluate_sequence(
                sequence=(
                    prepared[
                        name
                    ]
                ),

                split=(
                    split_lookup[
                        name
                    ]
                ),

                model=(
                    model
                ),

                feature_mean=(
                    feature_mean
                ),

                feature_std=(
                    feature_std
                ),

                args=(
                    args
                ),

                device=(
                    device
                ),

                output_dir=(
                    output_dir
                ),
            )

        except Exception as exc:

            nan = float(
                "nan"
            )

            result = EvalResult(
                sequence=(
                    name
                ),

                split=(
                    split_lookup[
                        name
                    ]
                ),

                status="failed",

                samples=0,

                duration_s=(
                    nan
                ),

                gnss_source=(
                    prepared[
                        name
                    ]
                    .files
                    .gnss_source
                ),

                odo_source=(
                    prepared[
                        name
                    ]
                    .odo_source
                ),

                ate_rmse_m=(
                    nan
                ),

                ate_median_m=(
                    nan
                ),

                ate_p95_m=(
                    nan
                ),

                ate_se2_rmse_m=(
                    nan
                ),

                heading_mae_deg=(
                    nan
                ),

                rpe_1s_trans_rmse_m=(
                    nan
                ),

                rpe_5s_trans_rmse_m=(
                    nan
                ),

                rpe_10s_trans_rmse_m=(
                    nan
                ),

                final_error_m=(
                    nan
                ),

                final_error_se2_m=(
                    nan
                ),

                final_drift_per_m=(
                    nan
                ),

                path_length_ratio=(
                    nan
                ),

                gnss_seen=0,
                gnss_normal=0,
                gnss_reacquired=0,
                gnss_rejected=0,

                gnss_rejection_rate_pct=(
                    nan
                ),

                gnss_max_coast_s=(
                    nan
                ),

                mean_abs_delta_v_mps=(
                    nan
                ),

                p95_abs_delta_v_mps=(
                    nan
                ),

                mean_abs_delta_omega_radps=(
                    nan
                ),

                p95_abs_delta_omega_radps=(
                    nan
                ),

                alpha_xy_mean=(
                    nan
                ),

                alpha_xy_p95=(
                    nan
                ),

                alpha_xy_max=(
                    nan
                ),

                alpha_heading_mean=(
                    nan
                ),

                alpha_heading_p95=(
                    nan
                ),

                alpha_heading_max=(
                    nan
                ),

                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        results.append(
            result
        )

    # -------------------------------------------------------
    # CSV results
    # -------------------------------------------------------

    fidelity_csv_path = (
        output_dir
        / "gru_fidelity.csv"
    )

    with fidelity_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,

            fieldnames=list(
                EvalResult
                .__annotations__
                .keys()
            ),
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                asdict(
                    result
                )
            )

    # -------------------------------------------------------
    # Split-level summary
    # -------------------------------------------------------

    def split_mean(
        split_name: str,
        field: str,
    ) -> float | None:

        values = [
            float(
                getattr(
                    result,
                    field,
                )
            )

            for result
            in results

            if (
                result.status
                == "ok"

                and

                result.split
                == split_name

                and

                np.isfinite(
                    float(
                        getattr(
                            result,
                            field,
                        )
                    )
                )
            )
        ]

        if not values:

            return None

        return float(
            np.mean(
                values
            )
        )

    summary = {
        "schema":
            "i2nav_gru_dualhead_v1",

        "device":
            str(
                device
            ),

        "splits":
            split_payload,

        "model": {
            "hidden_size":
                args.hidden_size,

            "num_layers":
                args.num_layers,

            "window":
                args.window,

            "window_seconds":
                args.window
                / args.rate_hz,

            "delta_v_limit_mps":
                delta_v_limit,

            "delta_omega_limit_radps":
                delta_omega_limit,

            "alpha_min":
                args.alpha_min,

            "alpha_max":
                args.alpha_max,
        },

        "training": {
            "epochs_completed":
                len(
                    history
                ),

            "best_validation_loss":
                min(
                    row[
                        "validation_loss"
                    ]

                    for row
                    in history
                ),
        },

        "test_metrics": {
            "ate_mean_m":
                split_mean(
                    "test",
                    "ate_rmse_m",
                ),

            "ate_se2_mean_m":
                split_mean(
                    "test",
                    "ate_se2_rmse_m",
                ),

            "rpe_1s_mean_m":
                split_mean(
                    "test",
                    "rpe_1s_trans_rmse_m",
                ),

            "heading_mae_mean_deg":
                split_mean(
                    "test",
                    "heading_mae_deg",
                ),
        },

        "validation_metrics": {
            "ate_mean_m":
                split_mean(
                    "validation",
                    "ate_rmse_m",
                ),

            "rpe_1s_mean_m":
                split_mean(
                    "validation",
                    "rpe_1s_trans_rmse_m",
                ),
        },

        "security_constraint": (
            "The neural network never receives GNSS-derived "
            "inputs. Its outputs are bounded dynamics corrections "
            "and bounded Q multipliers derived only from trusted "
            "ODO/IMU motion history."
        ),

        "important_reporting_note": (
            "Only held-out TEST sequence results should be treated "
            "as unbiased performance for this fixed split. "
            "Train-sequence results must not be used as evidence "
            "of generalization."
        ),
    }

    summary_path = (
        output_dir
        / "gru_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),

        encoding="utf-8",
    )

    # -------------------------------------------------------
    # Terminal result table
    # -------------------------------------------------------

    print()

    print(
        f"{'sequence':<14} "
        f"{'split':<12} "
        f"{'ATE':>8} "
        f"{'RPE1':>8} "
        f"{'heading':>9} "
        f"{'aXY':>6} "
        f"{'aH':>6}"
    )

    print(
        "-"
        * 72
    )

    for result in results:

        if (
            result.status
            == "ok"
        ):

            print(
                f"{result.sequence:<14} "
                f"{result.split:<12} "
                f"{result.ate_rmse_m:8.3f} "
                f"{result.rpe_1s_trans_rmse_m:8.3f} "
                f"{result.heading_mae_deg:9.2f} "
                f"{result.alpha_xy_mean:6.2f} "
                f"{result.alpha_heading_mean:6.2f}"
            )

        else:

            print(
                f"{result.sequence:<14} "
                f"{result.split:<12} "
                f"FAILED: {result.error}"
            )

    print()

    print(
        "Wrote:"
    )

    print(
        "  "
        f"{output_dir / 'gru_dualhead.pt'}"
    )

    print(
        "  "
        f"{output_dir / 'training_history.csv'}"
    )

    print(
        "  "
        f"{fidelity_csv_path}"
    )

    print(
        "  "
        f"{summary_path}"
    )

    print(
        "  "
        f"{output_dir / 'split.json'}"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )