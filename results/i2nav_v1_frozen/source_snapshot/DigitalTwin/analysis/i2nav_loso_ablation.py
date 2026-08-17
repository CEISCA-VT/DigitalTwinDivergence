#!/usr/bin/env python3
"""
Publication-oriented i2Nav LOSO + ablation experiment.

Outer evaluation
----------------
Each of the 10 i2Nav sequences serves exactly once as an unseen test sequence.

For every outer fold:

    1 test sequence
    2 validation sequences
    7 training sequences

Validation sequences are selected deterministically from the canonical
sequence ordering. The outer test sequence is NEVER used for:

    - training
    - feature normalization
    - target scaling
    - correction-limit derivation
    - early stopping
    - checkpoint selection

Methods
-------
fixed_v5
    Physics model + fixed Q + V5 robust GNSS gate/reacquisition.

heuristic_v6
    Physics model + trusted heuristic adaptive Q.
    No learned dynamics correction.

gru_dynamics
    Separately trained GRU predicts delta_v and delta_omega.
    Q remains fixed.

gru_q
    Separately trained GRU predicts alpha_xy and alpha_heading.
    No learned dynamics correction.

gru_dual
    Separately trained dual-head GRU predicts both:
        delta_v, delta_omega
        alpha_xy, alpha_heading

Security constraint
-------------------
Every adaptive/learned model receives ONLY trusted ODO/IMU-derived features.

GNSS position, GNSS residual, innovation, NIS, R, reported GNSS uncertainty,
and GNSS-derived features are NEVER model inputs.

Requires
--------
DigitalTwin.analysis.i2nav_gru_dualhead
DigitalTwin.analysis.i2nav_adaptive_q_baseline
DigitalTwin.ekf

Recommended smoke test
----------------------
python -m DigitalTwin.analysis.i2nav_loso_ablation \
    --root public_datasets/im2nav \
    --folds parking02 \
    --epochs 3

Full experiment
---------------
python -m DigitalTwin.analysis.i2nav_loso_ablation \
    --root public_datasets/im2nav

Outputs
-------
results/i2nav_loso_ablation/
    loso_results.csv
    loso_summary.json
    fold_splits.json
    folds/
        building00/
            gru_dynamics.pt
            gru_dynamics_history.csv
            gru_q.pt
            gru_q_history.csv
            gru_dual.pt
            gru_dual_history.csv
        ...
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

    from torch.utils.data import DataLoader

except ImportError as exc:
    raise SystemExit(
        "PyTorch is required.\n"
        "Install a CUDA-enabled PyTorch build if you want GPU training."
    ) from exc


# ===========================================================================
# Project imports
# ===========================================================================

try:
    from DigitalTwin.ekf import RoverEKF

    from DigitalTwin.analysis.i2nav_gru_dualhead import (
        FEATURE_NAMES,
        PreparedSequence,
        WindowDataset,
        prepare_sequence,
        compute_feature_stats,
        derive_correction_limits,
        derive_target_scales,
    )

    from DigitalTwin.analysis.i2nav_adaptive_q_baseline import (
        discover_files,
        summarize_errors,
        compute_trusted_adaptive_q,
    )

except ImportError:
    project_root = Path(__file__).resolve().parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(
            0,
            str(project_root),
        )

    from DigitalTwin.ekf import RoverEKF

    from DigitalTwin.analysis.i2nav_gru_dualhead import (
        FEATURE_NAMES,
        PreparedSequence,
        WindowDataset,
        prepare_sequence,
        compute_feature_stats,
        derive_correction_limits,
        derive_target_scales,
    )

    from DigitalTwin.analysis.i2nav_adaptive_q_baseline import (
        discover_files,
        summarize_errors,
        compute_trusted_adaptive_q,
    )


# ===========================================================================
# Canonical sequence ordering
# ===========================================================================

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


LEARNED_METHODS = (
    "gru_dynamics",
    "gru_q",
    "gru_dual",
)


ALL_METHODS = (
    "fixed_v5",
    "heuristic_v6",
    "gru_dynamics",
    "gru_q",
    "gru_dual",
)


# ===========================================================================
# Result structure
# ===========================================================================

@dataclass
class LOSOResult:
    fold: int

    test_sequence: str

    method: str

    status: str

    train_sequences: str
    validation_sequences: str

    samples: int
    duration_s: float

    gnss_source: str
    odo_source: str

    # -----------------------------------------------------------------------
    # Fidelity
    # -----------------------------------------------------------------------

    ate_rmse_m: float
    ate_median_m: float
    ate_p95_m: float
    ate_max_m: float

    ate_se2_rmse_m: float

    heading_mae_deg: float
    heading_p95_deg: float

    rpe_1s_trans_rmse_m: float
    rpe_5s_trans_rmse_m: float
    rpe_10s_trans_rmse_m: float

    final_error_m: float
    final_error_se2_m: float

    final_drift_per_m: float
    path_length_ratio: float

    # -----------------------------------------------------------------------
    # GNSS behavior
    # -----------------------------------------------------------------------

    gnss_seen: int
    gnss_normal: int
    gnss_reacquired: int
    gnss_rejected: int

    gnss_rejection_rate_pct: float
    gnss_max_coast_s: float

    # -----------------------------------------------------------------------
    # Dynamics correction
    # -----------------------------------------------------------------------

    mean_abs_delta_v_mps: float
    p95_abs_delta_v_mps: float
    max_abs_delta_v_mps: float

    mean_abs_delta_omega_radps: float
    p95_abs_delta_omega_radps: float
    max_abs_delta_omega_radps: float

    # -----------------------------------------------------------------------
    # Adaptive Q
    # -----------------------------------------------------------------------

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

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Fold construction
# ===========================================================================

def build_fold_split(
    test_sequence: str,
    validation_count: int,
) -> tuple[
    list[str],
    list[str],
]:
    """
    Deterministic nested split.

    Example:
        test = building00

        validation =
            building01
            building02

        training =
            remaining 7 sequences

    The choice depends ONLY on sequence ordering and test identity.
    It does not inspect any test metrics.
    """

    if test_sequence not in SEQUENCES:
        raise ValueError(
            f"Unknown sequence: {test_sequence}"
        )

    if validation_count < 1:
        raise ValueError(
            "validation_count must be >= 1"
        )

    if validation_count >= len(SEQUENCES) - 1:
        raise ValueError(
            "Too many validation sequences."
        )

    test_index = SEQUENCES.index(
        test_sequence
    )

    validation = []

    offset = 1

    while len(validation) < validation_count:

        candidate = SEQUENCES[
            (
                test_index
                + offset
            )
            % len(SEQUENCES)
        ]

        offset += 1

        if candidate == test_sequence:
            continue

        validation.append(
            candidate
        )

    training = [
        sequence
        for sequence
        in SEQUENCES
        if (
            sequence != test_sequence
            and sequence not in validation
        )
    ]

    return (
        training,
        validation,
    )


# ===========================================================================
# Ablation GRU
# ===========================================================================

class AblationGRU(
    nn.Module
):
    """
    GRU used for separately trained learned ablations.

    Modes
    -----
    dynamics
        Learn delta_v and delta_omega.
        alpha_xy = alpha_heading = 1.

    q
        Learn alpha_xy and alpha_heading.
        delta_v = delta_omega = 0.

    dual
        Learn both heads.
    """

    def __init__(
        self,
        mode: str,

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

        if mode not in (
            "dynamics",
            "q",
            "dual",
        ):
            raise ValueError(
                f"Invalid mode: {mode}"
            )

        self.mode = mode

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

        # -------------------------------------------------------------------
        # Temporal encoder
        # -------------------------------------------------------------------

        self.gru = nn.GRU(
            input_size=input_dim,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # -------------------------------------------------------------------
        # Shared representation
        # -------------------------------------------------------------------

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

        # -------------------------------------------------------------------
        # Only create heads actually required by the ablation.
        # -------------------------------------------------------------------

        if mode in (
            "dynamics",
            "dual",
        ):
            self.dynamics_head = nn.Linear(
                hidden_size,
                2,
            )

        else:
            self.dynamics_head = None

        if mode in (
            "q",
            "dual",
        ):
            self.q_head = nn.Linear(
                hidden_size,
                2,
            )

        else:
            self.q_head = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        batch_size = (
            x.shape[0]
        )

        _, hidden = self.gru(
            x
        )

        representation = self.trunk(
            hidden[-1]
        )

        # -------------------------------------------------------------------
        # Dynamics
        # -------------------------------------------------------------------

        if self.dynamics_head is not None:

            raw_dynamics = self.dynamics_head(
                representation
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

        else:

            corrections = torch.zeros(
                (
                    batch_size,
                    2,
                ),
                dtype=x.dtype,
                device=x.device,
            )

        # -------------------------------------------------------------------
        # Q multipliers
        # -------------------------------------------------------------------

        if self.q_head is not None:

            normalized = torch.sigmoid(
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
                * normalized
            )

        else:

            alphas = torch.ones(
                (
                    batch_size,
                    2,
                ),
                dtype=x.dtype,
                device=x.device,
            )

        return (
            corrections,
            alphas,
        )


# ===========================================================================
# Loss calculation
# ===========================================================================

def calculate_losses(
    model: AblationGRU,

    corrections: torch.Tensor,

    alphas: torch.Tensor,

    target: torch.Tensor,

    target_scale: torch.Tensor,

    q_xy_sigma_mps: float,

    q_heading_sigma_radps: float,

    alpha_reg_weight: float,

    nll_weight: float,

    mean_only: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    error = (
        target
        - corrections
    )

    # -----------------------------------------------------------------------
    # Dynamics correction objective
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Heteroscedastic residual likelihood
    # -----------------------------------------------------------------------

    sigma_v = torch.clamp(
        q_xy_sigma_mps
        * alphas[
            :,
            0,
        ],

        min=1e-4,
    )

    sigma_omega = torch.clamp(
        q_heading_sigma_radps
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

    # -----------------------------------------------------------------------
    # Prevent arbitrary covariance inflation.
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Ablation-specific training objectives
    # -----------------------------------------------------------------------

    if model.mode == "dynamics":

        loss = (
            mean_loss
        )

    elif model.mode == "q":

        loss = (
            nll

            +

            alpha_reg_weight
            * alpha_reg
        )

    else:

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

    return (
        loss,
        mean_loss,
        nll,
        alpha_reg,
    )


# ===========================================================================
# One training/validation epoch
# ===========================================================================

def run_epoch(
    model: AblationGRU,

    loader: DataLoader,

    optimizer:
        torch.optim.Optimizer
        | None,

    device: torch.device,

    target_scale: torch.Tensor,

    args: argparse.Namespace,

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

    total_loss = 0.0
    total_mean = 0.0
    total_nll = 0.0
    total_alpha_reg = 0.0

    total_samples = 0

    for (
        features,
        target,
    ) in loader:

        features = features.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        target = target.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
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

        (
            loss,
            mean_loss,
            nll,
            alpha_reg,
        ) = calculate_losses(
            model=model,

            corrections=corrections,

            alphas=alphas,

            target=target,

            target_scale=target_scale,

            q_xy_sigma_mps=(
                args.q_xy_sigma_mps
            ),

            q_heading_sigma_radps=(
                args.q_heading_sigma_radps
            ),

            alpha_reg_weight=(
                args.alpha_reg_weight
            ),

            nll_weight=(
                args.nll_weight
            ),

            mean_only=mean_only,
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

        total_samples += (
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

        total_mean += (
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
        total_samples,
        1,
    )

    return {
        "loss":
            total_loss
            / denominator,

        "mean_loss":
            total_mean
            / denominator,

        "nll":
            total_nll
            / denominator,

        "alpha_reg":
            total_alpha_reg
            / denominator,
    }


# ===========================================================================
# Train one learned ablation
# ===========================================================================

def train_ablation(
    mode: str,

    train_sequences: list[
        PreparedSequence
    ],

    validation_sequences: list[
        PreparedSequence
    ],

    feature_mean: np.ndarray,

    feature_std: np.ndarray,

    target_scale_numpy: np.ndarray,

    dv_limit: float,

    domega_limit: float,

    args: argparse.Namespace,

    device: torch.device,

    fold_dir: Path,
) -> AblationGRU:

    train_dataset = WindowDataset(
        sequences=train_sequences,

        feature_mean=feature_mean,

        feature_std=feature_std,

        window=args.window,

        stride=args.stride,
    )

    validation_dataset = WindowDataset(
        sequences=validation_sequences,

        feature_mean=feature_mean,

        feature_std=feature_std,

        window=args.window,

        stride=args.stride,
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    model = AblationGRU(
        mode=mode,

        input_dim=len(
            FEATURE_NAMES
        ),

        hidden_size=args.hidden_size,

        num_layers=args.num_layers,

        dropout=args.dropout,

        dv_limit=dv_limit,

        domega_limit=domega_limit,

        alpha_min=args.alpha_min,

        alpha_max=args.alpha_max,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=args.lr,

        weight_decay=args.weight_decay,
    )

    target_scale = torch.tensor(
        target_scale_numpy,

        dtype=torch.float32,

        device=device,
    )

    best_validation_loss = float(
        "inf"
    )

    best_state = None

    patience_remaining = (
        args.patience
    )

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ---------------------------------------------------------------
        # Dual model retains original warmup.
        #
        # q-only trains Q immediately.
        # dynamics-only always trains dynamics.
        # ---------------------------------------------------------------

        mean_only = (
            mode == "dual"
            and epoch
            <= args.warmup_epochs
        )

        training_stats = run_epoch(
            model=model,

            loader=train_loader,

            optimizer=optimizer,

            device=device,

            target_scale=target_scale,

            args=args,

            mean_only=mean_only,
        )

        with torch.no_grad():

            validation_stats = run_epoch(
                model=model,

                loader=validation_loader,

                optimizer=None,

                device=device,

                target_scale=target_scale,

                args=args,

                mean_only=False,
            )

        row = {
            "epoch":
                epoch,

            "train_loss":
                training_stats[
                    "loss"
                ],

            "train_mean_loss":
                training_stats[
                    "mean_loss"
                ],

            "train_nll":
                training_stats[
                    "nll"
                ],

            "validation_loss":
                validation_stats[
                    "loss"
                ],

            "validation_mean_loss":
                validation_stats[
                    "mean_loss"
                ],

            "validation_nll":
                validation_stats[
                    "nll"
                ],
        }

        history.append(
            row
        )

        print(
            f"      "
            f"{mode:<8} "
            f"epoch={epoch:03d} "
            f"train={row['train_loss']:.5f} "
            f"val={row['validation_loss']:.5f}"
        )

        validation_loss = (
            row[
                "validation_loss"
            ]
        )

        if (
            validation_loss
            <
            best_validation_loss
            - args.min_delta
        ):

            best_validation_loss = (
                validation_loss
            )

            best_state = {
                name:
                    tensor
                    .detach()
                    .cpu()
                    .clone()

                for (
                    name,
                    tensor,
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
                    f"      "
                    f"{mode}: early stop "
                    f"at epoch {epoch}; "
                    f"best={best_validation_loss:.6f}"
                )

                break

    if best_state is None:

        raise RuntimeError(
            f"No valid checkpoint for {mode}"
        )

    model.load_state_dict(
        best_state
    )

    # -----------------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------------

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        fold_dir
        / f"gru_{mode}_history.csv"
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

    checkpoint = {
        "mode":
            mode,

        "state_dict":
            model.state_dict(),

        "feature_mean":
            feature_mean,

        "feature_std":
            feature_std,

        "target_scale":
            target_scale_numpy,

        "dv_limit":
            dv_limit,

        "domega_limit":
            domega_limit,

        "hidden_size":
            args.hidden_size,

        "num_layers":
            args.num_layers,

        "window":
            args.window,

        "alpha_min":
            args.alpha_min,

        "alpha_max":
            args.alpha_max,

        "best_validation_loss":
            best_validation_loss,
    }

    torch.save(
        checkpoint,

        fold_dir
        / f"gru_{mode}.pt",
    )

    return (
        model
    )


# ===========================================================================
# Neural inference
# ===========================================================================

@torch.no_grad()
def predict_neural_sequence(
    model: AblationGRU,

    sequence: PreparedSequence,

    feature_mean: np.ndarray,

    feature_std: np.ndarray,

    window: int,

    batch_size: int,

    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    sample_count = len(
        sequence.grid
    )

    corrections = np.zeros(
        (
            sample_count,
            2,
        ),
        dtype=np.float32,
    )

    alphas = np.ones(
        (
            sample_count,
            2,
        ),
        dtype=np.float32,
    )

    model.eval()

    indices = list(
        range(
            window - 1,
            sample_count,
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

            values = sequence.features[
                start_index
                :
                end_index + 1
            ]

            values = (
                values
                - feature_mean
            ) / feature_std

            windows.append(
                values
            )

        tensor = torch.from_numpy(
            np.stack(
                windows
            ).astype(
                np.float32
            )
        ).to(
            device
        )

        (
            batch_corrections,
            batch_alphas,
        ) = model(
            tensor
        )

        batch_corrections = (
            batch_corrections
            .detach()
            .cpu()
            .numpy()
        )

        batch_alphas = (
            batch_alphas
            .detach()
            .cpu()
            .numpy()
        )

        for local_index, end_index in enumerate(
            batch_indices
        ):

            corrections[
                end_index
            ] = (
                batch_corrections[
                    local_index
                ]
            )

            alphas[
                end_index
            ] = (
                batch_alphas[
                    local_index
                ]
            )

    return (
        corrections,
        alphas,
    )


# ===========================================================================
# Baseline predictions
# ===========================================================================

def fixed_predictions(
    sequence: PreparedSequence,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    corrections = np.zeros(
        (
            len(
                sequence.grid
            ),
            2,
        ),
        dtype=np.float32,
    )

    alphas = np.ones(
        (
            len(
                sequence.grid
            ),
            2,
        ),
        dtype=np.float32,
    )

    return (
        corrections,
        alphas,
    )


def heuristic_predictions(
    sequence: PreparedSequence,

    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    corrections = np.zeros(
        (
            len(
                sequence.grid
            ),
            2,
        ),
        dtype=np.float32,
    )

    alphas = np.ones(
        (
            len(
                sequence.grid
            ),
            2,
        ),
        dtype=np.float32,
    )

    for k in range(
        1,
        len(
            sequence.grid
        ),
    ):

        dt = float(
            sequence.grid[k]
            - sequence.grid[
                k - 1
            ]
        )

        (
            _,
            alpha_xy,
            alpha_heading,
            _,
            _,
        ) = compute_trusted_adaptive_q(
            speed_now=float(
                sequence.odo_speed[k]
            ),

            speed_previous=float(
                sequence.odo_speed[
                    k - 1
                ]
            ),

            omega_now=float(
                sequence.imu_yaw_rate[k]
            ),

            omega_previous=float(
                sequence.imu_yaw_rate[
                    k - 1
                ]
            ),

            dt=dt,

            base_xy_sigma_mps=(
                args.q_xy_sigma_mps
            ),

            base_heading_sigma_radps=(
                args.q_heading_sigma_radps
            ),

            alpha_min=(
                args.alpha_min
            ),

            alpha_max=(
                args.alpha_max
            ),

            xy_speed_coeff=(
                args.alpha_xy_speed_coeff
            ),

            xy_turn_coeff=(
                args.alpha_xy_turn_coeff
            ),

            xy_accel_coeff=(
                args.alpha_xy_accel_coeff
            ),

            heading_turn_coeff=(
                args.alpha_heading_turn_coeff
            ),

            heading_yaw_accel_coeff=(
                args.alpha_heading_yaw_accel_coeff
            ),
        )

        alphas[
            k,
            0,
        ] = alpha_xy

        alphas[
            k,
            1,
        ] = alpha_heading

    return (
        corrections,
        alphas,
    )


# ===========================================================================
# EKF replay
# ===========================================================================

def evaluate_predictions(
    *,
    fold: int,

    method: str,

    sequence: PreparedSequence,

    training_names: list[str],

    validation_names: list[str],

    corrections: np.ndarray,

    alphas: np.ndarray,

    args: argparse.Namespace,

    trajectory_path:
        Path
        | None,
) -> LOSOResult:

    grid = (
        sequence.grid
    )

    truth_xy = np.column_stack(
        (
            sequence.gt_x,
            sequence.gt_y,
        )
    )

    # -----------------------------------------------------------------------
    # Same GT initialization used by previous i2Nav experiments.
    # -----------------------------------------------------------------------

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
        initial_state=initial_state,

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

    gnss_index = 0

    if gnss is not None:

        while (
            gnss_index
            < len(
                gnss[
                    "t"
                ]
            )

            and

            gnss[
                "t"
            ][gnss_index]
            < grid[0]
        ):

            gnss_index += 1

    gnss_seen = 0
    gnss_normal = 0
    gnss_reacquired = 0
    gnss_rejected = 0

    reacquisition_streak = 0

    last_accepted_gnss_time = float(
        grid[0]
    )

    max_coast_s = 0.0

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

        # -------------------------------------------------------------------
        # Corrected dynamics
        # -------------------------------------------------------------------

        corrected_speed = float(
            sequence.odo_speed[k]
            + corrections[
                k,
                0,
            ]
        )

        corrected_omega = float(
            sequence.imu_yaw_rate[k]
            + corrections[
                k,
                1,
            ]
        )

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

        # -------------------------------------------------------------------
        # Process covariance
        # -------------------------------------------------------------------

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

        # -------------------------------------------------------------------
        # Physics prediction
        # -------------------------------------------------------------------

        ekf.predict(
            corrected_speed,
            corrected_omega,
            dt,
            Q,
        )

        # -------------------------------------------------------------------
        # Robust GNSS measurement path
        # -------------------------------------------------------------------

        if gnss is not None:

            latest = None

            while (
                gnss_index
                < len(
                    gnss[
                        "t"
                    ]
                )

                and

                gnss[
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

                    innovation = (
                        z
                        - H
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

                    accepted = False

                    # -------------------------------------------------------
                    # Normal hard NIS gate
                    # -------------------------------------------------------

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

                        reacquisition_streak = 0

                        accepted = True

                    # -------------------------------------------------------
                    # Safe reacquisition
                    # -------------------------------------------------------

                    else:

                        coast_s = max(
                            0.0,

                            float(
                                grid[k]
                                - last_accepted_gnss_time
                            ),
                        )

                        candidate = False

                        extra_sigma = 0.0

                        if (
                            coast_s
                            >= args.reacq_start_s
                        ):

                            extra_sigma = min(
                                args.reacq_sigma_max_m,

                                args.reacq_sigma_growth_mps
                                * coast_s,
                            )

                            S_reacq = (
                                S

                                +

                                identity_2
                                * extra_sigma**2
                            )

                            try:

                                reacq_nis = float(
                                    innovation.T

                                    @

                                    np.linalg.solve(
                                        S_reacq,
                                        innovation,
                                    )
                                )

                            except np.linalg.LinAlgError:

                                reacq_nis = float(
                                    "inf"
                                )

                            candidate = (
                                np.isfinite(
                                    reacq_nis
                                )

                                and

                                reacq_nis
                                <= args.gnss_nis_gate
                            )

                        if candidate:

                            reacquisition_streak += 1

                        else:

                            reacquisition_streak = 0

                        if (
                            candidate

                            and

                            reacquisition_streak
                            >= args.reacq_consecutive
                        ):

                            R_reacq = (
                                R

                                +

                                identity_2
                                * extra_sigma**2
                            )

                            ekf.update_gps(
                                z,
                                R_reacq,
                            )

                            gnss_reacquired += 1

                            reacquisition_streak = 0

                            accepted = True

                        else:

                            gnss_rejected += 1

                    if accepted:

                        last_accepted_gnss_time = float(
                            gnss[
                                "t"
                            ][latest]
                        )

        if (
            sequence.files.gnss
            is not None
        ):

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

    gnss_rejection_rate = (
        100.0
        * gnss_rejected
        / gnss_seen

        if gnss_seen > 0
        else 0.0
    )

    # =========================================================================
    # Optional detailed trajectory
    # =========================================================================

    if trajectory_path is not None:

        trajectory_path.parent.mkdir(
            parents=True,
            exist_ok=True,
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

                    "estimate_east_m",
                    "estimate_north_m",
                    "estimate_heading_rad",

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

    # =========================================================================
    # Result
    # =========================================================================

    return LOSOResult(
        fold=fold,

        test_sequence=(
            sequence.name
        ),

        method=method,

        status="ok",

        train_sequences=",".join(
            training_names
        ),

        validation_sequences=",".join(
            validation_names
        ),

        samples=len(
            grid
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

        ate_rmse_m=float(
            metrics[
                "ate_rmse_m"
            ]
        ),

        ate_median_m=float(
            metrics[
                "ate_median_m"
            ]
        ),

        ate_p95_m=float(
            metrics[
                "ate_p95_m"
            ]
        ),

        ate_max_m=float(
            metrics[
                "ate_max_m"
            ]
        ),

        ate_se2_rmse_m=float(
            metrics[
                "ate_se2_rmse_m"
            ]
        ),

        heading_mae_deg=float(
            metrics[
                "heading_mae_deg"
            ]
        ),

        heading_p95_deg=float(
            metrics[
                "heading_p95_deg"
            ]
        ),

        rpe_1s_trans_rmse_m=float(
            metrics[
                "rpe_1s_trans_rmse_m"
            ]
        ),

        rpe_5s_trans_rmse_m=float(
            metrics[
                "rpe_5s_trans_rmse_m"
            ]
        ),

        rpe_10s_trans_rmse_m=float(
            metrics[
                "rpe_10s_trans_rmse_m"
            ]
        ),

        final_error_m=float(
            metrics[
                "final_error_m"
            ]
        ),

        final_error_se2_m=float(
            metrics[
                "final_error_se2_m"
            ]
        ),

        final_drift_per_m=float(
            metrics[
                "final_drift_per_m"
            ]
        ),

        path_length_ratio=float(
            metrics[
                "path_length_ratio"
            ]
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
            gnss_rejection_rate
        ),

        gnss_max_coast_s=(
            max_coast_s
        ),

        mean_abs_delta_v_mps=float(
            np.mean(
                np.abs(
                    corrections[
                        :,
                        0,
                    ]
                )
            )
        ),

        p95_abs_delta_v_mps=float(
            np.percentile(
                np.abs(
                    corrections[
                        :,
                        0,
                    ]
                ),
                95,
            )
        ),

        max_abs_delta_v_mps=float(
            np.max(
                np.abs(
                    corrections[
                        :,
                        0,
                    ]
                )
            )
        ),

        mean_abs_delta_omega_radps=float(
            np.mean(
                np.abs(
                    corrections[
                        :,
                        1,
                    ]
                )
            )
        ),

        p95_abs_delta_omega_radps=float(
            np.percentile(
                np.abs(
                    corrections[
                        :,
                        1,
                    ]
                ),
                95,
            )
        ),

        max_abs_delta_omega_radps=float(
            np.max(
                np.abs(
                    corrections[
                        :,
                        1,
                    ]
                )
            )
        ),

        alpha_xy_mean=float(
            np.mean(
                alphas[
                    :,
                    0,
                ]
            )
        ),

        alpha_xy_p95=float(
            np.percentile(
                alphas[
                    :,
                    0,
                ],
                95,
            )
        ),

        alpha_xy_max=float(
            np.max(
                alphas[
                    :,
                    0,
                ]
            )
        ),

        alpha_heading_mean=float(
            np.mean(
                alphas[
                    :,
                    1,
                ]
            )
        ),

        alpha_heading_p95=float(
            np.percentile(
                alphas[
                    :,
                    1,
                ],
                95,
            )
        ),

        alpha_heading_max=float(
            np.max(
                alphas[
                    :,
                    1,
                ]
            )
        ),
    )


# ===========================================================================
# Failed result
# ===========================================================================

def failed_result(
    fold: int,

    test_sequence: str,

    method: str,

    training_names: list[str],

    validation_names: list[str],

    sequence: PreparedSequence,

    exc: Exception,
) -> LOSOResult:

    nan = float(
        "nan"
    )

    return LOSOResult(
        fold=fold,

        test_sequence=(
            test_sequence
        ),

        method=method,

        status="failed",

        train_sequences=",".join(
            training_names
        ),

        validation_sequences=",".join(
            validation_names
        ),

        samples=0,

        duration_s=nan,

        gnss_source=(
            sequence
            .files
            .gnss_source
        ),

        odo_source=(
            sequence.odo_source
        ),

        ate_rmse_m=nan,
        ate_median_m=nan,
        ate_p95_m=nan,
        ate_max_m=nan,

        ate_se2_rmse_m=nan,

        heading_mae_deg=nan,
        heading_p95_deg=nan,

        rpe_1s_trans_rmse_m=nan,
        rpe_5s_trans_rmse_m=nan,
        rpe_10s_trans_rmse_m=nan,

        final_error_m=nan,
        final_error_se2_m=nan,

        final_drift_per_m=nan,
        path_length_ratio=nan,

        gnss_seen=0,
        gnss_normal=0,
        gnss_reacquired=0,
        gnss_rejected=0,

        gnss_rejection_rate_pct=nan,
        gnss_max_coast_s=nan,

        mean_abs_delta_v_mps=nan,
        p95_abs_delta_v_mps=nan,
        max_abs_delta_v_mps=nan,

        mean_abs_delta_omega_radps=nan,
        p95_abs_delta_omega_radps=nan,
        max_abs_delta_omega_radps=nan,

        alpha_xy_mean=nan,
        alpha_xy_p95=nan,
        alpha_xy_max=nan,

        alpha_heading_mean=nan,
        alpha_heading_p95=nan,
        alpha_heading_max=nan,

        error=(
            f"{type(exc).__name__}: {exc}"
        ),
    )


# ===========================================================================
# CSV
# ===========================================================================

def write_results_csv(
    path: Path,

    results: list[
        LOSOResult
    ],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,

            fieldnames=list(
                LOSOResult
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


# ===========================================================================
# Statistics
# ===========================================================================

def finite_method_values(
    results: list[
        LOSOResult
    ],

    method: str,

    field: str,

    *,
    gnss_mode:
        str
        | None
        = None,
) -> np.ndarray:

    values = []

    for result in results:

        if (
            result.status
            != "ok"

            or

            result.method
            != method
        ):
            continue

        if (
            gnss_mode
            == "fused"

            and

            result.gnss_source
            == "NONE"
        ):
            continue

        if (
            gnss_mode
            == "gnss_free"

            and

            result.gnss_source
            != "NONE"
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


def bootstrap_mean_ci(
    values: np.ndarray,

    seed: int,

    repetitions: int = 10000,
) -> tuple[
    float | None,
    float | None,
]:

    if len(
        values
    ) < 2:

        return (
            None,
            None,
        )

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        repetitions,
        dtype=float,
    )

    for index in range(
        repetitions
    ):

        sample = rng.choice(
            values,

            size=len(
                values
            ),

            replace=True,
        )

        means[index] = float(
            np.mean(
                sample
            )
        )

    lower = float(
        np.percentile(
            means,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            means,
            97.5,
        )
    )

    return (
        lower,
        upper,
    )


def method_summary(
    results: list[
        LOSOResult
    ],

    method: str,

    seed: int,
) -> dict:

    ate = finite_method_values(
        results,
        method,
        "ate_rmse_m",
    )

    se2 = finite_method_values(
        results,
        method,
        "ate_se2_rmse_m",
    )

    rpe1 = finite_method_values(
        results,
        method,
        "rpe_1s_trans_rmse_m",
    )

    rpe5 = finite_method_values(
        results,
        method,
        "rpe_5s_trans_rmse_m",
    )

    rpe10 = finite_method_values(
        results,
        method,
        "rpe_10s_trans_rmse_m",
    )

    heading = finite_method_values(
        results,
        method,
        "heading_mae_deg",
    )

    gnss_free_ate = finite_method_values(
        results,
        method,
        "ate_rmse_m",
        gnss_mode="gnss_free",
    )

    fused_ate = finite_method_values(
        results,
        method,
        "ate_rmse_m",
        gnss_mode="fused",
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            ate,
            seed=seed,
        )
    )

    def mean_or_none(
        values: np.ndarray,
    ) -> float | None:

        if not len(
            values
        ):
            return None

        return float(
            np.mean(
                values
            )
        )

    def median_or_none(
        values: np.ndarray,
    ) -> float | None:

        if not len(
            values
        ):
            return None

        return float(
            np.median(
                values
            )
        )

    def std_or_none(
        values: np.ndarray,
    ) -> float | None:

        if len(
            values
        ) < 2:
            return None

        return float(
            np.std(
                values,
                ddof=1,
            )
        )

    return {
        "held_out_sequences":
            int(
                len(
                    ate
                )
            ),

        "ate_macro_mean_m":
            mean_or_none(
                ate
            ),

        "ate_median_m":
            median_or_none(
                ate
            ),

        "ate_std_m":
            std_or_none(
                ate
            ),

        "ate_sequence_rms_m": (
            float(
                np.sqrt(
                    np.mean(
                        ate**2
                    )
                )
            )
            if len(
                ate
            )
            else None
        ),

        "ate_bootstrap_95ci_m": (
            [
                ci_low,
                ci_high,
            ]
            if ci_low
            is not None
            else None
        ),

        "ate_se2_macro_mean_m":
            mean_or_none(
                se2
            ),

        "rpe_1s_macro_mean_m":
            mean_or_none(
                rpe1
            ),

        "rpe_5s_macro_mean_m":
            mean_or_none(
                rpe5
            ),

        "rpe_10s_macro_mean_m":
            mean_or_none(
                rpe10
            ),

        "heading_mae_macro_mean_deg":
            mean_or_none(
                heading
            ),

        "gnss_free_ate_mean_m":
            mean_or_none(
                gnss_free_ate
            ),

        "gnss_fused_ate_mean_m":
            mean_or_none(
                fused_ate
            ),
    }


# ===========================================================================
# Summary
# ===========================================================================

def write_summary(
    path: Path,

    results: list[
        LOSOResult
    ],

    fold_splits: list[
        dict
    ],

    args: argparse.Namespace,

    device: torch.device,
) -> None:

    summary = {
        "schema":
            "i2nav_loso_ablation_v1",

        "device":
            str(
                device
            ),

        "outer_evaluation":
            "leave-one-sequence-out",

        "validation_count_per_fold":
            args.validation_count,

        "methods":
            list(
                args.methods
            ),

        "method_summaries": {
            method:
                method_summary(
                    results,
                    method,
                    args.seed,
                )

            for method
            in args.methods
        },

        "fold_splits":
            fold_splits,

        "model_configuration": {
            "rate_hz":
                args.rate_hz,

            "window_samples":
                args.window,

            "window_seconds":
                args.window
                / args.rate_hz,

            "hidden_size":
                args.hidden_size,

            "num_layers":
                args.num_layers,

            "dropout":
                args.dropout,

            "alpha_min":
                args.alpha_min,

            "alpha_max":
                args.alpha_max,

            "q_xy_sigma_mps":
                args.q_xy_sigma_mps,

            "q_heading_sigma_radps":
                args.q_heading_sigma_radps,

            "gnss_nis_gate":
                args.gnss_nis_gate,
        },

        "security_constraint": (
            "No adaptive or neural model receives GNSS position, "
            "GNSS innovation, residual, NIS, GNSS covariance, "
            "reported GNSS uncertainty, or any GNSS-derived feature. "
            "Learned and heuristic adaptation uses trusted ODO/IMU "
            "motion information only."
        ),

        "reporting_rule": (
            "Only the outer-fold test result for each sequence is used "
            "in LOSO aggregate performance. Training and validation "
            "trajectory performance are excluded from headline metrics."
        ),

        "ablation_note": (
            "GRU dynamics-only, GRU Q-only, and GRU dual-head models "
            "are separately trained in each fold. This avoids the weaker "
            "practice of training only a dual-head model and disabling "
            "heads after training."
        ),
    }

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


# ===========================================================================
# Terminal table
# ===========================================================================

def print_final_table(
    results: list[
        LOSOResult
    ],

    methods: list[
        str
    ],
) -> None:

    print()

    print(
        "="
        * 98
    )

    print(
        "FINAL OUTER-LOSO RESULTS"
    )

    print(
        "="
        * 98
    )

    print(
        f"{'sequence':<14} "
        f"{'method':<16} "
        f"{'ATE':>8} "
        f"{'SE2':>8} "
        f"{'RPE1':>8} "
        f"{'head':>8} "
        f"{'GNSS':>10}"
    )

    print(
        "-"
        * 98
    )

    for sequence in SEQUENCES:

        sequence_results = [
            result
            for result
            in results
            if (
                result.test_sequence
                == sequence

                and

                result.method
                in methods
            )
        ]

        for result in sequence_results:

            if (
                result.status
                == "ok"
            ):

                print(
                    f"{result.test_sequence:<14} "
                    f"{result.method:<16} "
                    f"{result.ate_rmse_m:8.3f} "
                    f"{result.ate_se2_rmse_m:8.3f} "
                    f"{result.rpe_1s_trans_rmse_m:8.3f} "
                    f"{result.heading_mae_deg:8.2f} "
                    f"{result.gnss_source:>10}"
                )

            else:

                print(
                    f"{result.test_sequence:<14} "
                    f"{result.method:<16} "
                    f"FAILED: "
                    f"{result.error}"
                )

    print()

    print(
        "Aggregate held-out LOSO:"
    )

    for method in methods:

        summary = method_summary(
            results,
            method,
            seed=42,
        )

        mean_ate = (
            summary[
                "ate_macro_mean_m"
            ]
        )

        gnss_free = (
            summary[
                "gnss_free_ate_mean_m"
            ]
        )

        fused = (
            summary[
                "gnss_fused_ate_mean_m"
            ]
        )

        print(
            f"  {method:<16} "
            f"ATE={mean_ate:.3f} m "
            if mean_ate
            is not None
            else
            f"  {method:<16} ATE=N/A"
        )

        if (
            mean_ate
            is not None
        ):

            print(
                f"      GNSS-free="
                f"{gnss_free:.3f} m "
                if gnss_free
                is not None
                else
                "      GNSS-free=N/A "
            )

            if fused is not None:

                print(
                    f"      GNSS-fused="
                    f"{fused:.3f} m"
                )


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "10-fold i2Nav LOSO with "
            "separately trained GRU ablations."
        )
    )

    # -----------------------------------------------------------------------
    # Data / output
    # -----------------------------------------------------------------------

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
            "results/i2nav_loso_ablation"
        ),
    )

    parser.add_argument(
        "--folds",

        nargs="*",

        default=None,

        help=(
            "Optional subset of outer test folds, "
            "e.g. --folds parking02 street02"
        ),
    )

    parser.add_argument(
        "--methods",

        nargs="*",

        choices=ALL_METHODS,

        default=list(
            ALL_METHODS
        ),
    )

    parser.add_argument(
        "--save-trajectories",

        action="store_true",
    )

    # -----------------------------------------------------------------------
    # Split design
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--validation-count",

        type=int,

        default=2,
    )

    # -----------------------------------------------------------------------
    # Sampling / GRU
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--rate-hz",

        type=float,

        default=10.0,
    )

    parser.add_argument(
        "--window",

        type=int,

        default=20,
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

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

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

        default=1024,
    )

    parser.add_argument(
        "--eval-batch-size",

        type=int,

        default=2048,
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
    )

    # -----------------------------------------------------------------------
    # Q
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
    # V6 heuristic
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

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--device",

        choices=(
            "auto",
            "cuda",
            "cpu",
        ),

        default="auto",
    )

    return (
        parser.parse_args()
    )


# ===========================================================================
# Device selection
# ===========================================================================

def select_device(
    requested: str,
) -> torch.device:

    if requested == "cpu":

        return torch.device(
            "cpu"
        )

    if requested == "cuda":

        if not torch.cuda.is_available():

            raise SystemExit(
                "ERROR: --device cuda requested, "
                "but torch.cuda.is_available() is False."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    device = select_device(
        args.device
    )

    print(
        f"Device: {device}"
    )

    if (
        device.type
        == "cuda"
    ):

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------------------------
    # Discover dataset
    # -----------------------------------------------------------------------

    discovered = {
        files.name:
            files

        for files
        in discover_files(
            root
        )
    }

    missing = [
        sequence
        for sequence
        in SEQUENCES
        if sequence
        not in discovered
    ]

    if missing:

        raise SystemExit(
            f"Missing sequences: {missing}"
        )

    requested_folds = (
        list(
            SEQUENCES
        )

        if not args.folds

        else list(
            args.folds
        )
    )

    invalid_folds = [
        sequence
        for sequence
        in requested_folds
        if sequence
        not in SEQUENCES
    ]

    if invalid_folds:

        raise SystemExit(
            f"Unknown folds: {invalid_folds}"
        )

    # -----------------------------------------------------------------------
    # Prepare every sequence once.
    #
    # This only loads raw data and constructs trusted features/GT targets.
    # Fold-specific normalization is still computed from TRAIN ONLY.
    # -----------------------------------------------------------------------

    print()

    print(
        "Preparing all i2Nav sequences..."
    )

    prepared: dict[
        str,
        PreparedSequence,
    ] = {}

    for sequence_name in SEQUENCES:

        print(
            f"  {sequence_name}"
        )

        prepared[
            sequence_name
        ] = prepare_sequence(
            discovered[
                sequence_name
            ],

            hz=args.rate_hz,

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

    all_results: list[
        LOSOResult
    ] = []

    fold_splits = []

    # =========================================================================
    # OUTER LOSO
    # =========================================================================

    for fold_number, test_name in enumerate(
        requested_folds,
        start=1,
    ):

        # -------------------------------------------------------------------
        # Reset RNG per fold deterministically.
        # -------------------------------------------------------------------

        fold_seed = (
            args.seed
            + SEQUENCES.index(
                test_name
            )
            * 100
        )

        set_seed(
            fold_seed
        )

        (
            training_names,
            validation_names,
        ) = build_fold_split(
            test_name,

            args.validation_count,
        )

        fold_splits.append(
            {
                "fold":
                    fold_number,

                "test":
                    test_name,

                "validation":
                    validation_names,

                "train":
                    training_names,
            }
        )

        print()

        print(
            "="
            * 100
        )

        print(
            f"OUTER FOLD {fold_number}"
            f" / {len(requested_folds)}"
        )

        print(
            f"TEST: {test_name}"
        )

        print(
            "VAL : "
            + ", ".join(
                validation_names
            )
        )

        print(
            "TRAIN: "
            + ", ".join(
                training_names
            )
        )

        print(
            "="
            * 100
        )

        training_sequences = [
            prepared[
                name
            ]
            for name
            in training_names
        ]

        validation_sequences = [
            prepared[
                name
            ]
            for name
            in validation_names
        ]

        test_sequence = (
            prepared[
                test_name
            ]
        )

        # -------------------------------------------------------------------
        # CRITICAL:
        # Statistics derived from OUTER TRAIN ONLY.
        # -------------------------------------------------------------------

        (
            feature_mean,
            feature_std,
        ) = compute_feature_stats(
            training_sequences
        )

        target_scale = derive_target_scales(
            training_sequences
        )

        (
            dv_limit,
            domega_limit,
        ) = derive_correction_limits(
            training_sequences
        )

        print(
            f"  dv limit     = "
            f"{dv_limit:.4f} m/s"
        )

        print(
            f"  domega limit = "
            f"{domega_limit:.4f} rad/s"
        )

        fold_dir = (
            output_dir
            / "folds"
            / test_name
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ===================================================================
        # Baseline: V5 fixed Q
        # ===================================================================

        if (
            "fixed_v5"
            in args.methods
        ):

            print()

            print(
                "  Evaluating fixed_v5..."
            )

            try:

                (
                    corrections,
                    alphas,
                ) = fixed_predictions(
                    test_sequence
                )

                trajectory_path = (
                    fold_dir
                    / "fixed_v5_trajectory.csv"
                    if args.save_trajectories
                    else None
                )

                result = evaluate_predictions(
                    fold=fold_number,

                    method="fixed_v5",

                    sequence=test_sequence,

                    training_names=(
                        training_names
                    ),

                    validation_names=(
                        validation_names
                    ),

                    corrections=(
                        corrections
                    ),

                    alphas=(
                        alphas
                    ),

                    args=args,

                    trajectory_path=(
                        trajectory_path
                    ),
                )

            except Exception as exc:

                result = failed_result(
                    fold_number,

                    test_name,

                    "fixed_v5",

                    training_names,

                    validation_names,

                    test_sequence,

                    exc,
                )

            all_results.append(
                result
            )

            if result.status == "ok":

                print(
                    f"      ATE = "
                    f"{result.ate_rmse_m:.3f} m"
                )

        # ===================================================================
        # Baseline: V6 heuristic Q
        # ===================================================================

        if (
            "heuristic_v6"
            in args.methods
        ):

            print()

            print(
                "  Evaluating heuristic_v6..."
            )

            try:

                (
                    corrections,
                    alphas,
                ) = heuristic_predictions(
                    test_sequence,
                    args,
                )

                trajectory_path = (
                    fold_dir
                    / "heuristic_v6_trajectory.csv"
                    if args.save_trajectories
                    else None
                )

                result = evaluate_predictions(
                    fold=fold_number,

                    method="heuristic_v6",

                    sequence=test_sequence,

                    training_names=(
                        training_names
                    ),

                    validation_names=(
                        validation_names
                    ),

                    corrections=(
                        corrections
                    ),

                    alphas=(
                        alphas
                    ),

                    args=args,

                    trajectory_path=(
                        trajectory_path
                    ),
                )

            except Exception as exc:

                result = failed_result(
                    fold_number,

                    test_name,

                    "heuristic_v6",

                    training_names,

                    validation_names,

                    test_sequence,

                    exc,
                )

            all_results.append(
                result
            )

            if result.status == "ok":

                print(
                    f"      ATE = "
                    f"{result.ate_rmse_m:.3f} m"
                )

        # ===================================================================
        # Separately trained learned ablations
        # ===================================================================

        for method in (
            "gru_dynamics",
            "gru_q",
            "gru_dual",
        ):

            if method not in args.methods:
                continue

            mode = (
                method
                .replace(
                    "gru_",
                    "",
                )
            )

            print()

            print(
                f"  Training {method}..."
            )

            try:

                # -----------------------------------------------------------
                # Reset RNG so each model begins reproducibly.
                # -----------------------------------------------------------

                method_seed = (
                    fold_seed

                    +

                    {
                        "dynamics":
                            1,

                        "q":
                            2,

                        "dual":
                            3,
                    }[
                        mode
                    ]
                )

                set_seed(
                    method_seed
                )

                model = train_ablation(
                    mode=mode,

                    train_sequences=(
                        training_sequences
                    ),

                    validation_sequences=(
                        validation_sequences
                    ),

                    feature_mean=(
                        feature_mean
                    ),

                    feature_std=(
                        feature_std
                    ),

                    target_scale_numpy=(
                        target_scale
                    ),

                    dv_limit=(
                        dv_limit
                    ),

                    domega_limit=(
                        domega_limit
                    ),

                    args=args,

                    device=device,

                    fold_dir=(
                        fold_dir
                    ),
                )

                (
                    corrections,
                    alphas,
                ) = predict_neural_sequence(
                    model=model,

                    sequence=(
                        test_sequence
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

                    batch_size=(
                        args.eval_batch_size
                    ),

                    device=device,
                )

                trajectory_path = (
                    fold_dir
                    / f"{method}_trajectory.csv"
                    if args.save_trajectories
                    else None
                )

                result = evaluate_predictions(
                    fold=fold_number,

                    method=method,

                    sequence=(
                        test_sequence
                    ),

                    training_names=(
                        training_names
                    ),

                    validation_names=(
                        validation_names
                    ),

                    corrections=(
                        corrections
                    ),

                    alphas=(
                        alphas
                    ),

                    args=args,

                    trajectory_path=(
                        trajectory_path
                    ),
                )

                del model

                if (
                    device.type
                    == "cuda"
                ):

                    torch.cuda.empty_cache()

            except Exception as exc:

                result = failed_result(
                    fold_number,

                    test_name,

                    method,

                    training_names,

                    validation_names,

                    test_sequence,

                    exc,
                )

            all_results.append(
                result
            )

            if (
                result.status
                == "ok"
            ):

                print(
                    f"      TEST ATE = "
                    f"{result.ate_rmse_m:.3f} m"
                )

                print(
                    f"      RPE1     = "
                    f"{result.rpe_1s_trans_rmse_m:.3f} m"
                )

                print(
                    f"      heading  = "
                    f"{result.heading_mae_deg:.2f} deg"
                )

        # -------------------------------------------------------------------
        # Write partial results after every fold.
        #
        # If a long run is interrupted, completed folds remain saved.
        # -------------------------------------------------------------------

        write_results_csv(
            output_dir
            / "loso_results.csv",

            all_results,
        )

        (
            output_dir
            / "fold_splits.json"
        ).write_text(
            json.dumps(
                fold_splits,
                indent=2,
            ),
            encoding="utf-8",
        )

        write_summary(
            output_dir
            / "loso_summary.json",

            all_results,

            fold_splits,

            args,

            device,
        )

    # =========================================================================
    # Done
    # =========================================================================

    print_final_table(
        all_results,

        list(
            args.methods
        ),
    )

    print()

    print(
        "Wrote:"
    )

    print(
        f"  "
        f"{output_dir / 'loso_results.csv'}"
    )

    print(
        f"  "
        f"{output_dir / 'loso_summary.json'}"
    )

    print(
        f"  "
        f"{output_dir / 'fold_splits.json'}"
    )

    print()

    print(
        "Important: headline numbers must come "
        "from the outer test folds only."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )