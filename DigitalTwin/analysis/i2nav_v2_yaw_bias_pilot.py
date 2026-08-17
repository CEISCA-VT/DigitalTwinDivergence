#!/usr/bin/env python3

"""
i2Nav Twin V2 yaw-bias pilot
============================

Physical motivation
-------------------
Frozen-V1 residual analysis showed that parking02's catastrophic long-term
divergence is dominated by small persistent yaw-rate error.

V1 can predict the shape of the yaw correction extremely well while leaving
a tiny DC offset. That offset integrates into large heading drift and global
ATE.

Twin V2 therefore decomposes yaw correction into:

    delta_omega =
        explicit_bias
        +
        fast_dynamic_residual

where:

    explicit_bias
        is allowed to carry persistent / low-frequency correction

    fast_dynamic_residual
        is intended to model transient turning/slip dynamics

This version adds an explicit mean-neutrality loss to prevent the fast
dynamic residual from silently carrying its own persistent DC bias:

    mean(delta_omega_fast) -> 0

over 5 s, 10 s, and 30 s windows.

Sensor inputs remain:
    ODO + IMU only.

Ground truth is used only during training/validation and evaluation.
The held-out test sequence is never used for checkpoint selection.

Frozen V1 remains read-only.

Pilot folds:
    parking02
    parking01
    street00

Seeds:
    42
    1042
    2042
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
)


# =============================================================================
# Pilot configuration
# =============================================================================

PILOT_FOLDS = (
    "parking02",
    "parking01",
    "street00",
)

BASE_SEEDS = (
    42,
    1042,
    2042,
)


# =============================================================================
# Generic utilities
# =============================================================================

def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:

    if (
        requested.lower().startswith("cuda")
        and not torch.cuda.is_available()
    ):

        print(
            "[warning] CUDA requested but unavailable; using CPU."
        )

        return torch.device("cpu")

    return torch.device(requested)


def wrap_tensor(angle: torch.Tensor) -> torch.Tensor:

    return torch.atan2(
        torch.sin(angle),
        torch.cos(angle),
    )


def safe_mean(values) -> float:

    array = np.asarray(
        list(values),
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return float("nan")

    return float(
        np.mean(array)
    )


def safe_std(values) -> float:

    array = np.asarray(
        list(values),
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) <= 1:
        return 0.0

    return float(
        np.std(
            array,
            ddof=1,
        )
    )


def write_json(
    path: Path,
    value: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
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

    fields = []
    seen = set()

    for row in rows:

        for key in row:

            if key not in seen:

                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def read_csv(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
        errors="ignore",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def original_default_args(original):

    old_argv = sys.argv[:]

    try:

        sys.argv = [
            "i2nav_loso_ablation.py"
        ]

        return original.parse_args()

    finally:

        sys.argv = old_argv


# =============================================================================
# Validate exact original PreparedSequence structure
# =============================================================================

def validate_sequence(sequence) -> None:

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
        "features",
        "target_corrections",
        "gnss",
        "odo_source",
    )

    missing = [

        name

        for name in required

        if not hasattr(
            sequence,
            name,
        )
    ]

    if missing:

        raise RuntimeError(
            f"{getattr(sequence, 'name', 'sequence')}: "
            f"missing PreparedSequence fields {missing}. "
            f"Actual fields={list(vars(sequence).keys())}"
        )

    n = len(
        sequence.grid
    )

    features = np.asarray(
        sequence.features
    )

    targets = np.asarray(
        sequence.target_corrections
    )

    if len(features) != n:

        raise RuntimeError(
            f"{sequence.name}: feature/grid length mismatch."
        )

    if targets.shape != (
        n,
        2,
    ):

        raise RuntimeError(
            f"{sequence.name}: target_corrections has shape "
            f"{targets.shape}; expected ({n}, 2)."
        )


# =============================================================================
# Train-only feature normalization
# =============================================================================

def feature_normalization(
    prepared,
    training_names,
):

    features = np.concatenate(
        [

            np.asarray(
                prepared[
                    name
                ].features,
                dtype=np.float32,
            )

            for name
            in training_names
        ],
        axis=0,
    )

    mean = np.mean(
        features,
        axis=0,
    ).astype(
        np.float32
    )

    std = np.std(
        features,
        axis=0,
    ).astype(
        np.float32
    )

    std = np.maximum(
        std,
        1e-4,
    )

    return (
        mean,
        std,
    )


# =============================================================================
# V1-compatible overlapping windows
# =============================================================================

def sliding_windows(
    features: np.ndarray,
    window: int,
) -> np.ndarray:

    features = np.ascontiguousarray(
        features,
        dtype=np.float32,
    )

    if len(features) < window:

        raise RuntimeError(
            "Sequence shorter than requested GRU window."
        )

    view = (
        np.lib.stride_tricks
        .sliding_window_view(
            features,
            window_shape=window,
            axis=0,
        )
    )

    if (
        view.ndim == 3
        and
        view.shape[1]
        == features.shape[1]
    ):

        view = np.transpose(
            view,
            (
                0,
                2,
                1,
            ),
        )

    if (
        view.ndim != 3
        or
        view.shape[1] != window
    ):

        raise RuntimeError(
            f"Unexpected sliding-window shape {view.shape}."
        )

    return np.ascontiguousarray(
        view,
        dtype=np.float32,
    )


# =============================================================================
# Cached sequence representation
# =============================================================================

@dataclass
class SequenceCache:

    name: str

    windows: np.ndarray

    target: np.ndarray

    odo_speed: np.ndarray

    imu_yaw_rate: np.ndarray

    gt_x: np.ndarray

    gt_y: np.ndarray

    gt_heading: np.ndarray

    grid: np.ndarray


def build_cache(
    sequence,
    feature_mean,
    feature_std,
    window,
):

    normalized = (

        np.asarray(
            sequence.features,
            dtype=np.float32,
        )

        -

        feature_mean[
            None,
            :
        ]

    ) / feature_std[
        None,
        :
    ]

    return SequenceCache(

        name=
            sequence.name,

        windows=
            sliding_windows(
                normalized,
                window,
            ),

        target=
            np.asarray(
                sequence.target_corrections,
                dtype=np.float32,
            ),

        odo_speed=
            np.asarray(
                sequence.odo_speed,
                dtype=np.float32,
            ),

        imu_yaw_rate=
            np.asarray(
                sequence.imu_yaw_rate,
                dtype=np.float32,
            ),

        gt_x=
            np.asarray(
                sequence.gt_x,
                dtype=np.float32,
            ),

        gt_y=
            np.asarray(
                sequence.gt_y,
                dtype=np.float32,
            ),

        gt_heading=
            np.asarray(
                sequence.gt_heading,
                dtype=np.float32,
            ),

        grid=
            np.asarray(
                sequence.grid,
                dtype=np.float64,
            ),
    )


# =============================================================================
# Long contiguous training chunks
# =============================================================================

class ChunkDataset(Dataset):

    def __init__(
        self,
        caches,
        names,
        *,
        window,
        chunk_steps,
        stride,
    ):

        self.caches = caches
        self.window = int(window)
        self.chunk_steps = int(chunk_steps)

        self.items = []

        for name in names:

            cache = caches[
                name
            ]

            n = len(
                cache.grid
            )

            first = (
                window
                - 1
            )

            last = (
                n
                - chunk_steps
                + 1
            )

            for start in range(
                first,
                last,
                stride,
            ):

                self.items.append(
                    (
                        name,
                        start,
                    )
                )

        if not self.items:

            raise RuntimeError(
                "No ChunkDataset items were generated."
            )


    def __len__(self):

        return len(
            self.items
        )


    def __getitem__(self, index):

        (
            name,
            start,
        ) = self.items[
            index
        ]

        cache = self.caches[
            name
        ]

        window_start = (

            start

            -

            (
                self.window
                - 1
            )
        )

        end = (
            start
            +
            self.chunk_steps
        )

        return {

            "windows":
                torch.from_numpy(

                    cache.windows[
                        window_start:
                        window_start
                        + self.chunk_steps
                    ]
                ),

            "target":
                torch.from_numpy(

                    cache.target[
                        start:end
                    ]
                ),

            "odo_speed":
                torch.from_numpy(

                    cache.odo_speed[
                        start:end
                    ]
                ),

            "imu_yaw_rate":
                torch.from_numpy(

                    cache.imu_yaw_rate[
                        start:end
                    ]
                ),

            "gt_x":
                torch.from_numpy(

                    cache.gt_x[
                        start:end
                    ]
                ),

            "gt_y":
                torch.from_numpy(

                    cache.gt_y[
                        start:end
                    ]
                ),

            "gt_heading":
                torch.from_numpy(

                    cache.gt_heading[
                        start:end
                    ]
                ),
        }


# =============================================================================
# Twin V2
# =============================================================================

class V2YawBiasGRU(nn.Module):

    """
    Physics-motivated correction model.

        corrected velocity:
            v* = v_odo + delta_v

        corrected yaw rate:
            omega* =
                omega_imu
                + b_omega
                + delta_omega_fast

    b_omega:
        explicit persistent/slow correction component

    delta_omega_fast:
        transient residual component

    The loss explicitly forces delta_omega_fast toward zero mean over
    long windows so it cannot become a hidden bias head.
    """

    def __init__(
        self,
        input_dim,
        *,
        hidden_size=64,
        num_layers=2,
        dropout=0.10,
        dv_limit=0.15,
        dw_dynamic_limit=0.020,
        yaw_bias_limit=0.005,
    ):

        super().__init__()

        self.dv_limit = float(
            dv_limit
        )

        self.dw_dynamic_limit = float(
            dw_dynamic_limit
        )

        self.yaw_bias_limit = float(
            yaw_bias_limit
        )

        self.gru = nn.GRU(

            input_size=
                input_dim,

            hidden_size=
                hidden_size,

            num_layers=
                num_layers,

            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),

            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.dv_head = nn.Linear(
            hidden_size,
            1,
        )

        self.dw_dynamic_head = nn.Linear(
            hidden_size,
            1,
        )

        self.yaw_bias_head = nn.Linear(
            hidden_size,
            1,
        )

        nn.init.zeros_(
            self.dv_head.bias
        )

        nn.init.zeros_(
            self.dw_dynamic_head.bias
        )

        nn.init.zeros_(
            self.yaw_bias_head.bias
        )

        # Keep explicit bias initially near neutral.
        nn.init.normal_(
            self.yaw_bias_head.weight,
            mean=0.0,
            std=1e-3,
        )


    def forward(
        self,
        windows,
    ):

        if windows.ndim == 4:

            (
                batch,
                horizon,
                window,
                features,
            ) = windows.shape

            x = windows.reshape(

                batch
                * horizon,

                window,

                features,
            )

            restore = (
                batch,
                horizon,
            )

        elif windows.ndim == 3:

            x = windows
            restore = None

        else:

            raise RuntimeError(
                f"Unexpected model input shape "
                f"{windows.shape}."
            )

        (
            _,
            hidden,
        ) = self.gru(
            x
        )

        latent = self.norm(
            hidden[-1]
        )

        dv = (

            self.dv_limit

            * torch.tanh(

                self.dv_head(
                    latent
                )
            )
        )

        dw_dynamic = (

            self.dw_dynamic_limit

            * torch.tanh(

                self.dw_dynamic_head(
                    latent
                )
            )
        )

        yaw_bias = (

            self.yaw_bias_limit

            * torch.tanh(

                self.yaw_bias_head(
                    latent
                )
            )
        )

        if restore is not None:

            (
                batch,
                horizon,
            ) = restore

            dv = dv.reshape(
                batch,
                horizon,
            )

            dw_dynamic = (
                dw_dynamic.reshape(
                    batch,
                    horizon,
                )
            )

            yaw_bias = (
                yaw_bias.reshape(
                    batch,
                    horizon,
                )
            )

        else:

            dv = dv.squeeze(
                -1
            )

            dw_dynamic = (
                dw_dynamic.squeeze(
                    -1
                )
            )

            yaw_bias = (
                yaw_bias.squeeze(
                    -1
                )
            )

        dw = (

            yaw_bias

            +

            dw_dynamic
        )

        return {

            "dv":
                dv,

            "dw_dynamic":
                dw_dynamic,

            "yaw_bias":
                yaw_bias,

            "dw":
                dw,
        }


# =============================================================================
# Differentiable local trajectory propagation
# =============================================================================

def propagate_chunk(
    corrected_v,
    corrected_w,
    gt_x,
    gt_y,
    gt_heading,
    dt,
):

    (
        batch,
        horizon,
    ) = corrected_v.shape

    x = gt_x[
        :,
        0
    ]

    y = gt_y[
        :,
        0
    ]

    heading = gt_heading[
        :,
        0
    ]

    xs = [
        x
    ]

    ys = [
        y
    ]

    headings = [
        heading
    ]

    for k in range(
        1,
        horizon,
    ):

        v = 0.5 * (

            corrected_v[
                :,
                k - 1
            ]

            +

            corrected_v[
                :,
                k
            ]
        )

        omega = 0.5 * (

            corrected_w[
                :,
                k - 1
            ]

            +

            corrected_w[
                :,
                k
            ]
        )

        heading_mid = (

            heading

            +

            0.5
            * omega
            * dt
        )

        x = (

            x

            +

            v
            * torch.cos(
                heading_mid
            )
            * dt
        )

        y = (

            y

            +

            v
            * torch.sin(
                heading_mid
            )
            * dt
        )

        heading = (

            heading

            +

            omega
            * dt
        )

        xs.append(
            x
        )

        ys.append(
            y
        )

        headings.append(
            heading
        )

    return (

        torch.stack(
            xs,
            dim=1,
        ),

        torch.stack(
            ys,
            dim=1,
        ),

        torch.stack(
            headings,
            dim=1,
        ),
    )


# =============================================================================
# V2 loss
# =============================================================================

def compute_loss(
    model,
    batch,
    *,
    dt,
    point_weight,
    trajectory_weight,
    bias_weight,
    bias_smooth_weight,
    dynamic_residual_weight,
    fast_mean_weight,
):

    prediction = model(
        batch[
            "windows"
        ]
    )

    dv = prediction[
        "dv"
    ]

    dw = prediction[
        "dw"
    ]

    yaw_bias = prediction[
        "yaw_bias"
    ]

    dw_dynamic = prediction[
        "dw_dynamic"
    ]

    target = batch[
        "target"
    ]

    true_dv = target[
        :,
        :,
        0
    ]

    true_dw = target[
        :,
        :,
        1
    ]

    # =========================================================================
    # 1. Pointwise correction accuracy
    # =========================================================================

    point_dv = torch.mean(

        (
            (
                dv
                -
                true_dv
            )

            /
            0.05
        ) ** 2
    )

    point_dw = torch.mean(

        (
            (
                dw
                -
                true_dw
            )

            /
            0.03
        ) ** 2
    )

    point_loss = (

        point_dv

        +

        point_dw
    )

    # =========================================================================
    # Corrected motion
    # =========================================================================

    corrected_v = (

        batch[
            "odo_speed"
        ]

        +

        dv
    )

    corrected_w = (

        batch[
            "imu_yaw_rate"
        ]

        +

        dw
    )

    (
        pred_x,
        pred_y,
        pred_heading,
    ) = propagate_chunk(

        corrected_v,

        corrected_w,

        batch[
            "gt_x"
        ],

        batch[
            "gt_y"
        ],

        batch[
            "gt_heading"
        ],

        dt,
    )

    # =========================================================================
    # 2. 1 / 5 / 10 second trajectory fidelity
    # =========================================================================

    trajectory_terms = []

    for seconds in (
        1.0,
        5.0,
        10.0,
    ):

        index = int(
            round(
                seconds
                / dt
            )
        )

        if index >= pred_x.shape[
            1
        ]:

            continue

        position_sq = (

            (
                pred_x[
                    :,
                    index
                ]

                -

                batch[
                    "gt_x"
                ][
                    :,
                    index
                ]
            ) ** 2

            +

            (
                pred_y[
                    :,
                    index
                ]

                -

                batch[
                    "gt_y"
                ][
                    :,
                    index
                ]
            ) ** 2
        )

        heading_error = wrap_tensor(

            pred_heading[
                :,
                index
            ]

            -

            batch[
                "gt_heading"
            ][
                :,
                index
            ]
        )

        heading_scale = math.radians(
            5.0
        )

        trajectory_terms.append(

            torch.mean(
                position_sq
            )

            +

            torch.mean(

                (
                    heading_error
                    /
                    heading_scale
                ) ** 2
            )
        )

    trajectory_loss = torch.stack(
        trajectory_terms
    ).mean()

    # =========================================================================
    # 3. Remaining persistent signed yaw error
    # =========================================================================

    remaining_dw = (

        true_dw

        -

        dw
    )

    persistent_terms = []

    for seconds in (
        5.0,
        10.0,
        30.0,
    ):

        steps = min(

            int(
                round(
                    seconds
                    / dt
                )
            ),

            remaining_dw.shape[
                1
            ],
        )

        mean_signed_error = torch.mean(

            remaining_dw[
                :,
                :steps
            ],

            dim=1,
        )

        persistent_terms.append(

            torch.mean(

                (
                    mean_signed_error
                    /
                    0.001
                ) ** 2
            )
        )

    persistent_bias_loss = torch.stack(
        persistent_terms
    ).mean()

    # =========================================================================
    # 4. Explicit bias smoothness
    # =========================================================================

    bias_delta = (

        yaw_bias[
            :,
            1:
        ]

        -

        yaw_bias[
            :,
            :-1
        ]
    )

    bias_smooth_loss = torch.mean(

        (
            bias_delta
            /
            0.0005
        ) ** 2
    )

    # =========================================================================
    # 5. Keep fast residual magnitude modest
    # =========================================================================

    dynamic_residual_loss = torch.mean(

        (
            dw_dynamic
            /
            0.01
        ) ** 2
    )

    # =========================================================================
    # 6. NEW: Fast-residual mean neutrality
    #
    # The fast head is not allowed to become a hidden persistent-bias head.
    #
    # For each 5 / 10 / 30 second interval:
    #
    #        mean(delta_omega_fast) -> 0
    #
    # Scale 0.001 rad/s means persistent offsets on the order already shown
    # to create large long-term drift receive a meaningful penalty.
    # =========================================================================

    fast_mean_terms = []

    for seconds in (
        5.0,
        10.0,
        30.0,
    ):

        steps = min(

            int(
                round(
                    seconds
                    / dt
                )
            ),

            dw_dynamic.shape[
                1
            ],
        )

        fast_mean = torch.mean(

            dw_dynamic[
                :,
                :steps
            ],

            dim=1,
        )

        fast_mean_terms.append(

            torch.mean(

                (
                    fast_mean
                    /
                    0.001
                ) ** 2
            )
        )

    fast_mean_loss = torch.stack(
        fast_mean_terms
    ).mean()

    # =========================================================================
    # Total
    # =========================================================================

    total = (

        point_weight
        * point_loss

        +

        trajectory_weight
        * trajectory_loss

        +

        bias_weight
        * persistent_bias_loss

        +

        bias_smooth_weight
        * bias_smooth_loss

        +

        dynamic_residual_weight
        * dynamic_residual_loss

        +

        fast_mean_weight
        * fast_mean_loss
    )

    parts = {

        "total":
            float(
                total
                .detach()
                .cpu()
            ),

        "point":
            float(
                point_loss
                .detach()
                .cpu()
            ),

        "trajectory":
            float(
                trajectory_loss
                .detach()
                .cpu()
            ),

        "persistent_bias":
            float(
                persistent_bias_loss
                .detach()
                .cpu()
            ),

        "bias_smooth":
            float(
                bias_smooth_loss
                .detach()
                .cpu()
            ),

        "dynamic_residual":
            float(
                dynamic_residual_loss
                .detach()
                .cpu()
            ),

        "fast_mean":
            float(
                fast_mean_loss
                .detach()
                .cpu()
            ),
    }

    return (
        total,
        parts,
    )


# =============================================================================
# Device helper
# =============================================================================

def move_batch(
    batch,
    device,
):

    return {

        key:
            value.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

        for (
            key,
            value,
        )
        in batch.items()
    }


# =============================================================================
# Normal chunk validation
# =============================================================================

def validation_loss(
    model,
    loader,
    device,
    loss_kwargs,
):

    model.eval()

    sums = {}
    count = 0

    with torch.no_grad():

        for batch in loader:

            batch = move_batch(
                batch,
                device,
            )

            (
                _,
                parts,
            ) = compute_loss(

                model,

                batch,

                **loss_kwargs,
            )

            for (
                key,
                value,
            ) in parts.items():

                sums[
                    key
                ] = (

                    sums.get(
                        key,
                        0.0,
                    )

                    +

                    value
                )

            count += 1

    return {

        key:
            value
            / max(
                count,
                1,
            )

        for (
            key,
            value,
        )
        in sums.items()
    }


# =============================================================================
# Training
# =============================================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    *,
    device,
    epochs,
    patience,
    learning_rate,
    weight_decay,
    loss_kwargs,
):

    """
    Return to the original V2 checkpoint policy:

        lowest validation chunk loss

    No extra heuristic validation guard.
    No open-loop validation trajectory evaluator.
    """

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=
            learning_rate,

        weight_decay=
            weight_decay,
    )

    best_state = None

    best_validation_loss = float(
        "inf"
    )

    history = []

    bad_epochs = 0

    for epoch in range(
        1,
        epochs + 1,
    ):

        # =====================================================================
        # Train
        # =====================================================================

        model.train()

        running = {}
        batch_count = 0

        for batch in train_loader:

            batch = move_batch(
                batch,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            (
                loss,
                parts,
            ) = compute_loss(

                model,

                batch,

                **loss_kwargs,
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            for (
                key,
                value,
            ) in parts.items():

                running[
                    key
                ] = (

                    running.get(
                        key,
                        0.0,
                    )

                    +

                    value
                )

            batch_count += 1

        train_mean = {

            key:
                value
                / max(
                    batch_count,
                    1,
                )

            for (
                key,
                value,
            )
            in running.items()
        }

        # =====================================================================
        # Validation
        # =====================================================================

        val_mean = validation_loss(

            model,

            validation_loader,

            device,

            loss_kwargs,
        )

        history_row = {

            "epoch":
                epoch,

            **{

                f"train_{key}":
                    value

                for (
                    key,
                    value,
                )
                in train_mean.items()
            },

            **{

                f"val_{key}":
                    value

                for (
                    key,
                    value,
                )
                in val_mean.items()
            },
        }

        history.append(
            history_row
        )

        print()

        print(
            f"      epoch {epoch:02d}  "
            f"train={train_mean['total']:.4f}  "
            f"val={val_mean['total']:.4f}  "
            f"bias={val_mean['persistent_bias']:.4f}  "
            f"fastmean={val_mean['fast_mean']:.4f}"
        )

        if (
            val_mean[
                "total"
            ]

            <

            best_validation_loss
            - 1e-6
        ):

            best_validation_loss = float(

                val_mean[
                    "total"
                ]
            )

            best_state = {

                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for (
                    key,
                    value,
                )
                in model.state_dict().items()
            }

            bad_epochs = 0

        else:

            bad_epochs += 1

        if (
            bad_epochs
            >= patience
        ):

            print(
                f"      early stopping: "
                f"best val={best_validation_loss:.4f}"
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No valid V2 checkpoint was produced."
        )

    return (
        best_state,
        history,
        best_validation_loss,
    )


# =============================================================================
# Full-sequence inference
# =============================================================================

def predict_sequence(
    model,
    sequence,
    *,
    feature_mean,
    feature_std,
    window,
    batch_size,
    device,
):

    model.eval()

    normalized = (

        np.asarray(
            sequence.features,
            dtype=np.float32,
        )

        -

        feature_mean[
            None,
            :
        ]

    ) / feature_std[
        None,
        :
    ]

    windows = sliding_windows(
        normalized,
        window,
    )

    chunks = {

        "dv":
            [],

        "dw":
            [],

        "yaw_bias":
            [],

        "dw_dynamic":
            [],
    }

    with torch.no_grad():

        for start in range(
            0,
            len(windows),
            batch_size,
        ):

            xb = (
                torch
                .from_numpy(

                    windows[
                        start:
                        start
                        + batch_size
                    ]
                )
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            output = model(
                xb
            )

            for key in chunks:

                chunks[
                    key
                ].append(

                    output[
                        key
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

    predicted = {

        key:
            np.concatenate(
                values
            ).astype(
                np.float32
            )

        for (
            key,
            values,
        )
        in chunks.items()
    }

    n = len(
        sequence.grid
    )

    offset = (
        window
        - 1
    )

    full = {

        key:
            np.zeros(
                n,
                dtype=np.float32,
            )

        for key
        in predicted
    }

    for key in predicted:

        full[
            key
        ][
            offset:
        ] = predicted[
            key
        ]

    full[
        "corrections"
    ] = (

        np.column_stack(
            [
                full[
                    "dv"
                ],

                full[
                    "dw"
                ],
            ]
        )
        .astype(
            np.float32
        )
    )

    return full


# =============================================================================
# Remaining yaw diagnostics
# =============================================================================

def yaw_diagnostics(
    sequence,
    prediction,
    window,
):

    start = (
        window
        - 1
    )

    true_dw = np.asarray(

        sequence.target_corrections[
            start:,
            1
        ],

        dtype=float,
    )

    predicted_dw = np.asarray(

        prediction[
            "dw"
        ][
            start:
        ],

        dtype=float,
    )

    fast_dw = np.asarray(

        prediction[
            "dw_dynamic"
        ][
            start:
        ],

        dtype=float,
    )

    explicit_bias = np.asarray(

        prediction[
            "yaw_bias"
        ][
            start:
        ],

        dtype=float,
    )

    residual = (

        true_dw

        -

        predicted_dw
    )

    mean_residual = float(
        np.mean(
            residual
        )
    )

    return {

        "remaining_yaw_bias_radps":
            mean_residual,

        "remaining_yaw_bias_deg_per_min":
            (

                mean_residual

                * 180.0
                / math.pi

                * 60.0
            ),

        "remaining_yaw_rmse_radps":
            float(

                np.sqrt(

                    np.mean(
                        residual**2
                    )
                )
            ),

        "mean_fast_yaw_residual_radps":
            float(
                np.mean(
                    fast_dw
                )
            ),

        "mean_fast_yaw_residual_deg_per_min":
            float(

                np.mean(
                    fast_dw
                )

                * 180.0
                / math.pi

                * 60.0
            ),

        "mean_explicit_yaw_bias_radps":
            float(
                np.mean(
                    explicit_bias
                )
            ),

        "mean_explicit_yaw_bias_deg_per_min":
            float(

                np.mean(
                    explicit_bias
                )

                * 180.0
                / math.pi

                * 60.0
            ),
    }


# =============================================================================
# Frozen manifest
# =============================================================================

def frozen_manifest_lookup(
    frozen_dir,
):

    path = (

        frozen_dir

        / "FROZEN_MANIFEST.json"
    )

    manifest = json.loads(

        path.read_text(
            encoding="utf-8"
        )
    )

    return {

        (
            run[
                "replicate"
            ],

            run[
                "test_sequence"
            ],
        ):
            run

        for run
        in manifest[
            "runs"
        ]
    }


# =============================================================================
# Frozen V1 metrics
# =============================================================================

def frozen_metric_lookup(
    frozen_dir,
):

    path = (

        frozen_dir

        / "canonical_metrics_per_run.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing frozen metrics:\n{path}"
        )

    result = {}

    for row in read_csv(
        path
    ):

        key = (

            row[
                "replicate"
            ],

            row[
                "test_sequence"
            ],
        )

        converted = {}

        for (
            field,
            value,
        ) in row.items():

            if value in (
                "",
                None,
            ):

                continue

            try:

                converted[
                    field
                ] = float(
                    value
                )

            except Exception:

                continue

        result[
            key
        ] = converted

    return result


# =============================================================================
# Frozen V1 alpha/Q predictions
# =============================================================================

def frozen_v1_alphas(
    original,
    sequence,
    frozen_dir,
    run_record,
    device,
    eval_batch_size,
):

    checkpoint_path = (

        frozen_dir

        / run_record[
            "frozen_checkpoint"
        ]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = original.AblationGRU(

        mode=
            "dual",

        input_dim=
            len(
                checkpoint[
                    "feature_mean"
                ]
            ),

        hidden_size=
            int(
                checkpoint[
                    "hidden_size"
                ]
            ),

        num_layers=
            int(
                checkpoint[
                    "num_layers"
                ]
            ),

        dropout=
            0.10,

        dv_limit=
            float(
                checkpoint[
                    "dv_limit"
                ]
            ),

        domega_limit=
            float(
                checkpoint[
                    "domega_limit"
                ]
            ),

        alpha_min=
            float(
                checkpoint[
                    "alpha_min"
                ]
            ),

        alpha_max=
            float(
                checkpoint[
                    "alpha_max"
                ]
            ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    model.eval()

    with torch.no_grad():

        (
            _,
            alphas,
        ) = original.predict_neural_sequence(

            model=
                model,

            sequence=
                sequence,

            feature_mean=
                np.asarray(
                    checkpoint[
                        "feature_mean"
                    ]
                ),

            feature_std=
                np.asarray(
                    checkpoint[
                        "feature_std"
                    ]
                ),

            window=
                int(
                    checkpoint[
                        "window"
                    ]
                ),

            batch_size=
                int(
                    eval_batch_size
                ),

            device=
                device,
        )

    del model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    return alphas


# =============================================================================
# Save correction trace
# =============================================================================

def save_prediction_trace(
    path,
    sequence,
    prediction,
):

    targets = np.asarray(
        sequence.target_corrections,
        dtype=float,
    )

    rows = []

    for k in range(
        len(
            sequence.grid
        )
    ):

        rows.append(
            {

                "time_s":
                    float(
                        sequence.grid[
                            k
                        ]
                    ),

                "true_delta_v_mps":
                    float(
                        targets[
                            k,
                            0
                        ]
                    ),

                "pred_delta_v_mps":
                    float(
                        prediction[
                            "dv"
                        ][
                            k
                        ]
                    ),

                "true_delta_omega_radps":
                    float(
                        targets[
                            k,
                            1
                        ]
                    ),

                "pred_total_delta_omega_radps":
                    float(
                        prediction[
                            "dw"
                        ][
                            k
                        ]
                    ),

                "pred_explicit_yaw_bias_radps":
                    float(
                        prediction[
                            "yaw_bias"
                        ][
                            k
                        ]
                    ),

                "pred_fast_yaw_residual_radps":
                    float(
                        prediction[
                            "dw_dynamic"
                        ][
                            k
                        ]
                    ),

                "remaining_yaw_error_radps":
                    float(

                        targets[
                            k,
                            1
                        ]

                        -

                        prediction[
                            "dw"
                        ][
                            k
                        ]
                    ),
            }
        )

    write_csv(
        path,
        rows,
    )


# =============================================================================
# Aggregate results
# =============================================================================

def aggregate_results(
    rows,
):

    output = []

    for sequence in PILOT_FOLDS:

        subset = [

            row

            for row in rows

            if (
                row[
                    "test_sequence"
                ]
                == sequence
            )
        ]

        if not subset:
            continue

        v1_values = [

            float(
                row[
                    "v1_ate_rmse_m"
                ]
            )

            for row
            in subset
        ]

        v2_values = [

            float(
                row[
                    "v2_ate_rmse_m"
                ]
            )

            for row
            in subset
        ]

        v1_mean = safe_mean(
            v1_values
        )

        v2_mean = safe_mean(
            v2_values
        )

        v1_std = safe_std(
            v1_values
        )

        v2_std = safe_std(
            v2_values
        )

        output.append(
            {

                "test_sequence":
                    sequence,

                "n_seeds":
                    len(
                        subset
                    ),

                "v1_ate_mean_m":
                    v1_mean,

                "v1_ate_std_m":
                    v1_std,

                "v1_ate_range_m":
                    float(

                        max(
                            v1_values
                        )

                        -

                        min(
                            v1_values
                        )
                    ),

                "v2_ate_mean_m":
                    v2_mean,

                "v2_ate_std_m":
                    v2_std,

                "v2_ate_range_m":
                    float(

                        max(
                            v2_values
                        )

                        -

                        min(
                            v2_values
                        )
                    ),

                "mean_ate_change_pct":
                    (

                        100.0

                        * (
                            v2_mean
                            -
                            v1_mean
                        )

                        / v1_mean
                    ),

                "seed_std_reduction_pct":
                    (

                        100.0

                        * (
                            v1_std
                            -
                            v2_std
                        )

                        / max(
                            v1_std,
                            1e-12,
                        )
                    ),

                "mean_abs_remaining_yaw_bias_deg_min":
                    safe_mean(

                        abs(
                            row[
                                "v2_remaining_yaw_bias_deg_per_min"
                            ]
                        )

                        for row
                        in subset
                    ),

                "mean_abs_fast_component_deg_min":
                    safe_mean(

                        abs(
                            row[
                                "v2_mean_fast_yaw_residual_deg_per_min"
                            ]
                        )

                        for row
                        in subset
                    ),
            }
        )

    return output


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
            "results/i2nav_v2_yaw_bias_pilot"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--train-stride",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--validation-stride",
        type=int,
        default=100,
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

    parser.add_argument(
        "--dv-limit",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--dw-dynamic-limit",
        type=float,
        default=0.020,
    )

    # Restored from the original successful V2 pilot.
    parser.add_argument(
        "--yaw-bias-limit",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--point-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--trajectory-weight",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--bias-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--bias-smooth-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--dynamic-residual-weight",
        type=float,
        default=0.01,
    )

    # NEW.
    #
    # Strong enough to discourage DC leakage into the fast residual,
    # but not so large that it dominates the existing physical losses.
    parser.add_argument(
        "--fast-mean-weight",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    root = args.root.resolve()

    frozen_dir = (
        args.frozen_dir.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    if not root.exists():

        raise FileNotFoundError(
            f"Dataset root not found:\n"
            f"{root}"
        )

    if not frozen_dir.exists():

        raise FileNotFoundError(
            f"Frozen V1 directory not found:\n"
            f"{frozen_dir}"
        )

    if output_dir == frozen_dir:

        raise RuntimeError(
            "Refusing to write into frozen V1 directory."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    # =========================================================================
    # Import exact original V1 implementation
    # =========================================================================

    original = importlib.import_module(
        "DigitalTwin.analysis.i2nav_loso_ablation"
    )

    defaults = original_default_args(
        original
    )

    rate = float(
        defaults.rate_hz
    )

    dt = (
        1.0
        / rate
    )

    chunk_steps = (

        int(
            round(
                args.chunk_seconds
                * rate
            )
        )

        + 1
    )

    if args.chunk_seconds < 30.0:

        raise RuntimeError(
            "chunk-seconds must be >= 30."
        )

    print()
    print("=" * 100)
    print("TWIN V2 YAW-BIAS PILOT — FAST-RESIDUAL MEAN NEUTRALITY")
    print("=" * 100)
    print()

    print(
        f"Dataset          : {root}"
    )

    print(
        f"Frozen V1        : {frozen_dir}"
    )

    print(
        f"Output           : {output_dir}"
    )

    print(
        f"Device           : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU              : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"Rate             : {rate:.3f} Hz"
    )

    print(
        f"Yaw bias limit   : {args.yaw_bias_limit:.6f} rad/s"
    )

    print(
        f"Fast mean weight : {args.fast_mean_weight:.3f}"
    )

    print(
        f"Pilot folds      : {PILOT_FOLDS}"
    )

    print()

    # =========================================================================
    # Exact V1 preprocessing
    # =========================================================================

    discovered_list = (
        original.discover_files(
            root
        )
    )

    discovered = {

        item.name:
            item

        for item
        in discovered_list
    }

    prepared = {}

    print(
        "Preparing exact V1 sequences..."
    )

    for name in original.SEQUENCES:

        if name not in discovered:

            raise RuntimeError(
                f"Dataset discovery did not find {name}."
            )

        sequence = original.prepare_sequence(

            discovered[
                name
            ],

            hz=
                defaults.rate_hz,

            imu_yaw_sign=
                defaults.imu_yaw_sign,

            gnss_sigma_max_m=
                defaults.gnss_sigma_max_m,

            gnss_anchor_count=
                defaults.gnss_anchor_count,
        )

        validate_sequence(
            sequence
        )

        prepared[
            name
        ] = sequence

        print(
            f"  {name:<12} "
            f"{len(sequence.grid):6d} samples"
        )

    manifest_lookup = (
        frozen_manifest_lookup(
            frozen_dir
        )
    )

    metric_lookup = (
        frozen_metric_lookup(
            frozen_dir
        )
    )

    results = []

    # =========================================================================
    # Pilot folds
    # =========================================================================

    for test_name in PILOT_FOLDS:

        print()
        print("=" * 100)
        print(
            f"FOLD {test_name}"
        )
        print("=" * 100)

        (
            training_names,
            validation_names,
        ) = original.build_fold_split(

            test_name,

            int(
                defaults.validation_count
            ),
        )

        print(
            f"Train      : {training_names}"
        )

        print(
            f"Validation : {validation_names}"
        )

        print(
            f"Test       : {test_name}"
        )

        # =====================================================================
        # Train-only normalization
        # =====================================================================

        (
            feature_mean,
            feature_std,
        ) = feature_normalization(

            prepared,

            training_names,
        )

        required_names = set(

            training_names

            +

            validation_names

            +

            [
                test_name
            ]
        )

        caches = {

            name:
                build_cache(

                    prepared[
                        name
                    ],

                    feature_mean,

                    feature_std,

                    args.window,
                )

            for name
            in required_names
        }

        train_dataset = ChunkDataset(

            caches,

            training_names,

            window=
                args.window,

            chunk_steps=
                chunk_steps,

            stride=
                args.train_stride,
        )

        validation_dataset = ChunkDataset(

            caches,

            validation_names,

            window=
                args.window,

            chunk_steps=
                chunk_steps,

            stride=
                args.validation_stride,
        )

        print(
            f"Training chunks   : "
            f"{len(train_dataset)}"
        )

        print(
            f"Validation chunks : "
            f"{len(validation_dataset)}"
        )

        # =====================================================================
        # Three seeds
        # =====================================================================

        for (
            replicate_number,
            base_seed,
        ) in enumerate(
            BASE_SEEDS,
            start=1,
        ):

            replicate = (

                f"replicate_"
                f"{replicate_number:02d}_"
                f"base{base_seed}"
            )

            actual_seed = (

                base_seed

                +

                original.SEQUENCES.index(
                    test_name
                )
                * 100

                +

                23
            )

            seed_everything(
                actual_seed
            )

            print()
            print(
                f"  [{replicate_number}/3] "
                f"{replicate} "
                f"seed={actual_seed}"
            )

            generator = (
                torch.Generator()
            )

            generator.manual_seed(
                actual_seed
            )

            train_loader = DataLoader(

                train_dataset,

                batch_size=
                    args.batch_size,

                shuffle=True,

                num_workers=0,

                pin_memory=(
                    device.type
                    == "cuda"
                ),

                generator=
                    generator,
            )

            validation_loader = DataLoader(

                validation_dataset,

                batch_size=
                    args.batch_size,

                shuffle=False,

                num_workers=0,

                pin_memory=(
                    device.type
                    == "cuda"
                ),
            )

            model = V2YawBiasGRU(

                input_dim=
                    int(
                        prepared[
                            test_name
                        ]
                        .features
                        .shape[1]
                    ),

                hidden_size=
                    args.hidden_size,

                num_layers=
                    args.num_layers,

                dropout=
                    args.dropout,

                dv_limit=
                    args.dv_limit,

                dw_dynamic_limit=
                    args.dw_dynamic_limit,

                yaw_bias_limit=
                    args.yaw_bias_limit,
            ).to(
                device
            )

            loss_kwargs = {

                "dt":
                    dt,

                "point_weight":
                    args.point_weight,

                "trajectory_weight":
                    args.trajectory_weight,

                "bias_weight":
                    args.bias_weight,

                "bias_smooth_weight":
                    args.bias_smooth_weight,

                "dynamic_residual_weight":
                    args.dynamic_residual_weight,

                "fast_mean_weight":
                    args.fast_mean_weight,
            }

            start_time = (
                time.time()
            )

            (
                best_state,
                history,
                best_validation_loss,
            ) = train_model(

                model,

                train_loader,

                validation_loader,

                device=
                    device,

                epochs=
                    args.epochs,

                patience=
                    args.patience,

                learning_rate=
                    args.learning_rate,

                weight_decay=
                    args.weight_decay,

                loss_kwargs=
                    loss_kwargs,
            )

            elapsed = (
                time.time()
                -
                start_time
            )

            model.load_state_dict(
                best_state,
                strict=True,
            )

            model.eval()

            fold_number = (

                original.SEQUENCES.index(
                    test_name
                )

                + 1
            )

            run_dir = (

                output_dir

                / replicate

                / (
                    f"fold_"
                    f"{fold_number:02d}_"
                    f"{test_name}"
                )
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            write_csv(

                run_dir
                / "training_history.csv",

                history,
            )

            torch.save(

                {

                    "schema":
                        "i2nav_twin_v2_fast_mean_neutral",

                    "state_dict":
                        best_state,

                    "feature_mean":
                        feature_mean,

                    "feature_std":
                        feature_std,

                    "window":
                        args.window,

                    "hidden_size":
                        args.hidden_size,

                    "num_layers":
                        args.num_layers,

                    "dropout":
                        args.dropout,

                    "dv_limit":
                        args.dv_limit,

                    "dw_dynamic_limit":
                        args.dw_dynamic_limit,

                    "yaw_bias_limit":
                        args.yaw_bias_limit,

                    "fast_mean_weight":
                        args.fast_mean_weight,

                    "best_validation_loss":
                        best_validation_loss,

                    "base_seed":
                        base_seed,

                    "actual_seed":
                        actual_seed,

                    "test_sequence":
                        test_name,

                    "training_names":
                        training_names,

                    "validation_names":
                        validation_names,
                },

                run_dir
                / "v2_yaw_bias.pt",
            )

            # =================================================================
            # Test prediction
            # =================================================================

            test_sequence = prepared[
                test_name
            ]

            prediction = predict_sequence(

                model,

                test_sequence,

                feature_mean=
                    feature_mean,

                feature_std=
                    feature_std,

                window=
                    args.window,

                batch_size=
                    int(
                        defaults.eval_batch_size
                    ),

                device=
                    device,
            )

            save_prediction_trace(

                run_dir
                / "v2_prediction_trace.csv",

                test_sequence,

                prediction,
            )

            # =================================================================
            # Keep frozen V1 uncertainty/Q behavior
            # =================================================================

            frozen_key = (

                replicate,
                test_name,
            )

            if frozen_key not in manifest_lookup:

                raise RuntimeError(
                    f"Frozen manifest missing {frozen_key}."
                )

            alphas = frozen_v1_alphas(

                original,

                test_sequence,

                frozen_dir,

                manifest_lookup[
                    frozen_key
                ],

                device,

                int(
                    defaults.eval_batch_size
                ),
            )

            # =================================================================
            # Original V1 evaluator
            # =================================================================

            evaluation = original.evaluate_predictions(

                fold=
                    fold_number,

                method=
                    "v2_yaw_bias",

                sequence=
                    test_sequence,

                training_names=
                    training_names,

                validation_names=
                    validation_names,

                corrections=
                    prediction[
                        "corrections"
                    ],

                alphas=
                    alphas,

                args=
                    defaults,

                trajectory_path=(

                    run_dir

                    / "v2_evaluated_trajectory.csv"
                ),
            )

            metrics = vars(
                evaluation
            )

            diagnostic = yaw_diagnostics(

                test_sequence,

                prediction,

                args.window,
            )

            if frozen_key not in metric_lookup:

                raise RuntimeError(
                    f"Frozen metric table missing {frozen_key}."
                )

            v1 = metric_lookup[
                frozen_key
            ]

            result = {

                "replicate":
                    replicate,

                "base_seed":
                    base_seed,

                "actual_v2_seed":
                    actual_seed,

                "test_sequence":
                    test_name,

                "training_seconds":
                    elapsed,

                "best_validation_loss":
                    best_validation_loss,

                # V1 ----------------------------------------------------------

                "v1_ate_rmse_m":
                    float(
                        v1[
                            "ate_rmse_m"
                        ]
                    ),

                "v1_heading_mae_deg":
                    float(
                        v1[
                            "heading_mae_deg"
                        ]
                    ),

                "v1_rpe_1s_m":
                    float(
                        v1[
                            "rpe_1s_trans_rmse_m"
                        ]
                    ),

                "v1_rpe_5s_m":
                    float(
                        v1[
                            "rpe_5s_trans_rmse_m"
                        ]
                    ),

                "v1_rpe_10s_m":
                    float(
                        v1[
                            "rpe_10s_trans_rmse_m"
                        ]
                    ),

                # V2 ----------------------------------------------------------

                "v2_ate_rmse_m":
                    float(
                        metrics[
                            "ate_rmse_m"
                        ]
                    ),

                "v2_heading_mae_deg":
                    float(
                        metrics[
                            "heading_mae_deg"
                        ]
                    ),

                "v2_rpe_1s_m":
                    float(
                        metrics[
                            "rpe_1s_trans_rmse_m"
                        ]
                    ),

                "v2_rpe_5s_m":
                    float(
                        metrics[
                            "rpe_5s_trans_rmse_m"
                        ]
                    ),

                "v2_rpe_10s_m":
                    float(
                        metrics[
                            "rpe_10s_trans_rmse_m"
                        ]
                    ),

                "v2_remaining_yaw_bias_radps":
                    diagnostic[
                        "remaining_yaw_bias_radps"
                    ],

                "v2_remaining_yaw_bias_deg_per_min":
                    diagnostic[
                        "remaining_yaw_bias_deg_per_min"
                    ],

                "v2_remaining_yaw_rmse_radps":
                    diagnostic[
                        "remaining_yaw_rmse_radps"
                    ],

                "v2_mean_fast_yaw_residual_radps":
                    diagnostic[
                        "mean_fast_yaw_residual_radps"
                    ],

                "v2_mean_fast_yaw_residual_deg_per_min":
                    diagnostic[
                        "mean_fast_yaw_residual_deg_per_min"
                    ],

                "v2_mean_explicit_yaw_bias_radps":
                    diagnostic[
                        "mean_explicit_yaw_bias_radps"
                    ],

                "v2_mean_explicit_yaw_bias_deg_per_min":
                    diagnostic[
                        "mean_explicit_yaw_bias_deg_per_min"
                    ],
            }

            result[
                "ate_change_pct"
            ] = (

                100.0

                * (
                    result[
                        "v2_ate_rmse_m"
                    ]

                    -

                    result[
                        "v1_ate_rmse_m"
                    ]
                )

                /

                result[
                    "v1_ate_rmse_m"
                ]
            )

            results.append(
                result
            )

            write_json(

                run_dir
                / "run_summary.json",

                result,
            )

            # Save partial results after every run.
            write_csv(

                output_dir
                / "pilot_run_results.csv",

                results,
            )

            print()
            print(
                "      RESULT"
            )

            print(
                f"      V1 ATE = "
                f"{result['v1_ate_rmse_m']:.4f} m"
            )

            print(
                f"      V2 ATE = "
                f"{result['v2_ate_rmse_m']:.4f} m "
                f"({result['ate_change_pct']:+.1f}%)"
            )

            print(
                f"      V2 remaining yaw bias = "
                f"{result['v2_remaining_yaw_bias_deg_per_min']:+.3f} "
                f"deg/min"
            )

            print(
                f"      fast residual mean = "
                f"{result['v2_mean_fast_yaw_residual_deg_per_min']:+.3f} "
                f"deg/min"
            )

            print(
                f"      explicit bias mean = "
                f"{result['v2_mean_explicit_yaw_bias_deg_per_min']:+.3f} "
                f"deg/min"
            )

            del model

            if device.type == "cuda":

                torch.cuda.empty_cache()

    # =========================================================================
    # Aggregate pilot
    # =========================================================================

    aggregate = aggregate_results(
        results
    )

    write_csv(

        output_dir
        / "pilot_aggregate_results.csv",

        aggregate,
    )

    write_json(

        output_dir
        / "pilot_configuration.json",

        {

            "schema":
                "i2nav_twin_v2_fast_mean_neutral",

            "pilot_folds":
                PILOT_FOLDS,

            "base_seeds":
                BASE_SEEDS,

            "rate_hz":
                rate,

            "window":
                args.window,

            "chunk_seconds":
                args.chunk_seconds,

            "epochs":
                args.epochs,

            "patience":
                args.patience,

            "model":
                {

                    "hidden_size":
                        args.hidden_size,

                    "num_layers":
                        args.num_layers,

                    "dropout":
                        args.dropout,

                    "dv_limit":
                        args.dv_limit,

                    "dw_dynamic_limit":
                        args.dw_dynamic_limit,

                    "yaw_bias_limit":
                        args.yaw_bias_limit,
                },

            "loss":
                {

                    "point_weight":
                        args.point_weight,

                    "trajectory_weight":
                        args.trajectory_weight,

                    "bias_weight":
                        args.bias_weight,

                    "bias_smooth_weight":
                        args.bias_smooth_weight,

                    "dynamic_residual_weight":
                        args.dynamic_residual_weight,

                    "fast_mean_weight":
                        args.fast_mean_weight,
                },

            "fast_mean_neutrality":
                (
                    "Mean fast yaw residual is penalized toward zero over "
                    "5 s, 10 s, and 30 s horizons."
                ),

            "checkpoint_selection":
                (
                    "Lowest ordinary validation chunk loss. "
                    "No test information and no heuristic bias guard."
                ),

            "frozen_v1":
                (
                    "Frozen V1 checkpoints and uncertainty predictions are "
                    "read-only and never modified."
                ),
        },
    )

    print()
    print("=" * 100)
    print("V2 PILOT SUMMARY")
    print("=" * 100)
    print()

    for row in aggregate:

        print(
            f"{row['test_sequence']}"
        )

        print(
            "  V1 ATE mean/std/range : "
            f"{row['v1_ate_mean_m']:.4f} / "
            f"{row['v1_ate_std_m']:.4f} / "
            f"{row['v1_ate_range_m']:.4f} m"
        )

        print(
            "  V2 ATE mean/std/range : "
            f"{row['v2_ate_mean_m']:.4f} / "
            f"{row['v2_ate_std_m']:.4f} / "
            f"{row['v2_ate_range_m']:.4f} m"
        )

        print(
            "  V2 mean ATE change    : "
            f"{row['mean_ate_change_pct']:+.1f}%"
        )

        print(
            "  seed-std reduction    : "
            f"{row['seed_std_reduction_pct']:+.1f}%"
        )

        print(
            "  mean |remaining bias| : "
            f"{row['mean_abs_remaining_yaw_bias_deg_min']:.3f} "
            f"deg/min"
        )

        print(
            "  mean |fast DC|        : "
            f"{row['mean_abs_fast_component_deg_min']:.3f} "
            f"deg/min"
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