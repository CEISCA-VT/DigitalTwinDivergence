#!/usr/bin/env python3

"""
Hard replay test for frozen i2Nav Twin V1.

Checks:

1. Frozen source snapshots still match their SHA-256 hashes.
2. All 30 frozen checkpoints still match their hashes.
3. Fresh inference reproduces canonical trajectories.
4. Fresh evaluation reproduces canonical metrics.

NO TRAINING is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


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


# =============================================================================
# Utilities
# =============================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def read_csv(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="ignore",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# =============================================================================
# Original V1 defaults
# =============================================================================

def original_default_args(
    original_module,
):

    saved_argv = sys.argv[:]

    try:

        sys.argv = [
            "i2nav_loso_ablation.py"
        ]

        return (
            original_module.parse_args()
        )

    finally:
        sys.argv = saved_argv


# =============================================================================
# Dataset
# =============================================================================

def prepare_all(
    original_module,
    root: Path,
    original_args,
):

    discovered = {

        f.name: f

        for f in
        original_module.discover_files(
            root
        )
    }

    prepared = {}

    for sequence_name in SEQUENCES:

        if sequence_name not in discovered:

            raise RuntimeError(
                "Dataset discovery missing "
                f"{sequence_name}"
            )

        prepared[sequence_name] = (
            original_module.prepare_sequence(

                discovered[
                    sequence_name
                ],

                hz=
                    original_args.rate_hz,

                imu_yaw_sign=
                    original_args.imu_yaw_sign,

                gnss_sigma_max_m=
                    original_args.gnss_sigma_max_m,

                gnss_anchor_count=
                    original_args.gnss_anchor_count,
            )
        )

    return prepared


# =============================================================================
# Load frozen model
# =============================================================================

def load_model(
    original_module,
    checkpoint_path: Path,
    device: torch.device,
):

    # -------------------------------------------------------------------------
    # PyTorch >= 2.6 defaults weights_only=True.
    #
    # Our original research checkpoints contain NumPy objects.
    #
    # These are our own trusted frozen checkpoints, so full checkpoint loading
    # is intentional here.
    # -------------------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = original_module.AblationGRU(

        mode="dual",

        input_dim=len(
            original_module.FEATURE_NAMES
        ),

        hidden_size=int(
            checkpoint["hidden_size"]
        ),

        num_layers=int(
            checkpoint["num_layers"]
        ),

        dropout=0.10,

        dv_limit=float(
            checkpoint["dv_limit"]
        ),

        domega_limit=float(
            checkpoint["domega_limit"]
        ),

        alpha_min=float(
            checkpoint["alpha_min"]
        ),

        alpha_max=float(
            checkpoint["alpha_max"]
        ),
    )

    model = model.to(device)

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad_(
            False
        )

    return (
        model,
        checkpoint,
    )


# =============================================================================
# Numeric trajectory reader
# =============================================================================

def load_numeric_csv(
    path: Path,
):

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
        errors="ignore",
    ) as f:

        reader = csv.reader(f)

        header = next(reader)

        rows = []

        for row in reader:

            if not row:
                continue

            rows.append(
                [
                    float(value)
                    for value in row
                ]
            )

    return (
        header,
        np.asarray(
            rows,
            dtype=float,
        ),
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
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--pred-rtol",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--pred-atol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--metric-atol",
        type=float,
        default=1e-6,
    )

    args = parser.parse_args()

    frozen_dir = (
        args.frozen_dir.resolve()
    )

    data_root = (
        args.root.resolve()
    )

    if not frozen_dir.exists():

        raise FileNotFoundError(
            "Frozen V1 directory not found:\n"
            f"{frozen_dir}"
        )

    if not data_root.exists():

        raise FileNotFoundError(
            "Dataset directory not found:\n"
            f"{data_root}"
        )

    manifest_path = (
        frozen_dir
        / "FROZEN_MANIFEST.json"
    )

    source_hashes_path = (
        frozen_dir
        / "SOURCE_HASHES.json"
    )

    canonical_metrics_path = (
        frozen_dir
        / "canonical_metrics_per_run.csv"
    )

    for required in (
        manifest_path,
        source_hashes_path,
        canonical_metrics_path,
    ):

        if not required.exists():

            raise FileNotFoundError(
                "Frozen V1 artifact incomplete. "
                f"Missing:\n{required}"
            )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    source_hashes = json.loads(
        source_hashes_path.read_text(
            encoding="utf-8"
        )
    )

    canonical_rows = read_csv(
        canonical_metrics_path
    )

    canonical_lookup = {

        (
            row["replicate"],
            int(float(row["fold"])),
            row["test_sequence"],
        ):
        row

        for row in canonical_rows
    }

    failures = []

    print()
    print("=" * 90)
    print("Twin V1 frozen replay")
    print("=" * 90)
    print()

    # =========================================================================
    # LEVEL 1: frozen source integrity
    # =========================================================================

    source_ok = 0

    for record in source_hashes:

        path = (
            frozen_dir
            / record["frozen_copy"]
        )

        if not path.exists():

            failures.append(
                f"Missing frozen source snapshot: "
                f"{path}"
            )

            continue

        actual_hash = (
            sha256_file(path)
        )

        if (
            actual_hash
            != record["sha256"]
        ):

            failures.append(
                "Source snapshot hash mismatch: "
                f"{path}"
            )

        else:

            source_ok += 1

    source_pass = (
        source_ok
        == len(source_hashes)
    )

    print(
        "Source snapshot integrity : "
        f"{'PASS' if source_pass else 'FAIL'} "
        f"{source_ok}/{len(source_hashes)}"
    )

    # =========================================================================
    # LEVEL 2: checkpoint integrity
    # =========================================================================

    checkpoint_ok = 0

    runs = manifest["runs"]

    for run in runs:

        checkpoint_path = (
            frozen_dir
            / run["frozen_checkpoint"]
        )

        if not checkpoint_path.exists():

            failures.append(
                "Missing frozen checkpoint: "
                f"{checkpoint_path}"
            )

            continue

        actual_hash = (
            sha256_file(
                checkpoint_path
            )
        )

        if (
            actual_hash
            != run["checkpoint_sha256"]
        ):

            failures.append(
                "Checkpoint SHA-256 mismatch: "
                f"{checkpoint_path}"
            )

        else:

            checkpoint_ok += 1

    checkpoint_pass = (
        checkpoint_ok
        == len(runs)
    )

    print(
        "Checkpoint integrity      : "
        f"{'PASS' if checkpoint_pass else 'FAIL'} "
        f"{checkpoint_ok}/{len(runs)}"
    )

    # =========================================================================
    # Device
    # =========================================================================

    if (
        args.device.lower().startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):

        print()
        print(
            "[warn] CUDA unavailable; "
            "using CPU."
        )

        device = torch.device(
            "cpu"
        )

    else:

        device = torch.device(
            args.device
        )

    print()
    print(
        f"Replay device             : {device}"
    )

    # =========================================================================
    # Import original implementation
    # =========================================================================

    original = importlib.import_module(
        "DigitalTwin.analysis.i2nav_loso_ablation"
    )

    original_args = (
        original_default_args(
            original
        )
    )

    original_args.root = (
        data_root
    )

    # =========================================================================
    # Prepare data
    # =========================================================================

    print()
    print(
        "Preparing i2Nav dataset..."
    )

    prepared = prepare_all(
        original,
        data_root,
        original_args,
    )

    # =========================================================================
    # Temporary replay directory
    # =========================================================================

    if args.work_dir is None:

        temporary = (
            tempfile.TemporaryDirectory(
                prefix="i2nav_v1_replay_"
            )
        )

        work_dir = Path(
            temporary.name
        )

    else:

        temporary = None

        work_dir = (
            args.work_dir.resolve()
        )

        work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================================
    # Replay
    # =========================================================================

    prediction_ok = 0
    metric_ok = 0

    number_runs = len(runs)

    print()
    print(
        f"Replaying {number_runs} "
        f"frozen V1 checkpoints..."
    )
    print()

    try:

        for index, run in enumerate(
            runs,
            start=1,
        ):

            replicate = (
                run["replicate"]
            )

            fold = int(
                run["fold"]
            )

            sequence_name = (
                run["test_sequence"]
            )

            key = (
                replicate,
                fold,
                sequence_name,
            )

            print(
                f"[{index:02d}/{number_runs:02d}] "
                f"{replicate} "
                f"fold={fold:02d} "
                f"{sequence_name}",
                end=" ",
                flush=True,
            )

            if key not in canonical_lookup:

                failures.append(
                    "No canonical metric row for "
                    f"{key}"
                )

                print("FAIL")
                continue

            checkpoint_path = (
                frozen_dir
                / run["frozen_checkpoint"]
            )

            model, checkpoint = (
                load_model(
                    original,
                    checkpoint_path,
                    device,
                )
            )

            sequence = (
                prepared[
                    sequence_name
                ]
            )

            # -----------------------------------------------------------------
            # Fresh frozen inference
            # -----------------------------------------------------------------

            with torch.no_grad():

                corrections, alphas = (
                    original.predict_neural_sequence(

                        model=model,

                        sequence=sequence,

                        feature_mean=np.asarray(
                            checkpoint[
                                "feature_mean"
                            ]
                        ),

                        feature_std=np.asarray(
                            checkpoint[
                                "feature_std"
                            ]
                        ),

                        window=int(
                            checkpoint[
                                "window"
                            ]
                        ),

                        batch_size=int(
                            original_args.eval_batch_size
                        ),

                        device=device,
                    )
                )

            training_names, validation_names = (
                original.build_fold_split(

                    sequence_name,

                    int(
                        original_args.validation_count
                    ),
                )
            )

            replay_trajectory = (

                work_dir
                / (
                    f"{replicate}_"
                    f"fold_{fold:02d}_"
                    f"{sequence_name}_"
                    f"trajectory.csv"
                )
            )

            result = (
                original.evaluate_predictions(

                    fold=fold,

                    method="gru_dual",

                    sequence=sequence,

                    training_names=
                        training_names,

                    validation_names=
                        validation_names,

                    corrections=
                        corrections,

                    alphas=
                        alphas,

                    args=
                        original_args,

                    trajectory_path=
                        replay_trajectory,
                )
            )

            # =================================================================
            # LEVEL 3: trajectory replay
            # =================================================================

            canonical_trajectory = (

                frozen_dir
                / run[
                    "canonical_trajectory"
                ]
            )

            canonical_header, canonical_data = (
                load_numeric_csv(
                    canonical_trajectory
                )
            )

            replay_header, replay_data = (
                load_numeric_csv(
                    replay_trajectory
                )
            )

            prediction_pass = True

            if (
                canonical_header
                != replay_header
            ):

                failures.append(
                    f"{key}: trajectory "
                    "header mismatch"
                )

                prediction_pass = False

            elif (
                canonical_data.shape
                != replay_data.shape
            ):

                failures.append(
                    f"{key}: trajectory shape "
                    f"{canonical_data.shape} != "
                    f"{replay_data.shape}"
                )

                prediction_pass = False

            else:

                try:

                    np.testing.assert_allclose(

                        replay_data,

                        canonical_data,

                        rtol=
                            args.pred_rtol,

                        atol=
                            args.pred_atol,

                        equal_nan=True,
                    )

                except AssertionError as exc:

                    difference = np.abs(
                        replay_data
                        - canonical_data
                    )

                    max_absolute = (
                        float(
                            np.nanmax(
                                difference
                            )
                        )
                        if difference.size
                        else 0.0
                    )

                    failures.append(
                        f"{key}: trajectory mismatch; "
                        f"max_abs={max_absolute:.3e}; "
                        f"{str(exc).splitlines()[0]}"
                    )

                    prediction_pass = False

            if prediction_pass:
                prediction_ok += 1

            # =================================================================
            # LEVEL 4: metric replay
            # =================================================================

            canonical_metric_row = (
                canonical_lookup[key]
            )

            fresh_metrics = vars(
                result
            )

            metric_fields = [

                "ate_rmse_m",

                "heading_mae_deg",

                "rpe_1s_trans_rmse_m",

                "rpe_5s_trans_rmse_m",

                "rpe_10s_trans_rmse_m",

                "final_error_m",

                "ate_se2_rmse_m",
            ]

            metric_pass = True

            for metric in metric_fields:

                canonical_value = float(
                    canonical_metric_row[
                        metric
                    ]
                )

                replay_value = float(
                    fresh_metrics[
                        metric
                    ]
                )

                if not np.isclose(

                    replay_value,

                    canonical_value,

                    rtol=0.0,

                    atol=
                        args.metric_atol,

                    equal_nan=True,
                ):

                    failures.append(
                        f"{key}: {metric} mismatch "
                        f"replay={replay_value:.12g} "
                        f"canonical={canonical_value:.12g}"
                    )

                    metric_pass = False

            if metric_pass:
                metric_ok += 1

            overall_pass = (
                prediction_pass
                and metric_pass
            )

            print(

                f"{'PASS' if overall_pass else 'FAIL'} "

                f"ATE="
                f"{float(fresh_metrics['ate_rmse_m']):.6f}"
            )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:

        if temporary is not None:
            temporary.cleanup()

    # =========================================================================
    # Final result
    # =========================================================================

    print()
    print("=" * 90)

    print(
        "Prediction replay         : "
        f"{'PASS' if prediction_ok == number_runs else 'FAIL'} "
        f"{prediction_ok}/{number_runs}"
    )

    print(
        "Metric replay             : "
        f"{'PASS' if metric_ok == number_runs else 'FAIL'} "
        f"{metric_ok}/{number_runs}"
    )

    print("=" * 90)

    if failures:

        print()
        print(
            "V1 STATUS: FAILED / CHANGED"
        )

        print()
        print("Failures:")

        for failure in failures:

            print(
                " -",
                failure,
            )

        return 1

    print()
    print(
        "V1 STATUS: VERIFIED / UNCHANGED"
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())