#!/usr/bin/env python3

"""
Freeze the completed i2Nav 10-fold x 3-seed dual-GRU study as Twin V1.

This script:

1. Finds the 30 existing gru_dual.pt checkpoints.
2. Copies them into a frozen V1 directory.
3. Hashes checkpoints and V1 source code.
4. Replays every checkpoint on its original held-out sequence.
5. Saves canonical trajectories.
6. Saves canonical per-run, per-fold, and overall metrics.

IMPORTANT:
- This script DOES NOT train anything.
- Existing V1 results are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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

CHECKPOINT_BASENAME = "gru_dual.pt"

EVIDENCE_BASENAMES = {
    "aggregate_summary.csv",
    "all_seed_fold_results.csv",
    "baseline_comparison.csv",
    "environment_manifest.json",
    "final_summary.json",
    "full_run.log",
    "historical_reproducibility.csv",
    "per_fold_seed_summary.csv",
    "per_fold_summary.csv",
    "fold_splits.json",
    "loso_results.csv",
    "loso_summary.json",
}

SOURCE_FILES = (
    "DigitalTwin/analysis/i2nav_loso_ablation.py",
    "DigitalTwin/analysis/i2nav_gru_dualhead.py",
    "DigitalTwin/analysis/i2nav_adaptive_q_baseline.py",
    "DigitalTwin/ekf.py",
)


# =============================================================================
# Utilities
# =============================================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Result-package parsing
# =============================================================================

def parse_rep_fold(
    checkpoint_path: Path,
) -> tuple[str, int, str]:

    parts = checkpoint_path.parts

    replicate = next(
        (
            p
            for p in parts
            if p.startswith("replicate_")
        ),
        "replicate_00",
    )

    sequence = None
    fold_number = None

    for part in reversed(parts[:-1]):

        match = re.match(
            r"fold_(\d+)_(.+)",
            part,
        )

        if match:
            possible_sequence = match.group(2)

            if possible_sequence in SEQUENCES:
                fold_number = int(match.group(1))
                sequence = possible_sequence
                break

        if part in SEQUENCES:
            sequence = part
            break

    if sequence is None:
        raise ValueError(
            "Could not infer held-out sequence from checkpoint path:\n"
            f"{checkpoint_path}"
        )

    if fold_number is None:
        fold_number = (
            SEQUENCES.index(sequence) + 1
        )

    return (
        replicate,
        fold_number,
        sequence,
    )


def infer_base_seed(
    replicate_label: str,
) -> int | None:

    match = re.search(
        r"base(\d+)",
        replicate_label,
    )

    if not match:
        return None

    return int(match.group(1))


# =============================================================================
# Original V1 argument handling
# =============================================================================

def original_default_args(
    original_module,
) -> argparse.Namespace:

    saved_argv = sys.argv[:]

    try:

        sys.argv = [
            "i2nav_loso_ablation.py"
        ]

        return original_module.parse_args()

    finally:
        sys.argv = saved_argv


# =============================================================================
# Dataset preparation
# =============================================================================

def discover_dataset(
    original_module,
    root: Path,
) -> dict[str, Any]:

    files = original_module.discover_files(root)

    return {
        f.name: f
        for f in files
    }


def prepare_all(
    original_module,
    root: Path,
    original_args: argparse.Namespace,
) -> dict[str, Any]:

    discovered = discover_dataset(
        original_module,
        root,
    )

    missing = [
        s
        for s in SEQUENCES
        if s not in discovered
    ]

    if missing:
        raise RuntimeError(
            "Dataset discovery did not find:\n"
            f"{missing}"
        )

    prepared = {}

    print()
    print("Preparing i2Nav sequences once...")
    print()

    for sequence_name in SEQUENCES:

        print(
            f"  preparing {sequence_name}"
        )

        prepared[sequence_name] = (
            original_module.prepare_sequence(
                discovered[sequence_name],

                hz=original_args.rate_hz,

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
# Load V1 checkpoint
# =============================================================================

def load_v1_model(
    original_module,
    checkpoint_path: Path,
    device: torch.device,
):

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # PyTorch >=2.6 defaults weights_only=True.
    #
    # These V1 checkpoints contain NumPy objects in addition to tensors.
    #
    # Because these are OUR OWN TRUSTED research checkpoints we explicitly
    # load the complete checkpoint.
    # -------------------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = [
        "state_dict",
        "feature_mean",
        "feature_std",
        "dv_limit",
        "domega_limit",
        "hidden_size",
        "num_layers",
        "window",
        "alpha_min",
        "alpha_max",
    ]

    missing = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing:
        raise KeyError(
            f"{checkpoint_path}\n"
            f"Missing checkpoint keys: {missing}"
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

        # Original V1 default.
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
        parameter.requires_grad_(False)

    return (
        model,
        checkpoint,
    )


# =============================================================================
# Preserve evidence
# =============================================================================

def copy_evidence(
    source_dir: Path,
    frozen_dir: Path,
) -> list[dict[str, Any]]:

    evidence_dir = (
        frozen_dir
        / "evidence"
    )

    records = []

    for path in sorted(
        source_dir.rglob("*")
    ):

        if not path.is_file():
            continue

        keep = (

            path.name
            in EVIDENCE_BASENAMES

            or path.name
            in {
                "gru_dual.pt",
                "gru_dual_checkpoint.pt",
                "gru_dual_history.csv",
            }
        )

        if not keep:
            continue

        relative = path.relative_to(
            source_dir
        )

        destination = (
            evidence_dir
            / relative
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            path,
            destination,
        )

        records.append(
            {
                "source_relative_path":
                    str(relative),

                "frozen_relative_path":
                    str(
                        destination.relative_to(
                            frozen_dir
                        )
                    ),

                "sha256":
                    sha256_file(destination),

                "size_bytes":
                    destination.stat().st_size,
            }
        )

    return records


# =============================================================================
# Metric aggregation
# =============================================================================

def summarize(
    rows: list[dict[str, Any]],
):

    by_sequence = defaultdict(list)

    for row in rows:

        by_sequence[
            str(row["test_sequence"])
        ].append(row)

    metric_fields = [

        "ate_rmse_m",

        "heading_mae_deg",

        "rpe_1s_trans_rmse_m",

        "rpe_5s_trans_rmse_m",

        "rpe_10s_trans_rmse_m",
    ]

    fold_rows = []

    for sequence_name in SEQUENCES:

        sequence_rows = (
            by_sequence.get(
                sequence_name,
                [],
            )
        )

        if not sequence_rows:
            continue

        fold_row = {

            "test_sequence":
                sequence_name,

            "n_seeds":
                len(sequence_rows),
        }

        for metric in metric_fields:

            values = np.asarray(
                [
                    float(row[metric])
                    for row in sequence_rows
                ],
                dtype=float,
            )

            fold_row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            fold_row[
                f"{metric}_std"
            ] = (

                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )

                if len(values) > 1

                else 0.0
            )

        fold_rows.append(
            fold_row
        )

    summary = {

        "schema":
            "i2nav_twin_v1_canonical_summary_v1",

        "n_runs":
            len(rows),

        "n_folds":
            len(fold_rows),

        "aggregation":
            (
                "mean across seeds within each held-out fold, "
                "then macro mean across the 10 held-out folds"
            ),
    }

    for metric in metric_fields:

        values = [

            float(
                row[
                    f"{metric}_mean"
                ]
            )

            for row
            in fold_rows
        ]

        summary[metric] = (

            float(
                np.mean(values)
            )

            if values

            else None
        )

    return (
        fold_rows,
        summary,
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
        "--source-dir",
        type=Path,
        required=True,
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
        "--expected-checkpoints",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    repository_root = (
        Path.cwd().resolve()
    )

    source_dir = (
        args.source_dir.resolve()
    )

    frozen_dir = (
        args.frozen_dir.resolve()
    )

    data_root = (
        args.root.resolve()
    )

    # -------------------------------------------------------------------------
    # Validate paths
    # -------------------------------------------------------------------------

    if not source_dir.exists():

        raise FileNotFoundError(
            f"Source directory not found:\n"
            f"{source_dir}"
        )

    if not data_root.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{data_root}"
        )

    if (
        frozen_dir.exists()
        and any(frozen_dir.iterdir())
        and not args.overwrite
    ):

        raise RuntimeError(
            "\nFrozen V1 directory already exists and is non-empty:\n"
            f"{frozen_dir}\n\n"
            "Refusing to modify an existing frozen artifact.\n"
            "Delete the incomplete directory first if this is a failed freeze."
        )

    frozen_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load original V1 implementation
    # -------------------------------------------------------------------------

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

    original_args.output_dir = (
        frozen_dir
        / "_unused_original_output"
    )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    if (
        args.device.lower().startswith("cuda")
        and not torch.cuda.is_available()
    ):

        print(
            "[warn] CUDA requested but unavailable. "
            "Using CPU."
        )

        device = torch.device(
            "cpu"
        )

    else:

        device = torch.device(
            args.device
        )

    print()
    print("=" * 90)
    print("FREEZE i2Nav TWIN V1")
    print("=" * 90)
    print()
    print(
        f"Source results : {source_dir}"
    )
    print(
        f"Frozen output  : {frozen_dir}"
    )
    print(
        f"Dataset        : {data_root}"
    )
    print(
        f"Device         : {device}"
    )
    print()

    # -------------------------------------------------------------------------
    # Find the 30 authoritative checkpoints
    # -------------------------------------------------------------------------

    checkpoints = sorted(
        source_dir.rglob(
            CHECKPOINT_BASENAME
        )
    )

    print(
        f"Found {len(checkpoints)} "
        f"{CHECKPOINT_BASENAME} checkpoints."
    )

    if (
        len(checkpoints)
        != args.expected_checkpoints
    ):

        raise RuntimeError(
            "\nExpected "
            f"{args.expected_checkpoints} "
            f"checkpoints but found "
            f"{len(checkpoints)}.\n"
        )

    parsed = []
    seen = set()

    for checkpoint_path in checkpoints:

        replicate, fold, sequence = (
            parse_rep_fold(
                checkpoint_path
            )
        )

        key = (
            replicate,
            fold,
            sequence,
        )

        if key in seen:

            raise RuntimeError(
                f"Duplicate V1 checkpoint identity:\n"
                f"{key}\n"
                f"{checkpoint_path}"
            )

        seen.add(key)

        parsed.append(
            (
                checkpoint_path,
                replicate,
                fold,
                sequence,
            )
        )

    replicates = sorted(
        {
            item[1]
            for item in parsed
        }
    )

    if len(replicates) != 3:

        raise RuntimeError(
            "Expected exactly 3 replicates.\n"
            f"Found: {replicates}"
        )

    for replicate in replicates:

        sequences = {

            item[3]

            for item in parsed

            if item[1]
            == replicate
        }

        if sequences != set(SEQUENCES):

            raise RuntimeError(
                f"{replicate} does not contain "
                "all 10 held-out sequences.\n"
                f"Found: {sorted(sequences)}"
            )

    print()
    print(
        "Verified 3 replicates x 10 folds."
    )

    # -------------------------------------------------------------------------
    # Copy original evidence
    # -------------------------------------------------------------------------

    print()
    print(
        "Copying original V1 evidence..."
    )

    evidence_records = (
        copy_evidence(
            source_dir,
            frozen_dir,
        )
    )

    # -------------------------------------------------------------------------
    # Snapshot source code
    # -------------------------------------------------------------------------

    print()
    print(
        "Snapshotting V1 source code..."
    )

    source_hashes = []

    for relative_path in SOURCE_FILES:

        source_path = (
            repository_root
            / relative_path
        )

        # i2nav_loso_ablation.py is mandatory.
        # Other files are preserved if present.
        if not source_path.exists():

            if relative_path.endswith(
                "i2nav_loso_ablation.py"
            ):

                raise FileNotFoundError(
                    "Authoritative V1 source missing:\n"
                    f"{source_path}"
                )

            print(
                f"  [skip] not present: {relative_path}"
            )

            continue

        destination = (
            frozen_dir
            / "source_snapshot"
            / relative_path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_path,
            destination,
        )

        digest = sha256_file(
            source_path
        )

        source_hashes.append(
            {
                "path":
                    relative_path,

                "sha256":
                    digest,

                "frozen_copy":
                    str(
                        destination.relative_to(
                            frozen_dir
                        )
                    ),
            }
        )

        print(
            f"  frozen {relative_path}"
        )

    write_json(
        frozen_dir
        / "SOURCE_HASHES.json",

        source_hashes,
    )

    # -------------------------------------------------------------------------
    # Prepare dataset
    # -------------------------------------------------------------------------

    prepared = prepare_all(
        original,
        data_root,
        original_args,
    )

    # -------------------------------------------------------------------------
    # Replay all 30 models
    # -------------------------------------------------------------------------

    canonical_rows = []
    manifest_runs = []

    print()
    print(
        f"Replaying {len(parsed)} "
        f"V1 checkpoints on {device}..."
    )
    print()

    for index, (
        checkpoint_path,
        replicate,
        fold,
        sequence_name,
    ) in enumerate(
        parsed,
        start=1,
    ):

        print(
            f"[{index:02d}/30] "
            f"{replicate} "
            f"fold={fold:02d} "
            f"test={sequence_name}"
        )

        checkpoint_destination = (

            frozen_dir
            / "checkpoints"
            / replicate
            / f"fold_{fold:02d}_{sequence_name}"
            / CHECKPOINT_BASENAME
        )

        checkpoint_destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            checkpoint_path,
            checkpoint_destination,
        )

        model, checkpoint = (
            load_v1_model(
                original,
                checkpoint_destination,
                device,
            )
        )

        sequence = prepared[
            sequence_name
        ]

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

        trajectory_path = (

            frozen_dir
            / "canonical_predictions"
            / replicate
            / (
                f"fold_{fold:02d}_"
                f"{sequence_name}_"
                f"gru_dual_trajectory.csv"
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
                    trajectory_path,
            )
        )

        result_dict = dict(
            vars(result)
        )

        checkpoint_hash = (
            sha256_file(
                checkpoint_destination
            )
        )

        trajectory_hash = (
            sha256_file(
                trajectory_path
            )
        )

        result_dict.update(
            {
                "replicate":
                    replicate,

                "base_seed_inferred":
                    infer_base_seed(
                        replicate
                    ),

                "test_sequence":
                    sequence_name,

                "checkpoint_sha256":
                    checkpoint_hash,

                "checkpoint_relative_path":
                    str(
                        checkpoint_destination.relative_to(
                            frozen_dir
                        )
                    ),

                "trajectory_relative_path":
                    str(
                        trajectory_path.relative_to(
                            frozen_dir
                        )
                    ),

                "trajectory_sha256":
                    trajectory_hash,
            }
        )

        canonical_rows.append(
            result_dict
        )

        manifest_runs.append(
            {
                "replicate":
                    replicate,

                "base_seed_inferred":
                    infer_base_seed(
                        replicate
                    ),

                "fold":
                    fold,

                "test_sequence":
                    sequence_name,

                "source_checkpoint":
                    str(
                        checkpoint_path.relative_to(
                            source_dir
                        )
                    ),

                "frozen_checkpoint":
                    str(
                        checkpoint_destination.relative_to(
                            frozen_dir
                        )
                    ),

                "checkpoint_sha256":
                    checkpoint_hash,

                "canonical_trajectory":
                    str(
                        trajectory_path.relative_to(
                            frozen_dir
                        )
                    ),

                "canonical_trajectory_sha256":
                    trajectory_hash,
            }
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Save canonical metrics
    # -------------------------------------------------------------------------

    canonical_rows.sort(
        key=lambda row: (
            str(row["replicate"]),
            int(row["fold"]),
        )
    )

    write_csv(
        frozen_dir
        / "canonical_metrics_per_run.csv",

        canonical_rows,
    )

    fold_rows, summary = summarize(
        canonical_rows
    )

    write_csv(
        frozen_dir
        / "canonical_metrics_per_fold.csv",

        fold_rows,
    )

    write_json(
        frozen_dir
        / "canonical_metrics_summary.json",

        summary,
    )

    # -------------------------------------------------------------------------
    # Save manifest
    # -------------------------------------------------------------------------

    manifest = {

        "schema":
            "i2nav_twin_v1_frozen_manifest_v1",

        "artifact":
            "Twin V1",

        "status":
            "FROZEN",

        "model":
            "gru_dual",

        "protocol":
            "10-fold LOSO x 3 seeds",

        "training_runs":
            len(manifest_runs),

        "authoritative_training_evaluation_code":
            (
                "DigitalTwin/analysis/"
                "i2nav_loso_ablation.py"
            ),

        "retraining_allowed":
            False,

        "source_package":
            str(source_dir),

        "data_root":
            str(data_root),

        "device_used_for_canonical_replay":
            str(device),

        "source_hashes_file":
            "SOURCE_HASHES.json",

        "evidence_file_count":
            len(evidence_records),

        "runs":
            manifest_runs,
    }

    write_json(
        frozen_dir
        / "FROZEN_MANIFEST.json",

        manifest,
    )

    write_json(
        frozen_dir
        / "EVIDENCE_HASHES.json",

        evidence_records,
    )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("FROZEN TWIN V1 COMPLETE")
    print("=" * 90)
    print()

    print(
        f"Directory   : {frozen_dir}"
    )

    print(
        f"Checkpoints : "
        f"{len(manifest_runs)}"
    )

    print()

    print(
        "Canonical macro metrics:"
    )

    print(
        "  ATE RMSE       : "
        f"{summary['ate_rmse_m']:.6f} m"
    )

    print(
        "  RPE 1 s        : "
        f"{summary['rpe_1s_trans_rmse_m']:.6f} m"
    )

    print(
        "  RPE 5 s        : "
        f"{summary['rpe_5s_trans_rmse_m']:.6f} m"
    )

    print(
        "  RPE 10 s       : "
        f"{summary['rpe_10s_trans_rmse_m']:.6f} m"
    )

    print(
        "  Heading MAE    : "
        f"{summary['heading_mae_deg']:.6f} deg"
    )

    print()
    print(
        "Next: run test_i2nav_v1_replay.py."
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())