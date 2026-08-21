#!/usr/bin/env python3
"""Train/run the external maintenance baselines under strict i2Nav LOSO.

Default source: the frozen V2 evaluated-trajectory archive. Only its raw wheel /
IMU channels and ground truth are used as the common physical sequence corpus;
V2 estimates are ignored by these baseline algorithms.

Outputs use one canonical trajectory schema so all fidelity evaluators can be
applied identically.
"""
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .common import discover_i2nav_corpus, save_json
from .fixed_physics import run_fixed_physics, METHOD_NAME as FIXED_NAME
from .ekf_iw import fit_config as fit_ekf_config, run_ekf_iw, METHOD_NAME as EKF_NAME
from .lwoi_imu import LWOIConfig, SparseRBFResidual, run_lwoi_imu, METHOD_NAME as LWOI_NAME
from .ynet_reduced import YNetConfig, fit_ynet, run_ynet, METHOD_NAME as YNET_NAME

MODEL_KEYS = ["fixed_recomputed", "ekf_iw", "lwoi_imu", "ynet_reduced"]


def csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-root", default="results/i2nav_v2_full_loso/i2nav_v2_full_loso")
    p.add_argument("--glob", default="**/v2_evaluated_trajectory.csv")
    p.add_argument("--output-root", default="results/i2nav_external_baselines")
    p.add_argument("--models", default="ekf_iw,lwoi_imu,ynet_reduced", help=f"Comma list from {MODEL_KEYS}; fixed_recomputed is optional sanity baseline")
    p.add_argument("--seeds", default="42,52,62")
    p.add_argument("--test-sequences", default="", help="Optional comma list; blank means all")
    p.add_argument("--expected-sequences", type=int, default=10)
    p.add_argument("--no-verify-duplicates", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Fast validation: parking00 + parking02 when available, tiny learned configs")
    p.add_argument("--lwoi-centers", type=int, default=256)
    p.add_argument("--lwoi-max-train-samples", type=int, default=30000)
    p.add_argument("--ynet-epochs", type=int, default=20)
    p.add_argument("--ynet-max-windows", type=int, default=80000)
    p.add_argument("--ynet-window-samples", type=int, default=20)
    return p.parse_args()


def _write_trajectory(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    a = parse_args()
    models = csv_list(a.models)
    bad = [m for m in models if m not in MODEL_KEYS]
    if bad:
        raise SystemExit(f"Unknown model keys {bad}; allowed {MODEL_KEYS}")
    seeds = int_list(a.seeds)
    if not seeds:
        raise SystemExit("At least one seed is required")

    corpus = discover_i2nav_corpus(a.input_root, a.glob, verify_duplicates=not a.no_verify_duplicates)
    seqs = sorted(corpus)
    print(f"Discovered {len(seqs)} unique physical sequences: {seqs}")
    if a.expected_sequences > 0 and len(seqs) != a.expected_sequences:
        raise SystemExit(f"Expected {a.expected_sequences} unique sequences, found {len(seqs)}. Refusing to run ambiguous LOSO.")

    requested = csv_list(a.test_sequences)
    if requested:
        missing = sorted(set(requested) - set(seqs))
        if missing:
            raise SystemExit(f"Requested test sequences not found: {missing}")
        test_seqs = requested
    elif a.smoke:
        preferred = [s for s in ["parking00", "parking02"] if s in corpus]
        test_seqs = preferred if preferred else seqs[:2]
    else:
        test_seqs = seqs

    lcfg = LWOIConfig(n_centers=a.lwoi_centers, max_train_samples=a.lwoi_max_train_samples)
    ycfg = YNetConfig(epochs=a.ynet_epochs, max_train_windows=a.ynet_max_windows, window_samples=a.ynet_window_samples)
    if a.smoke:
        lcfg.n_centers = min(lcfg.n_centers, 64)
        lcfg.max_train_samples = min(lcfg.max_train_samples, 5000)
        ycfg.epochs = min(ycfg.epochs, 2)
        ycfg.max_train_windows = min(ycfg.max_train_windows, 5000)

    out = Path(a.output_root)
    out.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for s in seqs:
        source_rows.append({
            "sequence": s,
            "canonical_source": str(corpus[s].path),
            "n_duplicate_sources": len(corpus[s].duplicates),
            "raw_fingerprint": corpus[s].raw_fingerprint,
        })
    pd.DataFrame(source_rows).to_csv(out / "source_corpus_manifest.csv", index=False)

    manifest = []

    def record(method_key, method_name, seq, seed, traj_path, train_seq, adaptation, status="ok", error=""):
        manifest.append({
            "method_key": method_key,
            "method": method_name,
            "sequence": seq,
            "seed": seed,
            "trajectory": str(traj_path),
            "test_sequence": seq,
            "train_sequences": ";".join(train_seq),
            "n_train_sequences": len(train_seq),
            "adaptation_level": adaptation,
            "status": status,
            "error": error,
        })
        pd.DataFrame(manifest).to_csv(out / "baseline_manifest.csv", index=False)

    for fold_idx, test_seq in enumerate(test_seqs, 1):
        train_seq = [s for s in seqs if s != test_seq]
        train_frames = [corpus[s].data for s in train_seq]
        test_df = corpus[test_seq].data
        print("=" * 88)
        print(f"FOLD {fold_idx}/{len(test_seqs)} test={test_seq} train={train_seq}")

        jobs = []
        if "fixed_recomputed" in models:
            jobs.append(("fixed_recomputed", None))
        if "ekf_iw" in models:
            jobs.append(("ekf_iw", None))
        if "lwoi_imu" in models:
            jobs.extend(("lwoi_imu", seed) for seed in seeds)
        if "ynet_reduced" in models:
            jobs.extend(("ynet_reduced", seed) for seed in seeds)

        # EKF config is also used as the YNet fusion back-end; fit once per fold.
        ekf_cfg = fit_ekf_config(train_frames) if any(k in models for k in ["ekf_iw", "ynet_reduced"]) else None
        if ekf_cfg is not None:
            save_json(out / "EKF_IW" / test_seq / "ekf_config.json", asdict(ekf_cfg))

        for model_key, seed in jobs:
            try:
                if model_key == "fixed_recomputed":
                    method_name = FIXED_NAME
                    traj = out / "Fixed_Physics_Recomputed" / test_seq / "evaluated_trajectory.csv"
                    adaptation = "recomputed raw-input sanity baseline; prefer frozen official Fixed Physics for headline comparison"
                    if a.skip_existing and traj.exists():
                        print(f"  SKIP existing {traj}")
                    else:
                        _write_trajectory(run_fixed_physics(test_df), traj)

                elif model_key == "ekf_iw":
                    method_name = EKF_NAME
                    traj = out / "EKF_IW" / test_seq / "evaluated_trajectory.csv"
                    adaptation = "planar classical EKF-IW compatible with frozen wheel-speed/yaw-rate channels; not WING full-state reproduction"
                    if a.skip_existing and traj.exists():
                        print(f"  SKIP existing {traj}")
                    else:
                        _write_trajectory(run_ekf_iw(test_df, ekf_cfg), traj)

                elif model_key == "lwoi_imu":
                    method_name = LWOI_NAME
                    traj = out / "LWOI_IMU_Adaptation" / f"seed_{seed}" / test_seq / "evaluated_trajectory.csv"
                    adaptation = "LWOI-style sparse-RBF residual adaptation to available i2Nav channels; not exact official LWOI reproduction"
                    if a.skip_existing and traj.exists():
                        print(f"  SKIP existing {traj}")
                    else:
                        print(f"  Training {method_name}, seed={seed}")
                        model = SparseRBFResidual(lcfg, seed=seed).fit(train_frames)
                        model.save(traj.with_name("lwoi_model.npz"))
                        _write_trajectory(run_lwoi_imu(test_df, model), traj)

                elif model_key == "ynet_reduced":
                    method_name = YNET_NAME
                    traj = out / "YNet_Reduced" / f"seed_{seed}" / test_seq / "evaluated_trajectory.csv"
                    adaptation = "YNet-style reduced-input TCN+attention + planar EKF; not exact original YNet reproduction"
                    if a.skip_existing and traj.exists():
                        print(f"  SKIP existing {traj}")
                    else:
                        print(f"  Training {method_name}, seed={seed}")
                        trained = fit_ynet(train_frames, ycfg, seed=seed)
                        trained.save(traj.with_name("ynet_model.pt"))
                        _write_trajectory(run_ynet(test_df, trained, ekf_cfg), traj)
                else:
                    raise AssertionError(model_key)

                record(model_key, method_name, test_seq, seed if seed is not None else "deterministic", traj, train_seq, adaptation)
                print(f"  WROTE {traj}")
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR {model_key} {test_seq} seed={seed}: {err}")
                traceback.print_exc()
                method_name = {"fixed_recomputed": FIXED_NAME, "ekf_iw": EKF_NAME, "lwoi_imu": LWOI_NAME, "ynet_reduced": YNET_NAME}.get(model_key, model_key)
                dummy = out / "FAILED" / model_key / str(seed) / test_seq / "evaluated_trajectory.csv"
                record(model_key, method_name, test_seq, seed if seed is not None else "deterministic", dummy, train_seq, "failed", status="error", error=err)
                if not a.continue_on_error:
                    raise

    print("=" * 88)
    print(f"Completed baseline generation. Manifest: {out / 'baseline_manifest.csv'}")
    print("Publication boundary: LWOI and YNet outputs are explicit reduced-input adaptations; do not label them exact reproductions.")


if __name__ == "__main__":
    main()
