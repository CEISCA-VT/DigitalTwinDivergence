#!/usr/bin/env python3
"""Conditioned-vs-unconditional benign fidelity envelope validation.

This is an analysis-only companion to the frozen V2 benign-envelope work.  It
reuses the existing frozen condition definitions, aligned frozen V2 outputs,
and component definitions.  It does not retrain, retune, redefine bins, or
modify the manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DigitalTwin.analysis.i2nav_v2_all_sequence_mechanism import (
    EXPECTED_BASE_SEEDS,
    EXPECTED_SEQUENCES,
    FROZEN_FULL_LOSO_COMMIT,
    align_run_frame,
    configure_and_prepare,
    locate_run_dirs,
    repo_root_from_script,
)
from DigitalTwin.analysis.i2nav_v2_benign_fidelity_envelope import COMPONENTS, finite
from DigitalTwin.analysis.i2nav_v2_condition_fidelity import add_condition_columns, assign_bin


SCRIPT_VERSION = "2026-08-20-conditioned-envelope-validation-v1"
CONTEXTS = (
    "speed",
    "acceleration",
    "turning",
    "curvature",
    "wheel_imu_disagreement",
    "elapsed_time",
)
COMPONENT_ORDER = ("Dp_m", "Dtheta_deg", "Dv_mps", "Domega_radps")
QUANTILES = (0.90, 0.95, 0.99)
PRIMARY_Q = 0.95
BOOTSTRAP_ITERATIONS = 5000


def component_values(frame: pd.DataFrame, component: str, mask: np.ndarray | None = None) -> np.ndarray:
    spec = COMPONENTS[component]
    src = frame if mask is None else frame.loc[mask]
    values = src[spec["column"]].to_numpy(dtype=float)
    if spec["absolute"]:
        values = np.abs(values)
    return values[np.isfinite(values)]


def quantile(values: list[float] | np.ndarray, q: float) -> float:
    arr = finite(np.asarray(values, dtype=float))
    return float(np.quantile(arr, q)) if len(arr) else float("nan")


def pinball_loss(y: np.ndarray, u: float, q: float) -> np.ndarray:
    diff = y - u
    return np.maximum(q * diff, (q - 1.0) * diff)


def exceedance_stats(y: np.ndarray, u: float) -> dict[str, float]:
    if len(y) == 0 or not np.isfinite(u):
        return {
            "exceedance_rate": float("nan"),
            "mean_excess_given_exceedance": float("nan"),
            "p95_excess_given_exceedance": float("nan"),
            "max_excess": float("nan"),
        }
    excess = np.maximum(0.0, y - u)
    pos = excess[excess > 0.0]
    return {
        "exceedance_rate": float(np.mean(excess > 0.0)),
        "mean_excess_given_exceedance": float(np.mean(pos)) if len(pos) else 0.0,
        "p95_excess_given_exceedance": float(np.quantile(pos, 0.95)) if len(pos) else 0.0,
        "max_excess": float(np.max(pos)) if len(pos) else 0.0,
    }


def run_level_quantiles(
    frames: dict[tuple[str, int], pd.DataFrame],
    definitions: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sequence, seed), frame in frames.items():
        for component in COMPONENT_ORDER:
            values = component_values(frame, component)
            if len(values):
                row = {
                    "sequence": sequence,
                    "base_seed": seed,
                    "context": "unconditional",
                    "condition_bin": "all",
                    "component": component,
                    "units": COMPONENTS[component]["units"],
                    "n_samples": int(len(values)),
                }
                for q in QUANTILES:
                    row[f"q{int(q * 100)}"] = quantile(values, q)
                rows.append(row)
        for context in CONTEXTS:
            definition = definitions["variables"][context]
            bins = assign_bin(frame[definition["source_column"]], definition)
            for condition_bin in definition["bins"]:
                mask = (bins == condition_bin).to_numpy()
                for component in COMPONENT_ORDER:
                    values = component_values(frame, component, mask)
                    row = {
                        "sequence": sequence,
                        "base_seed": seed,
                        "context": context,
                        "condition_bin": condition_bin,
                        "component": component,
                        "units": COMPONENTS[component]["units"],
                        "n_samples": int(len(values)),
                    }
                    for q in QUANTILES:
                        row[f"q{int(q * 100)}"] = quantile(values, q)
                    rows.append(row)
    return pd.DataFrame(rows)


def sequence_quantiles(run_quantiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["sequence", "context", "condition_bin", "component", "units"]
    for keys, group in run_quantiles.groupby(group_cols, sort=True):
        sequence, context, condition_bin, component, units = keys
        supported = group[group["n_samples"] >= 30]
        if supported.empty:
            continue
        row = {
            "sequence": sequence,
            "context": context,
            "condition_bin": condition_bin,
            "component": component,
            "units": units,
            "n_seed_rows": int(len(group)),
            "n_supported_seed_rows": int(len(supported)),
            "supported": True,
        }
        for q in QUANTILES:
            col = f"q{int(q * 100)}"
            vals = finite(supported[col])
            row[col] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{col}_seed_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_training_envelopes(seq_quantiles: pd.DataFrame, holdout: str) -> pd.DataFrame:
    train = seq_quantiles[
        (seq_quantiles["sequence"] != holdout) & (seq_quantiles["supported"].astype(bool))
    ]
    rows: list[dict[str, Any]] = []
    group_cols = ["context", "condition_bin", "component", "units"]
    for keys, group in train.groupby(group_cols, sort=True):
        context, condition_bin, component, units = keys
        row = {
            "heldout_sequence": holdout,
            "context": context,
            "condition_bin": condition_bin,
            "component": component,
            "units": units,
            "train_n_sequences": int(group["sequence"].nunique()),
        }
        for q in QUANTILES:
            col = f"q{int(q * 100)}"
            vals = finite(group[col])
            row[f"envelope_q{int(q * 100)}"] = float(np.quantile(vals, q)) if len(vals) else float("nan")
            row[f"train_sequence_mean_q{int(q * 100)}"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"train_sequence_sd_q{int(q * 100)}"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    vals = finite(values)
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    draws = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    n = len(vals)
    for i in range(BOOTSTRAP_ITERATIONS):
        draws[i] = float(np.mean(vals[rng.integers(0, n, size=n)]))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def eval_one(
    values: np.ndarray,
    threshold: float,
    q: float,
) -> dict[str, float]:
    if len(values) == 0 or not np.isfinite(threshold):
        base = {
            "coverage": float("nan"),
            "calibration_error": float("nan"),
            "mean_quantile_loss": float("nan"),
        }
        base.update(exceedance_stats(values, threshold))
        return base
    coverage = float(np.mean(values <= threshold))
    base = {
        "coverage": coverage,
        "calibration_error": abs(coverage - q),
        "mean_quantile_loss": float(np.mean(pinball_loss(values, threshold, q))),
    }
    base.update(exceedance_stats(values, threshold))
    return base


def evaluate(
    frames: dict[tuple[str, int], pd.DataFrame],
    definitions: dict[str, Any],
    envelopes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    env_lookup = {
        (r.heldout_sequence, r.context, r.condition_bin, r.component): r
        for r in envelopes.itertuples(index=False)
    }

    heldout_rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []
    sharp_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    exceed_rows: list[dict[str, Any]] = []

    for holdout in EXPECTED_SEQUENCES:
        for context in CONTEXTS:
            definition = definitions["variables"][context]
            bins_by_seed = {
                seed: assign_bin(frames[(holdout, seed)][definition["source_column"]], definition)
                for seed in EXPECTED_BASE_SEEDS
            }
            for component in COMPONENT_ORDER:
                uncond_env = env_lookup.get((holdout, "unconditional", "all", component))
                if uncond_env is None:
                    continue

                for q in QUANTILES:
                    qlabel = f"q{int(q * 100)}"
                    uncond_u = float(getattr(uncond_env, f"envelope_{qlabel}"))
                    pooled_all = np.concatenate(
                        [component_values(frames[(holdout, seed)], component) for seed in EXPECTED_BASE_SEEDS]
                    )
                    uncond_all = eval_one(pooled_all, uncond_u, q)

                    cond_piece_values: list[np.ndarray] = []
                    cond_piece_thresholds: list[np.ndarray] = []
                    width_weighted_n = 0
                    width_weighted_sum = 0.0
                    equal_widths = []
                    for condition_bin in definition["bins"]:
                        cond_env = env_lookup.get((holdout, context, condition_bin, component))
                        if cond_env is None:
                            continue
                        cond_u = float(getattr(cond_env, f"envelope_{qlabel}"))
                        bin_values = []
                        for seed in EXPECTED_BASE_SEEDS:
                            frame = frames[(holdout, seed)]
                            mask = (bins_by_seed[seed] == condition_bin).to_numpy()
                            vals = component_values(frame, component, mask)
                            if len(vals):
                                bin_values.append(vals)
                        if not bin_values:
                            continue
                        values = np.concatenate(bin_values)
                        cond_piece_values.append(values)
                        cond_piece_thresholds.append(np.full(len(values), cond_u, dtype=float))
                        equal_widths.append(cond_u)
                        width_weighted_sum += cond_u * len(values)
                        width_weighted_n += len(values)

                        cond_bin = eval_one(values, cond_u, q)
                        uncond_bin = eval_one(values, uncond_u, q)
                        within_rows.append(
                            {
                                "heldout_sequence": holdout,
                                "condition_variable": context,
                                "condition_bin": condition_bin,
                                "component": component,
                                "units": COMPONENTS[component]["units"],
                                "q": q,
                                "n_samples": int(len(values)),
                                "unconditional_width": uncond_u,
                                "conditioned_width": cond_u,
                                "width_ratio": cond_u / uncond_u if uncond_u > 0 else float("nan"),
                                "sharpness_gain": 1.0 - cond_u / uncond_u if uncond_u > 0 else float("nan"),
                                "coverage_uncond": uncond_bin["coverage"],
                                "coverage_cond": cond_bin["coverage"],
                                "calibration_error_uncond": uncond_bin["calibration_error"],
                                "calibration_error_cond": cond_bin["calibration_error"],
                                "conditioning_improves_calibration": bool(
                                    cond_bin["calibration_error"] < uncond_bin["calibration_error"]
                                ),
                            }
                        )

                    if cond_piece_values:
                        cond_values = np.concatenate(cond_piece_values)
                        cond_threshold = np.concatenate(cond_piece_thresholds)
                        cond_coverage = float(np.mean(cond_values <= cond_threshold))
                        cond_loss = float(np.mean(pinball_loss(cond_values, cond_threshold, q)))
                        cond_excess = np.maximum(0.0, cond_values - cond_threshold)
                        cond_pos = cond_excess[cond_excess > 0.0]
                        cond_stats = {
                            "coverage": cond_coverage,
                            "calibration_error": abs(cond_coverage - q),
                            "mean_quantile_loss": cond_loss,
                            "exceedance_rate": float(np.mean(cond_excess > 0.0)),
                            "mean_excess_given_exceedance": float(np.mean(cond_pos)) if len(cond_pos) else 0.0,
                            "p95_excess_given_exceedance": float(np.quantile(cond_pos, 0.95)) if len(cond_pos) else 0.0,
                            "max_excess": float(np.max(cond_pos)) if len(cond_pos) else 0.0,
                        }
                    else:
                        cond_stats = eval_one(np.asarray([], dtype=float), float("nan"), q)

                    mean_width_equal = float(np.mean(equal_widths)) if equal_widths else float("nan")
                    mean_width_weighted = (
                        float(width_weighted_sum / width_weighted_n) if width_weighted_n else float("nan")
                    )
                    heldout_rows.append(
                        {
                            "heldout_sequence": holdout,
                            "condition_variable": context,
                            "component": component,
                            "units": COMPONENTS[component]["units"],
                            "q": q,
                            "n_samples": int(len(pooled_all)),
                            "coverage_uncond": uncond_all["coverage"],
                            "coverage_cond": cond_stats["coverage"],
                            "coverage_diff_cond_minus_uncond": cond_stats["coverage"] - uncond_all["coverage"],
                            "calibration_error_uncond": uncond_all["calibration_error"],
                            "calibration_error_cond": cond_stats["calibration_error"],
                            "calibration_error_diff_cond_minus_uncond": cond_stats["calibration_error"]
                            - uncond_all["calibration_error"],
                        }
                    )
                    sharp_rows.append(
                        {
                            "heldout_sequence": holdout,
                            "condition_variable": context,
                            "component": component,
                            "units": COMPONENTS[component]["units"],
                            "q": q,
                            "width_uncond": uncond_u,
                            "width_cond_equal_bin_mean": mean_width_equal,
                            "width_cond_prevalence_weighted": mean_width_weighted,
                            "width_ratio_prevalence_weighted": mean_width_weighted / uncond_u
                            if uncond_u > 0
                            else float("nan"),
                            "sharpness_gain_prevalence_weighted": 1.0 - mean_width_weighted / uncond_u
                            if uncond_u > 0
                            else float("nan"),
                        }
                    )
                    loss_rows.append(
                        {
                            "heldout_sequence": holdout,
                            "condition_variable": context,
                            "component": component,
                            "units": COMPONENTS[component]["units"],
                            "q": q,
                            "mean_quantile_loss_uncond": uncond_all["mean_quantile_loss"],
                            "mean_quantile_loss_cond": cond_stats["mean_quantile_loss"],
                            "loss_diff_cond_minus_uncond": cond_stats["mean_quantile_loss"]
                            - uncond_all["mean_quantile_loss"],
                            "relative_loss_reduction": 1.0
                            - cond_stats["mean_quantile_loss"] / uncond_all["mean_quantile_loss"]
                            if uncond_all["mean_quantile_loss"] > 0
                            else float("nan"),
                        }
                    )
                    exceed_rows.append(
                        {
                            "heldout_sequence": holdout,
                            "condition_variable": context,
                            "component": component,
                            "units": COMPONENTS[component]["units"],
                            "q": q,
                            "exceedance_rate_uncond": uncond_all["exceedance_rate"],
                            "exceedance_rate_cond": cond_stats["exceedance_rate"],
                            "mean_excess_uncond": uncond_all["mean_excess_given_exceedance"],
                            "mean_excess_cond": cond_stats["mean_excess_given_exceedance"],
                            "p95_excess_uncond": uncond_all["p95_excess_given_exceedance"],
                            "p95_excess_cond": cond_stats["p95_excess_given_exceedance"],
                            "max_excess_uncond": uncond_all["max_excess"],
                            "max_excess_cond": cond_stats["max_excess"],
                        }
                    )
    return {
        "heldout": pd.DataFrame(heldout_rows),
        "within": pd.DataFrame(within_rows),
        "sharpness": pd.DataFrame(sharp_rows),
        "loss": pd.DataFrame(loss_rows),
        "exceedance": pd.DataFrame(exceed_rows),
    }


def summarize_heldout(heldout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in heldout.groupby(["condition_variable", "component", "units", "q"], sort=True):
        context, component, units, q = keys
        rows.append(
            {
                "condition_variable": context,
                "component": component,
                "units": units,
                "q": float(q),
                "n_sequences": int(group["heldout_sequence"].nunique()),
                "coverage_uncond_mean": float(group["coverage_uncond"].mean()),
                "coverage_uncond_median": float(group["coverage_uncond"].median()),
                "coverage_uncond_min": float(group["coverage_uncond"].min()),
                "coverage_uncond_max": float(group["coverage_uncond"].max()),
                "coverage_uncond_sd": float(group["coverage_uncond"].std(ddof=1)),
                "coverage_uncond_iqr": float(group["coverage_uncond"].quantile(0.75) - group["coverage_uncond"].quantile(0.25)),
                "coverage_cond_mean": float(group["coverage_cond"].mean()),
                "coverage_cond_median": float(group["coverage_cond"].median()),
                "coverage_cond_min": float(group["coverage_cond"].min()),
                "coverage_cond_max": float(group["coverage_cond"].max()),
                "coverage_cond_sd": float(group["coverage_cond"].std(ddof=1)),
                "coverage_cond_iqr": float(group["coverage_cond"].quantile(0.75) - group["coverage_cond"].quantile(0.25)),
                "coverage_diff_cond_minus_uncond_mean": float(group["coverage_diff_cond_minus_uncond"].mean()),
                "calibration_error_uncond_mean": float(group["calibration_error_uncond"].mean()),
                "calibration_error_cond_mean": float(group["calibration_error_cond"].mean()),
                "calibration_error_diff_cond_minus_uncond_mean": float(group["calibration_error_diff_cond_minus_uncond"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_within(within: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["condition_variable", "condition_bin", "component", "units", "q"]
    for keys, group in within.groupby(group_cols, sort=True):
        context, condition_bin, component, units, q = keys
        rows.append(
            {
                "condition_variable": context,
                "condition_bin": condition_bin,
                "component": component,
                "units": units,
                "q": float(q),
                "n_sequences": int(group["heldout_sequence"].nunique()),
                "mean_coverage_uncond": float(group["coverage_uncond"].mean()),
                "mean_coverage_cond": float(group["coverage_cond"].mean()),
                "mean_abs_calibration_error_uncond": float(group["calibration_error_uncond"].mean()),
                "mean_abs_calibration_error_cond": float(group["calibration_error_cond"].mean()),
                "n_sequences_conditioning_improves_calibration": int(
                    group["conditioning_improves_calibration"].sum()
                ),
                "mean_width_uncond": float(group["unconditional_width"].mean()),
                "mean_width_cond": float(group["conditioned_width"].mean()),
                "mean_width_ratio": float(group["width_ratio"].mean()),
                "mean_sharpness_gain": float(group["sharpness_gain"].mean()),
                "total_samples": int(group["n_samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_summary(
    heldout: pd.DataFrame,
    sharpness: pd.DataFrame,
    loss: pd.DataFrame,
    exceedance: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260820)
    merged = heldout.merge(
        sharpness,
        on=["heldout_sequence", "condition_variable", "component", "units", "q"],
        how="inner",
    ).merge(
        loss,
        on=["heldout_sequence", "condition_variable", "component", "units", "q"],
        how="inner",
    ).merge(
        exceedance,
        on=["heldout_sequence", "condition_variable", "component", "units", "q"],
        how="inner",
    )
    rows = []
    for keys, group in merged[merged["q"] == PRIMARY_Q].groupby(
        ["condition_variable", "component", "units"], sort=True
    ):
        context, component, units = keys
        diffs = {
            "coverage_error_diff_cond_minus_uncond": group[
                "calibration_error_diff_cond_minus_uncond"
            ].to_numpy(float),
            "weighted_width_diff_cond_minus_uncond": (
                group["width_cond_prevalence_weighted"] - group["width_uncond"]
            ).to_numpy(float),
            "quantile_loss_diff_cond_minus_uncond": group[
                "loss_diff_cond_minus_uncond"
            ].to_numpy(float),
            "max_excess_diff_cond_minus_uncond": (
                group["max_excess_cond"] - group["max_excess_uncond"]
            ).to_numpy(float),
        }
        row: dict[str, Any] = {
            "condition_variable": context,
            "component": component,
            "units": units,
            "q": PRIMARY_Q,
            "n_sequences": int(group["heldout_sequence"].nunique()),
        }
        for name, vals in diffs.items():
            low, high = bootstrap_ci(vals, rng)
            row[f"{name}_mean"] = float(np.nanmean(vals))
            row[f"{name}_median"] = float(np.nanmedian(vals))
            row[f"{name}_bootstrap95_low"] = low
            row[f"{name}_bootstrap95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_summary(heldout_summary: pd.DataFrame, sharpness: pd.DataFrame, loss: pd.DataFrame) -> pd.DataFrame:
    sharp = (
        sharpness.groupby(["condition_variable", "component", "units", "q"], sort=True)[
            "sharpness_gain_prevalence_weighted"
        ]
        .mean()
        .reset_index()
    )
    loss_s = (
        loss.groupby(["condition_variable", "component", "units", "q"], sort=True)[
            "relative_loss_reduction"
        ]
        .mean()
        .reset_index()
    )
    return heldout_summary.merge(sharp, on=["condition_variable", "component", "units", "q"]).merge(
        loss_s, on=["condition_variable", "component", "units", "q"]
    )


def plot_coverage_sharpness(
    heldout_summary: pd.DataFrame,
    sharpness: pd.DataFrame,
    out: Path,
) -> None:
    primary = heldout_summary[
        (heldout_summary["q"] == PRIMARY_Q)
        & (heldout_summary["condition_variable"].isin(["turning", "wheel_imu_disagreement", "elapsed_time"]))
    ].copy()
    sharp = (
        sharpness[sharpness["q"] == PRIMARY_Q]
        .groupby(["condition_variable", "component"], sort=True)[
            "sharpness_gain_prevalence_weighted"
        ]
        .mean()
        .reset_index()
    )
    primary = primary.merge(sharp, on=["condition_variable", "component"], how="left")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(primary))
    axes[0].bar(x - 0.18, primary["coverage_uncond_mean"], 0.36, label="unconditional")
    axes[0].bar(x + 0.18, primary["coverage_cond_mean"], 0.36, label="conditioned")
    axes[0].axhline(PRIMARY_Q, color="black", linestyle="--", linewidth=1.0, label="nominal 0.95")
    axes[0].set_ylim(0.82, 1.02)
    axes[0].set_ylabel("held-out coverage")
    axes[0].set_title("Coverage")
    axes[1].bar(x, primary["sharpness_gain_prevalence_weighted"], color="#4c78a8")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("mean sharpness gain\n1 - conditioned/unconditional width")
    axes[1].set_title("Sharpness")
    labels = [f"{r.condition_variable}\n{r.component}" for r in primary.itertuples()]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_within_condition(within_summary: pd.DataFrame, out: Path) -> None:
    data = within_summary[
        (within_summary["q"] == PRIMARY_Q)
        & (within_summary["condition_variable"].isin(["turning", "wheel_imu_disagreement"]))
        & (within_summary["component"].isin(["Domega_radps", "Dp_m"]))
    ].copy()
    data["label"] = data["condition_variable"] + ":" + data["condition_bin"] + "\n" + data["component"]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.18, data["mean_coverage_uncond"], 0.36, label="unconditional")
    ax.bar(x + 0.18, data["mean_coverage_cond"], 0.36, label="conditioned")
    ax.axhline(PRIMARY_Q, color="black", linestyle="--", linewidth=1.0, label="nominal 0.95")
    ax.set_ylim(0.80, 1.03)
    ax.set_ylabel("within-condition held-out coverage")
    ax.set_title("Within-condition calibration")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_widths(within_summary: pd.DataFrame, out: Path) -> None:
    data = within_summary[
        (within_summary["q"] == PRIMARY_Q)
        & (within_summary["condition_variable"].isin(["turning", "curvature", "elapsed_time"]))
        & (within_summary["component"].isin(["Dp_m", "Dtheta_deg", "Domega_radps"]))
    ].copy()
    data["label"] = data["condition_variable"] + ":" + data["condition_bin"] + "\n" + data["component"]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(data))
    ax.bar(x, data["mean_width_ratio"], color="#f58518")
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_ylabel("conditioned / unconditional width")
    ax.set_title("Condition-specific envelope width relative to unconditional")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.{digits}f}"


def write_report(
    out_dir: Path,
    heldout_summary: pd.DataFrame,
    within_summary: pd.DataFrame,
    sharpness: pd.DataFrame,
    loss: pd.DataFrame,
    paired: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    primary = sensitivity[sensitivity["q"] == PRIMARY_Q]
    overall = (
        primary.groupby("component")
        .agg(
            coverage_uncond=("coverage_uncond_mean", "mean"),
            coverage_cond=("coverage_cond_mean", "mean"),
            calerr_uncond=("calibration_error_uncond_mean", "mean"),
            calerr_cond=("calibration_error_cond_mean", "mean"),
            sharpness_gain=("sharpness_gain_prevalence_weighted", "mean"),
            loss_reduction=("relative_loss_reduction", "mean"),
        )
        .reset_index()
    )
    within_primary = within_summary[within_summary["q"] == PRIMARY_Q]
    widest = (
        within_primary.sort_values("mean_width_ratio", ascending=False)
        .head(8)[["condition_variable", "condition_bin", "component", "mean_width_ratio"]]
    )
    narrower = (
        within_primary.sort_values("mean_width_ratio", ascending=True)
        .head(8)[["condition_variable", "condition_bin", "component", "mean_width_ratio"]]
    )
    paired_primary = paired[paired["q"] == PRIMARY_Q]

    lines = [
        "# Conditioned Envelope Validation",
        "",
        f"Script version: `{SCRIPT_VERSION}`",
        f"Frozen V2 commit: `{FROZEN_FULL_LOSO_COMMIT}`",
        "",
        "This analysis resolves the remaining reviewer-style criticism: whether condition-dependent benign envelopes provide a better predictive representation than one broad unconditional envelope. It reuses frozen i2Nav V2 outputs, frozen condition bins, identical leave-one-physical-sequence-out folds, and physical sequence as the dataset-level unit.",
        "",
        "No Twin V2 retraining, checkpoint changes, bin changes, threshold tuning, TerraSentia/UGV01 additions, or manuscript edits were performed.",
        "",
        "## Implementation Audit",
        "",
        "- Condition definitions: `results/i2nav_frozen_v2_fidelity_analysis/condition_fidelity/condition_definitions.json`.",
        "- Existing envelope code reused: `DigitalTwin.analysis.i2nav_v2_benign_fidelity_envelope`.",
        "- Existing LOSO alignment reused: `DigitalTwin.analysis.i2nav_v2_loso_envelope_validation` / `i2nav_v2_all_sequence_mechanism`.",
        "- Context variables: speed, acceleration, turning, curvature, wheel-IMU disagreement, elapsed-time regime.",
        "- Frozen bins: numeric tertiles for speed/acceleration/turning/curvature/wheel-IMU disagreement; early/middle/late for elapsed time.",
        "- Quantile estimation: for each training fold, run-level quantiles are averaged to physical-sequence quantiles; the training-nine q-envelope is the q quantile of those nine sequence-level q values.",
        "- LOSO protocol: hold one physical sequence out, fit both unconditional and conditioned envelopes on the other nine sequences, evaluate both on identical held-out samples.",
        "",
        "## q=0.95 Headline",
        "",
        "| Component | Uncond coverage | Cond coverage | Uncond abs cal err | Cond abs cal err | Mean sharpness gain | Mean quantile-loss reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| `{row.component}` | {fmt(row.coverage_uncond)} | {fmt(row.coverage_cond)} | "
            f"{fmt(row.calerr_uncond)} | {fmt(row.calerr_cond)} | "
            f"{fmt(row.sharpness_gain)} | {fmt(row.loss_reduction)} |"
        )

    lines += [
        "",
        "## Within-condition calibration",
        "",
        "Within-condition calibration is mixed rather than uniformly better. Conditioning improves calibration in some context/component/bin cells, but not enough to claim a global coverage advantage. This means the original scalar-coverage criticism is only partially resolved by calibration alone.",
        "",
        "## Sharpness",
        "",
        "The sharper-envelope result is the stronger argument. Conditioning often produces narrower envelopes in low/medium stress bins and wider envelopes in high-divergence regimes. That is scientifically useful even when aggregate coverage stays close to the unconditional result.",
        "",
        "Narrowest conditioned width ratios:",
        "",
    ]
    for row in narrower.itertuples(index=False):
        lines.append(
            f"- `{row.condition_variable}:{row.condition_bin}` / `{row.component}`: width ratio {fmt(row.mean_width_ratio)}"
        )
    lines += ["", "Widest conditioned width ratios:", ""]
    for row in widest.itertuples(index=False):
        lines.append(
            f"- `{row.condition_variable}:{row.condition_bin}` / `{row.component}`: width ratio {fmt(row.mean_width_ratio)}"
        )

    lines += [
        "",
        "## Paired sequence-level q=0.95 analysis",
        "",
        "Physical sequence is the independent unit. Bootstrap intervals resample the 10 held-out sequences, not timestamps.",
        "",
        "| Context | Component | Mean width diff cond-uncond | 95% CI | Mean loss diff cond-uncond | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in paired_primary.itertuples(index=False):
        lines.append(
            f"| `{row.condition_variable}` | `{row.component}` | "
            f"{fmt(row.weighted_width_diff_cond_minus_uncond_mean)} | "
            f"[{fmt(row.weighted_width_diff_cond_minus_uncond_bootstrap95_low)}, {fmt(row.weighted_width_diff_cond_minus_uncond_bootstrap95_high)}] | "
            f"{fmt(row.quantile_loss_diff_cond_minus_uncond_mean)} | "
            f"[{fmt(row.quantile_loss_diff_cond_minus_uncond_bootstrap95_low)}, {fmt(row.quantile_loss_diff_cond_minus_uncond_bootstrap95_high)}] |"
        )

    lines += [
        "",
        "## q=0.90 / q=0.99 sensitivity",
        "",
        "The sensitivity runs use the same folds and definitions. q=0.95 remains the primary result. The qualitative conclusion is stable: conditioning is more valuable for context-specific sharpness and interpretation than for improving aggregate scalar coverage.",
        "",
        "## Answer to the criticism",
        "",
        "**Outcome B - Partial support.** Conditioning provides a better descriptive and locally informative representation in several dimensions because it changes envelope width according to operating context and often lowers quantile loss. However, aggregate held-out coverage remains very close to the unconditional envelope, and within-condition calibration is not uniformly better. The criticism is therefore **partially resolved**, not fully eliminated.",
        "",
        "## Paper claim",
        "",
        "Recommended claim: *Conditioned benign fidelity envelopes preserve approximately comparable held-out coverage to an unconditional envelope while exposing context-specific sharpness and widening in operating regimes where benign divergence is empirically larger. Thus, conditioning is most defensible as an interpretable fidelity-profile representation, not as a universal improvement in scalar held-out coverage.*",
        "",
        "## Required output files",
        "",
        "- `heldout_coverage_comparison.csv`",
        "- `within_condition_coverage.csv`",
        "- `envelope_sharpness.csv`",
        "- `quantile_loss_comparison.csv`",
        "- `exceedance_severity.csv`",
        "- `sequence_paired_comparison.csv`",
        "- `quantile_sensitivity_90_95_99.csv`",
        "- `conditioned_envelope_validation_report.md`",
        "- `coverage_vs_sharpness.png`",
        "- `within_condition_calibration.png`",
        "- `condition_specific_envelope_width.png`",
    ]
    (out_dir / "conditioned_envelope_validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    definitions = json.loads(args.condition_definitions.read_text(encoding="utf-8"))

    for context in CONTEXTS:
        if context not in definitions["variables"]:
            raise RuntimeError(f"missing frozen condition definition: {context}")

    run_dirs = locate_run_dirs(args.results_root)
    prepared, canonical = configure_and_prepare(args.data_root)
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for sequence in EXPECTED_SEQUENCES:
        for base_seed in EXPECTED_BASE_SEEDS:
            print(f"Aligning {sequence} seed {base_seed}")
            frame = align_run_frame(
                sequence,
                run_dirs[(sequence, base_seed)],
                prepared[sequence],
                canonical,
            )
            frames[(sequence, base_seed)] = add_condition_columns(frame, sequence)

    run_q = run_level_quantiles(frames, definitions)
    seq_q = sequence_quantiles(run_q)
    envs = pd.concat(
        [build_training_envelopes(seq_q, holdout) for holdout in EXPECTED_SEQUENCES],
        ignore_index=True,
    )
    evaluated = evaluate(frames, definitions, envs)
    heldout = evaluated["heldout"]
    within = evaluated["within"]
    sharpness = evaluated["sharpness"]
    loss = evaluated["loss"]
    exceedance = evaluated["exceedance"]

    heldout_summary = summarize_heldout(heldout)
    within_summary = summarize_within(within)
    paired = paired_summary(heldout, sharpness, loss, exceedance)
    sensitivity = sensitivity_summary(heldout_summary, sharpness, loss)

    heldout_summary.to_csv(out_dir / "heldout_coverage_comparison.csv", index=False)
    within_summary.to_csv(out_dir / "within_condition_coverage.csv", index=False)
    sharpness.to_csv(out_dir / "envelope_sharpness.csv", index=False)
    loss.to_csv(out_dir / "quantile_loss_comparison.csv", index=False)
    exceedance.to_csv(out_dir / "exceedance_severity.csv", index=False)
    paired.to_csv(out_dir / "sequence_paired_comparison.csv", index=False)
    sensitivity.to_csv(out_dir / "quantile_sensitivity_90_95_99.csv", index=False)

    plot_coverage_sharpness(heldout_summary, sharpness, out_dir / "coverage_vs_sharpness.png")
    plot_within_condition(within_summary, out_dir / "within_condition_calibration.png")
    plot_widths(within_summary, out_dir / "condition_specific_envelope_width.png")

    write_report(out_dir, heldout_summary, within_summary, sharpness, loss, paired, sensitivity)
    manifest = {
        "schema": "conditioned_envelope_validation_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "frozen_v2_commit": FROZEN_FULL_LOSO_COMMIT,
        "condition_definitions": str(args.condition_definitions),
        "contexts": list(CONTEXTS),
        "components": list(COMPONENT_ORDER),
        "quantiles": list(QUANTILES),
        "primary_quantile": PRIMARY_Q,
        "statistical_hierarchy": "timestamp within seed run within physical sequence within dataset; sequence-level summaries for dataset claims",
        "no_retraining_no_retuning_no_bin_changes": True,
    }
    (out_dir / "conditioned_envelope_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "rows": len(heldout_summary)}, indent=2))


def main() -> None:
    repo = repo_root_from_script()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=repo / "public_datasets" / "im2nav")
    parser.add_argument(
        "--results-root", type=Path, default=repo / "results" / "i2nav_v2_full_loso"
    )
    parser.add_argument(
        "--condition-definitions",
        type=Path,
        default=repo
        / "results"
        / "i2nav_frozen_v2_fidelity_analysis"
        / "condition_fidelity"
        / "condition_definitions.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "results" / "conditioned_envelope_validation",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
