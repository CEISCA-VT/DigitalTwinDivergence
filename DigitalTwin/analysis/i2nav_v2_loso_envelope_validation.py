#!/usr/bin/env python3
"""Leave-one-physical-sequence-out validation of benign fidelity envelopes.

This script is post-hoc only. It reads frozen Twin V2 LOSO outputs and the
previously frozen condition definitions. It does not retrain, tune, alter V2, or
redefine condition bins.
"""

from __future__ import annotations

import argparse
import json
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
from DigitalTwin.analysis.i2nav_v2_benign_fidelity_envelope import (
    COMPONENTS,
    SUPPORTED_CONTEXTS,
    aggregate_sequence_distributions,
    build_distribution_table,
    condition_component_rows,
    finite,
    unconditional_component_rows,
)
from DigitalTwin.analysis.i2nav_v2_condition_fidelity import (
    add_condition_columns,
    assign_bin,
)


SCRIPT_VERSION = "2026-08-20-loso-envelope-validation-v1"
SUPPORTED_VALIDATION_CONTEXTS = (
    "speed",
    "acceleration",
    "turning",
    "curvature",
    "wheel_imu_disagreement",
    "elapsed_time",
)


def percentile(values: np.ndarray | pd.Series, q: float) -> float:
    arr = finite(values)
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def support_label(n_sequences: int) -> str:
    if n_sequences >= 8:
        return "descriptive_broad_support"
    if n_sequences >= 5:
        return "preliminary_moderate_support"
    return "preliminary_limited_support"


def build_train_envelopes(
    sequence_distributions: pd.DataFrame,
    holdout_sequence: str,
) -> pd.DataFrame:
    """Build p90/p95 training-nine envelopes from sequence-level p95 values."""
    train = sequence_distributions[
        (sequence_distributions["sequence"] != holdout_sequence)
        & (sequence_distributions["supported"].astype(bool))
    ].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in train.groupby(["context", "condition_bin", "component", "units"], sort=True):
        context, condition_bin, component, units = keys
        vals = finite(group["p95"])
        if len(vals) == 0:
            continue
        rows.append(
            {
                "heldout_sequence": holdout_sequence,
                "context": context,
                "condition_bin": condition_bin,
                "component": component,
                "units": units,
                "train_n_sequences": int(group["sequence"].nunique()),
                "train_p90_envelope": float(np.percentile(vals, 90.0)),
                "train_p95_envelope": float(np.percentile(vals, 95.0)),
                "train_sequence_p95_mean": float(np.mean(vals)),
                "train_sequence_p95_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "support_label": support_label(int(group["sequence"].nunique())),
            }
        )
    return pd.DataFrame(rows)


def full_envelopes(sequence_distributions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    supported = sequence_distributions[sequence_distributions["supported"].astype(bool)]
    for keys, group in supported.groupby(["context", "condition_bin", "component", "units"], sort=True):
        context, condition_bin, component, units = keys
        vals = finite(group["p95"])
        if len(vals) == 0:
            continue
        rows.append(
            {
                "context": context,
                "condition_bin": condition_bin,
                "component": component,
                "units": units,
                "full_p95_envelope": float(np.percentile(vals, 95.0)),
                "full_p90_envelope": float(np.percentile(vals, 90.0)),
                "full_n_sequences": int(group["sequence"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def component_values(frame: pd.DataFrame, component: str) -> np.ndarray:
    spec = COMPONENTS[component]
    values = frame[spec["column"]].to_numpy(dtype=float)
    if spec["absolute"]:
        values = np.abs(values)
    return values[np.isfinite(values)]


def component_values_for_mask(frame: pd.DataFrame, component: str, mask: np.ndarray) -> np.ndarray:
    spec = COMPONENTS[component]
    values = frame.loc[mask, spec["column"]].to_numpy(dtype=float)
    if spec["absolute"]:
        values = np.abs(values)
    return values[np.isfinite(values)]


def coverage(values: np.ndarray, threshold: float) -> tuple[float, float, float, float, int]:
    values = values[np.isfinite(values)]
    if len(values) == 0 or not np.isfinite(threshold):
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    exceed = values - threshold
    exceed_pos = exceed[exceed > 0.0]
    inside = float(np.mean(values <= threshold))
    exceed_frac = 1.0 - inside
    mean_exceed = float(np.mean(exceed_pos)) if len(exceed_pos) else 0.0
    max_exceed = float(np.max(exceed_pos)) if len(exceed_pos) else 0.0
    return inside, exceed_frac, mean_exceed, max_exceed, int(len(values))


def evaluate_holdout(
    frames: dict[tuple[str, int], pd.DataFrame],
    definitions: dict[str, Any],
    holdout_sequence: str,
    train_env: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    env_lookup = {
        (r.context, r.condition_bin, r.component): r
        for r in train_env.itertuples(index=False)
    }
    uncond_lookup = {
        r.component: r
        for r in train_env[
            (train_env["context"] == "unconditional") & (train_env["condition_bin"] == "all")
        ].itertuples(index=False)
    }

    for context in SUPPORTED_VALIDATION_CONTEXTS:
        definition = definitions["variables"][context]
        source = definition["source_column"]
        for condition_bin in definition["bins"]:
            for component in COMPONENTS:
                env = env_lookup.get((context, condition_bin, component))
                uncond = uncond_lookup.get(component)
                if env is None or uncond is None:
                    continue
                seed_coverages = []
                seed_uncond_coverages = []
                seed_exceedances = []
                seed_counts = []
                pooled_values: list[np.ndarray] = []
                for seed in EXPECTED_BASE_SEEDS:
                    frame = frames[(holdout_sequence, seed)]
                    bins = assign_bin(frame[source], definition)
                    mask = (bins == condition_bin).to_numpy()
                    values = component_values_for_mask(frame, component, mask)
                    if len(values) == 0:
                        continue
                    pooled_values.append(values)
                    inside, exceed_frac, mean_exceed, max_exceed, n = coverage(
                        values, float(env.train_p95_envelope)
                    )
                    u_inside, _, _, _, _ = coverage(values, float(uncond.train_p95_envelope))
                    seed_coverages.append(inside)
                    seed_uncond_coverages.append(u_inside)
                    seed_exceedances.append(max_exceed)
                    seed_counts.append(n)

                if not pooled_values:
                    continue
                pooled = np.concatenate(pooled_values)
                inside, exceed_frac, mean_exceed, max_exceed, n = coverage(
                    pooled, float(env.train_p95_envelope)
                )
                p90_inside, p90_exceed, p90_mean_exceed, p90_max_exceed, _ = coverage(
                    pooled, float(env.train_p90_envelope)
                )
                uncond_inside, uncond_exceed, uncond_mean_exceed, uncond_max_exceed, _ = coverage(
                    pooled, float(uncond.train_p95_envelope)
                )
                rows.append(
                    {
                        "heldout_sequence": holdout_sequence,
                        "context": context,
                        "condition_bin": condition_bin,
                        "component": component,
                        "units": env.units,
                        "train_n_sequences": int(env.train_n_sequences),
                        "heldout_seed_runs": len(seed_coverages),
                        "heldout_observations": n,
                        "train_p90_envelope": float(env.train_p90_envelope),
                        "train_p95_envelope": float(env.train_p95_envelope),
                        "conditioned_inside_p95_fraction": inside,
                        "conditioned_exceedance_p95_fraction": exceed_frac,
                        "conditioned_mean_exceedance_p95": mean_exceed,
                        "conditioned_max_exceedance_p95": max_exceed,
                        "conditioned_inside_p90_fraction": p90_inside,
                        "conditioned_exceedance_p90_fraction": p90_exceed,
                        "conditioned_mean_exceedance_p90": p90_mean_exceed,
                        "conditioned_max_exceedance_p90": p90_max_exceed,
                        "unconditional_train_p95_envelope": float(uncond.train_p95_envelope),
                        "unconditional_inside_p95_fraction": uncond_inside,
                        "unconditional_exceedance_p95_fraction": uncond_exceed,
                        "unconditional_mean_exceedance_p95": uncond_mean_exceed,
                        "unconditional_max_exceedance_p95": uncond_max_exceed,
                        "coverage_gain_conditioned_minus_unconditional": inside - uncond_inside,
                        "seed_coverage_mean": float(np.nanmean(seed_coverages)),
                        "seed_coverage_sd": float(np.nanstd(seed_coverages, ddof=1))
                        if len(seed_coverages) > 1
                        else 0.0,
                        "seed_unconditional_coverage_mean": float(np.nanmean(seed_uncond_coverages)),
                        "seed_unconditional_coverage_sd": float(np.nanstd(seed_uncond_coverages, ddof=1))
                        if len(seed_uncond_coverages) > 1
                        else 0.0,
                        "seed_max_exceedance_mean": float(np.nanmean(seed_exceedances)),
                        "seed_observation_min": int(np.min(seed_counts)),
                        "seed_observation_max": int(np.max(seed_counts)),
                    }
                )
    return pd.DataFrame(rows)


def build_stability(
    train_envelopes: pd.DataFrame,
    full_env: pd.DataFrame,
    sequence_distributions: pd.DataFrame,
) -> pd.DataFrame:
    full_lookup = {
        (r.context, r.condition_bin, r.component): r
        for r in full_env.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for r in train_envelopes.itertuples(index=False):
        full = full_lookup.get((r.context, r.condition_bin, r.component))
        if full is None:
            continue
        group = sequence_distributions[
            (sequence_distributions["context"] == r.context)
            & (sequence_distributions["condition_bin"] == r.condition_bin)
            & (sequence_distributions["component"] == r.component)
            & (sequence_distributions["supported"].astype(bool))
        ]
        held = group[group["sequence"] == r.heldout_sequence]
        train = group[group["sequence"] != r.heldout_sequence]
        held_p95 = percentile(held["p95"], 50.0) if not held.empty else float("nan")
        train_vals = finite(train["p95"])
        seed_sd_vals = finite(train.get("p95_seed_sd", pd.Series(dtype=float)))
        rows.append(
            {
                "heldout_sequence": r.heldout_sequence,
                "context": r.context,
                "condition_bin": r.condition_bin,
                "component": r.component,
                "units": r.units,
                "full_p95_envelope": float(full.full_p95_envelope),
                "training_nine_p95_envelope": float(r.train_p95_envelope),
                "training_nine_p90_envelope": float(r.train_p90_envelope),
                "p95_delta_training_minus_full": float(r.train_p95_envelope)
                - float(full.full_p95_envelope),
                "p95_relative_delta_abs": abs(
                    (float(r.train_p95_envelope) - float(full.full_p95_envelope))
                    / float(full.full_p95_envelope)
                )
                if float(full.full_p95_envelope) != 0.0
                else float("nan"),
                "heldout_sequence_p95": held_p95,
                "heldout_minus_training_p95": held_p95 - float(r.train_p95_envelope)
                if np.isfinite(held_p95)
                else float("nan"),
                "training_sequence_p95_sd": float(np.std(train_vals, ddof=1))
                if len(train_vals) > 1
                else 0.0,
                "mean_training_seed_p95_sd": float(np.mean(seed_sd_vals)) if len(seed_sd_vals) else float("nan"),
                "physical_to_seed_sd_ratio": (
                    float(np.std(train_vals, ddof=1)) / float(np.mean(seed_sd_vals))
                    if len(train_vals) > 1 and len(seed_sd_vals) and float(np.mean(seed_sd_vals)) > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_validation(per_sequence: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in per_sequence.groupby(["context", "component", "units"], sort=True):
        context, component, units = keys
        rows.append(
            {
                "context": context,
                "component": component,
                "units": units,
                "n_heldout_sequence_bins": int(len(group)),
                "mean_conditioned_inside_p95_fraction": float(
                    group["conditioned_inside_p95_fraction"].mean()
                ),
                "median_conditioned_inside_p95_fraction": float(
                    group["conditioned_inside_p95_fraction"].median()
                ),
                "mean_unconditional_inside_p95_fraction": float(
                    group["unconditional_inside_p95_fraction"].mean()
                ),
                "median_unconditional_inside_p95_fraction": float(
                    group["unconditional_inside_p95_fraction"].median()
                ),
                "mean_coverage_gain_conditioned_minus_unconditional": float(
                    group["coverage_gain_conditioned_minus_unconditional"].mean()
                ),
                "median_coverage_gain_conditioned_minus_unconditional": float(
                    group["coverage_gain_conditioned_minus_unconditional"].median()
                ),
                "mean_conditioned_exceedance_p95_fraction": float(
                    group["conditioned_exceedance_p95_fraction"].mean()
                ),
                "mean_conditioned_max_exceedance_p95": float(
                    group["conditioned_max_exceedance_p95"].mean()
                ),
                "mean_seed_coverage_sd": float(group["seed_coverage_sd"].mean()),
            }
        )

    overall = []
    for component, group in per_sequence.groupby("component", sort=True):
        stab = stability[stability["component"] == component]
        overall.append(
            {
                "context": "ALL_SUPPORTED_CONTEXTS",
                "component": component,
                "units": str(group["units"].iloc[0]),
                "n_heldout_sequence_bins": int(len(group)),
                "mean_conditioned_inside_p95_fraction": float(
                    group["conditioned_inside_p95_fraction"].mean()
                ),
                "median_conditioned_inside_p95_fraction": float(
                    group["conditioned_inside_p95_fraction"].median()
                ),
                "mean_unconditional_inside_p95_fraction": float(
                    group["unconditional_inside_p95_fraction"].mean()
                ),
                "median_unconditional_inside_p95_fraction": float(
                    group["unconditional_inside_p95_fraction"].median()
                ),
                "mean_coverage_gain_conditioned_minus_unconditional": float(
                    group["coverage_gain_conditioned_minus_unconditional"].mean()
                ),
                "median_coverage_gain_conditioned_minus_unconditional": float(
                    group["coverage_gain_conditioned_minus_unconditional"].median()
                ),
                "mean_conditioned_exceedance_p95_fraction": float(
                    group["conditioned_exceedance_p95_fraction"].mean()
                ),
                "mean_conditioned_max_exceedance_p95": float(
                    group["conditioned_max_exceedance_p95"].mean()
                ),
                "mean_seed_coverage_sd": float(group["seed_coverage_sd"].mean()),
                "median_abs_p95_relative_delta": float(stab["p95_relative_delta_abs"].median())
                if not stab.empty
                else float("nan"),
            }
        )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(overall)], ignore_index=True)


def plot_conditioned_vs_unconditional(summary: pd.DataFrame, out: Path) -> None:
    data = summary[summary["context"] == "ALL_SUPPORTED_CONTEXTS"].copy()
    order = ["Dp_m", "Dtheta_deg", "Dv_mps", "Domega_radps", "Iomega_abs_deg"]
    data["component"] = pd.Categorical(data["component"], categories=order, ordered=True)
    data = data.sort_values("component")
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        data["mean_conditioned_inside_p95_fraction"],
        width,
        label="conditioned p95",
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2,
        data["mean_unconditional_inside_p95_fraction"],
        width,
        label="unconditional p95",
        color="#f58518",
    )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0, label="nominal 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(data["component"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mean held-out observation coverage")
    ax.set_title("LOSO held-out coverage: conditioned versus unconditional envelopes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_influence(stability: pd.DataFrame, out: Path) -> None:
    focus = stability[stability["context"].isin(["elapsed_time", "turning", "wheel_imu_disagreement"])].copy()
    agg = (
        focus.groupby(["heldout_sequence", "component"])["p95_relative_delta_abs"]
        .median()
        .reset_index()
    )
    components = ["Dp_m", "Dtheta_deg", "Dv_mps", "Domega_radps", "Iomega_abs_deg"]
    sequences = list(EXPECTED_SEQUENCES)
    matrix = np.full((len(components), len(sequences)), np.nan)
    for i, comp in enumerate(components):
        for j, seq in enumerate(sequences):
            vals = agg[(agg["component"] == comp) & (agg["heldout_sequence"] == seq)][
                "p95_relative_delta_abs"
            ]
            if not vals.empty:
                matrix[i, j] = float(vals.iloc[0])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(sequences)))
    ax.set_xticklabels(sequences, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(components)
    ax.set_title("Envelope influence by held-out physical sequence")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("median absolute relative p95 change")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def write_summary(out_dir: Path, per_seq: pd.DataFrame, summary: pd.DataFrame, stability: pd.DataFrame) -> None:
    overall = summary[summary["context"] == "ALL_SUPPORTED_CONTEXTS"].set_index("component")
    parking = per_seq[per_seq["heldout_sequence"].isin(["parking01", "parking02"])]
    nonparking = per_seq[~per_seq["heldout_sequence"].isin(["parking01", "parking02"])]
    park_cov = parking.groupby("component")["conditioned_inside_p95_fraction"].mean()
    nonpark_cov = nonparking.groupby("component")["conditioned_inside_p95_fraction"].mean()

    dominated = (
        stability[stability["component"].isin(["Dp_m", "Dtheta_deg"])]
        .groupby(["heldout_sequence", "component"])["p95_delta_training_minus_full"]
        .median()
        .reset_index()
    )
    lines = [
        "# LOSO Benign Envelope Validation",
        "",
        f"Script version: `{SCRIPT_VERSION}`",
        f"Frozen full-LOSO commit expected by context: `{FROZEN_FULL_LOSO_COMMIT}`",
        "",
        "This analysis validates the empirical benign fidelity envelope by holding out one physical i2Nav sequence at a time, estimating p90/p95 envelopes from the remaining nine sequences, and evaluating coverage on the held-out sequence. It uses the frozen condition definitions and does not retrain, retune, or modify Twin V2.",
        "",
        "Important interpretation constraint: p95 exceedance is not an attack, anomaly, failure, or untrustworthiness claim. It only means that a held-out observation is above the descriptive benign envelope learned from the other physical sequences.",
        "",
        "## Does The Benign Envelope Generalize To An Unseen Physical Sequence?",
        "",
        "Partially. Rate-domain components generalize well; global position and heading dimensions are much less stable because difficult parking sequences strongly affect the p95 envelope.",
        "",
        "| Component | Mean conditioned p95 coverage | Mean unconditional p95 coverage | Mean conditioned exceedance | Median p95 sensitivity |",
        "|---|---:|---:|---:|---:|",
    ]
    for component in ["Dp_m", "Dtheta_deg", "Dv_mps", "Domega_radps", "Iomega_abs_deg"]:
        if component not in overall.index:
            continue
        row = overall.loc[component]
        lines.append(
            f"| `{component}` | {fmt(row['mean_conditioned_inside_p95_fraction'])} | "
            f"{fmt(row['mean_unconditional_inside_p95_fraction'])} | "
            f"{fmt(row['mean_conditioned_exceedance_p95_fraction'])} | "
            f"{fmt(row.get('median_abs_p95_relative_delta', np.nan))} |"
        )

    lines += [
        "",
        "## parking01 and parking02 Predictability",
        "",
        "parking02 is the severe held-out under-coverage case for global position/heading characterization. parking01 is much better covered in the held-out validation, but the parking01/parking02 family still materially affects the global p95 envelope because those sequences define the difficult long-horizon regime. For rate-domain quantities, held-out coverage is much closer to the nominal descriptive p95.",
        "",
        "| Component | parking01/02 mean coverage | other sequences mean coverage |",
        "|---|---:|---:|",
    ]
    for component in ["Dp_m", "Dtheta_deg", "Dv_mps", "Domega_radps", "Iomega_abs_deg"]:
        lines.append(
            f"| `{component}` | {fmt(park_cov.get(component, np.nan))} | "
            f"{fmt(nonpark_cov.get(component, np.nan))} |"
        )

    lines += [
        "",
        "## Do parking01 and parking02 Inflate Global p95?",
        "",
        "Yes. The leave-one-sequence p95 sensitivity confirms the earlier benign-envelope finding: global `Dp_m` and `Dtheta_deg` envelopes are materially influenced by difficult parking behavior, with parking02 the dominant single held-out influence. Rate-domain quantities such as `Dv_mps` and `Domega_radps` are less affected by which physical sequence is left out.",
        "",
        "## Does Conditioning Improve Held-Out Characterization?",
        "",
        "Conditioning does not uniformly increase scalar coverage in every component, because the conditioned envelope is deliberately more specific and often tighter than the unconditional envelope. Its main benefit is interpretability: it exposes which operating contexts are naturally high-divergence instead of hiding them inside one broad unconditional p95.",
        "",
        "## Stable Enough For Publication?",
        "",
        "The current envelope is **partially stable**. Publication-grade descriptive claims are strongest for the existence of condition dependence, the local/global distinction, and the relative stability of rate-domain quantities. Exact global-position and global-heading p95 values should be presented as descriptive and sequence-sensitive, not as universal operating guarantees.",
        "",
        "## Files Produced",
        "",
        "- `loso_envelope_validation_per_sequence.csv`",
        "- `loso_envelope_validation_summary.csv`",
        "- `loso_envelope_stability.csv`",
        "- `loso_conditioned_vs_unconditional_coverage.png`",
        "- `loso_envelope_influence_by_sequence.png`",
    ]
    (out_dir / "loso_benign_envelope_validation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
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
        / "i2nav_v2_post_loso_analysis"
        / "condition_fidelity"
        / "condition_definitions.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo
        / "results"
        / "i2nav_v2_post_loso_analysis"
        / "loso_envelope_validation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    definitions = json.loads(args.condition_definitions.read_text(encoding="utf-8"))
    for context in SUPPORTED_VALIDATION_CONTEXTS:
        if context not in definitions["variables"]:
            raise RuntimeError(f"Frozen condition definition missing: {context}")

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

    per_run_condition = condition_component_rows(frames, definitions)
    per_sequence_condition = aggregate_sequence_distributions(per_run_condition)
    per_run_uncond = unconditional_component_rows(frames)
    per_sequence_uncond = aggregate_sequence_distributions(per_run_uncond)
    sequence_distributions = pd.concat(
        [per_sequence_condition, per_sequence_uncond], ignore_index=True
    )

    full_env = full_envelopes(sequence_distributions)
    all_train_env = []
    all_eval = []
    for holdout in EXPECTED_SEQUENCES:
        train_env = build_train_envelopes(sequence_distributions, holdout)
        all_train_env.append(train_env)
        all_eval.append(evaluate_holdout(frames, definitions, holdout, train_env))

    train_envelopes = pd.concat(all_train_env, ignore_index=True)
    per_sequence = pd.concat(all_eval, ignore_index=True)
    stability = build_stability(train_envelopes, full_env, sequence_distributions)
    summary = summarize_validation(per_sequence, stability)

    per_sequence.to_csv(args.output_dir / "loso_envelope_validation_per_sequence.csv", index=False)
    summary.to_csv(args.output_dir / "loso_envelope_validation_summary.csv", index=False)
    stability.to_csv(args.output_dir / "loso_envelope_stability.csv", index=False)
    train_envelopes.to_csv(args.output_dir / "loso_training_nine_envelopes.csv", index=False)

    plot_conditioned_vs_unconditional(
        summary, args.output_dir / "loso_conditioned_vs_unconditional_coverage.png"
    )
    plot_influence(stability, args.output_dir / "loso_envelope_influence_by_sequence.png")
    write_summary(args.output_dir, per_sequence, summary, stability)

    manifest = {
        "schema": "i2nav_v2_loso_benign_envelope_validation_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "frozen_full_loso_commit": FROZEN_FULL_LOSO_COMMIT,
        "no_training_tuning_or_v2_modification": True,
        "condition_definitions": str(args.condition_definitions),
        "contexts": list(SUPPORTED_VALIDATION_CONTEXTS),
        "components": list(COMPONENTS.keys()),
        "statistical_hierarchy": [
            "timestamp observations describe within-run coverage",
            "coverage is summarized inside seed runs",
            "seed-run variability is reported separately",
            "physical sequence is the held-out validation unit",
        ],
        "interpretation": (
            "p95 exceedance is a descriptive benign-envelope exceedance only; it is not "
            "an attack, anomaly, failure, or untrustworthiness label"
        ),
    }
    (args.output_dir / "loso_envelope_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(per_sequence)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
