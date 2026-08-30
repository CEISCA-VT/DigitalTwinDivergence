#!/usr/bin/env python3
"""Empirical benign fidelity characterization for frozen Twin V2.

This script constructs descriptive, componentwise benign fidelity profiles and
conditioned p95 envelopes from saved frozen V2 LOSO artifacts. It does not
train, tune, modify V2, redefine condition bins, or construct a detector.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from DigitalTwin.analysis.i2nav_v2_condition_fidelity import (
    SCRIPT_VERSION as CONDITION_SCRIPT_VERSION,
    add_condition_columns,
    assign_bin,
)


SCRIPT_VERSION = "2026-08-20-benign-fidelity-envelope-v1"
SUPPORTED_CONTEXTS = (
    "speed",
    "acceleration",
    "turning",
    "curvature",
    "wheel_imu_disagreement",
    "elapsed_time",
)
PROFILE_COLUMNS = (
    "sequence",
    "base_seed",
    "replicate",
    "ATE_m",
    "heading_MAE_deg",
    "RPE1_m",
    "RPE5_m",
    "RPE10_m",
    "persistent_signed_yaw_residual_radps",
    "persistent_abs_yaw_mismatch_deg_per_min",
    "Iomega_max_abs_deg",
    "Dp_p95_m",
    "Dp_max_m",
    "Dtheta_p95_deg",
    "Dtheta_max_deg",
)
COMPONENTS = {
    "Dp_m": {"column": "Dp_m", "units": "m", "absolute": False},
    "Dtheta_deg": {"column": "Dtheta_deg", "units": "deg", "absolute": False},
    "Dv_mps": {"column": "rv_mps", "units": "m/s", "absolute": True},
    "Domega_radps": {"column": "romega_radps", "units": "rad/s", "absolute": True},
    "Iomega_abs_deg": {"column": "Iomega_deg", "units": "deg", "absolute": True},
}


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def percentile(values: pd.Series | np.ndarray, q: float) -> float:
    arr = finite(values)
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def read_profile_rows(mechanism_per_run: Path) -> pd.DataFrame:
    src = pd.read_csv(mechanism_per_run)
    out = pd.DataFrame(
        {
            "sequence": src["sequence"],
            "base_seed": src["base_seed"].astype(int),
            "replicate": src["base_seed"].map(
                {42: "replicate_01_base42", 1042: "replicate_02_base1042", 2042: "replicate_03_base2042"}
            ),
            "ATE_m": src["ATE_m"],
            "heading_MAE_deg": src["heading_MAE_deg"],
            "RPE1_m": src["RPE1_m"],
            "RPE5_m": src["RPE5_m"],
            "RPE10_m": src["RPE10_m"],
            "persistent_signed_yaw_residual_radps": src[
                "persistent_signed_yaw_residual_radps"
            ],
            "persistent_abs_yaw_mismatch_deg_per_min": src[
                "persistent_abs_yaw_residual_deg_per_min"
            ],
            "Iomega_max_abs_deg": src["Iomega_max_abs_deg"],
            "Dp_p95_m": src["Dp_p95_m"],
            "Dp_max_m": src["Dp_max_m"],
            "Dtheta_p95_deg": src["Dtheta_p95_deg"],
            "Dtheta_max_deg": src["Dtheta_max_deg"],
        }
    )
    return out[list(PROFILE_COLUMNS)].sort_values(["sequence", "base_seed"])


def condition_component_rows(
    frames: dict[tuple[str, int], pd.DataFrame],
    definitions: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sequence, seed), frame in frames.items():
        for context in SUPPORTED_CONTEXTS:
            definition = definitions["variables"][context]
            source = definition["source_column"]
            bins = assign_bin(frame[source], definition)
            for bin_name in definition["bins"]:
                mask = (bins == bin_name).to_numpy()
                subset = frame.loc[mask]
                dt = float(np.median(np.diff(frame["time_s"].to_numpy(dtype=float))))
                base = {
                    "sequence": sequence,
                    "base_seed": int(seed),
                    "context": context,
                    "condition_bin": bin_name,
                    "n_timestamps": int(len(subset)),
                    "duration_s_approx": float(len(subset) * dt),
                }
                for component, spec in COMPONENTS.items():
                    values = subset[spec["column"]].to_numpy(dtype=float)
                    if spec["absolute"]:
                        values = np.abs(values)
                    values = values[np.isfinite(values)]
                    row = dict(base)
                    row.update(
                        {
                            "component": component,
                            "units": spec["units"],
                            "median": float(np.median(values)) if len(values) else np.nan,
                            "mean": float(np.mean(values)) if len(values) else np.nan,
                            "p90": float(np.percentile(values, 90.0)) if len(values) else np.nan,
                            "p95": float(np.percentile(values, 95.0)) if len(values) else np.nan,
                            "max": float(np.max(values)) if len(values) else np.nan,
                            "supported": bool(len(values) >= 30),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def aggregate_sequence_distributions(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["median", "mean", "p90", "p95", "max", "n_timestamps", "duration_s_approx"]
    rows: list[dict[str, Any]] = []
    group_cols = ["sequence", "context", "condition_bin", "component", "units"]
    for keys, group in per_run.groupby(group_cols, sort=True):
        sequence, context, condition_bin, component, units = keys
        supported = group[group["supported"].astype(bool)]
        row: dict[str, Any] = {
            "sequence": sequence,
            "context": context,
            "condition_bin": condition_bin,
            "component": component,
            "units": units,
            "n_seed_rows": int(len(group)),
            "n_supported_seed_rows": int(len(supported)),
            "supported": bool(len(supported) > 0),
        }
        if len(supported):
            for col in metric_cols:
                vals = finite(supported[col])
                if len(vals):
                    row[col] = float(np.mean(vals))
                    row[f"{col}_seed_sd"] = (
                        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def build_distribution_table(per_sequence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["context", "condition_bin", "component", "units"]
    for keys, group in per_sequence.groupby(group_cols, sort=True):
        context, condition_bin, component, units = keys
        supported = group[group["supported"].astype(bool)]
        row: dict[str, Any] = {
            "context": context,
            "condition_bin": condition_bin,
            "component": component,
            "units": units,
            "n_sequences": int(supported["sequence"].nunique()),
            "n_sequence_condition_rows": int(len(supported)),
            "support_label": support_label(supported["sequence"].nunique()),
        }
        for stat in ("median", "mean", "p90", "p95", "max"):
            vals = finite(supported[stat]) if len(supported) else np.array([])
            row[f"sequence_mean_of_{stat}"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"sequence_median_of_{stat}"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"sequence_p95_of_{stat}"] = (
                float(np.percentile(vals, 95.0)) if len(vals) else np.nan
            )
            row[f"sequence_sd_of_{stat}"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if len(vals) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def support_label(n_sequences: int) -> str:
    if n_sequences >= 8:
        return "descriptive_broad_support"
    if n_sequences >= 5:
        return "preliminary_moderate_support"
    return "preliminary_limited_support"


def build_envelope(distributions: pd.DataFrame, q: float = 0.95) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    q_col = f"sequence_p{int(q * 100)}_of_p95"
    if q_col not in distributions.columns:
        raise ValueError(f"Missing distribution column {q_col}")
    for _, row in distributions.iterrows():
        rows.append(
            {
                "context": row["context"],
                "condition_bin": row["condition_bin"],
                "component": row["component"],
                "units": row["units"],
                "q": q,
                "benign_envelope_p95": row[q_col],
                "n_sequences": int(row["n_sequences"]),
                "support_label": row["support_label"],
                "interpretation": (
                    "descriptive 95th percentile of sequence-level within-condition p95; "
                    "not a detector or universal trust threshold"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_stability(per_sequence: pd.DataFrame, envelope: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["context", "condition_bin", "component", "units"]
    for keys, group in per_sequence.groupby(group_cols, sort=True):
        context, condition_bin, component, units = keys
        supported = group[group["supported"].astype(bool)].copy()
        vals = supported[["sequence", "p95"]].dropna()
        if vals.empty:
            continue
        all_values = vals["p95"].to_numpy(dtype=float)
        full = float(np.percentile(all_values, 95.0))
        loo_values = []
        for seq in vals["sequence"].unique():
            hold = vals[vals["sequence"] != seq]["p95"].to_numpy(dtype=float)
            if len(hold):
                loo_values.append((seq, float(np.percentile(hold, 95.0))))
        loo_arr = np.array([v for _, v in loo_values], dtype=float)
        max_loo_delta = float(np.max(np.abs(loo_arr - full))) if len(loo_arr) else np.nan
        dominant = ""
        if len(loo_values):
            dominant = max(loo_values, key=lambda item: abs(item[1] - full))[0]

        p01p02 = vals[vals["sequence"].isin(["parking01", "parking02"])]["p95"].to_numpy(dtype=float)
        without_p = vals[~vals["sequence"].isin(["parking01", "parking02"])]["p95"].to_numpy(dtype=float)
        without_p95 = float(np.percentile(without_p, 95.0)) if len(without_p) else np.nan

        seed_sd = per_sequence[
            (per_sequence["context"] == context)
            & (per_sequence["condition_bin"] == condition_bin)
            & (per_sequence["component"] == component)
        ]["p95_seed_sd"]
        rows.append(
            {
                "context": context,
                "condition_bin": condition_bin,
                "component": component,
                "units": units,
                "full_sequence_level_p95": full,
                "loo_min_p95": float(np.min(loo_arr)) if len(loo_arr) else np.nan,
                "loo_max_p95": float(np.max(loo_arr)) if len(loo_arr) else np.nan,
                "loo_max_abs_delta": max_loo_delta,
                "loo_most_influential_sequence": dominant,
                "without_parking01_02_p95": without_p95,
                "parking01_02_influence_delta": full - without_p95
                if np.isfinite(without_p95)
                else np.nan,
                "parking01_02_max_sequence_p95": float(np.max(p01p02)) if len(p01p02) else np.nan,
                "mean_seed_sd_of_sequence_p95": float(np.nanmean(seed_sd.to_numpy(dtype=float))),
                "n_sequences": int(vals["sequence"].nunique()),
                "support_label": support_label(vals["sequence"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def trust_coverage(values: np.ndarray, tolerance: float) -> float:
    """rho_j(c; delta_j) = P(D_j <= delta_j | H0, c).

    This is intentionally parameterized. The script does not invent tolerances.
    """
    arr = finite(values)
    if not len(arr):
        return float("nan")
    return float(np.mean(arr <= float(tolerance)))


def plot_envelope_by_condition(envelope: pd.DataFrame, out: Path) -> None:
    components = ["Dp_m", "Dtheta_deg", "Dv_mps", "Domega_radps"]
    contexts = ["speed", "turning", "wheel_imu_disagreement", "elapsed_time"]
    fig, axes = plt.subplots(len(components), len(contexts), figsize=(16, 11), squeeze=False)
    for i, component in enumerate(components):
        for j, context in enumerate(contexts):
            ax = axes[i][j]
            q = envelope[
                (envelope["component"] == component)
                & (envelope["context"] == context)
            ]
            if q.empty:
                ax.axis("off")
                continue
            order = [b for b in ("low", "medium", "high", "early", "middle", "late") if b in set(q["condition_bin"])]
            vals = [
                float(q[q["condition_bin"] == b]["benign_envelope_p95"].iloc[0])
                for b in order
            ]
            ax.bar(order, vals, color="#4c78a8")
            ax.set_title(f"{component} | {context}", fontsize=9)
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle("Descriptive benign p95 envelope by supported condition")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_unconditional_vs_conditioned(envelope: pd.DataFrame, out: Path) -> None:
    components = ["Dp_m", "Dtheta_deg", "Dv_mps", "Domega_radps"]
    contexts = ["speed", "turning", "wheel_imu_disagreement", "elapsed_time"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    for ax, component in zip(axes.ravel(), components):
        q = envelope[envelope["component"] == component]
        unconditional = q[q["context"] == "unconditional"]
        if unconditional.empty:
            continue
        base = float(unconditional["benign_envelope_p95"].iloc[0])
        labels = ["uncond"]
        values = [base]
        for context in contexts:
            cq = q[q["context"] == context]
            if cq.empty:
                continue
            values.append(float(cq["benign_envelope_p95"].max()))
            labels.append(f"{context}\nmax")
        ax.bar(labels, values, color=["#999999"] + ["#f58518"] * (len(values) - 1))
        ax.set_title(component)
        ax.set_ylabel("p95 envelope")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Unconditional p95 can hide condition-dependent envelope expansion")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def unconditional_component_rows(frames: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sequence, seed), frame in frames.items():
        dt = float(np.median(np.diff(frame["time_s"].to_numpy(dtype=float))))
        base = {
            "sequence": sequence,
            "base_seed": int(seed),
            "context": "unconditional",
            "condition_bin": "all",
            "n_timestamps": int(len(frame)),
            "duration_s_approx": float(len(frame) * dt),
        }
        for component, spec in COMPONENTS.items():
            values = frame[spec["column"]].to_numpy(dtype=float)
            if spec["absolute"]:
                values = np.abs(values)
            values = values[np.isfinite(values)]
            row = dict(base)
            row.update(
                {
                    "component": component,
                    "units": spec["units"],
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "p90": float(np.percentile(values, 90.0)) if len(values) else np.nan,
                    "p95": float(np.percentile(values, 95.0)) if len(values) else np.nan,
                    "max": float(np.max(values)) if len(values) else np.nan,
                    "supported": bool(len(values) >= 30),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    definitions: dict[str, Any],
    profile_count: int,
) -> None:
    manifest = {
        "schema": "i2nav_v2_benign_fidelity_characterization_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "frozen_full_loso_commit": FROZEN_FULL_LOSO_COMMIT,
        "analysis_git_commit": git_commit(repo_root_from_script()),
        "source_artifacts": {
            "full_loso_results_root": str(args.results_root),
            "condition_definitions": str(args.condition_definitions),
            "all_sequence_mechanism_per_run": str(args.mechanism_per_run),
            "condition_fidelity_outputs": str(args.condition_dir),
            "architecture_independent_evaluator": "DigitalTwin/analysis/i2nav_fidelity_evaluator.py",
        },
        "frozen_condition_variables_used": list(SUPPORTED_CONTEXTS),
        "excluded_variables": {
            "lateral_slip_proxy": "unavailable in frozen canonical i2Nav context",
            "persistent_yaw_mismatch": "retained in profile/stability context but not used as a supported operating-context envelope variable",
        },
        "aggregation_hierarchy": [
            "within-run timestamp distributions are descriptive",
            "condition metrics are computed within each seed run",
            "three seeds are aggregated within each physical sequence",
            "physical sequence is the primary dataset-level unit",
        ],
        "percentile_definition": (
            "benign_envelope_p95 is the empirical 95th percentile across sequence-level "
            "seed-aggregated within-condition p95 values for each component"
        ),
        "limitations": [
            "Only 10 physical sequences are available for dataset-level inference.",
            "p95 envelope values are descriptive benign percentiles, not detection thresholds.",
            "parking01/parking02 can strongly influence long-horizon position and heading dimensions.",
            "The lateral/slip proxy is unavailable in the current frozen canonical context.",
            "Exceeding a p95 envelope only means uncommon relative to characterized benign data.",
        ],
        "profile_rows": profile_count,
        "condition_definitions_digest": definitions,
    }
    (out_dir / "benign_fidelity_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fmt(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def write_summary(
    out_dir: Path,
    envelope: pd.DataFrame,
    stability: pd.DataFrame,
    distributions: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    unconditional = envelope[envelope["context"] == "unconditional"]
    elapsed_dp = envelope[
        (envelope["context"] == "elapsed_time") & (envelope["component"] == "Dp_m")
    ]
    turning_rpe_proxy = envelope[
        (envelope["context"] == "turning") & (envelope["component"] == "Domega_radps")
    ]
    p_dp = stability[
        (stability["component"] == "Dp_m")
        & (stability["context"].isin(["speed", "turning", "elapsed_time", "wheel_imu_disagreement"]))
    ].copy()
    p_dp["abs_parking_delta"] = np.abs(p_dp["parking01_02_influence_delta"])
    top_parking = p_dp.sort_values("abs_parking_delta", ascending=False).head(3)

    stable = stability.copy()
    stable["relative_loo_delta"] = stable["loo_max_abs_delta"] / stable[
        "full_sequence_level_p95"
    ].replace(0.0, np.nan)
    stable_dims = (
        stable.groupby("component")["relative_loo_delta"].median().sort_values()
    )

    lines = [
        "# Empirical Benign Digital-Twin Fidelity Characterization",
        "",
        f"Script version: `{SCRIPT_VERSION}`",
        f"Frozen full-LOSO commit expected by context: `{FROZEN_FULL_LOSO_COMMIT}`",
        "",
        "This is a descriptive benign characterization, not a detector. Exceeding the "
        "p95 envelope does not mean attack, anomaly, failure, or untrustworthiness. It "
        "only means the observation is uncommon relative to the characterized benign data.",
        "",
        "## Formal Object",
        "",
        "The empirical object is `D(t) | H0, c ~ P_benign(D | c)`, where `c` is a "
        "benign operating context. The componentwise descriptive envelope is:",
        "",
        "`E_benign^(q)(c) = {D : D_j <= Q_q[D_j | H0, c], for all j}` with `q = 0.95`.",
        "",
        "Meters, degrees, m/s, and rad/s are kept as separate physical dimensions.",
        "",
        "## Is Benign Divergence Sufficiently Bounded To Characterize Empirically?",
        "",
        "Partly. The data are sufficient for a descriptive empirical characterization "
        "across the 10 i2Nav physical sequences and three seeds, especially for broad "
        "components such as `Dp_m`, `Dtheta_deg`, `Dv_mps`, and `Domega_radps`. The "
        "values should not be presented as universal stable operating limits because "
        "only 10 physical sequences define the dataset-level support.",
        "",
        "## Does The Envelope Depend On Operating Condition?",
        "",
    ]
    if not elapsed_dp.empty:
        early = elapsed_dp[elapsed_dp["condition_bin"] == "early"]["benign_envelope_p95"]
        late = elapsed_dp[elapsed_dp["condition_bin"] == "late"]["benign_envelope_p95"]
        if not early.empty and not late.empty:
            lines.append(
                f"Yes. For example, `Dp_m` p95 under the elapsed-time envelope "
                f"changes from {fmt(early.iloc[0])} m in the early-run bin to "
                f"{fmt(late.iloc[0])} m in the late-run bin."
            )
    lines += [
        "The unconditional envelope therefore hides condition dependence, particularly "
        "for long-horizon position/heading dimensions and time-accumulating quantities.",
        "",
        "## Stable Versus Highly Variable Dimensions",
        "",
        "| Component | median relative LOSO p95 sensitivity |",
        "|---|---:|",
    ]
    for component, value in stable_dims.items():
        lines.append(f"| `{component}` | {fmt(value)} |")

    lines += [
        "",
        "Lower relative LOSO sensitivity means the envelope component is more stable "
        "under leave-one-sequence-out perturbation. Larger values mean the p95 estimate "
        "is more dependent on which physical sequence is included.",
        "",
        "## parking01/parking02 Influence",
        "",
        "parking01 and parking02 materially influence long-horizon global dimensions, "
        "especially `Dp_m` and `Dtheta_deg`. The strongest `Dp_m` examples are:",
        "",
        "| Context | Bin | Full p95 | Without parking01/02 p95 | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in top_parking.iterrows():
        lines.append(
            f"| {row['context']} | {row['condition_bin']} | "
            f"{fmt(row['full_sequence_level_p95'])} | "
            f"{fmt(row['without_parking01_02_p95'])} | "
            f"{fmt(row['parking01_02_influence_delta'])} |"
        )

    lines += [
        "",
        "## Would An Unconditional p95 Envelope Obscure Important Behavior?",
        "",
        "Yes. A single unconditional p95 mixes easy street/playground behavior with hard "
        "parking behavior and hides the difference between local finite-horizon fidelity "
        "and accumulated global synchronization error. The conditioned envelope is more "
        "scientifically honest because it records where benign divergence is naturally "
        "larger.",
        "",
        "## Publication-Grade Now Versus Preliminary",
        "",
        "Publication-grade now:",
        "- the Twin Fidelity Profile database for the 30 frozen V2 runs;",
        "- the componentwise condition-dependent descriptive distributions;",
        "- the conclusion that unconditional p95 values obscure condition-dependent behavior;",
        "- the finding that parking01/parking02 strongly influence global envelope dimensions.",
        "",
        "Preliminary/descriptive rather than final operating guarantees:",
        "- per-condition p95 values with only a few effective physical-sequence units;",
        "- any condition/bin with fewer than 8 sequence-level supports;",
        "- any envelope dimension dominated by parking01/parking02;",
        "- lateral/slip behavior, because the supported canonical lateral proxy is unavailable here.",
        "",
        "## Application-Specific Trust Coverage Interface",
        "",
        "`rho_j(c; delta_j) = P(D_j <= delta_j | H0, c)` is implemented as a parameterized "
        "function in the analysis script. No application-independent tolerances are "
        "invented in this report.",
        "",
        "## Files Produced",
        "",
        "- `twin_fidelity_profiles.csv`",
        "- `benign_condition_distributions.csv`",
        "- `benign_envelope_p95.csv`",
        "- `benign_envelope_stability.csv`",
        "- `benign_envelope_by_condition.png`",
        "- `unconditional_vs_conditioned_envelope.png`",
        "- `benign_fidelity_manifest.json`",
    ]
    (out_dir / "benign_fidelity_framework_summary.md").write_text(
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
        "--condition-dir",
        type=Path,
        default=repo / "results" / "i2nav_frozen_v2_fidelity_analysis" / "condition_fidelity",
    )
    parser.add_argument(
        "--mechanism-per-run",
        type=Path,
        default=repo
        / "results"
        / "i2nav_frozen_v2_fidelity_analysis"
        / "all_sequence_mechanism"
        / "per_run_mechanism.csv",
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
        default=repo
        / "results"
        / "i2nav_frozen_v2_fidelity_analysis"
        / "benign_fidelity_characterization",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    definitions = json.loads(args.condition_definitions.read_text(encoding="utf-8"))
    for context in SUPPORTED_CONTEXTS:
        if context not in definitions["variables"]:
            raise RuntimeError(f"Frozen condition definition missing: {context}")
    if definitions["variables"].get("lateral_slip_proxy", {}).get("type") != "unavailable":
        raise RuntimeError("Expected lateral_slip_proxy to remain unavailable for this characterization")

    profiles = read_profile_rows(args.mechanism_per_run)
    profiles.to_csv(args.output_dir / "twin_fidelity_profiles.csv", index=False)

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

    per_run = condition_component_rows(frames, definitions)
    per_sequence = aggregate_sequence_distributions(per_run)
    uncond_run = unconditional_component_rows(frames)
    uncond_seq = aggregate_sequence_distributions(uncond_run)
    all_seq = pd.concat([per_sequence, uncond_seq], ignore_index=True)

    distributions = build_distribution_table(all_seq)
    envelope = build_envelope(distributions, q=0.95)
    stability = build_stability(all_seq, envelope)

    distributions.to_csv(args.output_dir / "benign_condition_distributions.csv", index=False)
    envelope.to_csv(args.output_dir / "benign_envelope_p95.csv", index=False)
    stability.to_csv(args.output_dir / "benign_envelope_stability.csv", index=False)

    plot_envelope_by_condition(envelope, args.output_dir / "benign_envelope_by_condition.png")
    plot_unconditional_vs_conditioned(
        envelope, args.output_dir / "unconditional_vs_conditioned_envelope.png"
    )
    write_manifest(args.output_dir, args, definitions, len(profiles))
    write_summary(args.output_dir, envelope, stability, distributions, profiles)

    print("Saved outputs:")
    for path in sorted(args.output_dir.iterdir()):
        print(" ", path.name)
    print("\nUnconditional envelope:")
    q = envelope[envelope["context"] == "unconditional"][
        ["component", "benign_envelope_p95", "units", "support_label"]
    ]
    print(q.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
