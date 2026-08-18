#!/usr/bin/env python3
"""
i2nav_context_length_identifiability.py
=======================================

Targeted diagnostic for Twin V2 persistent-yaw identifiability.

This script does NOT train V1 or V2 and does not modify any frozen results.

Question
--------
Does a longer causal ODO+IMU history make the required persistent yaw
correction easier to distinguish between parking01 and parking02?

It repeats the parking01 <-> parking02 nearest-history diagnostic at
2, 5, 10, 20, and 30 seconds of context while keeping the target fixed:

    target = causal 30-second mean of
             (GT yaw rate - V1 nominal IMU yaw rate)

For each context length, it:
  * builds exact V1 ODO+IMU feature histories,
  * standardizes the history dimensions for diagnostic distance only,
  * finds the nearest history in the other parking sequence,
  * measures how different the required 30 s yaw correction is,
  * reports ambiguity rates among the closest 50% and closest 25% matches.

Interpretation
--------------
If longer context sharply reduces the persistent-bias gap among close
cross-sequence histories, a slow causal estimator with longer memory is
scientifically justified.

If even 20-30 s histories remain highly ambiguous, simply increasing GRU
memory is unlikely to solve parking02; the next step should be additional
physically informative lightweight observables/state.

This is a diagnostic, not a formal proof of identifiability.

Example (Windows PowerShell)
----------------------------
python -u -m DigitalTwin.analysis.i2nav_context_length_identifiability `
    --root ./public_datasets/im2nav `
    --output-dir ./results/i2nav_context_length_identifiability
"""

from __future__ import annotations

import argparse
import importlib
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
        description=(
            "Test whether longer causal ODO+IMU histories reduce "
            "parking01/parking02 persistent-yaw ambiguity."
        )
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
        default=Path("results/i2nav_context_length_identifiability"),
        help="Output directory.",
    )
    p.add_argument(
        "--contexts-s",
        type=str,
        default="2,5,10,20,30",
        help="Comma-separated causal history lengths in seconds.",
    )
    p.add_argument(
        "--target-s",
        type=float,
        default=30.0,
        help="Causal persistent-yaw target window in seconds.",
    )
    p.add_argument(
        "--step-s",
        type=float,
        default=1.0,
        help=(
            "Time spacing between sampled history endpoints. "
            "Default 1 s keeps the diagnostic fast on a laptop."
        ),
    )
    p.add_argument(
        "--query-chunk",
        type=int,
        default=256,
        help="Query chunk size for exact cross-sequence nearest-neighbor search.",
    )
    return p.parse_args()


def original_default_args(original):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["i2nav_loso_ablation.py"]
        return original.parse_args()
    finally:
        sys.argv = old_argv


def rolling_mean_samples(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=samples, min_periods=samples)
        .mean()
        .to_numpy()
    )


def load_exact_sequences(root: Path):
    original = importlib.import_module(
        "DigitalTwin.analysis.i2nav_loso_ablation"
    )
    defaults = original_default_args(original)

    files_by_name = {x.name: x for x in original.discover_files(root)}

    prepared = {}
    for name in ("parking01", "parking02"):
        if name not in files_by_name:
            raise RuntimeError(
                f"Could not discover required sequence '{name}' under {root}."
            )

        prepared[name] = original.prepare_sequence(
            files_by_name[name],
            hz=defaults.rate_hz,
            imu_yaw_sign=defaults.imu_yaw_sign,
            gnss_sigma_max_m=defaults.gnss_sigma_max_m,
            gnss_anchor_count=defaults.gnss_anchor_count,
        )

    return prepared


def build_history_matrix(
    features: np.ndarray,
    context_samples: int,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Flatten causal histories ending at each index:
      [t-context+1, ..., t] x feature_dim.
    """
    features = np.asarray(features, dtype=np.float32)
    rows = []

    for idx in indices:
        idx = int(idx)
        start = idx - context_samples + 1
        rows.append(features[start : idx + 1].reshape(-1))

    return np.asarray(rows, dtype=np.float32)


def exact_nearest_cross_sequence(
    query: np.ndarray,
    reference: np.ndarray,
    query_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact Euclidean nearest neighbor, chunked so no huge NxM matrix persists.

    Returns:
      rms_distance_per_dimension, nearest_reference_index
    """
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)

    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("Query/reference history matrices are incompatible.")

    r_norm = np.sum(r * r, axis=1)
    best_d2 = np.full(len(q), np.inf, dtype=np.float64)
    best_idx = np.full(len(q), -1, dtype=np.int64)

    for start in range(0, len(q), max(1, int(query_chunk))):
        stop = min(start + max(1, int(query_chunk)), len(q))
        qc = q[start:stop]

        q_norm = np.sum(qc * qc, axis=1)[:, None]
        d2 = q_norm + r_norm[None, :] - 2.0 * (qc @ r.T)
        d2 = np.maximum(d2, 0.0)

        idx = np.argmin(d2, axis=1)
        vals = d2[np.arange(len(idx)), idx]

        best_d2[start:stop] = vals
        best_idx[start:stop] = idx

    # RMS standardized distance per dimension makes context lengths somewhat
    # more comparable than raw Euclidean norm. Do not over-interpret absolute
    # distance across dimensions; ambiguity fractions are the primary result.
    rms_distance = np.sqrt(best_d2 / max(q.shape[1], 1))

    return rms_distance, best_idx


def build_context_payload(
    seq,
    context_s: float,
    target_s: float,
    step_s: float,
):
    t = np.asarray(seq.grid, dtype=float)
    if len(t) < 2:
        raise RuntimeError(f"{seq.name}: time grid is too short.")

    dt_all = np.diff(t)
    good = np.isfinite(dt_all) & (dt_all > 0)
    if not np.any(good):
        raise RuntimeError(f"{seq.name}: invalid time grid.")

    dt = float(np.median(dt_all[good]))
    hz = 1.0 / dt

    context_samples = max(1, int(round(context_s * hz)))
    target_samples = max(1, int(round(target_s * hz)))
    step_samples = max(1, int(round(step_s * hz)))

    true_yaw_residual = np.asarray(
        seq.target_corrections[:, 1],
        dtype=float,
    )

    persistent_target = rolling_mean_samples(
        true_yaw_residual,
        target_samples,
    )

    # Need both a complete causal context and complete target window.
    first = max(context_samples - 1, target_samples - 1)

    indices = np.arange(
        first,
        len(t),
        step_samples,
        dtype=int,
    )

    valid = np.isfinite(persistent_target[indices])
    indices = indices[valid]

    if len(indices) < 20:
        raise RuntimeError(
            f"{seq.name}: only {len(indices)} valid histories for "
            f"{context_s:g}s context."
        )

    X = build_history_matrix(
        np.asarray(seq.features, dtype=np.float32),
        context_samples,
        indices,
    )

    y = persistent_target[indices]

    return {
        "X": X,
        "y": y,
        "indices": indices,
        "time_s": t[indices],
        "dt": dt,
        "hz": hz,
        "context_samples": context_samples,
        "target_samples": target_samples,
        "step_samples": step_samples,
    }


def summarize_direction(
    context_s: float,
    query_name: str,
    reference_name: str,
    query_payload: dict,
    reference_payload: dict,
    distance: np.ndarray,
    nearest_idx: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    q_bias = np.asarray(query_payload["y"], dtype=float)
    r_bias = np.asarray(reference_payload["y"], dtype=float)[nearest_idx]

    signed_gap = (q_bias - r_bias) * RADPS_TO_DEG_PER_MIN
    abs_gap = np.abs(signed_gap)

    median_dist = float(np.median(distance))
    q25_dist = float(np.quantile(distance, 0.25))

    close50 = distance <= median_dist
    close25 = distance <= q25_dist

    row = {
        "context_s": float(context_s),
        "query_sequence": query_name,
        "reference_sequence": reference_name,
        "n_query_histories": int(len(distance)),
        "history_dimensions": int(query_payload["X"].shape[1]),
        "median_nearest_history_distance": median_dist,
        "p90_nearest_history_distance": float(np.percentile(distance, 90)),
        "median_abs_persistent_bias_gap_deg_per_min": float(
            np.median(abs_gap)
        ),
        "p90_abs_persistent_bias_gap_deg_per_min": float(
            np.percentile(abs_gap, 90)
        ),
        "close50_median_abs_bias_gap_deg_per_min": float(
            np.median(abs_gap[close50])
        ),
        "close25_median_abs_bias_gap_deg_per_min": float(
            np.median(abs_gap[close25])
        ),
        "close50_fraction_bias_gap_gt_0p5_deg_min": float(
            np.mean(abs_gap[close50] > 0.5)
        ),
        "close50_fraction_bias_gap_gt_1p0_deg_min": float(
            np.mean(abs_gap[close50] > 1.0)
        ),
        "close25_fraction_bias_gap_gt_0p5_deg_min": float(
            np.mean(abs_gap[close25] > 0.5)
        ),
        "close25_fraction_bias_gap_gt_1p0_deg_min": float(
            np.mean(abs_gap[close25] > 1.0)
        ),
        "nearest_target_mae_deg_per_min": float(np.mean(abs_gap)),
        "nearest_target_rmse_deg_per_min": float(
            np.sqrt(np.mean(signed_gap ** 2))
        ),
        "query_target_std_deg_per_min": float(
            np.std(q_bias * RADPS_TO_DEG_PER_MIN)
        ),
        "nearest_reference_target_std_deg_per_min": float(
            np.std(r_bias * RADPS_TO_DEG_PER_MIN)
        ),
    }

    pairs = pd.DataFrame(
        {
            "context_s": float(context_s),
            "query_sequence": query_name,
            "reference_sequence": reference_name,
            "query_time_s": query_payload["time_s"],
            "reference_time_s": reference_payload["time_s"][nearest_idx],
            "history_rms_standardized_distance": distance,
            "query_true_persistent_bias_deg_per_min": (
                q_bias * RADPS_TO_DEG_PER_MIN
            ),
            "nearest_reference_true_persistent_bias_deg_per_min": (
                r_bias * RADPS_TO_DEG_PER_MIN
            ),
            "signed_bias_gap_deg_per_min": signed_gap,
            "abs_bias_gap_deg_per_min": abs_gap,
            "is_closest_50pct": close50,
            "is_closest_25pct": close25,
        }
    )

    return row, pairs


def analyze_context(
    prepared: dict,
    context_s: float,
    target_s: float,
    step_s: float,
    query_chunk: int,
):
    payload = {
        name: build_context_payload(
            seq,
            context_s=context_s,
            target_s=target_s,
            step_s=step_s,
        )
        for name, seq in prepared.items()
    }

    # Diagnostic-only standardization across parking01 + parking02.
    # This is NOT a training transform and does not enter V2.
    combined = np.vstack(
        [
            payload["parking01"]["X"],
            payload["parking02"]["X"],
        ]
    )

    mean = np.mean(combined, axis=0)
    std = np.maximum(np.std(combined, axis=0), 1e-5)

    for name in payload:
        payload[name]["Xz"] = (
            payload[name]["X"] - mean
        ) / std

    summaries = []
    pair_tables = []

    for query_name, reference_name in (
        ("parking02", "parking01"),
        ("parking01", "parking02"),
    ):
        distance, nearest_idx = exact_nearest_cross_sequence(
            payload[query_name]["Xz"],
            payload[reference_name]["Xz"],
            query_chunk=query_chunk,
        )

        row, pairs = summarize_direction(
            context_s=context_s,
            query_name=query_name,
            reference_name=reference_name,
            query_payload=payload[query_name],
            reference_payload=payload[reference_name],
            distance=distance,
            nearest_idx=nearest_idx,
        )

        summaries.append(row)
        pair_tables.append(pairs)

    return pd.DataFrame(summaries), pd.concat(
        pair_tables,
        ignore_index=True,
    )


def make_combined_summary(direction_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "median_abs_persistent_bias_gap_deg_per_min",
        "p90_abs_persistent_bias_gap_deg_per_min",
        "close50_median_abs_bias_gap_deg_per_min",
        "close25_median_abs_bias_gap_deg_per_min",
        "close50_fraction_bias_gap_gt_0p5_deg_min",
        "close50_fraction_bias_gap_gt_1p0_deg_min",
        "close25_fraction_bias_gap_gt_0p5_deg_min",
        "close25_fraction_bias_gap_gt_1p0_deg_min",
        "nearest_target_mae_deg_per_min",
        "nearest_target_rmse_deg_per_min",
    ]

    rows = []
    for context_s, g in direction_summary.groupby("context_s", sort=True):
        row = {
            "context_s": float(context_s),
            "n_directions": int(len(g)),
        }
        for metric in metrics:
            vals = pd.to_numeric(
                g[metric],
                errors="coerce",
            ).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = (
                float(np.mean(vals)) if len(vals) else float("nan")
            )

        rows.append(row)

    combined = pd.DataFrame(rows).sort_values("context_s")

    if len(combined):
        baseline = combined.iloc[0]

        for metric in (
            "close50_fraction_bias_gap_gt_1p0_deg_min_mean",
            "close25_fraction_bias_gap_gt_1p0_deg_min_mean",
            "close50_median_abs_bias_gap_deg_per_min_mean",
            "close25_median_abs_bias_gap_deg_per_min_mean",
        ):
            base = float(baseline[metric])
            if abs(base) > EPS:
                combined[
                    metric.replace("_mean", "_reduction_vs_shortest_pct")
                ] = (
                    100.0
                    * (base - combined[metric].astype(float))
                    / base
                )
            else:
                combined[
                    metric.replace("_mean", "_reduction_vs_shortest_pct")
                ] = float("nan")

    return combined


def plot_ambiguity_vs_context(
    combined: pd.DataFrame,
    output_dir: Path,
) -> None:
    x = combined["context_s"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        x,
        100.0
        * combined[
            "close50_fraction_bias_gap_gt_1p0_deg_min_mean"
        ].to_numpy(dtype=float),
        marker="o",
        label="Closest 50%",
    )
    ax.plot(
        x,
        100.0
        * combined[
            "close25_fraction_bias_gap_gt_1p0_deg_min_mean"
        ].to_numpy(dtype=float),
        marker="o",
        label="Closest 25%",
    )

    ax.set_xlabel("Causal ODO+IMU context length (s)")
    ax.set_ylabel("Cross-sequence matches with >1 deg/min target gap (%)")
    ax.set_title("Persistent-yaw ambiguity vs causal context length")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "ambiguity_gt1degmin_vs_context.png",
        dpi=170,
    )
    plt.close(fig)


def plot_bias_gap_vs_context(
    combined: pd.DataFrame,
    output_dir: Path,
) -> None:
    x = combined["context_s"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        x,
        combined[
            "close50_median_abs_bias_gap_deg_per_min_mean"
        ].to_numpy(dtype=float),
        marker="o",
        label="Closest 50%",
    )
    ax.plot(
        x,
        combined[
            "close25_median_abs_bias_gap_deg_per_min_mean"
        ].to_numpy(dtype=float),
        marker="o",
        label="Closest 25%",
    )

    ax.set_xlabel("Causal ODO+IMU context length (s)")
    ax.set_ylabel("Median |30 s persistent-yaw target gap| (deg/min)")
    ax.set_title("Required yaw-correction mismatch vs context length")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "median_bias_gap_vs_context.png",
        dpi=170,
    )
    plt.close(fig)


def plot_directional_ambiguity(
    direction_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for query_name, g in direction_summary.groupby(
        "query_sequence",
        sort=False,
    ):
        g = g.sort_values("context_s")
        ax.plot(
            g["context_s"].to_numpy(dtype=float),
            100.0
            * g[
                "close50_fraction_bias_gap_gt_1p0_deg_min"
            ].to_numpy(dtype=float),
            marker="o",
            label=f"{query_name} query",
        )

    ax.set_xlabel("Causal ODO+IMU context length (s)")
    ax.set_ylabel("Closest-half matches with >1 deg/min target gap (%)")
    ax.set_title("Directional parking01/parking02 ambiguity")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "directional_ambiguity_vs_context.png",
        dpi=170,
    )
    plt.close(fig)


def write_findings(
    combined: pd.DataFrame,
    direction_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = []
    lines.append("Context-length persistent-yaw identifiability diagnostic")
    lines.append("=" * 76)
    lines.append("")
    lines.append(
        "Target: causal 30 s mean of GT yaw rate - V1 nominal IMU yaw rate."
    )
    lines.append(
        "Interpretation: this is diagnostic evidence, not a formal proof of "
        "identifiability or non-identifiability."
    )
    lines.append("")

    for _, r in combined.sort_values("context_s").iterrows():
        lines.append(f"{r['context_s']:.0f} s context")
        lines.append(
            "  closest-50% fraction with >1 deg/min target gap: "
            f"{100*r['close50_fraction_bias_gap_gt_1p0_deg_min_mean']:.1f}%"
        )
        lines.append(
            "  closest-25% fraction with >1 deg/min target gap: "
            f"{100*r['close25_fraction_bias_gap_gt_1p0_deg_min_mean']:.1f}%"
        )
        lines.append(
            "  closest-50% median |target gap|: "
            f"{r['close50_median_abs_bias_gap_deg_per_min_mean']:.3f} deg/min"
        )
        lines.append(
            "  closest-25% median |target gap|: "
            f"{r['close25_median_abs_bias_gap_deg_per_min_mean']:.3f} deg/min"
        )
        lines.append("")

    if len(combined) >= 2:
        first = combined.sort_values("context_s").iloc[0]
        last = combined.sort_values("context_s").iloc[-1]

        base_rate = float(
            first["close50_fraction_bias_gap_gt_1p0_deg_min_mean"]
        )
        last_rate = float(
            last["close50_fraction_bias_gap_gt_1p0_deg_min_mean"]
        )

        base_gap = float(
            first["close50_median_abs_bias_gap_deg_per_min_mean"]
        )
        last_gap = float(
            last["close50_median_abs_bias_gap_deg_per_min_mean"]
        )

        lines.append("Shortest-to-longest context change")
        lines.append(
            f"  >1 deg/min ambiguity: "
            f"{100*base_rate:.1f}% -> {100*last_rate:.1f}%"
        )
        lines.append(
            f"  median target gap: {base_gap:.3f} -> {last_gap:.3f} deg/min"
        )
        lines.append("")

        # Heuristic recommendation only.
        if (
            last_rate <= 0.15
            and last_rate <= 0.5 * base_rate
            and last_gap <= 0.6 * base_gap
        ):
            lines.append("HEURISTIC RESULT: LONGER CONTEXT HELPS STRONGLY")
            lines.append(
                "A separate slow causal persistent-bias estimator using "
                "longer history is justified for the next V2 pilot."
            )
        elif (
            last_rate <= 0.30
            and last_rate < 0.75 * base_rate
        ):
            lines.append("HEURISTIC RESULT: LONGER CONTEXT HELPS PARTIALLY")
            lines.append(
                "Longer memory is promising, but the next V2 should still "
                "treat the persistent component as a separately supervised "
                "physical target rather than a free additive head."
            )
        else:
            lines.append("HEURISTIC RESULT: LONGER CONTEXT DOES NOT RESOLVE AMBIGUITY")
            lines.append(
                "Do not simply make the GRU longer/larger. Inspect additional "
                "lightweight physically informative observables/state, such as "
                "wheel differential or wheel-vs-IMU yaw disagreement if available."
            )

    lines.append("")
    lines.append("Directional details")
    for _, r in direction_summary.sort_values(
        ["context_s", "query_sequence"]
    ).iterrows():
        lines.append(
            f"  {r['context_s']:.0f}s {r['query_sequence']} -> "
            f"{r['reference_sequence']}: "
            f"close50 >1deg/min="
            f"{100*r['close50_fraction_bias_gap_gt_1p0_deg_min']:.1f}%, "
            f"close50 median gap="
            f"{r['close50_median_abs_bias_gap_deg_per_min']:.3f} deg/min"
        )

    (output_dir / "context_length_findings.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts_s = sorted(
        {
            float(x.strip())
            for x in args.contexts_s.split(",")
            if x.strip()
        }
    )

    if not contexts_s:
        raise ValueError("No context lengths were provided.")

    if any(x <= 0 for x in contexts_s):
        raise ValueError("All context lengths must be > 0.")

    if args.target_s <= 0 or args.step_s <= 0:
        raise ValueError("--target-s and --step-s must be > 0.")

    print("=" * 88)
    print("I2NAV CONTEXT-LENGTH PERSISTENT-YAW IDENTIFIABILITY")
    print("=" * 88)
    print("Dataset root :", root)
    print("Output       :", output_dir)
    print("Contexts (s) :", contexts_s)
    print("Target (s)   :", args.target_s)
    print("Endpoint step:", args.step_s, "s")
    print()

    prepared = load_exact_sequences(root)

    for name, seq in prepared.items():
        print(
            f"{name}: {len(seq.grid)} samples, "
            f"{seq.features.shape[1]} exact V1 features"
        )

    all_summaries = []
    all_pairs = []

    for context_s in contexts_s:
        print()
        print("-" * 88)
        print(f"CONTEXT = {context_s:g} s")
        print("-" * 88)

        summary, pairs = analyze_context(
            prepared=prepared,
            context_s=context_s,
            target_s=args.target_s,
            step_s=args.step_s,
            query_chunk=args.query_chunk,
        )

        for _, r in summary.iterrows():
            print(
                f"{r['query_sequence']} -> {r['reference_sequence']}: "
                f"n={int(r['n_query_histories'])}, "
                f"close50 >1deg/min="
                f"{100*r['close50_fraction_bias_gap_gt_1p0_deg_min']:.1f}%, "
                f"close50 median gap="
                f"{r['close50_median_abs_bias_gap_deg_per_min']:.3f} deg/min"
            )

        all_summaries.append(summary)
        all_pairs.append(pairs)

    direction_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    ).sort_values(
        ["context_s", "query_sequence"],
        kind="stable",
    )

    pairs_all = pd.concat(
        all_pairs,
        ignore_index=True,
    ).sort_values(
        ["context_s", "query_sequence", "query_time_s"],
        kind="stable",
    )

    combined = make_combined_summary(direction_summary)

    direction_summary.to_csv(
        output_dir / "context_length_direction_summary.csv",
        index=False,
    )
    combined.to_csv(
        output_dir / "context_length_combined_summary.csv",
        index=False,
    )
    pairs_all.to_csv(
        output_dir / "context_length_neighbor_pairs.csv",
        index=False,
    )

    plot_ambiguity_vs_context(combined, output_dir)
    plot_bias_gap_vs_context(combined, output_dir)
    plot_directional_ambiguity(direction_summary, output_dir)

    write_findings(
        combined=combined,
        direction_summary=direction_summary,
        output_dir=output_dir,
    )

    print()
    print("=" * 88)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print("Primary outputs:")
    print(" ", output_dir / "context_length_combined_summary.csv")
    print(" ", output_dir / "context_length_direction_summary.csv")
    print(" ", output_dir / "context_length_findings.txt")
    print(" ", output_dir / "ambiguity_gt1degmin_vs_context.png")
    print(" ", output_dir / "median_bias_gap_vs_context.png")
    print(" ", output_dir / "directional_ambiguity_vs_context.png")
    print()
    print(
        "The large context_length_neighbor_pairs.csv is optional; "
        "you usually do not need to send it unless we want deeper inspection."
    )


if __name__ == "__main__":
    main()
