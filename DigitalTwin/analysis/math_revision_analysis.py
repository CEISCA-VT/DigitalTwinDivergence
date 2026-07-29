"""Summarize the revised paired-replay mathematical diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import read_rows, write_rows
from .real_data_study import PRIMARY_VARIANTS, VARIANT_LABELS


METRICS = (
    "mean_normalization_credit",
    "mean_reference_metric_innovation_change",
    "mean_paired_nis_delta",
    "innovation_covariance_order_fraction",
    "normalization_credit_nonnegative_fraction",
    "nis_suppression_condition_fraction",
    "observed_nis_suppression_fraction",
    "mean_directional_detectability_loss_factor",
    "mean_worst_direction_detectability_loss_factor",
    "mean_rolling_gate_pass_fraction",
    "max_residual_cover_bound_violation_m",
    "max_score_decomposition_identity_error",
)


def _number(row: dict[str, object], field: str) -> float | None:
    value = row.get(field, "")
    if value in {"", None}:
        return None
    return float(value)


def _condition_key(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        str(row.get(field, ""))
        for field in (
            "detector_variant",
            "attack",
            "direction",
            "magnitude_m",
            "rate_mps",
            "replay_delay_s",
            "transport",
        )
    )


def aggregate_math_diagnostics(
    campaign: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in campaign:
        if row.get("attack") != "none":
            groups[_condition_key(row)].append(row)

    output: list[dict[str, object]] = []
    key_fields = (
        "detector_variant",
        "attack",
        "direction",
        "magnitude_m",
        "rate_mps",
        "replay_delay_s",
        "transport",
    )
    for key, rows in sorted(groups.items()):
        summary: dict[str, object] = dict(zip(key_fields, key))
        summary["detector_label"] = VARIANT_LABELS.get(key[0], key[0])
        summary["detector_run_evaluations"] = len(rows)
        for metric in METRICS:
            values = [
                value
                for row in rows
                if (value := _number(row, metric)) is not None
            ]
            summary[metric] = float(np.mean(values)) if values else ""
        output.append(summary)
    return output


def validate_math_diagnostics(
    campaign: list[dict[str, object]],
    out_dir: Path,
) -> dict[str, object]:
    attacked = [row for row in campaign if row.get("attack") != "none"]
    identity_errors = [
        value
        for row in attacked
        if (value := _number(row, "max_score_decomposition_identity_error"))
        is not None
    ]
    cover_violations = [
        value
        for row in attacked
        if (value := _number(row, "max_residual_cover_bound_violation_m"))
        is not None
    ]
    required_fields_present = all(
        all(field in row for field in METRICS) for row in attacked
    )
    payload = {
        "schema": "ugv01_math_revision_validation_v1",
        "status": "pass",
        "attacked_detector_run_evaluations": len(attacked),
        "required_fields_present": required_fields_present,
        "max_score_decomposition_identity_error": max(identity_errors, default=0.0),
        "max_residual_cover_bound_violation_m": max(
            cover_violations, default=0.0
        ),
        "score_decomposition_identity_tolerance": 1e-8,
        "residual_cover_bound_tolerance_m": 1e-10,
    }
    if (
        not required_fields_present
        or payload["max_score_decomposition_identity_error"] > 1e-8
        or payload["max_residual_cover_bound_violation_m"] > 1e-10
    ):
        payload["status"] = "fail"
    (out_dir / "math_revision_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if payload["status"] != "pass":
        raise RuntimeError(f"revised-math validation failed: {payload}")
    return payload


def _mean_for_variant(
    rows: list[dict[str, object]],
    variant: str,
    metric: str,
) -> float:
    values = [
        value
        for row in rows
        if row["detector_variant"] == variant
        and row["attack"] in {"drift", "strategic_drift"}
        and (value := _number(row, metric)) is not None
    ]
    return float(np.mean(values)) if values else 0.0


def plot_score_decomposition(
    summaries: list[dict[str, object]],
    out_dir: Path,
) -> None:
    variants = list(PRIMARY_VARIANTS)
    x = np.arange(len(variants), dtype=float)
    width = 0.25
    credits = [
        _mean_for_variant(summaries, variant, "mean_normalization_credit")
        for variant in variants
    ]
    changes = [
        _mean_for_variant(
            summaries, variant, "mean_reference_metric_innovation_change"
        )
        for variant in variants
    ]
    deltas = [
        _mean_for_variant(summaries, variant, "mean_paired_nis_delta")
        for variant in variants
    ]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width, credits, width, label="Normalization credit C")
    ax.bar(x, changes, width, label="Innovation change G")
    ax.bar(x + width, deltas, width, label="NIS delta G-C")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [VARIANT_LABELS[variant] for variant in variants], rotation=18)
    ax.set_ylabel("Mean paired score term")
    ax.set_title("Paired NIS decomposition across drift attacks")
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(out_dir / "math_score_decomposition.png", dpi=180)
    plt.close(fig)


def render_report(
    summaries: list[dict[str, object]],
    validation: dict[str, object],
) -> str:
    lines = [
        "# Revised Mathematical Diagnostics",
        "",
        "These artifacts evaluate the exact paired NIS identity and the rolling "
        "residual-cover bound introduced by the revised manuscript.",
        "",
        "## Validation",
        "",
        f"- Status: **{validation['status']}**.",
        f"- Attacked detector-run evaluations: "
        f"**{validation['attacked_detector_run_evaluations']}**.",
        f"- Maximum score-decomposition identity error: "
        f"`{validation['max_score_decomposition_identity_error']:.3e}`.",
        f"- Maximum residual-cover bound violation: "
        f"`{validation['max_residual_cover_bound_violation_m']:.3e} m`.",
        "",
        "## Interpretation",
        "",
        "- `normalization credit C` measures how much the attacked innovation "
        "score falls when evaluated with the attacked covariance instead of "
        "the paired reference covariance.",
        "- `innovation change G` measures how much the attacked innovation grows "
        "under the paired reference metric.",
        "- The identity `d_a-d_0 = G-C` is checked numerically for every paired "
        "attack window.",
        "- Innovation-covariance ordering is reported empirically because the "
        "nonlinear EKF branches need not share identical linearizations.",
        "- Directional detectability-loss factors are conditional envelopes, "
        "not run-level detection probabilities.",
        "",
        "## Outputs",
        "",
        "- `math_mechanism_summary.csv`: condition-level revised-math metrics.",
        "- `math_revision_validation.json`: numerical identity and bound checks.",
        "- `math_score_decomposition.png`: primary-variant drift decomposition.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="DigitalTwin/datasets/analysis/real_data_study",
    )
    parser.add_argument(
        "--campaign-file",
        default="campaign_summary.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    campaign = read_rows(out_dir / args.campaign_file)
    summaries = aggregate_math_diagnostics(campaign)
    if not summaries:
        raise RuntimeError("campaign contains no attacked rows")
    validation = validate_math_diagnostics(campaign, out_dir)
    write_rows(
        out_dir / "math_mechanism_summary.csv",
        summaries,
        summaries[0].keys(),
    )
    plot_score_decomposition(summaries, out_dir)
    report = render_report(summaries, validation)
    report_path = out_dir / "math_revision_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
