"""Run the paired residual-coupled covariance-poisoning analysis."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .common import read_rows, write_rows
from .real_data_study import (
    ATTACK_START_FRACTIONS,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    DRIFT_RATES_MPS,
    VARIANTS,
    AttackSpec,
    _bootstrap_interval,
    _group_seed,
    build_benign_manifest,
    run_campaign,
)


COMPARATORS = ("fixed", "frozen_clean", "gps_independent", "evidence_gated")
SIGN_FLIP_ITERATIONS = 10000
MINIMUM_OPERATIONAL_STATE_ERROR_INCREASE_M = 0.10

EFFECT_FIELDS = {
    "q_trace_ratio_delta": "attack_window_q_trace_ratio",
    "s_trace_ratio_delta": "attack_window_s_trace_ratio",
    "nis_ratio_delta": "attack_window_nis_ratio",
    "max_undetected_state_deviation_delta_m": "max_undetected_state_deviation_m",
    "harmful_but_stealthy_probability_delta": "harmful_but_stealthy",
    "detection_probability_delta": "run_detected",
}


def _f(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value in {"", None} else float(value)


def _drift_attacks() -> list[AttackSpec]:
    attacks = [
        AttackSpec("drift", direction, rate_mps=rate)
        for direction in ("along", "cross")
        for rate in DRIFT_RATES_MPS
    ]
    attacks.extend(
        AttackSpec("strategic_drift", direction, rate_mps=0.03)
        for direction in ("along", "cross")
    )
    return attacks


def _scenario_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["run_id"],
        row["attack"],
        row["direction"],
        row["rate_mps"],
        row["attack_start_fraction"],
        row["transport"],
    )


def build_paired_effects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    attacked = [row for row in rows if row["attack"] in {"drift", "strategic_drift"}]
    indexed = {(_scenario_key(row), row["detector_variant"]): row for row in attacked}
    effects: list[dict[str, object]] = []
    for naive in attacked:
        if naive["detector_variant"] != "naive_adaptive":
            continue
        key = _scenario_key(naive)
        for comparator in COMPARATORS:
            control = indexed[(key, comparator)]
            output: dict[str, object] = {
                field: naive[field]
                for field in (
                    "run_id",
                    "speed",
                    "surface",
                    "trial",
                    "source_csv",
                    "attack",
                    "direction",
                    "rate_mps",
                    "attack_start_fraction",
                    "attack_start_s",
                    "evaluation_horizon_s",
                )
            }
            output["comparison"] = f"naive_adaptive_vs_{comparator}"
            for output_field, source_field in EFFECT_FIELDS.items():
                naive_value = _f(naive, source_field)
                control_value = _f(control, source_field)
                output[f"naive_{source_field}"] = naive_value
                output[f"control_{source_field}"] = control_value
                output[output_field] = naive_value - control_value
            effects.append(output)
    return effects


def validate_targeted_matrix(
    replay_rows: list[dict[str, object]],
    pairs: list[dict[str, object]],
    out_dir: Path,
) -> dict[str, object]:
    attacked = [row for row in replay_rows if row["attack"] in {"drift", "strategic_drift"}]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in attacked:
        grouped[
            (
                row["detector_variant"],
                row["attack"],
                row["direction"],
                row["rate_mps"],
            )
        ].append(row)
    expected_groups = len(VARIANTS) * len(_drift_attacks())
    expected_per_group = 20 * len(ATTACK_START_FRACTIONS)
    invalid = [
        list(key)
        for key, group in grouped.items()
        if len(group) != expected_per_group
        or len({str(row["run_id"]) for row in group}) != 20
        or {
            round(_f(row, "attack_start_fraction"), 8) for row in group
        }
        != set(ATTACK_START_FRACTIONS)
    ]
    expected_attacked = expected_groups * expected_per_group
    expected_pairs = 20 * len(_drift_attacks()) * len(ATTACK_START_FRACTIONS) * len(
        COMPARATORS
    )
    payload = {
        "schema": "ugv01_covariance_poisoning_validation_v1",
        "status": "pass"
        if len(attacked) == expected_attacked
        and len(grouped) == expected_groups
        and len(pairs) == expected_pairs
        and not invalid
        else "fail",
        "expected_attacked_scenarios": expected_attacked,
        "observed_attacked_scenarios": len(attacked),
        "expected_condition_groups": expected_groups,
        "observed_condition_groups": len(grouped),
        "expected_scenarios_per_condition": expected_per_group,
        "expected_paired_effect_rows": expected_pairs,
        "observed_paired_effect_rows": len(pairs),
        "invalid_condition_groups": invalid,
    }
    (out_dir / "covariance_poisoning_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if payload["status"] != "pass":
        raise RuntimeError(f"covariance-poisoning matrix validation failed: {payload}")
    return payload


def _run_mean(rows: list[dict[str, object]], field: str) -> float:
    by_run: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run_id"])].append(_f(row, field))
    return float(np.mean([np.mean(values) for values in by_run.values()]))


def _sign_flip_p_value(
    rows: list[dict[str, object]],
    field: str,
    *,
    iterations: int = SIGN_FLIP_ITERATIONS,
    seed: int,
) -> float:
    by_run: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run_id"])].append(_f(row, field))
    run_effects = np.asarray([np.mean(values) for _, values in sorted(by_run.items())])
    observed = abs(float(run_effects.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(iterations, len(run_effects)))
    permuted = np.abs(np.mean(signs * run_effects[None, :], axis=1))
    extreme = int(np.sum(permuted >= observed - 1e-15))
    return (extreme + 1.0) / (iterations + 1.0)


def _scopes(rows: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    scopes = [
        ("standard_drift_pooled", [row for row in rows if row["attack"] == "drift"]),
        (
            "strategic_drift_pooled",
            [row for row in rows if row["attack"] == "strategic_drift"],
        ),
    ]
    for rate in DRIFT_RATES_MPS:
        scopes.append(
            (
                f"standard_drift_{rate:g}mps_pooled_direction",
                [
                    row
                    for row in rows
                    if row["attack"] == "drift" and math.isclose(_f(row, "rate_mps"), rate)
                ],
            )
        )
        for direction in ("along", "cross"):
            scopes.append(
                (
                    f"standard_drift_{direction}_{rate:g}mps",
                    [
                        row
                        for row in rows
                        if row["attack"] == "drift"
                        and row["direction"] == direction
                        and math.isclose(_f(row, "rate_mps"), rate)
                    ],
                )
            )
    for direction in ("along", "cross"):
        scopes.append(
            (
                f"strategic_drift_{direction}_0.03mps",
                [
                    row
                    for row in rows
                    if row["attack"] == "strategic_drift" and row["direction"] == direction
                ],
            )
        )
    return scopes


def aggregate_effects(
    effects: list[dict[str, object]],
    *,
    bootstrap_iterations: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for comparator in COMPARATORS:
        comparison = f"naive_adaptive_vs_{comparator}"
        comparison_rows = [row for row in effects if row["comparison"] == comparison]
        for scope, scope_rows in _scopes(comparison_rows):
            for metric in EFFECT_FIELDS:
                seed = _group_seed(("covariance_poisoning", comparison, scope, metric))
                low, high = _bootstrap_interval(
                    scope_rows,
                    lambda sample, field=metric: _run_mean(sample, field),
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
                output.append(
                    {
                        "comparison": comparison,
                        "scope": scope,
                        "metric": metric,
                        "mean_paired_effect": _run_mean(scope_rows, metric),
                        "cluster_bootstrap_ci95_low": low,
                        "cluster_bootstrap_ci95_high": high,
                        "sign_flip_p_value": _sign_flip_p_value(
                            scope_rows,
                            metric,
                            seed=seed + BOOTSTRAP_SEED,
                        ),
                        "physical_runs": len({str(row["run_id"]) for row in scope_rows}),
                        "paired_scenarios": len(scope_rows),
                    }
                )
    return output


def aggregate_evidence_gate(
    replay_rows: list[dict[str, object]],
    *,
    bootstrap_iterations: int,
) -> list[dict[str, object]]:
    gate_rows = [
        row
        for row in replay_rows
        if row["detector_variant"] == "evidence_gated"
        and row["attack"] in {"drift", "strategic_drift"}
    ]
    output: list[dict[str, object]] = []
    for scope, scope_rows in _scopes(gate_rows):
        seed = _group_seed(("evidence_gate", scope))
        delta_low, delta_high = _bootstrap_interval(
            scope_rows,
            lambda sample: _run_mean(sample, "attack_induced_feedback_activation_delta"),
            iterations=bootstrap_iterations,
            seed=seed,
        )
        output.append(
            {
                "scope": scope,
                "attacked_feedback_activation_fraction": _run_mean(
                    scope_rows, "residual_feedback_activation_fraction"
                ),
                "clean_feedback_activation_fraction": _run_mean(
                    scope_rows, "clean_feedback_activation_fraction"
                ),
                "attack_induced_activation_delta": _run_mean(
                    scope_rows, "attack_induced_feedback_activation_delta"
                ),
                "attack_induced_delta_ci95_low": delta_low,
                "attack_induced_delta_ci95_high": delta_high,
                "attack_induced_delta_sign_flip_p_value": _sign_flip_p_value(
                    scope_rows,
                    "attack_induced_feedback_activation_delta",
                    seed=seed + BOOTSTRAP_SEED,
                ),
                "independent_evidence_fraction": _run_mean(
                    scope_rows, "independent_evidence_fraction"
                ),
                "physical_runs": len({str(row["run_id"]) for row in scope_rows}),
                "scenarios": len(scope_rows),
            }
        )
    return output


def _effect_index(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {(str(row["comparison"]), str(row["scope"]), str(row["metric"])): row for row in rows}


def poisoning_conclusion(aggregates: list[dict[str, object]]) -> dict[str, object]:
    indexed = _effect_index(aggregates)
    comparison = "naive_adaptive_vs_frozen_clean"
    scope = "standard_drift_pooled"
    q = indexed[(comparison, scope, "q_trace_ratio_delta")]
    nis = indexed[(comparison, scope, "nis_ratio_delta")]
    state = indexed[(comparison, scope, "max_undetected_state_deviation_delta_m")]
    harmful = indexed[(comparison, scope, "harmful_but_stealthy_probability_delta")]
    detection = indexed[(comparison, scope, "detection_probability_delta")]
    covariance_inflation = float(q["cluster_bootstrap_ci95_low"]) > 0.0
    nis_suppression = float(nis["cluster_bootstrap_ci95_high"]) < 0.0
    mechanism_supported = covariance_inflation and nis_suppression
    operational_advantage = (
        float(state["cluster_bootstrap_ci95_low"])
        > MINIMUM_OPERATIONAL_STATE_ERROR_INCREASE_M
        or float(harmful["cluster_bootstrap_ci95_low"]) > 0.0
        or float(detection["cluster_bootstrap_ci95_high"]) < 0.0
    )
    return {
        "schema": "ugv01_covariance_poisoning_conclusion_v1",
        "primary_comparison": comparison,
        "primary_scope": scope,
        "mechanism_decision_rule": (
            "Q-ratio delta CI lower bound > 0 and NIS-ratio delta CI upper "
            "bound < 0"
        ),
        "operational_advantage_decision_rule": (
            "detection-probability delta CI upper bound < 0, harmful-probability "
            "delta CI lower bound > 0, or state-error delta CI lower bound > 0.10 m"
        ),
        "covariance_inflation_supported": covariance_inflation,
        "nis_suppression_supported": nis_suppression,
        "covariance_poisoning_mechanism_supported": mechanism_supported,
        "operational_attacker_advantage_supported": operational_advantage,
        "residual_adaptation_increases_operational_attacker_advantage": operational_advantage,
        "minimum_operational_state_error_increase_m": MINIMUM_OPERATIONAL_STATE_ERROR_INCREASE_M,
        "interpretation": (
            "covariance poisoning is measurable, but operational attacker "
            "advantage is not established on the current design corpus"
            if mechanism_supported and not operational_advantage
            else "operational attacker advantage is supported on the current design corpus"
            if operational_advantage
            else "covariance poisoning is not supported on the current design corpus"
        ),
    }


def _plot_effects(aggregates: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indexed = _effect_index(aggregates)
    labels = [value.replace("_", "\n") for value in COMPARATORS]
    metrics = [
        ("q_trace_ratio_delta", "Delta attacked/clean trace(Q)"),
        ("s_trace_ratio_delta", "Delta attacked/clean trace(S)"),
        ("nis_ratio_delta", "Delta attacked/clean mean NIS"),
        ("max_undetected_state_deviation_delta_m", "Delta max undetected state error (m)"),
        ("harmful_but_stealthy_probability_delta", "Delta harmful-but-stealthy probability"),
        ("detection_probability_delta", "Delta detection probability"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = []
        lower = []
        upper = []
        for comparator in COMPARATORS:
            row = indexed[
                (f"naive_adaptive_vs_{comparator}", "standard_drift_pooled", metric)
            ]
            value = float(row["mean_paired_effect"])
            values.append(value)
            lower.append(value - float(row["cluster_bootstrap_ci95_low"]))
            upper.append(float(row["cluster_bootstrap_ci95_high"]) - value)
        axis.errorbar(
            range(len(values)), values, yerr=np.asarray([lower, upper]), fmt="o", capsize=4
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(range(len(labels)), labels, fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Naive residual-coupled adaptation minus control: pooled standard drift")
    fig.tight_layout()
    fig.savefig(out_dir / "covariance_poisoning_paired_effects.png", dpi=180)
    plt.close(fig)


def _fmt_effect(row: dict[str, object], digits: int = 4) -> str:
    return (
        f"{float(row['mean_paired_effect']):.{digits}f} "
        f"[{float(row['cluster_bootstrap_ci95_low']):.{digits}f}, "
        f"{float(row['cluster_bootstrap_ci95_high']):.{digits}f}]"
    )


def render_report(
    effects: list[dict[str, object]],
    gate: list[dict[str, object]],
    conclusion: dict[str, object],
) -> str:
    indexed = _effect_index(effects)
    lines = [
        "# Covariance-Poisoning Analysis",
        "",
        (
            "This paired analysis uses all 20 accepted physical runs, three "
            "injection times, ordinary drift in both directions at 0.01, 0.03, "
            "and 0.05 m/s, and strategic drift in both directions at 0.03 m/s."
        ),
        "",
        (
            "The primary causal comparison is naive residual-coupled adaptation "
            "versus frozen-clean covariance on the same run, attack, direction, "
            "rate, and start time. Confidence intervals resample complete runs."
        ),
        "",
        "## Pooled Standard-Drift Effects",
        "",
        (
            "Positive values mean the naive variant is larger than the control. "
            "For NIS and detection probability, a negative value favors the attacker."
        ),
        "",
        (
            "| Control | Delta Q ratio | Delta S ratio | Delta NIS ratio | "
            "Delta max undetected error | Delta harmful probability | "
            "Delta detection probability |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparator in COMPARATORS:
        comparison = f"naive_adaptive_vs_{comparator}"
        key = lambda metric: indexed[(comparison, "standard_drift_pooled", metric)]
        lines.append(
            f"| {comparator} | {_fmt_effect(key('q_trace_ratio_delta'))} | "
            f"{_fmt_effect(key('s_trace_ratio_delta'))} | "
            f"{_fmt_effect(key('nis_ratio_delta'))} | "
            f"{_fmt_effect(key('max_undetected_state_deviation_delta_m'), 3)} m | "
            f"{_fmt_effect(key('harmful_but_stealthy_probability_delta'), 3)} | "
            f"{_fmt_effect(key('detection_probability_delta'), 3)} |"
        )

    primary = "naive_adaptive_vs_frozen_clean"
    lines.extend(
        [
            "",
            "## Primary Decision",
            "",
            f"- Covariance inflation supported: **{conclusion['covariance_inflation_supported']}**.",
            f"- NIS suppression supported: **{conclusion['nis_suppression_supported']}**.",
            (
                "- Covariance-poisoning mechanism supported: "
                f"**{conclusion['covariance_poisoning_mechanism_supported']}**."
            ),
            (
                "- Operational attacker advantage supported: "
                f"**{conclusion['operational_attacker_advantage_supported']}**."
            ),
            f"- Interpretation: **{conclusion['interpretation']}**.",
            "",
            (
                "The mechanism decision requires covariance inflation together "
                "with NIS suppression. Operational attacker advantage requires "
                "reduced detection, increased harmful probability, or at least "
                "0.10 m additional undetected state error. This prevents a "
                "statistically detectable but tiny state difference from being "
                "presented as a safety failure."
            ),
            "",
            "## Evidence-Gate Activity",
            "",
            (
                "| Scope | Attacked activation | Clean activation | "
                "Attack-induced delta (95% CI) | Independent evidence |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gate:
        if row["scope"] not in {
            "standard_drift_pooled",
            "standard_drift_0.03mps_pooled_direction",
            "standard_drift_0.05mps_pooled_direction",
            "strategic_drift_pooled",
        }:
            continue
        lines.append(
            f"| {row['scope']} | {float(row['attacked_feedback_activation_fraction']):.3f} | "
            f"{float(row['clean_feedback_activation_fraction']):.3f} | "
            f"{float(row['attack_induced_activation_delta']):.4f} "
            f"[{float(row['attack_induced_delta_ci95_low']):.4f}, "
            f"{float(row['attack_induced_delta_ci95_high']):.4f}] | "
            f"{float(row['independent_evidence_fraction']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- Results are counterfactual offline replays, not live compromise of the rover.",
            "- State error is paired against the clean digital-twin replay, not independent overhead ground truth.",
            (
                "- The 20 physical runs are the existing design corpus; a "
                "prospective dataset is still required for external confirmation."
            ),
            "- Sign-flip p-values and clustered intervals use the physical run as the sampling unit.",
            "",
            "## Artifacts",
            "",
            "- `covariance_replay_summary.csv`: targeted replay metrics.",
            "- `covariance_poisoning_pairs.csv`: scenario-matched naive-minus-control effects.",
            "- `covariance_poisoning_aggregate.csv`: clustered effects by scope and metric.",
            "- `evidence_gate_activation.csv`: gate activity and attack-induced changes.",
            "- `covariance_poisoning_conclusion.json`: machine-readable primary decision.",
            "- `covariance_poisoning_validation.json`: targeted-matrix completeness certificate.",
            "- `covariance_poisoning_paired_effects.png`: pooled causal effect figure.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="raw_logs/telemetry")
    parser.add_argument(
        "--out-dir", default="DigitalTwin/datasets/analysis/covariance_poisoning"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_benign_manifest(Path(args.input_dir), out_dir)
    replay_path = out_dir / "covariance_replay_summary.csv"
    if args.summarize_existing:
        replay_rows = read_rows(replay_path)
    else:
        policy = json.loads(
            Path("DigitalTwin/configs/locked_alarm_policy.json").read_text(encoding="utf-8")
        )
        thresholds = {
            mode: {"threshold": float(policy["thresholds"][mode])} for mode in VARIANTS
        }
        replay_rows = run_campaign(
            manifest,
            thresholds,
            out_dir,
            attacks=_drift_attacks(),
            start_fractions=ATTACK_START_FRACTIONS,
            include_buffered=False,
            summary_filename=replay_path.name,
        )

    pairs = build_paired_effects(replay_rows)
    validate_targeted_matrix(replay_rows, pairs, out_dir)
    effects = aggregate_effects(pairs, bootstrap_iterations=args.bootstrap_iterations)
    gate = aggregate_evidence_gate(
        replay_rows, bootstrap_iterations=args.bootstrap_iterations
    )
    conclusion = poisoning_conclusion(effects)
    write_rows(out_dir / "covariance_poisoning_pairs.csv", pairs, pairs[0].keys())
    write_rows(
        out_dir / "covariance_poisoning_aggregate.csv", effects, effects[0].keys()
    )
    write_rows(out_dir / "evidence_gate_activation.csv", gate, gate[0].keys())
    (out_dir / "covariance_poisoning_conclusion.json").write_text(
        json.dumps(conclusion, indent=2), encoding="utf-8"
    )
    _plot_effects(effects, out_dir)
    report_path = out_dir / "covariance_poisoning_report.md"
    report_path.write_text(render_report(effects, gate, conclusion), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
