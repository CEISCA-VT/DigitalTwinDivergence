"""Post-process the completed real-data campaign without replaying logs.

This script consumes ``campaign_summary.csv`` and emits paper-facing tables
that do not require new rover data: multiple paired-divergence tolerances,
paired detector differences with clustered bootstrap intervals, direct gate
behavior summaries, and result provenance hashes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from DigitalTwin.analysis.common import parse_float, read_rows, write_rows


BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260723
TOLERANCES_M = (0.25, 0.5, 1.0, 2.0, 5.0)
PAIRWISE_COMPARISONS = (
    ("naive_adaptive", "fixed"),
    ("naive_adaptive", "frozen_clean"),
    ("naive_adaptive", "gps_independent"),
    ("naive_adaptive", "evidence_gated"),
    ("naive_adaptive", "robust_innovation_gate"),
    ("naive_adaptive", "huber_ekf"),
    ("naive_adaptive", "cusum_whitened_innovation"),
    ("evidence_gated", "gps_independent"),
    ("gps_bias_evidence_gated", "gps_bias_fixed"),
    ("gps_bias_evidence_gated", "evidence_gated"),
    ("gps_bias_evidence_gated", "fixed"),
    ("gps_bias_evidence_gated", "naive_adaptive"),
)
GROUP_FIELDS = (
    "transport",
    "attack",
    "direction",
    "magnitude_m",
    "rate_mps",
    "replay_delay_s",
)
VARIANT_LABELS = {
    "fixed": "B3 fixed NIS",
    "naive_adaptive": "B7 naive adaptive",
    "frozen_clean": "frozen clean oracle",
    "gps_independent": "B8 GPS-independent adaptive",
    "evidence_gated": "B9 evidence-gated adaptive",
    "gps_bias_fixed": "R1 GPS-bias fixed-Q EKF",
    "gps_bias_evidence_gated": "R2 GPS-bias evidence-gated EKF",
    "gps_jump": "B1 GPS jump",
    "raw_position_residual": "B2 raw DT residual",
    "robust_innovation_gate": "B4 robust innovation gate",
    "huber_ekf": "B5 Huber EKF",
    "cusum_whitened_innovation": "B6 CUSUM whitened innovation",
    "innovation_matching_adaptive": "standard innovation-matching adaptive",
}


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    return float(parse_float(row.get(key, ""), default) or default)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in GROUP_FIELDS)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return BOOTSTRAP_SEED ^ int(digest[:8], 16)


def _clustered_bootstrap(
    rows: list[dict[str, str]],
    statistic: Callable[[list[dict[str, str]]], float],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    strata: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[(row.get("speed", ""), row.get("surface", ""))][row.get("run_id", "")].append(row)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample: list[dict[str, str]] = []
        for runs in strata.values():
            run_ids = sorted(runs)
            if not run_ids:
                continue
            sampled = rng.integers(0, len(run_ids), size=len(run_ids))
            for index in sampled:
                sample.extend(runs[run_ids[int(index)]])
        value = statistic(sample)
        if math.isfinite(value):
            values.append(float(value))
    if not values:
        return 0.0, 0.0
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def write_tolerance_summary(
    rows: list[dict[str, str]],
    out_dir: Path,
    *,
    iterations: int,
) -> list[dict[str, object]]:
    attack_rows = [
        row for row in rows if row.get("transport") == "baseline" and row.get("attack") != "none"
    ]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in attack_rows:
        grouped[(_group_key(row), row.get("detector_variant", ""))].append(row)

    output: list[dict[str, object]] = []
    for (attack_key, variant), group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        for tolerance in TOLERANCES_M:
            def statistic(sample: list[dict[str, str]], tol: float = tolerance) -> float:
                return _mean(
                    _f(row, "max_undetected_state_deviation_m") > tol for row in sample
                )

            probability = statistic(group)
            low, high = _clustered_bootstrap(
                group,
                statistic,
                iterations=iterations,
                seed=_stable_seed(attack_key, variant, tolerance),
            )
            output.append(
                {
                    **dict(zip(GROUP_FIELDS, attack_key)),
                    "detector_variant": variant,
                    "detector_label": VARIANT_LABELS.get(variant, variant),
                    "paired_divergence_tolerance_m": tolerance,
                    "tolerance_exceeding_paired_divergence_before_alarm": probability,
                    "ci95_low": low,
                    "ci95_high": high,
                    "physical_runs": len({row.get("run_id", "") for row in group}),
                    "scenarios": len(group),
                }
            )
    write_rows(out_dir / "paired_divergence_tolerance_summary.csv", output, output[0].keys())
    return output


def _paired_units(group: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    units: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in group:
        key = (
            row.get("run_id", ""),
            row.get("attack_start_fraction", ""),
            row.get("transport", ""),
        )
        units[key][row.get("detector_variant", "")] = row
    return units


def write_paired_comparisons(
    rows: list[dict[str, str]],
    out_dir: Path,
    *,
    iterations: int,
) -> list[dict[str, object]]:
    attack_rows = [
        row for row in rows if row.get("transport") == "baseline" and row.get("attack") != "none"
    ]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in attack_rows:
        grouped[_group_key(row)].append(row)

    output: list[dict[str, object]] = []
    for attack_key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        units = _paired_units(group)
        for left, right in PAIRWISE_COMPARISONS:
            matched = [
                {
                    **unit[left],
                    "_left_exceeds_5m": str(
                        int(_f(unit[left], "max_undetected_state_deviation_m") > 5.0)
                    ),
                    "_right_exceeds_5m": str(
                        int(_f(unit[right], "max_undetected_state_deviation_m") > 5.0)
                    ),
                    "_left_detected": unit[left].get("run_detected", "0"),
                    "_right_detected": unit[right].get("run_detected", "0"),
                }
                for unit in units.values()
                if left in unit and right in unit
            ]
            if not matched:
                continue

            def diff_exceedance(sample: list[dict[str, str]]) -> float:
                return _mean(
                    int(row["_left_exceeds_5m"]) - int(row["_right_exceeds_5m"])
                    for row in sample
                )

            def diff_detection(sample: list[dict[str, str]]) -> float:
                return _mean(
                    int(row["_left_detected"]) - int(row["_right_detected"])
                    for row in sample
                )

            exceedance = diff_exceedance(matched)
            ex_low, ex_high = _clustered_bootstrap(
                matched,
                diff_exceedance,
                iterations=iterations,
                seed=_stable_seed(attack_key, left, right, "exceedance"),
            )
            detection = diff_detection(matched)
            det_low, det_high = _clustered_bootstrap(
                matched,
                diff_detection,
                iterations=iterations,
                seed=_stable_seed(attack_key, left, right, "detection"),
            )
            output.append(
                {
                    **dict(zip(GROUP_FIELDS, attack_key)),
                    "left_variant": left,
                    "left_label": VARIANT_LABELS.get(left, left),
                    "right_variant": right,
                    "right_label": VARIANT_LABELS.get(right, right),
                    "delta_paired_divergence_exceeds_5m": exceedance,
                    "delta_paired_divergence_exceeds_5m_ci95_low": ex_low,
                    "delta_paired_divergence_exceeds_5m_ci95_high": ex_high,
                    "delta_detection_probability": detection,
                    "delta_detection_probability_ci95_low": det_low,
                    "delta_detection_probability_ci95_high": det_high,
                    "matched_run_start_units": len(matched),
                    "physical_runs": len({row.get("run_id", "") for row in matched}),
                }
            )
    write_rows(out_dir / "paired_detector_differences.csv", output, output[0].keys())
    return output


def write_gate_behavior(rows: list[dict[str, str]], out_dir: Path) -> list[dict[str, object]]:
    attack_rows = [
        row for row in rows if row.get("transport") == "baseline" and row.get("attack") != "none"
    ]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in attack_rows:
        grouped[(_group_key(row), row.get("detector_variant", ""))].append(row)
    output: list[dict[str, object]] = []
    for (attack_key, variant), group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        output.append(
            {
                **dict(zip(GROUP_FIELDS, attack_key)),
                "detector_variant": variant,
                "detector_label": VARIANT_LABELS.get(variant, variant),
                "mean_residual_feedback_activation_fraction": _mean(
                    _f(row, "residual_feedback_activation_fraction") for row in group
                ),
                "mean_clean_feedback_activation_fraction": _mean(
                    _f(row, "clean_feedback_activation_fraction") for row in group
                ),
                "mean_attack_induced_feedback_activation_delta": _mean(
                    _f(row, "attack_induced_feedback_activation_delta") for row in group
                ),
                "mean_independent_evidence_fraction": _mean(
                    _f(row, "independent_evidence_fraction") for row in group
                ),
                "mean_q_trace_ratio": _mean(_f(row, "mean_q_trace_ratio") for row in group),
                "mean_s_trace_ratio": _mean(_f(row, "mean_s_trace_ratio") for row in group),
                "mean_attack_window_q_trace_ratio": _mean(
                    _f(row, "attack_window_q_trace_ratio") for row in group
                ),
                "mean_attack_window_s_trace_ratio": _mean(
                    _f(row, "attack_window_s_trace_ratio") for row in group
                ),
                "physical_runs": len({row.get("run_id", "") for row in group}),
                "scenarios": len(group),
            }
        )
    write_rows(out_dir / "gate_behavior_summary.csv", output, output[0].keys())
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(out_dir: Path) -> dict[str, object]:
    files = [
        out_dir / "benign_manifest.csv",
        out_dir / "campaign_summary.csv",
        out_dir / "campaign_aggregate.csv",
        out_dir / "epsilon_summary.csv",
        out_dir / "campaign_validation.json",
        out_dir / "locked_thresholds.json",
        Path("DigitalTwin/configs/attack_campaign.json"),
        Path("DigitalTwin/configs/locked_alarm_policy.json"),
        Path("DigitalTwin/configs/uncertainty_policies.json"),
    ]
    payload = {
        "schema": "ugv01_results_provenance_v1",
        "files": [
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": _sha256(path) if path.exists() else "",
            }
            for path in files
        ],
    }
    (out_dir / "results_provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def render_report(
    out_dir: Path,
    rows: list[dict[str, str]],
    tolerances: list[dict[str, object]],
    paired: list[dict[str, object]],
    gate: list[dict[str, object]],
    provenance: dict[str, object],
) -> str:
    baseline_attacks = [
        row for row in rows if row.get("transport") == "baseline" and row.get("attack") != "none"
    ]
    physical_runs = len({row.get("run_id", "") for row in rows if row.get("run_id", "")})
    detector_variants = len({row.get("detector_variant", "") for row in baseline_attacks})
    unique_attack_run_start_combinations = len(
        {
            (
                row.get("run_id", ""),
                row.get("attack", ""),
                row.get("direction", ""),
                row.get("magnitude_m", ""),
                row.get("rate_mps", ""),
                row.get("replay_delay_s", ""),
                row.get("attack_start_fraction", ""),
            )
            for row in baseline_attacks
        }
    )
    detector_run_evaluations = len(baseline_attacks)
    lines = [
        "# Post-Campaign Paper Tables",
        "",
        "Generated without new rover data from `campaign_summary.csv`.",
        "",
        "## Terminology",
        "",
        f"- Physical runs: {physical_runs}.",
        f"- Unique attack-run-start combinations: {unique_attack_run_start_combinations:,}.",
        f"- Detector variants: {detector_variants}.",
        f"- Detector-run evaluations: {detector_run_evaluations:,}.",
        f"- The {detector_run_evaluations:,} rows are not independent physical attacks; they are detector evaluations clustered within physical runs.",
        "",
        "## Generated Tables",
        "",
        "- `paired_divergence_tolerance_summary.csv`: tolerance-exceeding paired divergence before alarm at 0.25, 0.5, 1, 2, and 5 m.",
        "- `paired_detector_differences.csv`: paired detector differences with physical-run-clustered bootstrap intervals.",
        "- `gate_behavior_summary.csv`: feedback activation, independent evidence, and covariance-response summaries.",
        "- `results_provenance.json`: hashes for manifests, configs, and campaign artifacts.",
        "",
        "## Example: 0.05 m/s Cross-Track Drift at 5 m Tolerance",
        "",
        "| Variant | Tolerance-exceeding paired divergence before alarm | 95% CI |",
        "| --- | ---: | ---: |",
    ]
    example = [
        row for row in tolerances
        if row["transport"] == "baseline"
        and row["attack"] == "drift"
        and row["direction"] == "cross"
        and str(row["rate_mps"]) == "0.05"
        and float(row["paired_divergence_tolerance_m"]) == 5.0
    ]
    for row in sorted(example, key=lambda item: str(item["detector_variant"])):
        lines.append(
            f"| {row['detector_label']} | "
            f"{float(row['tolerance_exceeding_paired_divergence_before_alarm']):.3f} | "
            f"[{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}] |"
        )
    lines.extend([
        "",
        "## Provenance",
        "",
        f"- Hashed files: {len(provenance['files'])}.",
    ])
    (out_dir / "post_campaign_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/real_data_study")
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    rows = read_rows(out_dir / "campaign_summary.csv")
    tolerances = write_tolerance_summary(rows, out_dir, iterations=args.bootstrap_iterations)
    paired = write_paired_comparisons(rows, out_dir, iterations=args.bootstrap_iterations)
    gate = write_gate_behavior(rows, out_dir)
    provenance = write_provenance(out_dir)
    render_report(out_dir, rows, tolerances, paired, gate, provenance)
    print(out_dir / "post_campaign_report.md")


if __name__ == "__main__":
    main()
