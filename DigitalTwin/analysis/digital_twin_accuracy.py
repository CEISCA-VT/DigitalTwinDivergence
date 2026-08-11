"""Report digital-twin sensor agreement and internal consistency on benign logs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from DigitalTwin.analysis.common import quantile, read_rows, write_rows
from DigitalTwin.analysis.real_data_study import AttackSpec, _prepare_run, replay
from DigitalTwin.detector import chi_square_threshold
from DigitalTwin.motion import DEFAULT_MOTION_FUSION_POLICY
from DigitalTwin.security import initial_security_heading


DEFAULT_MANIFEST = Path(
    "DigitalTwin/datasets/analysis/real_data_study/benign_manifest.csv"
)
DEFAULT_OUTPUT = Path(
    "DigitalTwin/datasets/analysis/digital_twin_accuracy"
)
def _rmse(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.square(values))))) if len(values) else 0.0


def _lag_one_correlation(values: np.ndarray) -> float | None:
    data = np.asarray(values, dtype=float)
    if len(data) < 3 or float(np.std(data[:-1])) <= 1e-12 or float(np.std(data[1:])) <= 1e-12:
        return None
    return float(np.corrcoef(data[:-1], data[1:])[0, 1])


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _run_summary(manifest_row: dict[str, str]) -> dict[str, object]:
    path = Path(manifest_row["source_csv"])
    prepared = _prepare_run(path)
    result = replay(path, "evidence_gated", AttackSpec(), prepared=prepared)
    encoder_prepared = _prepare_run(
        path,
        replace(DEFAULT_MOTION_FUSION_POLICY, gyro_weight=0.0),
    )
    encoder_result = replay(
        path,
        "evidence_gated",
        AttackSpec(),
        prepared=encoder_prepared,
    )
    simple_heading = initial_security_heading(
        prepared.clean_gps_xy,
        prepared.mission_start_index,
        lookahead_updates=5,
    )
    simple_initialization_result = replay(
        path,
        "evidence_gated",
        AttackSpec(),
        prepared=replace(prepared, initial_heading_rad=simple_heading),
    )
    mask = np.asarray(result.alarm_enabled, dtype=bool)
    gps = result.clean_gps_xy[mask]
    operational = result.states_xy[mask]
    if result.security_states_xy is None:
        raise RuntimeError("replay did not produce the security-predictor trajectory")
    security = result.security_states_xy[mask]
    innovations = np.asarray(result.innovations, dtype=float)[mask]
    innovation_covariances = np.asarray(result.s_matrices, dtype=float)[mask]
    scores = result.scores[mask]

    operational_error = np.linalg.norm(operational - gps, axis=1)
    security_error = np.linalg.norm(security - gps, axis=1)
    encoder_operational_error = np.linalg.norm(
        encoder_result.states_xy[mask] - gps, axis=1
    )
    encoder_security_error = np.linalg.norm(
        encoder_result.security_states_xy[mask] - gps, axis=1
    )
    simple_initialization_security_error = np.linalg.norm(
        simple_initialization_result.security_states_xy[mask] - gps,
        axis=1,
    )
    threshold_95 = chi_square_threshold(2, 0.05)
    gate_allowed = [
        int(result.rows[index].get("gate_allowed", 0))
        for index in np.flatnonzero(mask)
    ]
    moving_straight = (
        (np.abs(prepared.encoder_controls[:, 0]) > 0.03)
        & (np.abs(prepared.encoder_controls[:, 1]) < 0.15)
        & mask
    )
    turning = (np.abs(prepared.encoder_controls[:, 1]) > 0.20) & mask
    gps_length = _path_length(gps)
    operational_length = _path_length(operational)
    security_length = _path_length(security)
    empirical_innovation_covariance = (
        np.cov(innovations, rowvar=False, bias=True)
        if len(innovations) > 1
        else np.zeros((2, 2), dtype=float)
    )
    mean_innovation_covariance = (
        np.mean(innovation_covariances, axis=0)
        if len(innovation_covariances)
        else np.eye(2)
    )
    mismatch_trace = float(
        np.trace(
            np.linalg.solve(
                mean_innovation_covariance,
                empirical_innovation_covariance,
            )
        )
    )
    innovation_mean = np.mean(innovations, axis=0) if len(innovations) else np.zeros(2)
    normalized_bias_energy = float(
        innovation_mean
        @ np.linalg.solve(mean_innovation_covariance, innovation_mean)
    )

    return {
        "run_id": manifest_row["run_id"],
        "speed": manifest_row["speed"],
        "surface": manifest_row["surface"],
        "trial": manifest_row["trial"],
        "split": manifest_row["split"],
        "evaluated_updates": int(mask.sum()),
        "operational_gps_rmse_m": _rmse(operational_error),
        "operational_gps_median_m": quantile(operational_error.tolist(), 0.5),
        "operational_gps_p95_m": quantile(operational_error.tolist(), 0.95),
        "operational_within_0p25_count": int(np.sum(operational_error <= 0.25)),
        "operational_within_0p50_count": int(np.sum(operational_error <= 0.50)),
        "operational_within_1p00_count": int(np.sum(operational_error <= 1.00)),
        "operational_within_2p00_count": int(np.sum(operational_error <= 2.00)),
        "security_gps_rmse_m": _rmse(security_error),
        "security_gps_median_m": quantile(security_error.tolist(), 0.5),
        "security_gps_p95_m": quantile(security_error.tolist(), 0.95),
        "encoder_only_operational_gps_rmse_m": _rmse(encoder_operational_error),
        "encoder_only_security_gps_rmse_m": _rmse(encoder_security_error),
        "simple_initialization_security_gps_rmse_m": _rmse(
            simple_initialization_security_error
        ),
        "gps_loop_closure_m": float(np.linalg.norm(gps[-1] - gps[0])) if len(gps) else 0.0,
        "operational_loop_closure_m": (
            float(np.linalg.norm(operational[-1] - operational[0])) if len(operational) else 0.0
        ),
        "security_loop_closure_m": (
            float(np.linalg.norm(security[-1] - security[0])) if len(security) else 0.0
        ),
        "gps_path_length_m": gps_length,
        "operational_path_length_m": operational_length,
        "security_path_length_m": security_length,
        "operational_to_gps_path_length_ratio": (
            operational_length / gps_length if gps_length > 1e-12 else ""
        ),
        "security_to_gps_path_length_ratio": (
            security_length / gps_length if gps_length > 1e-12 else ""
        ),
        "mean_nis": float(np.mean(scores)) if len(scores) else 0.0,
        "median_nis": quantile(scores.tolist(), 0.5),
        "covariance_mismatch_trace": mismatch_trace,
        "normalized_innovation_bias_energy": normalized_bias_energy,
        "mismatch_predicted_mean_nis": mismatch_trace + normalized_bias_energy,
        "nis_95_coverage_fraction": (
            float(np.mean(scores <= threshold_95)) if len(scores) else 0.0
        ),
        "innovation_x_lag1": _lag_one_correlation(innovations[:, 0]),
        "innovation_y_lag1": _lag_one_correlation(innovations[:, 1]),
        "gate_pass_fraction": float(np.mean(gate_allowed)) if gate_allowed else 0.0,
        "gyro_bias_deg_s": math.degrees(prepared.gyro_bias_radps),
        "yaw_disagreement_median_radps": quantile(
            prepared.yaw_disagreement_radps[mask].tolist(), 0.5
        ),
        "yaw_disagreement_p95_radps": quantile(
            prepared.yaw_disagreement_radps[mask].tolist(), 0.95
        ),
        "slip_indicator_median": quantile(
            prepared.slip_indicator[mask].tolist(), 0.5
        ),
        "slip_indicator_p95": quantile(
            prepared.slip_indicator[mask].tolist(), 0.95
        ),
        "straight_imu_yaw_rate_median_deg_s": (
            math.degrees(float(np.median(prepared.corrected_gyro_radps[moving_straight])))
            if moving_straight.any()
            else ""
        ),
        "turn_imu_encoder_rate_ratio": (
            float(
                np.sum(np.abs(prepared.corrected_gyro_radps[turning]))
                / max(
                    np.sum(np.abs(prepared.encoder_controls[turning, 1])),
                    1e-12,
                )
            )
            if turning.any()
            else ""
        ),
        "initial_heading_deg": math.degrees(prepared.initial_heading_rad),
        "initialization_updates": (
            prepared.initialization_end_index - prepared.mission_start_index
        ),
        "source_csv": str(path),
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) not in {"", None}]

    total_updates = sum(int(row["evaluated_updates"]) for row in rows)
    operational_rmse_pooled = math.sqrt(
        sum(
            int(row["evaluated_updates"]) * float(row["operational_gps_rmse_m"]) ** 2
            for row in rows
        )
        / total_updates
    )
    security_rmse_pooled = math.sqrt(
        sum(
            int(row["evaluated_updates"]) * float(row["security_gps_rmse_m"]) ** 2
            for row in rows
        )
        / total_updates
    )
    encoder_operational_rmse_pooled = math.sqrt(
        sum(
            int(row["evaluated_updates"])
            * float(row["encoder_only_operational_gps_rmse_m"]) ** 2
            for row in rows
        )
        / total_updates
    )
    encoder_security_rmse_pooled = math.sqrt(
        sum(
            int(row["evaluated_updates"])
            * float(row["encoder_only_security_gps_rmse_m"]) ** 2
            for row in rows
        )
        / total_updates
    )
    simple_initialization_security_rmse_pooled = math.sqrt(
        sum(
            int(row["evaluated_updates"])
            * float(row["simple_initialization_security_gps_rmse_m"]) ** 2
            for row in rows
        )
        / total_updates
    )
    split_summary: dict[str, dict[str, object]] = {}
    for split in ("development", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        updates = sum(int(row["evaluated_updates"]) for row in selected)
        split_summary[split] = {
            "runs": len(selected),
            "updates": updates,
            "operational_gps_rmse_m_pooled": math.sqrt(
                sum(
                    int(row["evaluated_updates"])
                    * float(row["operational_gps_rmse_m"]) ** 2
                    for row in selected
                )
                / updates
            ),
            "security_gps_rmse_m_pooled": math.sqrt(
                sum(
                    int(row["evaluated_updates"])
                    * float(row["security_gps_rmse_m"]) ** 2
                    for row in selected
                )
                / updates
            ),
            "nis_95_coverage_fraction": (
                sum(
                    int(row["evaluated_updates"])
                    * float(row["nis_95_coverage_fraction"])
                    for row in selected
                )
                / updates
            ),
            "mean_nis": (
                sum(
                    int(row["evaluated_updates"]) * float(row["mean_nis"])
                    for row in selected
                )
                / updates
            ),
            "gate_pass_fraction": (
                sum(
                    int(row["evaluated_updates"])
                    * float(row["gate_pass_fraction"])
                    for row in selected
                )
                / updates
            ),
        }
    condition_summary: dict[str, dict[str, object]] = {}
    conditions = sorted({(str(row["speed"]), str(row["surface"])) for row in rows})
    for speed, surface in conditions:
        selected = [
            row
            for row in rows
            if row["speed"] == speed and row["surface"] == surface
        ]
        condition_summary[f"{speed}|{surface}"] = {
            "runs": len(selected),
            "gyro_bias_deg_s_mean": float(
                np.mean([float(row["gyro_bias_deg_s"]) for row in selected])
            ),
            "yaw_disagreement_median_radps_mean": float(
                np.mean(
                    [
                        float(row["yaw_disagreement_median_radps"])
                        for row in selected
                    ]
                )
            ),
            "slip_indicator_p95_mean": float(
                np.mean([float(row["slip_indicator_p95"]) for row in selected])
            ),
            "straight_imu_yaw_rate_median_deg_s_mean": float(
                np.mean(
                    [
                        float(row["straight_imu_yaw_rate_median_deg_s"])
                        for row in selected
                        if row["straight_imu_yaw_rate_median_deg_s"] != ""
                    ]
                )
            ),
            "turn_imu_encoder_rate_ratio_mean": float(
                np.mean(
                    [
                        float(row["turn_imu_encoder_rate_ratio"])
                        for row in selected
                        if row["turn_imu_encoder_rate_ratio"] != ""
                    ]
                )
            ),
        }
    return {
        "schema": "ugv01_digital_twin_accuracy_v2",
        "architecture": "GPS-fused operational EKF plus GPS-independent security predictor",
        "runs": len(rows),
        "evaluated_updates": total_updates,
        "operational_gps_rmse_m_pooled": operational_rmse_pooled,
        "operational_gps_rmse_m_mean_across_runs": float(
            np.mean(values("operational_gps_rmse_m"))
        ),
        "operational_gps_rmse_m_median_across_runs": quantile(
            values("operational_gps_rmse_m"), 0.5
        ),
        "operational_gps_rmse_m_range": [
            min(values("operational_gps_rmse_m")),
            max(values("operational_gps_rmse_m")),
        ],
        "operational_gps_median_error_m_median_across_runs": quantile(
            values("operational_gps_median_m"), 0.5
        ),
        "security_gps_rmse_m_mean_across_runs": float(
            np.mean(values("security_gps_rmse_m"))
        ),
        "security_gps_rmse_m_pooled": security_rmse_pooled,
        "encoder_only_operational_gps_rmse_m_pooled": encoder_operational_rmse_pooled,
        "encoder_only_security_gps_rmse_m_pooled": encoder_security_rmse_pooled,
        "simple_initialization_security_gps_rmse_m_pooled": (
            simple_initialization_security_rmse_pooled
        ),
        "security_gps_rmse_m_median_across_runs": quantile(
            values("security_gps_rmse_m"), 0.5
        ),
        "security_gps_rmse_m_range": [
            min(values("security_gps_rmse_m")),
            max(values("security_gps_rmse_m")),
        ],
        "nis_95_coverage_fraction_mean_across_runs": float(
            np.mean(values("nis_95_coverage_fraction"))
        ),
        "mean_nis_mean_across_runs": float(np.mean(values("mean_nis"))),
        "covariance_mismatch_trace_median_across_runs": quantile(
            values("covariance_mismatch_trace"), 0.5
        ),
        "normalized_innovation_bias_energy_median_across_runs": quantile(
            values("normalized_innovation_bias_energy"), 0.5
        ),
        "gate_pass_fraction_mean_across_runs": float(
            np.mean(values("gate_pass_fraction"))
        ),
        "operational_loop_closure_m_median": quantile(
            values("operational_loop_closure_m"), 0.5
        ),
        "security_loop_closure_m_median": quantile(
            values("security_loop_closure_m"), 0.5
        ),
        "operational_within_0p25_fraction": (
            sum(int(row["operational_within_0p25_count"]) for row in rows)
            / total_updates
        ),
        "operational_within_0p50_fraction": (
            sum(int(row["operational_within_0p50_count"]) for row in rows)
            / total_updates
        ),
        "operational_within_1p00_fraction": (
            sum(int(row["operational_within_1p00_count"]) for row in rows)
            / total_updates
        ),
        "operational_within_2p00_fraction": (
            sum(int(row["operational_within_2p00_count"]) for row in rows)
            / total_updates
        ),
        "gyro_bias_deg_s_median": quantile(values("gyro_bias_deg_s"), 0.5),
        "gyro_bias_deg_s_range": [
            min(values("gyro_bias_deg_s")),
            max(values("gyro_bias_deg_s")),
        ],
        "yaw_disagreement_median_radps_across_runs": quantile(
            values("yaw_disagreement_median_radps"), 0.5
        ),
        "slip_indicator_p95_median_across_runs": quantile(
            values("slip_indicator_p95"), 0.5
        ),
        "split_summary": split_summary,
        "condition_summary": condition_summary,
        "interpretation": (
            "These are sensor-agreement and internal-consistency metrics. "
            "They are not physical localization accuracy because no independent "
            "camera/AprilTag ground truth is present."
        ),
    }


def _markdown(summary: dict[str, object]) -> str:
    op_range = summary["operational_gps_rmse_m_range"]
    sec_range = summary["security_gps_rmse_m_range"]
    lines = [
            "# Digital-Twin Accuracy and Consistency",
            "",
            "This report evaluates the revised architecture from the project PDF: a "
            "GPS-fused operational EKF and a separate GPS-independent security predictor.",
            "",
            "## Current benign-log results",
            "",
            f"- Runs: **{summary['runs']}** ({summary['evaluated_updates']} evaluated updates).",
            f"- Operational EKF-to-GPS mean run RMSE: **{summary['operational_gps_rmse_m_mean_across_runs']:.3f} m**.",
            f"- Pooled operational EKF-to-GPS RMSE: **{summary['operational_gps_rmse_m_pooled']:.3f} m**.",
            f"- Operational EKF-to-GPS median run RMSE: **{summary['operational_gps_rmse_m_median_across_runs']:.3f} m** "
            f"(range {op_range[0]:.3f}-{op_range[1]:.3f} m).",
            f"- Median of per-run operational median errors: **{summary['operational_gps_median_error_m_median_across_runs']:.3f} m**.",
            f"- Security-predictor-to-GPS median run RMSE: **{summary['security_gps_rmse_m_median_across_runs']:.3f} m** "
            f"(range {sec_range[0]:.3f}-{sec_range[1]:.3f} m).",
            f"- Mean 95% chi-square NIS coverage: **{100.0 * summary['nis_95_coverage_fraction_mean_across_runs']:.1f}%**.",
            f"- Median covariance-mismatch trace: **{summary['covariance_mismatch_trace_median_across_runs']:.3f}** "
            "(ideal calibrated value is approximately 2 for two GPS coordinates).",
            f"- Median normalized clean-innovation bias energy: "
            f"**{summary['normalized_innovation_bias_energy_median_across_runs']:.3f}**.",
            f"- Mean trusted-gate pass fraction: **{100.0 * summary['gate_pass_fraction_mean_across_runs']:.1f}%**.",
            f"- Median operational loop closure: **{summary['operational_loop_closure_m_median']:.3f} m**.",
            f"- Median security-predictor loop closure: **{summary['security_loop_closure_m_median']:.3f} m**.",
            f"- Operational samples within 0.25/0.50/1.00/2.00 m of GPS: "
            f"**{100.0 * summary['operational_within_0p25_fraction']:.1f}% / "
            f"{100.0 * summary['operational_within_0p50_fraction']:.1f}% / "
            f"{100.0 * summary['operational_within_1p00_fraction']:.1f}% / "
            f"{100.0 * summary['operational_within_2p00_fraction']:.1f}%**.",
            f"- Median per-run gyro bias: **{summary['gyro_bias_deg_s_median']:.3f} deg/s** "
            f"(range {summary['gyro_bias_deg_s_range'][0]:.3f} to "
            f"{summary['gyro_bias_deg_s_range'][1]:.3f} deg/s).",
            f"- Median yaw-rate disagreement across runs: "
            f"**{summary['yaw_disagreement_median_radps_across_runs']:.3f} rad/s**.",
            f"- Median run-level p95 slip indicator: "
            f"**{summary['slip_indicator_p95_median_across_runs']:.3f}**.",
            "",
            "## Split evaluation",
            "",
            "| Split | Runs | Updates | Operational-GPS RMSE | Security-GPS RMSE | NIS 95% coverage | Mean NIS | Gate pass |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    for split in ("development", "validation", "test"):
        values_for_split = summary["split_summary"][split]
        lines.append(
            f"| {split} | {values_for_split['runs']} | {values_for_split['updates']} | "
            f"{values_for_split['operational_gps_rmse_m_pooled']:.3f} m | "
            f"{values_for_split['security_gps_rmse_m_pooled']:.3f} m | "
            f"{100.0 * values_for_split['nis_95_coverage_fraction']:.1f}% | "
            f"{values_for_split['mean_nis']:.3f} | "
            f"{100.0 * values_for_split['gate_pass_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Motion diagnostics by condition",
            "",
            "| Speed / surface | Runs | Gyro bias | Yaw disagreement | Slip p95 | Straight IMU yaw | Turn IMU/encoder ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, condition in summary["condition_summary"].items():
        speed, surface = key.split("|", 1)
        lines.append(
            f"| {speed} / {surface.replace('_', ' ')} | {condition['runs']} | "
            f"{condition['gyro_bias_deg_s_mean']:.3f} deg/s | "
            f"{condition['yaw_disagreement_median_radps_mean']:.3f} rad/s | "
            f"{condition['slip_indicator_p95_mean']:.3f} | "
            f"{condition['straight_imu_yaw_rate_median_deg_s_mean']:.3f} deg/s | "
            f"{condition['turn_imu_encoder_rate_ratio_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Matched motion and initialization ablations",
            "",
            f"- Operational-GPS RMSE, encoder-only versus fused yaw: "
            f"**{summary['encoder_only_operational_gps_rmse_m_pooled']:.3f} -> "
            f"{summary['operational_gps_rmse_m_pooled']:.3f} m**.",
            f"- Security-GPS RMSE, encoder-only versus fused yaw: "
            f"**{summary['encoder_only_security_gps_rmse_m_pooled']:.3f} -> "
            f"{summary['security_gps_rmse_m_pooled']:.3f} m**.",
            f"- Security-GPS RMSE, five-fix displacement initialization versus "
            f"16-update shape alignment: "
            f"**{summary['simple_initialization_security_gps_rmse_m_pooled']:.3f} -> "
            f"{summary['security_gps_rmse_m_pooled']:.3f} m**.",
            "",
            "## Interpretation",
            "",
            "The operational EKF-to-GPS values quantify agreement with the sensor it "
            "fuses; they do not prove physical localization accuracy. The protected "
            "predictor is intentionally not corrected by GPS during evaluation, so its "
            "GPS disagreement measures dead-reckoning/model consistency and detector "
            "reference drift, not navigation-output accuracy.",
            "",
            "Physical position, heading, cross-track, and route-shape accuracy still "
            "require synchronized independent camera/AprilTag ground truth. Until that "
            "reference exists, centimetre-level or percentage localization accuracy "
            "must not be claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def run(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    manifest = read_rows(manifest_path)
    if not manifest:
        raise RuntimeError(f"{manifest_path} is empty")
    rows = [_run_summary(row) for row in manifest]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "per_run_accuracy.csv", rows, rows[0].keys())
    summary = _aggregate(rows)
    (output_dir / "accuracy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "accuracy_report.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.manifest, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
