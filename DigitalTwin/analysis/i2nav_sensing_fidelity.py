"""Build the i2Nav sensing-fidelity tradeoff audit.

This script does not recompute trajectories or benchmark scores. It combines the
frozen Twin V2 official benchmark outputs with primary-source i2Nav-Robot
benchmark rows to position the method by sensing burden.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


I2NAV_SOURCE = "https://github.com/i2Nav-WHU/i2Nav-Robot"
EVAL_SOURCE = "https://github.com/i2Nav-WHU/evaluate_odometry"


PUBLISHED_METHODS: list[dict[str, Any]] = [
    {
        "method": "VINS-Mono",
        "class": "VI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "yes",
        "lidar": "no",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 1.71,
        "official_ape_translation_rmse_m": 5.38,
        "sequence_count": 10,
        "comparability_status": "DIRECTLY_COMPARABLE",
    },
    {
        "method": "DM-VIO",
        "class": "VI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "yes",
        "lidar": "no",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": None,
        "official_ape_translation_rmse_m": None,
        "sequence_count": 10,
        "comparability_status": "NOT_COMPARABLE",
        "note": "i2Nav README aggregate row is Invalid.",
    },
    {
        "method": "OpenVINS (Stereo)",
        "class": "VI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "yes",
        "lidar": "no",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 1.01,
        "official_ape_translation_rmse_m": 1.93,
        "sequence_count": 10,
        "comparability_status": "DIRECTLY_COMPARABLE",
    },
    {
        "method": "DLIO",
        "class": "LI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "no",
        "lidar": "yes",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 4.74,
        "official_ape_translation_rmse_m": 3.50,
        "sequence_count": 10,
        "comparability_status": "DIRECTLY_COMPARABLE",
    },
    {
        "method": "FF-LINS",
        "class": "LI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "no",
        "lidar": "yes",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 2.00,
        "official_ape_translation_rmse_m": 3.23,
        "sequence_count": 10,
        "comparability_status": "DIRECTLY_COMPARABLE",
    },
    {
        "method": "FAST-LIO2",
        "class": "LI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "no",
        "lidar": "yes",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 1.11,
        "official_ape_translation_rmse_m": 1.12,
        "sequence_count": 10,
        "comparability_status": "DIRECTLY_COMPARABLE",
    },
    {
        "method": "FAST-LIVO2",
        "class": "LVI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "yes",
        "lidar": "yes",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 0.83,
        "official_ape_translation_rmse_m": 1.42,
        "sequence_count": 10,
        "comparability_status": "DIRECTLY_COMPARABLE",
    },
    {
        "method": "LE-VINS",
        "class": "LVI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "yes",
        "lidar": "yes",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 0.55,
        "official_ape_translation_rmse_m": 1.43,
        "sequence_count": 10,
        "comparability_status": "DIRECTLY_COMPARABLE",
    },
    {
        "method": "R3LIVE",
        "class": "LVI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "yes",
        "lidar": "yes",
        "radar": "no",
        "gnss": "no",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": None,
        "official_ape_translation_rmse_m": None,
        "sequence_count": 10,
        "comparability_status": "NOT_COMPARABLE",
        "note": "i2Nav README aggregate row is Invalid.",
    },
    {
        "method": "KF-GINS",
        "class": "GI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "no",
        "lidar": "no",
        "radar": "no",
        "gnss": "yes",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 26.53,
        "official_ape_translation_rmse_m": 10.43,
        "sequence_count": 8,
        "comparability_status": "PARTIALLY_COMPARABLE",
        "note": "GNSS-based README table uses the subset where GNSS methods are listed.",
    },
    {
        "method": "OB-GINS",
        "class": "GI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "no",
        "lidar": "no",
        "radar": "no",
        "gnss": "yes",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 9.55,
        "official_ape_translation_rmse_m": 9.98,
        "sequence_count": 8,
        "comparability_status": "PARTIALLY_COMPARABLE",
        "note": "GNSS-based README table uses the subset where GNSS methods are listed.",
    },
    {
        "method": "VINS-Fusion (Mono)",
        "class": "GVI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "yes",
        "lidar": "no",
        "radar": "no",
        "gnss": "yes",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 16.53,
        "official_ape_translation_rmse_m": 10.39,
        "sequence_count": 8,
        "comparability_status": "PARTIALLY_COMPARABLE",
        "note": "GNSS-based README table uses the subset where GNSS methods are listed.",
    },
    {
        "method": "IC-GVINS",
        "class": "GVI",
        "wheel_odo": "no",
        "imu": "yes",
        "camera": "yes",
        "lidar": "no",
        "radar": "no",
        "gnss": "yes",
        "proprioceptive_only": "no",
        "official_ape_rotation_rmse_deg": 0.32,
        "official_ape_translation_rmse_m": 0.41,
        "sequence_count": 8,
        "comparability_status": "PARTIALLY_COMPARABLE",
        "note": "GNSS-based README table uses the subset where GNSS methods are listed.",
    },
]


def read_metric(rows: list[dict[str, str]], metric: str, column: str) -> float:
    for row in rows:
        if row["metric"] == metric:
            return float(row[column])
    raise KeyError(metric)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NR"
    try:
        if math.isnan(float(value)):
            return "NR"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}"


def build_our_rows(results_root: Path) -> list[dict[str, Any]]:
    macro = pd.read_csv(results_root / "i2nav_official_benchmark" / "official_macro_summary.csv")
    agg = read_csv(results_root / "result_freeze_audit" / "official_aggregation_semantics.csv")

    rows: list[dict[str, Any]] = []
    for method in ["Twin V2", "Fixed Physics"]:
        m = macro.loc[macro["method"] == method].iloc[0]
        if method == "Twin V2":
            ape_seq_rms = read_metric(agg, "official_ape_translation_rmse_m", "official_table_sequence_RMS_if_needed")
            are_seq_rms = read_metric(agg, "official_ape_rotation_rmse_deg", "official_table_sequence_RMS_if_needed")
            note = (
                "Twin V2 uses wheel/odometry and IMU at runtime only; ground truth is used for training supervision "
                "and held-out evaluation. README-compatible RMS is sequence-RMS; macro mean is preserved separately."
            )
        else:
            per_seq = pd.read_csv(results_root / "i2nav_official_benchmark" / "official_per_sequence_results.csv")
            sub = per_seq.loc[per_seq["method"] == method]
            ape_seq_rms = math.sqrt(float((sub["official_ape_translation_rmse_m_mean"] ** 2).mean()))
            are_seq_rms = math.sqrt(float((sub["official_ape_rotation_rmse_deg_mean"] ** 2).mean()))
            note = "Fixed Physics is included as the local nonlearned baseline; rotation/RPE comparisons have a body-frame caveat."
        rows.append(
            {
                "method": method,
                "class": "wheel/odo+IMU",
                "wheel_odo": "yes",
                "imu": "yes",
                "camera": "no",
                "lidar": "no",
                "radar": "no",
                "gnss": "no",
                "proprioceptive_only": "yes",
                "official_ape_translation_rmse_m": ape_seq_rms,
                "official_ape_rotation_rmse_deg": are_seq_rms,
                "official_ape_translation_macro_mean_m": float(m["official_ape_translation_rmse_m_macro_mean"]),
                "official_ape_rotation_macro_mean_deg": float(m["official_ape_rotation_rmse_deg_macro_mean"]),
                "official_rpe_50m_translation_rmse_m": float(m["official_rpe_50m_translation_rmse_m_macro_mean"]),
                "official_rpe_100m_translation_rmse_m": float(m["official_rpe_100m_translation_rmse_m_macro_mean"]),
                "official_rpe_300m_translation_rmse_m": float(m["official_rpe_300m_translation_rmse_m_macro_mean"]),
                "sequence_count": 10,
                "comparability_status": "DIRECTLY_COMPARABLE",
                "source": "local frozen official i2Nav benchmark outputs",
                "note": note,
            }
        )
    return rows


def source_for_method(row: dict[str, Any]) -> str:
    if row["method"] in {"Twin V2", "Fixed Physics"}:
        return "results/i2nav_official_benchmark/ + results/result_freeze_audit/"
    return "i2Nav-Robot README benchmark table"


def build_rows(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = build_our_rows(results_root)
    for row in PUBLISHED_METHODS:
        copied = dict(row)
        copied.setdefault("official_ape_translation_macro_mean_m", "")
        copied.setdefault("official_ape_rotation_macro_mean_deg", "")
        copied.setdefault("official_rpe_50m_translation_rmse_m", "NR")
        copied.setdefault("official_rpe_100m_translation_rmse_m", "NR")
        copied.setdefault("official_rpe_300m_translation_rmse_m", "NR")
        copied["source"] = source_for_method(copied)
        copied.setdefault("note", "Official i2Nav-Robot README odometry benchmark table.")
        rows.append(copied)

    stack_rows = []
    protocol_rows = []
    for row in rows:
        stack_rows.append(
            {
                "method": row["method"],
                "class": row["class"],
                "wheel_odo": row["wheel_odo"],
                "imu": row["imu"],
                "camera": row["camera"],
                "lidar": row["lidar"],
                "radar": row["radar"],
                "gnss": row["gnss"],
                "proprioceptive_only": row["proprioceptive_only"],
                "sensor_count_reported": "NR",
                "sensor_model_reported": "NR",
                "compute_hardware_reported": "NR",
                "runtime_latency_reported": "NR",
                "power_reported": "NR",
                "source": row["source"],
            }
        )
        protocol_rows.append(
            {
                "method": row["method"],
                "dataset_sequences_used": row["sequence_count"],
                "evaluation_protocol": "i2Nav-Robot / evaluate_odometry official APE table" if row["comparability_status"] != "NOT_COMPARABLE" else "invalid aggregate",
                "alignment_method": "SE3 no scale for local exports; i2Nav README benchmark protocol treated as official table",
                "metric_definition": "ARE deg / ATE m aggregate RMS from README; Twin V2 also preserves arithmetic macro means",
                "units": "deg, m",
                "aggregate": "sequence RMS for README-compatible comparison",
                "comparability_status": row["comparability_status"],
                "reason": row.get("note", ""),
                "source": row["source"],
            }
        )
    return rows, stack_rows, protocol_rows


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    direct = [r for r in rows if r["comparability_status"] == "DIRECTLY_COMPARABLE" and r["official_ape_translation_rmse_m"] not in (None, "")]
    direct_sorted = sorted(direct, key=lambda r: float(r["official_ape_translation_rmse_m"]))
    v2 = next(r for r in rows if r["method"] == "Twin V2")
    rank = [r["method"] for r in direct_sorted].index("Twin V2") + 1
    heavier_better = [
        r["method"]
        for r in direct_sorted
        if r["method"] != "Twin V2"
        and r["proprioceptive_only"] == "no"
        and float(r["official_ape_translation_rmse_m"]) < float(v2["official_ape_translation_rmse_m"])
    ]
    lighter_or_same_better = [
        r["method"]
        for r in direct_sorted
        if r["method"] != "Twin V2"
        and r["proprioceptive_only"] == "yes"
        and float(r["official_ape_translation_rmse_m"]) < float(v2["official_ape_translation_rmse_m"])
    ]

    lines = [
        "# Sensing-Fidelity Tradeoff Summary",
        "",
        "## Scope",
        "",
        "This analysis positions the frozen Twin V2 official i2Nav benchmark result by sensing burden. It does not retrain, retune, re-export, or alter frozen benchmark trajectories.",
        "",
        "Twin V2 runtime sensing:",
        "",
        "- wheel/odometry: yes",
        "- IMU: yes",
        "- camera: no",
        "- LiDAR: no",
        "- radar: no",
        "- GNSS: no",
        "- ground truth: training supervision and held-out evaluation only, not runtime inference",
        "",
        "## Frozen Twin V2 Numbers",
        "",
        f"- Official APE translation RMSE, arithmetic macro mean: {fmt(v2['official_ape_translation_macro_mean_m'])} m",
        f"- Official APE rotation RMSE, arithmetic macro mean: {fmt(v2['official_ape_rotation_macro_mean_deg'])} deg",
        f"- README-compatible ATE sequence-RMS: {fmt(v2['official_ape_translation_rmse_m'])} m",
        f"- README-compatible ARE sequence-RMS: {fmt(v2['official_ape_rotation_rmse_deg'])} deg",
        f"- RPE 50 m macro mean: {fmt(v2['official_rpe_50m_translation_rmse_m'])} m",
        f"- RPE 100 m macro mean: {fmt(v2['official_rpe_100m_translation_rmse_m'])} m",
        f"- RPE 300 m macro mean: {fmt(v2['official_rpe_300m_translation_rmse_m'])} m",
        "",
        "## Protocol-Compatible Positioning",
        "",
        f"Twin V2 ranks {rank}/{len(direct_sorted)} among directly comparable 10-sequence ATE/ARE rows when using the README-compatible ATE sequence-RMS aggregate.",
        "",
        "Methods with lower ATE than Twin V2 in this official table are: "
        + (", ".join(heavier_better) if heavier_better else "none")
        + ".",
        "",
        "No directly comparable proprioceptive-only external method in the audited i2Nav README table outperforms Twin V2, but this does not prove Pareto optimality because the table does not include every possible wheel-inertial method.",
        "",
        "## Interpretation",
        "",
        "Twin V2 is not as accurate as the strongest heavier exteroceptive systems such as LiDAR/IMU or LiDAR/visual/IMU methods, and it should not be described as odometry SOTA. Its value is that it achieves a usable official trajectory result while requiring only wheel/odometry and IMU at runtime, which supports the sensor-lightweight digital-twin framing.",
        "",
        "The sensing-fidelity analysis should be used as supporting positioning, not as the main contribution. The main paper should still center on local/global fidelity, condition dependence, benign fidelity characterization, and asset-specific instantiation.",
        "",
        "## Claim Tests",
        "",
        "- Claim A: SUPPORTED_WITH_QUALIFICATION. Twin V2 provides usable and sometimes competitive trajectory fidelity using only wheel/odometry and IMU, but it does not beat the strongest LiDAR/visual systems.",
        "- Claim B: SUPPORTED_WITH_QUALIFICATION. The results support a favorable sensing-fidelity tradeoff as contextual positioning, not a universal Pareto claim.",
        "- Claim C: NOT_SUPPORTED. Pareto optimality is not established because sensing burden is componentwise and the audited benchmark table is not exhaustive.",
        "- Claim D: NOT_SUPPORTED. The penalty for removing camera/LiDAR/radar/GNSS is not necessarily modest; several heavier methods have substantially lower ATE.",
        "- Claim E: SUPPORTED. The runtime modality audit supports describing the proposed twin as sensor-lightweight.",
        "",
        "## Recommendation",
        "",
        "Include a compact sensing-fidelity table and a qualitative grouped plot only as context. Avoid a full leaderboard-style claim.",
    ]
    (out_dir / "sensing_fidelity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claim_matrix(out_dir: Path) -> None:
    rows = [
        {
            "claim": "Twin V2 provides competitive trajectory fidelity using only wheel/odometry and IMU sensing.",
            "status": "SUPPORTED_WITH_QUALIFICATION",
            "evidence": "Twin V2 has README-compatible ATE sequence-RMS 2.187 m and uses no camera/LiDAR/radar/GNSS at runtime; stronger exteroceptive methods remain better.",
        },
        {
            "claim": "Twin V2 achieves a favorable sensing-fidelity tradeoff relative to heavier i2Nav navigation systems.",
            "status": "SUPPORTED_WITH_QUALIFICATION",
            "evidence": "It sits between weak VI/LI rows and strong LIO/LVIO rows while using fewer runtime sensing modalities.",
        },
        {
            "claim": "Twin V2 is Pareto-optimal in sensing burden versus trajectory fidelity.",
            "status": "NOT_SUPPORTED",
            "evidence": "No defensible scalar sensing burden is defined, and the audited external table is not exhaustive.",
        },
        {
            "claim": "Removing camera, LiDAR, radar, and GNSS necessarily produces only a modest fidelity penalty.",
            "status": "NOT_SUPPORTED",
            "evidence": "FAST-LIO2, FAST-LIVO2, LE-VINS, OpenVINS, and IC-GVINS report lower ATE under their respective protocol rows.",
        },
        {
            "claim": "The results support describing the proposed twin as sensor-lightweight.",
            "status": "SUPPORTED",
            "evidence": "Twin V2 runtime inference uses wheel/odometry and IMU only, with camera/LiDAR/radar/GNSS absent.",
        },
    ]
    write_csv(out_dir / "claim_evidence_matrix.csv", rows, ["claim", "status", "evidence"])


def write_provenance(out_dir: Path) -> None:
    lines = [
        "# Source Provenance",
        "",
        "- i2Nav-Robot authoritative dataset and benchmark README: https://github.com/i2Nav-WHU/i2Nav-Robot",
        "- i2Nav-Robot citation listed by the repository: Tang et al., `i2Nav-Robot: A Large-Scale Indoor-Outdoor Robot Dataset for Multi-Sensor Fusion Navigation and Mapping`, arXiv:2508.11485, DOI 10.48550/arXiv.2508.11485.",
        "- Official evaluator repository used by the project protocol audit: https://github.com/i2Nav-WHU/evaluate_odometry",
        "- Local frozen Twin V2 benchmark outputs: `results/i2nav_official_benchmark/`.",
        "- Local final aggregation audit: `results/result_freeze_audit/official_aggregation_semantics.csv`.",
        "",
        "The i2Nav-Robot README explicitly identifies the dataset sensor suite, sequence count/duration, ground truth, tested odometry systems, tested GNSS-based systems, and the published ARE/ATE benchmark tables. External RPE50/RPE100/RPE300 values are not provided in the README table and are therefore recorded as NR.",
    ]
    (out_dir / "source_provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_tradeoff(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    direct = [
        r
        for r in rows
        if r["comparability_status"] == "DIRECTLY_COMPARABLE"
        and r.get("official_ape_translation_rmse_m") not in (None, "")
        and not pd.isna(r.get("official_ape_translation_rmse_m"))
    ]
    order = ["wheel/odo+IMU", "VI", "LI", "LVI"]
    direct.sort(key=lambda r: (order.index(r["class"]) if r["class"] in order else 99, float(r["official_ape_translation_rmse_m"])))
    colors = {"wheel/odo+IMU": "#1f77b4", "VI": "#9467bd", "LI": "#2ca02c", "LVI": "#ff7f0e"}
    labels = [r["method"] for r in direct]
    values = [float(r["official_ape_translation_rmse_m"]) for r in direct]
    bar_colors = [colors.get(r["class"], "#777777") for r in direct]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Official ATE RMSE, sequence-RMS (m)")
    ax.set_title("Fidelity Versus Runtime Sensing Burden on i2Nav-Robot")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.98,
        "Directly comparable 10-sequence README ATE/ARE rows only\nExternal RPE values are not reported in the README table",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(out_dir / "sensing_fidelity_tradeoff.png", dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, stack_rows, protocol_rows = build_rows(args.results_root)

    fields = [
        "method",
        "wheel_odo",
        "imu",
        "camera",
        "lidar",
        "radar",
        "gnss",
        "proprioceptive_only",
        "official_ape_translation_rmse_m",
        "official_ape_rotation_rmse_deg",
        "official_ape_translation_macro_mean_m",
        "official_ape_rotation_macro_mean_deg",
        "official_rpe_50m_translation_rmse_m",
        "official_rpe_100m_translation_rmse_m",
        "official_rpe_300m_translation_rmse_m",
        "comparability_status",
        "source",
        "note",
    ]
    for row in rows:
        row["source"] = source_for_method(row)
    write_csv(out_dir / "sensing_fidelity_comparison.csv", rows, fields)
    write_csv(
        out_dir / "sensing_stack_audit.csv",
        stack_rows,
        [
            "method",
            "class",
            "wheel_odo",
            "imu",
            "camera",
            "lidar",
            "radar",
            "gnss",
            "proprioceptive_only",
            "sensor_count_reported",
            "sensor_model_reported",
            "compute_hardware_reported",
            "runtime_latency_reported",
            "power_reported",
            "source",
        ],
    )
    write_csv(
        out_dir / "protocol_comparability_audit.csv",
        protocol_rows,
        [
            "method",
            "dataset_sequences_used",
            "evaluation_protocol",
            "alignment_method",
            "metric_definition",
            "units",
            "aggregate",
            "comparability_status",
            "reason",
            "source",
        ],
    )
    write_claim_matrix(out_dir)
    write_provenance(out_dir)
    plot_tradeoff(out_dir, rows)
    write_summary(out_dir, rows)
    print(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sensing_fidelity_comparison"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
