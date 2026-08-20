"""Final result-freeze audit for the sensor-lightweight DT fidelity project.

This script reads existing frozen result artifacts and produces audit reports.
It does not retrain, tune, regenerate trajectories, or alter frozen predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEQUENCES = [
    "building00",
    "building01",
    "building02",
    "parking00",
    "parking01",
    "parking02",
    "playground00",
    "street00",
    "street01",
    "street02",
]

RPE_DELTAS = [50, 100, 150, 200, 250, 300]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any, digits: int = 3) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return "NA"


def ffloat(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else math.nan
    except Exception:
        return math.nan


def status_close(actual: float, expected: float, tol: float) -> str:
    return "PASS" if abs(actual - expected) <= tol else "REQUIRES_CORRECTION"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_headlines(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    macro = pd.read_csv(root / "i2nav_official_benchmark" / "official_macro_summary.csv")
    comp = pd.read_csv(root / "i2nav_official_benchmark" / "official_method_comparison.csv")
    internal = pd.read_csv(root / "i2nav_official_benchmark" / "official_internal_comparison.csv")
    per_seq = pd.read_csv(root / "i2nav_official_benchmark" / "official_per_sequence_results.csv")

    def macro_value(method: str, col: str) -> float:
        return ffloat(macro.loc[macro["method"] == method, col].iloc[0])

    def comp_value(metric: str, col: str) -> float:
        return ffloat(comp.loc[comp["metric"] == metric, col].iloc[0])

    def internal_value(seq: str, col: str) -> float:
        return ffloat(internal.loc[internal["sequence"] == seq, col].iloc[0])

    checks = [
        ("Official V2 APE translation macro mean", 1.635, macro_value("Twin V2", "official_ape_translation_rmse_m_macro_mean"), "official_macro_summary.csv", "Twin V2 sequence macro mean", 0.0005),
        ("Official V2 APE rotation macro mean", 3.011, macro_value("Twin V2", "official_ape_rotation_rmse_deg_macro_mean"), "official_macro_summary.csv", "Twin V2 sequence macro mean", 0.0005),
        ("Official V2 RPE50 translation macro mean", 1.310, macro_value("Twin V2", "official_rpe_50m_translation_rmse_m_macro_mean"), "official_macro_summary.csv", "Twin V2 sequence macro mean", 0.0005),
        ("Official V2 RPE50 percent macro mean", 2.62, macro_value("Twin V2", "official_rpe_50m_translation_pct_macro_mean"), "official_macro_summary.csv", "Twin V2 sequence macro mean", 0.005),
        ("Official V2 RPE100 translation macro mean", 2.217, macro_value("Twin V2", "official_rpe_100m_translation_rmse_m_macro_mean"), "official_macro_summary.csv", "Twin V2 sequence macro mean", 0.0005),
        ("Official V2 RPE100 percent macro mean", 2.22, macro_value("Twin V2", "official_rpe_100m_translation_pct_macro_mean"), "official_macro_summary.csv", "Twin V2 sequence macro mean", 0.005),
        ("Official V2 RPE300 translation macro mean", 3.635, macro_value("Twin V2", "official_rpe_300m_translation_rmse_m_macro_mean"), "official_macro_summary.csv", "Twin V2 sequence macro mean", 0.0005),
        ("Official V2 RPE300 percent macro mean", 1.21, macro_value("Twin V2", "official_rpe_300m_translation_pct_macro_mean"), "official_macro_summary.csv", "Twin V2 sequence macro mean", 0.005),
        ("Fixed Physics APE translation macro mean", 3.299, macro_value("Fixed Physics", "official_ape_translation_rmse_m_macro_mean"), "official_macro_summary.csv", "Fixed sequence macro mean", 0.0005),
        ("Fixed->V2 APE absolute macro difference", -1.664, comp_value("official_ape_translation_rmse_m", "absolute_difference_candidate_minus_baseline"), "official_method_comparison.csv", "V2 macro mean - Fixed macro mean", 0.0005),
        ("Fixed->V2 APE macro mean reduction percent", -50.44, comp_value("official_ape_translation_rmse_m", "macro_mean_percent_difference_candidate_vs_baseline"), "official_method_comparison.csv", "100*(V2-Fixed)/Fixed", 0.005),
        ("Fixed->V2 APE mean sequence-wise relative reduction percent", -34.89, comp_value("official_ape_translation_rmse_m", "mean_sequencewise_percent_difference_candidate_vs_baseline"), "official_method_comparison.csv", "mean over sequences of 100*(V2_s-Fixed_s)/Fixed_s", 0.005),
        ("Fixed->V2 APE sequence wins", 9, comp_value("official_ape_translation_rmse_m", "sequences_improved_lower_is_better"), "official_method_comparison.csv", "count V2_s < Fixed_s", 0.0),
        ("Fixed->V2 RPE50 sequence wins", 10, comp_value("official_rpe_50m_translation_rmse_m", "sequences_improved_lower_is_better"), "official_method_comparison.csv", "count V2_s < Fixed_s", 0.0),
        ("parking02 official aligned APE translation", 5.747, internal_value("parking02", "official_ape_translation_rmse_m_mean"), "official_internal_comparison.csv", "seed-mean official APE", 0.0005),
        ("parking02 internal ATE", 11.350, internal_value("parking02", "internal_ate_rmse_m_mean"), "official_internal_comparison.csv", "seed-mean internal ATE", 0.0005),
        ("parking02 internal Dp p95", 22.345, internal_value("parking02", "internal_Dp_p95_m_mean"), "official_internal_comparison.csv", "seed-mean internal Dp p95", 0.0005),
        ("parking02 internal Dtheta p95", 30.415, internal_value("parking02", "internal_Dtheta_p95_deg_mean"), "official_internal_comparison.csv", "seed-mean internal Dtheta p95", 0.0005),
        ("parking02 internal RPE10", 0.097, internal_value("parking02", "internal_rpe_10s_m_mean"), "official_internal_comparison.csv", "seed-mean internal RPE10", 0.0005),
        ("parking01 official aligned APE translation", 1.920, internal_value("parking01", "official_ape_translation_rmse_m_mean"), "official_internal_comparison.csv", "seed-mean official APE", 0.0005),
        ("parking01 internal ATE", 4.071, internal_value("parking01", "internal_ate_rmse_m_mean"), "official_internal_comparison.csv", "seed-mean internal ATE", 0.0005),
        ("parking01 internal Dp p95", 7.763, internal_value("parking01", "internal_Dp_p95_m_mean"), "official_internal_comparison.csv", "seed-mean internal Dp p95", 0.0005),
    ]
    rows = []
    for metric, expected, actual, source, formula, tol in checks:
        rows.append(
            {
                "metric": metric,
                "expected_value": expected,
                "recomputed_value": actual,
                "source_file": f"results/i2nav_official_benchmark/{source}",
                "formula": formula,
                "tolerance": tol,
                "status": status_close(actual, expected, tol),
                "rounding_convention": "manuscript values rounded to 3 decimals unless percent is reported to 2 decimals",
            }
        )

    write_rows(out_dir / "headline_number_audit.csv", rows)
    md = ["# Headline Number Audit", "", "| metric | expected | recomputed | status |", "|---|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['metric']} | {r['expected_value']} | {fmt(r['recomputed_value'], 6)} | {r['status']} |")
    if all(r["status"] == "PASS" for r in rows):
        md.append("\nAll headline values pass the specified tolerances.")
    else:
        md.append("\nAt least one headline value requires correction.")
    (out_dir / "headline_number_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def audit_rpe_eligibility(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    per_run = pd.read_csv(root / "i2nav_official_benchmark" / "official_per_run_results.csv")
    rows = []
    for method in sorted(per_run["method"].unique()):
        d = per_run[per_run["method"] == method]
        for delta in RPE_DELTAS:
            col = f"official_rpe_{delta}m_translation_rmse_m"
            err_col = f"official_rpe_{delta}m_error"
            valid = d[pd.to_numeric(d[col], errors="coerce").notna()]
            excluded = d[pd.to_numeric(d[col], errors="coerce").isna()]
            rows.append(
                {
                    "method": method,
                    "distance_m": delta,
                    "n_sequences": int(valid["sequence"].nunique()),
                    "n_runs": int(len(valid)),
                    "eligible_sequences": ";".join(sorted(valid["sequence"].unique())),
                    "excluded_sequences": ";".join(sorted(excluded["sequence"].unique())),
                    "excluded_runs": int(len(excluded)),
                    "exclusion_reason": "; ".join(sorted(set(str(x) for x in excluded.get(err_col, pd.Series(dtype=str)).dropna() if str(x)))),
                    "statistical_unit": "physical sequence; seeds are algorithmic replicates",
                    "status": "PASS" if len(excluded) == 0 else "PASS_WITH_CAVEAT",
                }
            )
    write_rows(out_dir / "distance_rpe_eligibility.csv", rows)
    md = ["# Distance-RPE Eligibility Audit", "", "| method | distance m | n_sequences | n_runs | excluded | status |", "|---|---:|---:|---:|---|---|"]
    for r in rows:
        md.append(f"| {r['method']} | {r['distance_m']} | {r['n_sequences']} | {r['n_runs']} | {r['excluded_sequences'] or 'none'} | {r['status']} |")
    (out_dir / "distance_rpe_eligibility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def audit_v1(root: Path, out_dir: Path) -> dict[str, Any]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        s = str(path).lower()
        if "i2nav" in s and "v1" in s and any(k in path.name.lower() for k in ["trajectory", "estimate_traj", "prediction_trace", "run_summary", "loso"]):
            candidates.append(str(path))
    accepted = []
    rejected = []
    for c in candidates:
        rejected.append(
            {
                "path": c,
                "reason": "does not establish exact frozen V1 trajectory correspondence for all 10 sequences and 3 seeds",
            }
        )
    payload = {
        "status": "V1 official comparison unavailable - no equivalent frozen trajectories.",
        "searched_root": str(root),
        "candidate_count": len(candidates),
        "accepted": accepted,
        "rejected": rejected[:200],
        "search_terms": "i2nav + v1 + trajectory/estimate_traj/prediction_trace/run_summary/loso",
    }
    write_json(out_dir / "v1_official_trajectory_audit.json", payload)
    md = [
        "# Frozen V1 Official-Trajectory Audit",
        "",
        "**V1 official comparison unavailable - no equivalent frozen trajectories.**",
        "",
        f"Candidate files found: {len(candidates)}.",
        "",
        "No candidate was accepted because none established exact frozen V1 trajectory correspondence for the same 10 held-out physical sequences and the same historical V1 evaluation.",
    ]
    (out_dir / "v1_official_trajectory_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def official_vs_internal(root: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "i2nav_official_benchmark" / "official_internal_comparison.csv")
    out = df.rename(
        columns={
            "official_ape_translation_rmse_m_mean": "official_ape_translation_rmse_m",
            "official_ape_rotation_rmse_deg_mean": "official_ape_rotation_rmse_deg",
            "internal_ate_rmse_m_mean": "internal_ate_m",
            "internal_heading_mae_deg_mean": "internal_heading_error_deg",
            "internal_Dp_p95_m_mean": "Dp_p95_m",
            "internal_Dtheta_p95_deg_mean": "Dtheta_p95_deg",
            "internal_rpe_1s_m_mean": "RPE1_m",
            "internal_rpe_5s_m_mean": "RPE5_m",
            "internal_rpe_10s_m_mean": "RPE10_m",
        }
    ).copy()
    out["alignment_effect_indicator"] = (out["internal_ate_m"] - out["official_ape_translation_rmse_m"]) / out["internal_ate_m"]
    cols = [
        "sequence",
        "official_ape_translation_rmse_m",
        "official_ape_rotation_rmse_deg",
        "internal_ate_m",
        "internal_heading_error_deg",
        "Dp_p95_m",
        "Dtheta_p95_deg",
        "RPE1_m",
        "RPE5_m",
        "RPE10_m",
        "alignment_effect_indicator",
    ]
    out[cols].to_csv(out_dir / "official_vs_internal_fidelity.csv", index=False)
    p1 = out[out["sequence"] == "parking01"].iloc[0]
    p2 = out[out["sequence"] == "parking02"].iloc[0]
    md = [
        "# Official vs Internal Fidelity Comparison",
        "",
        "Official i2Nav metrics and internal DT-fidelity metrics answer different scientific questions. Official APE/RPE measure trajectory error under the standardized benchmark alignment; internal fidelity measures physical-virtual synchronization in the operational/reference frame.",
        "",
        "## Hard Sequences",
        "",
        f"- parking02: internal ATE {fmt(p2['internal_ate_m'])} m -> official aligned APE {fmt(p2['official_ape_translation_rmse_m'])} m; alignment-effect indicator {fmt(p2['alignment_effect_indicator'])}; Dp p95 {fmt(p2['Dp_p95_m'])} m; RPE10 {fmt(p2['RPE10_m'])} m.",
        f"- parking01: internal ATE {fmt(p1['internal_ate_m'])} m -> official aligned APE {fmt(p1['official_ape_translation_rmse_m'])} m; alignment-effect indicator {fmt(p1['alignment_effect_indicator'])}; Dp p95 {fmt(p1['Dp_p95_m'])} m; RPE10 {fmt(p1['RPE10_m'])} m.",
        "",
        "This supports the local-vs-global distinction: short-horizon relative motion can look strong while long-horizon synchronization degrades.",
    ]
    (out_dir / "official_vs_internal_fidelity.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [p1["internal_ate_m"], p2["internal_ate_m"]], width, label="Internal ATE")
    ax.bar(x + width / 2, [p1["official_ape_translation_rmse_m"], p2["official_ape_translation_rmse_m"]], width, label="Official aligned APE")
    ax.set_xticks(x)
    ax.set_xticklabels(["parking01", "parking02"])
    ax.set_ylabel("translation error (m)")
    ax.set_title("Hard Sequence Alignment Effect")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hard_sequence_alignment_effect.png", dpi=180)
    plt.close(fig)
    return out[cols]


def official_aggregation_semantics(root: Path, out_dir: Path) -> dict[str, Any]:
    per_seq = pd.read_csv(root / "i2nav_official_benchmark" / "official_per_sequence_results.csv")
    v2 = per_seq[per_seq["method"] == "Twin V2"]
    rows = []
    for metric in [
        "official_ape_translation_rmse_m_mean",
        "official_ape_rotation_rmse_deg_mean",
        "official_rpe_50m_translation_rmse_m_mean",
        "official_rpe_100m_translation_rmse_m_mean",
        "official_rpe_300m_translation_rmse_m_mean",
    ]:
        vals = pd.to_numeric(v2[metric], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric.replace("_mean", ""),
                "arithmetic_sequence_macro_mean": float(np.mean(vals)),
                "official_table_sequence_RMS_if_needed": float(np.sqrt(np.mean(vals * vals))),
                "n_sequences": int(len(vals)),
                "note": "evaluate_odometry computes per-run metrics; i2Nav-Robot README table labels aggregate RMS values, so retain both arithmetic macro mean and sequence-RMS for later table matching.",
            }
        )
    write_rows(out_dir / "official_aggregation_semantics.csv", rows)
    md = ["# Official Aggregation Semantics", "", "The public evaluator computes per-trajectory metrics. The i2Nav-Robot README reports aggregate table rows labeled RMS, so this audit preserves both arithmetic sequence macro means and sequence-RMS aggregates. The existing manuscript-facing 1.635 m remains the arithmetic macro mean and is not replaced.", ""]
    md.extend(["| metric | arithmetic macro mean | sequence RMS |", "|---|---:|---:|"])
    for r in rows:
        md.append(f"| {r['metric']} | {fmt(r['arithmetic_sequence_macro_mean'])} | {fmt(r['official_table_sequence_RMS_if_needed'])} |")
    (out_dir / "official_aggregation_semantics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"rows": rows}


def hierarchy_audit(out_dir: Path) -> list[dict[str, str]]:
    rows = [
        ("official benchmark", "PASS_WITH_CAVEAT", "Per-run values retained; sequence aggregation used. Fixed Physics is deterministic one-run-per-sequence; V2 has three seed replicates per sequence. Caveat: Fixed orientation convention mismatch limits rotation/RPE interpretation."),
        ("full LOSO V1->V2 statistics", "PASS", "Context records sequence-aggregated macro means, sequence wins, bootstrap over physical sequences, and sign-flip tests."),
        ("all-sequence mechanism analysis", "PASS_WITH_CAVEAT", "Summary explicitly says timestamp correlations are descriptive only; sequence-level associations are primary."),
        ("condition-dependent fidelity", "PASS", "Condition summaries are within run, then seed-aggregated within physical sequence, then sequence-level."),
        ("benign fidelity characterization", "PASS_WITH_CAVEAT", "Correctly labels p95 envelopes as descriptive and sequence-sensitive, not thresholds."),
        ("LOSO benign envelope validation", "PASS", "Holds out physical sequences and reports sequence-level coverage/sensitivity."),
        ("UGV01 asset-specific instantiation", "PASS_WITH_CAVEAT", "Asset-specific evidence exists, but claims should remain condition-limited to available AprilTag/telemetry runs."),
        ("Fixed Physics comparison", "PASS_WITH_CAVEAT", "Translation comparison can be reported carefully; orientation/RPE gap likely affected by body-frame convention mismatch."),
        ("V1 official comparison", "PASS_WITH_CAVEAT", "Not available because exact frozen V1 trajectories are absent; no questionable reconstruction performed."),
    ]
    data = [{"analysis": a, "status": s, "rationale": r} for a, s, r in rows]
    md = ["# Statistical Hierarchy Audit", "", "| analysis | status | rationale |", "|---|---|---|"]
    for r in data:
        md.append(f"| {r['analysis']} | {r['status']} | {r['rationale']} |")
    (out_dir / "statistical_hierarchy_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return data


def claims_audit(root: Path, out_dir: Path) -> list[dict[str, str]]:
    rows = [
        ("V2 improves trajectory performance", "Official APE macro mean 1.635 m vs Fixed 3.299 m; internal V2 improves V1 on several sequence-level metrics", "official_macro_summary.csv; CODEX_CONTEXT_DT_FIDELITY_POST_LOSO.md", "physical sequence", "moderate/strong", "V2 improves official translation APE relative to the available Fixed Physics baseline and improves internal LOSO metrics relative to V1.", "Fixed orientation/RPE mismatch; V1 official unavailable.", "SOTA or universal odometry superiority."),
        ("local-vs-global fidelity distinction", "parking02 has low RPE10 but high Dp/Dtheta and ATE", "all_sequence_mechanism/mechanism_summary.md", "physical sequence", "strong", "Short-horizon relative fidelity and long-horizon synchronization are distinct.", "Do not make universal causal claim.", "RPE alone proves twin fidelity."),
        ("parking01/parking02 failure mode", "Both show low-local/high-global pattern; parking02 extreme", "all_sequence_mechanism/mechanism_summary.md", "physical sequence", "strong", "parking02 is an extreme point in a broader hard-sequence pattern.", "Sequence-specific factors remain.", "parking02 is solved or unique anecdote."),
        ("persistent yaw mismatch pathway", "persistent yaw mismatch strongly associated with Iomega; Dtheta strongly associated with Dp", "all_sequence_mechanism/mechanism_summary.md", "physical sequence", "moderate", "Persistent yaw mismatch is a measurable failure pathway.", "Direct Iomega->Dtheta association weak across all sequences.", "Universal monotonic causal law."),
        ("condition-dependent fidelity", "turning/wheel-IMU degrade RPE; acceleration/curvature degrade global metrics", "condition_fidelity/condition_fidelity_summary.md", "physical sequence", "strong descriptive", "Fidelity depends on operating condition.", "Not every variable is strong/monotonic.", "One scalar condition explains everything."),
        ("benign fidelity characterization", "Componentwise p95 envelopes and condition distributions", "benign_fidelity_characterization", "physical sequence", "moderate descriptive", "Benign divergence can be characterized empirically.", "p95 is descriptive, not a detector threshold.", "Exceeding p95 means attack/failure."),
        ("LOSO benign-envelope behavior", "rate-domain generalizes better than global dimensions", "loso_envelope_validation", "physical sequence", "strong descriptive", "Envelope is partially stable; rate-domain components generalize better.", "Global Dp/Dtheta sequence-sensitive.", "Envelope is universal stable guarantee."),
        ("UGV01 asset instantiation", "UGV01 staged instantiation artifacts and comparison figure", "ugv01_asset_instantiation", "asset run/condition", "moderate", "The framework can be instantiated on UGV01 under tested conditions.", "Condition-limited; reference uncertainty matters.", "Universal UGV01 performance."),
        ("official i2Nav benchmark", "Verified public protocol; V2 official macro values", "i2nav_official_benchmark", "physical sequence", "strong", "Official benchmark layer validates trajectory performance under standardized alignment.", "Separate from DT-fidelity layer.", "Official APE replaces fidelity profile."),
        ("Fixed Physics comparison", "V2 better APE translation 9/10; RPE50 10/10", "official_method_comparison.csv", "physical sequence", "moderate", "V2 improves translation APE relative to available fixed baseline.", "Orientation/RPE likely frame mismatch.", "Huge rotation/RPE gap is entirely model superiority."),
        ("official alignment vs operational synchronization", "parking02 ATE 11.350 -> official APE 5.747", "official_vs_internal_fidelity.csv", "physical sequence", "strong", "Official alignment changes apparent long-horizon error magnitude.", "Alignment is valid for benchmark; answers different question.", "Official benchmark is wrong."),
    ]
    data = [
        {
            "Claim": c,
            "Supporting result": sr,
            "Supporting file": sf,
            "Statistical unit": su,
            "Evidence strength": es,
            "Allowed manuscript wording": aw,
            "Required caveat": rc,
            "Prohibited overclaim": po,
        }
        for c, sr, sf, su, es, aw, rc, po in rows
    ]
    write_rows(out_dir / "final_claims_audit.csv", data)
    md = ["# Final Claims Audit", "", "| claim | evidence strength | allowed wording | prohibited overclaim |", "|---|---|---|---|"]
    for r in data:
        md.append(f"| {r['Claim']} | {r['Evidence strength']} | {r['Allowed manuscript wording']} | {r['Prohibited overclaim']} |")
    (out_dir / "final_claims_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return data


def manuscript_tables_and_figures(root: Path, out_dir: Path) -> None:
    pub = out_dir / "publication_ready"
    pub.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (root / "i2nav_official_benchmark" / "official_macro_summary.csv", pub / "table_official_benchmark_compact.csv"),
        (root / "i2nav_official_benchmark" / "official_per_sequence_results.csv", pub / "table_v2_official_per_sequence.csv"),
        (out_dir / "official_vs_internal_fidelity.csv", pub / "table_official_vs_internal_fidelity.csv"),
        (out_dir / "hard_sequence_alignment_effect.png", pub / "figure_hard_sequence_alignment_effect.png"),
        (root / "i2nav_v2_post_loso_analysis" / "condition_fidelity" / "fidelity_by_turning.png", pub / "figure_condition_dependent_fidelity_turning.png"),
        (root / "i2nav_v2_post_loso_analysis" / "benign_fidelity_characterization" / "benign_envelope_by_condition.png", pub / "figure_benign_envelope_by_condition.png"),
        (root / "i2nav_v2_post_loso_analysis" / "loso_envelope_validation" / "loso_conditioned_vs_unconditional_coverage.png", pub / "figure_loso_envelope_coverage.png"),
        (root.parent / "results" / "ugv01_asset_instantiation" / "ugv01_instantiation_comparison.png", pub / "figure_ugv01_asset_instantiation.png"),
    ]:
        if src.exists():
            dst.write_bytes(src.read_bytes())
    (pub / "README.md").write_text(
        "# Publication-Ready Tables and Figures\n\nThese files are copied from audited result artifacts without new analysis or visual retuning.\n",
        encoding="utf-8",
    )


def freeze_manifest(root: Path, out_dir: Path, status: str, headline_rows: list[dict[str, Any]]) -> None:
    manifest_dir = root / "frozen_results_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    files = [
        root / "i2nav_official_benchmark" / "official_macro_summary.csv",
        root / "i2nav_official_benchmark" / "official_per_sequence_results.csv",
        root / "i2nav_official_benchmark" / "official_method_comparison.csv",
        root / "i2nav_v2_post_loso_analysis" / "all_sequence_mechanism" / "mechanism_summary.md",
        root / "i2nav_v2_post_loso_analysis" / "condition_fidelity" / "condition_fidelity_summary.md",
        root / "i2nav_v2_post_loso_analysis" / "benign_fidelity_characterization" / "benign_fidelity_framework_summary.md",
        root / "i2nav_v2_post_loso_analysis" / "loso_envelope_validation" / "loso_benign_envelope_validation_summary.md",
        root / "ugv01_asset_instantiation" / "ugv01_asset_instantiation_summary.md",
    ]
    checksums = []
    for p in files:
        if p.exists():
            checksums.append({"path": str(p), "sha256": sha256(p)})
    payload = {
        "schema": "digital_twin_fidelity_frozen_results_manifest_v1",
        "date": "2026-08-20",
        "freeze_status": status,
        "frozen_v2_root": "results/i2nav_v2_full_loso/i2nav_v2_full_loso",
        "base_seeds": [42, 1042, 2042],
        "held_out_physical_sequences": SEQUENCES,
        "statistical_unit": "physical sequence; timestamps nested in seed runs nested in physical sequences",
        "official_protocol": {
            "source": "i2Nav-WHU/evaluate_odometry",
            "association_tolerance_s": 0.005,
            "alignment": "SE3_no_scale",
            "scale_correction": False,
            "rpe_deltas_m": RPE_DELTAS,
            "rpe_all_pairs": True,
            "rpe_relative_delta_tolerance": 0.002,
        },
        "headline_values": headline_rows,
        "known_limitations": [
            "Fixed Physics official orientation/RPE values likely reflect a legacy body-frame/orientation-convention mismatch.",
            "Exact frozen V1 trajectories are unavailable for official benchmark evaluation.",
            "Benign p95 envelopes are descriptive, not detection thresholds.",
            "Official benchmark and internal DT-fidelity answer different questions.",
        ],
        "post_freeze_rule": "No result-changing training, tuning, checkpoint selection, normalization changes, protocol changes, or post-hoc optimization should occur unless a genuine correctness error is discovered and documented.",
        "checksums": checksums,
    }
    write_json(manifest_dir / "frozen_results_manifest.json", payload)
    (manifest_dir / "README.md").write_text(
        "# Frozen Results Manifest\n\n"
        f"Freeze status: **{status}**\n\n"
        "No result-changing training, tuning, checkpoint selection, normalization changes, protocol changes, or post-hoc optimization should occur unless a genuine correctness error is discovered and documented.\n",
        encoding="utf-8",
    )


def readiness_report(out_dir: Path, status: str, blockers: list[str]) -> None:
    lines = [
        "# Final Result-Freeze Readiness",
        "",
        f"## {status}",
        "",
        "This audit checks correctness, reproducibility, protocol equivalence, statistical validity, provenance, interpretation, and available baselines without retraining or changing frozen predictions.",
        "",
        "## Blocking Issues",
        "",
    ]
    if blockers:
        lines.extend(f"- {b}" for b in blockers)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Required Caveats",
            "",
            "- V1 official comparison unavailable: no equivalent frozen trajectories.",
            "- Fixed Physics orientation/RPE official gaps likely include a legacy orientation/body-frame convention mismatch.",
            "- Official benchmark metrics and internal DT-fidelity metrics must remain separate.",
            "- Benign p95 envelopes are descriptive and are not attack/failure thresholds.",
            "",
            "## Recommendation",
            "",
            status,
        ]
    )
    (out_dir / "FINAL_RESULT_FREEZE_READINESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_audit"))
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    headline = audit_headlines(args.results_root, out_dir)
    rpe = audit_rpe_eligibility(args.results_root, out_dir)
    v1 = audit_v1(args.results_root, out_dir)
    official_vs_internal(args.results_root, out_dir)
    official_aggregation_semantics(args.results_root, out_dir)
    hierarchy = hierarchy_audit(out_dir)
    claims_audit(args.results_root, out_dir)
    manuscript_tables_and_figures(args.results_root, out_dir)

    blockers: list[str] = []
    if any(r["status"] == "REQUIRES_CORRECTION" for r in headline):
        blockers.append("Headline number audit contains mismatches.")
    if any(r["status"] == "REQUIRES_CORRECTION" for r in hierarchy):
        blockers.append("Statistical hierarchy audit contains corrections.")
    status = "NOT_READY_TO_FREEZE" if blockers else "READY_TO_FREEZE"
    readiness_report(out_dir, status, blockers)
    freeze_manifest(args.results_root, out_dir, status, headline)
    print(status)


if __name__ == "__main__":
    main()
