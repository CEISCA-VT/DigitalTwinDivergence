"""Build a polished pre-benchmark presentation results review.

This document intentionally excludes official i2Nav benchmark scoring,
sensing-fidelity comparisons, portability datasets, and security experiments.
It uses completed frozen DT-fidelity artifacts only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image


@dataclass
class Img:
    path: Path
    rel_id: str
    width_in: float = 5.8


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: float) -> str:
    return f"{float(value):.1f}%"


def copy_fig(src: Path, dst_dir: Path, name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / name
    shutil.copy2(src, dst)
    return dst


def gather(results_root: Path) -> dict:
    macro = read_json(results_root / "i2nav_v2_full_loso" / "i2nav_v2_full_loso_summary" / "full_loso_macro_summary.json")
    mech_seq = pd.read_csv(results_root / "i2nav_frozen_v2_fidelity_analysis" / "all_sequence_mechanism" / "per_sequence_mechanism.csv")
    assoc = pd.read_csv(results_root / "i2nav_frozen_v2_fidelity_analysis" / "all_sequence_mechanism" / "mechanism_sequence_associations.csv")
    cond = pd.read_csv(results_root / "i2nav_frozen_v2_fidelity_analysis" / "condition_fidelity" / "condition_degradation_summary.csv")
    loso = pd.read_csv(results_root / "i2nav_frozen_v2_fidelity_analysis" / "loso_envelope_validation" / "loso_envelope_validation_summary.csv")
    ugv = pd.read_csv(results_root / "ugv01_physical_instantiation" / "ugv01_fidelity_by_stage.csv")
    return {"macro": macro, "mech_seq": mech_seq, "assoc": assoc, "cond": cond, "loso": loso, "ugv": ugv}


def metric_row(macro: dict, key: str, label: str, units: str) -> list[str]:
    m = macro["metrics"][key]
    ci = m["V2_minus_V1"]
    return [
        label,
        f(m["V1"]["mean"]),
        f(m["V2"]["mean"]),
        pct(m["macro_change_pct"]),
        f'{m["sequences_v2_better_than_v1"]}/10',
        f'[{f(ci["ci95_low"])}, {f(ci["ci95_high"])}] {units}',
    ]


def assoc_value(df: pd.DataFrame, name: str, col: str) -> float:
    return float(df.loc[df["association"] == name, col].iloc[0])


def cond_lookup(df: pd.DataFrame, var: str, comp: str, metric: str) -> pd.Series:
    match = df[(df["condition_variable"] == var) & (df["comparison_bin"] == comp) & (df["metric"] == metric)]
    if match.empty:
        raise KeyError((var, comp, metric))
    return match.iloc[0]


def loso_all(df: pd.DataFrame, component: str) -> pd.Series:
    row = df[(df["context"] == "ALL_SUPPORTED_CONTEXTS") & (df["component"] == component)]
    if row.empty:
        raise KeyError(component)
    return row.iloc[0]


def build_figures(results_root: Path, out_dir: Path) -> list[Path]:
    fig_dir = out_dir / "figures"
    return [
        copy_fig(
            results_root / "i2nav_frozen_v2_fidelity_analysis" / "all_sequence_mechanism" / "local_vs_global_fidelity.png",
            fig_dir,
            "fig1_local_vs_global_fidelity.png",
        ),
        copy_fig(
            results_root / "i2nav_frozen_v2_fidelity_analysis" / "all_sequence_mechanism" / "persistent_yaw_vs_global_divergence.png",
            fig_dir,
            "fig2_persistent_yaw_mechanism.png",
        ),
        copy_fig(
            results_root / "i2nav_frozen_v2_fidelity_analysis" / "condition_fidelity" / "fidelity_by_turning.png",
            fig_dir,
            "fig3_condition_turning.png",
        ),
        copy_fig(
            results_root / "i2nav_frozen_v2_fidelity_analysis" / "benign_fidelity_characterization" / "benign_envelope_by_condition.png",
            fig_dir,
            "fig4_benign_envelope_by_condition.png",
        ),
        copy_fig(
            results_root / "i2nav_frozen_v2_fidelity_analysis" / "loso_envelope_validation" / "loso_conditioned_vs_unconditional_coverage.png",
            fig_dir,
            "fig5_loso_envelope_coverage.png",
        ),
        copy_fig(
            results_root / "ugv01_physical_instantiation" / "ugv01_instantiation_comparison.png",
            fig_dir,
            "fig6_ugv01_asset_instantiation.png",
        ),
    ]


def build_markdown(out_dir: Path, data: dict, figs: list[Path]) -> Path:
    macro = data["macro"]
    mech = data["mech_seq"]
    assoc = data["assoc"]
    cond = data["cond"]
    loso = data["loso"]
    ugv = data["ugv"]

    result_rows = [
        metric_row(macro, "ate_m", "ATE RMSE", "m"),
        metric_row(macro, "heading_mae_deg", "Heading MAE", "deg"),
        metric_row(macro, "rpe_1s_m", "RPE1", "m"),
        metric_row(macro, "rpe_5s_m", "RPE5", "m"),
        metric_row(macro, "rpe_10s_m", "RPE10", "m"),
    ]
    local_global = mech[mech["sequence"].isin(["parking00", "parking01", "parking02"])][
        ["sequence", "ATE_m", "RPE10_m", "Dp_p95_m", "Dtheta_p95_deg"]
    ]
    c1 = cond_lookup(cond, "wheel_imu_disagreement", "high", "RPE1_m")
    c5 = cond_lookup(cond, "turning", "high", "RPE5_m")
    c10 = cond_lookup(cond, "turning", "high", "RPE10_m")
    cdp = cond_lookup(cond, "acceleration", "high", "Dp_p95_m")
    cdth = cond_lookup(cond, "curvature", "high", "Dtheta_p95_deg")
    s1 = ugv[ugv["stage_id"] == "S1"].iloc[0]
    s2 = ugv[ugv["stage_id"] == "S2"].iloc[0]
    s3 = ugv[ugv["stage_id"] == "S3"].iloc[0]

    def md_table(headers: list[str], rows: list[list[str]]) -> str:
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])

    text = f"""# Sensor-Lightweight Digital Twin Fidelity for Mobile Robots - Consolidated Experimental Results

**Pre-Manuscript Research Review**  
Date: 2026-08-20

## Executive Summary

This project studies the fidelity of the digital twin itself: how closely a sensor-lightweight computational replica remains synchronized with a physical mobile robot, how that synchronization degrades, and what asset-specific information is required before a generic model can reasonably be called a twin of a specific robot.

The core experiments are now coherent enough to support manuscript writing. The completed evidence includes a frozen 10-sequence x 3-seed i2Nav evaluation, local/global fidelity analysis, mechanism analysis, condition-dependent fidelity analysis, empirical benign fidelity characterization, leave-one-sequence-out envelope validation, and UGV01 asset-specific physical instantiation.

**Central result:** short-horizon relative-motion fidelity can remain strong while long-horizon physical-virtual synchronization deteriorates substantially. This means digital-twin fidelity should be evaluated as a multidimensional, condition-dependent profile rather than as one trajectory-error score.

## Core Experimental Protocol

The frozen i2Nav study uses 10 held-out physical sequences and three seeds per held-out sequence, giving 30 frozen Twin V2 runs. The physical sequence is the primary statistical unit. The three seeds quantify algorithmic variability within a sequence; they are not treated as independent physical experiments.

The statistical hierarchy is:

`timestamps subset seed run subset physical sequence subset dataset`

This avoids pseudo-replication because timestamp-level samples are correlated and are not counted as independent evidence for dataset-level claims.

## Main Frozen V1 -> V2 Results

{md_table(["Metric", "V1", "V2", "V2 change", "Sequences improved", "95% CI for V2 - V1"], result_rows)}

Interpretation: heading and longer-horizon fidelity improve clearly. RPE1 is essentially preserved. ATE improvement is meaningful but not universal across all sequences, so the safe wording is that V2 reduces an important long-horizon failure mode rather than solving global fidelity everywhere.

## Local Versus Global Fidelity

Good local relative-motion fidelity does not necessarily imply good long-horizon digital-twin synchronization. parking02 is the clearest example; parking01 provides corroborating evidence.

{md_table(["Sequence", "ATE (m)", "RPE10 (m)", "Dp p95 (m)", "Dtheta p95 (deg)"], [[r.sequence, f(r.ATE_m), f(r.RPE10_m), f(r.Dp_p95_m), f(r.Dtheta_p95_deg)] for r in local_global.itertuples()])}

![Local versus global fidelity](figures/{figs[0].name})

**Figure 1.** Local finite-horizon error and global divergence are empirically distinct in the frozen Twin V2 evaluation.

## Mechanism of Accumulated Divergence

The all-sequence mechanism analysis supports the following measured pathway:

`persistent yaw mismatch -> accumulated yaw residual -> heading divergence -> position divergence`

The strongest sequence-level associations are:

{md_table(["Association", "Pearson r", "Spearman r", "Interpretation"], [
    ["persistent yaw mismatch -> accumulated yaw residual", f(assoc_value(assoc, "persistent yaw mismatch -> accumulated yaw residual", "pearson_r_sequence_level")), f(assoc_value(assoc, "persistent yaw mismatch -> accumulated yaw residual", "spearman_r_sequence_level")), "strong"],
    ["accumulated yaw residual -> heading divergence", f(assoc_value(assoc, "Iomega -> heading divergence", "pearson_r_sequence_level")), f(assoc_value(assoc, "Iomega -> heading divergence", "spearman_r_sequence_level")), "weak/non-monotonic across sequences"],
    ["heading divergence -> position divergence", f(assoc_value(assoc, "heading divergence -> position divergence", "pearson_r_sequence_level")), f(assoc_value(assoc, "heading divergence -> position divergence", "spearman_r_sequence_level")), "strong"],
])}

Persistent yaw mismatch is therefore a measurable failure pathway associated with long-horizon divergence. It should not be phrased as a universal monotonic causal law because the accumulated-yaw-residual to heading-divergence relationship is weaker across all sequences.

![Persistent yaw mechanism](figures/{figs[1].name})

**Figure 2.** Persistent yaw mismatch is strongly associated with accumulated yaw residual and global divergence in the sequence-level analysis.

## Condition-Dependent Fidelity

The condition analysis uses frozen condition definitions. It computes metrics within each run, aggregates the three seeds within each physical sequence, and treats the 10 physical sequences as the dataset-level units.

{md_table(["Fidelity component", "Condition contrast", "Median delta", "Sequences degraded"], [
    ["RPE1", "high vs low wheel-IMU disagreement", f(c1.median_delta_vs_nominal), f"{int(c1.n_sequences_degraded)}/10"],
    ["RPE5", "high vs low turning", f(c5.median_delta_vs_nominal), f"{int(c5.n_sequences_degraded)}/10"],
    ["RPE10", "high vs low turning", f(c10.median_delta_vs_nominal), f"{int(c10.n_sequences_degraded)}/10"],
    ["Dp p95", "high vs low acceleration", f(cdp.median_delta_vs_nominal), f"{int(cdp.n_sequences_degraded)}/10"],
    ["Dtheta p95", "high vs low curvature", f(cdth.median_delta_vs_nominal), f"{int(cdth.n_sequences_degraded)}/10"],
])}

The conditions associated with degraded local fidelity are not identical to those associated with degraded global synchronization. This is an important digital-twin result because it shows that one scalar accuracy score is insufficient to describe fidelity.

![Condition-dependent fidelity](figures/{figs[2].name})

**Figure 3.** Turning intensity is one of the clearest operating contexts for local finite-horizon fidelity degradation.

## Empirical Benign Fidelity Characterization

The empirical object is:

`D(t) | H0, c ~ P_benign(D | c)`

In plain language, the project characterizes how much physical-virtual disagreement is normally observed under benign operating conditions. The p95 envelope is componentwise: meters, degrees, m/s, and rad/s are kept separate rather than collapsed into one arbitrary score.

For elapsed-time context, `Dp` p95 changes from **5.828 m** in the early-run bin to **17.118 m** in the late-run bin. This shows why an unconditional envelope can hide important condition dependence.

The p95 envelope is descriptive. It is not an attack threshold, not a universal failure threshold, and not a claim of untrustworthiness.

![Benign envelope](figures/{figs[3].name})

**Figure 4.** Componentwise benign divergence envelopes vary by operating condition.

## Held-Out Envelope Validation

The leave-one-sequence-out envelope validation holds out one physical sequence, estimates p95 envelopes from the remaining nine sequences, and evaluates the held-out sequence against that envelope.

{md_table(["Component", "Mean conditioned p95 coverage", "Mean unconditional p95 coverage", "Mean conditioned exceedance"], [
    ["Dv", f(loso_all(loso, "Dv_mps").mean_conditioned_inside_p95_fraction), f(loso_all(loso, "Dv_mps").mean_unconditional_inside_p95_fraction), f(loso_all(loso, "Dv_mps").mean_conditioned_exceedance_p95_fraction)],
    ["Domega", f(loso_all(loso, "Domega_radps").mean_conditioned_inside_p95_fraction), f(loso_all(loso, "Domega_radps").mean_unconditional_inside_p95_fraction), f(loso_all(loso, "Domega_radps").mean_conditioned_exceedance_p95_fraction)],
    ["Dp", f(loso_all(loso, "Dp_m").mean_conditioned_inside_p95_fraction), f(loso_all(loso, "Dp_m").mean_unconditional_inside_p95_fraction), f(loso_all(loso, "Dp_m").mean_conditioned_exceedance_p95_fraction)],
    ["Dtheta", f(loso_all(loso, "Dtheta_deg").mean_conditioned_inside_p95_fraction), f(loso_all(loso, "Dtheta_deg").mean_unconditional_inside_p95_fraction), f(loso_all(loso, "Dtheta_deg").mean_conditioned_exceedance_p95_fraction)],
])}

Interpretation: the envelope is partially stable. Rate-domain benign disagreement generalizes strongly, while global position and heading are more sequence-sensitive. parking02 remains the dominant hard held-out global case. Conditioning improves interpretability of where divergence grows, but it does not universally maximize raw held-out coverage.

![LOSO envelope validation](figures/{figs[4].name})

**Figure 5.** Held-out p95 coverage is strongest for rate-domain quantities and weaker for global divergence dimensions.

## UGV01 Asset-Specific Instantiation

The real-robot section asks whether a generic twin representation becomes a better twin after being bound to one specific physical UGV01 rover. The independent reference is AprilTag-based physical trajectory estimation.

{md_table(["Metric", "Current UGV01 model", "Asset-specific fitted", "Change"], [
    ["ATE RMSE", f(s1.ate_rmse_m) + " m", f(s2.ate_rmse_m) + " m", pct((s2.ate_rmse_m - s1.ate_rmse_m) / s1.ate_rmse_m * 100.0)],
    ["RPE1 RMSE", f(s1.rpe1_rmse_m) + " m", f(s2.rpe1_rmse_m) + " m", pct((s2.rpe1_rmse_m - s1.rpe1_rmse_m) / s1.rpe1_rmse_m * 100.0)],
    ["Heading MAE", f(s1.heading_mae_deg) + " deg", f(s2.heading_mae_deg) + " deg", pct((s2.heading_mae_deg - s1.heading_mae_deg) / s1.heading_mae_deg * 100.0)],
])}

Current full repaired UGV01 headline metrics are: ATE RMSE **{f(s3.ate_rmse_m)} m**, RPE1 **{f(s3.rpe1_rmse_m)} m**, RPE5 **{f(s3.rpe5_rmse_m)} m**, RPE10 **{f(s3.rpe10_rmse_m)} m**, and heading MAE **{f(s3.heading_mae_deg)} deg**.

The supported claim is scoped: asset-specific calibration materially improves physical-virtual fidelity under the validated low-speed indoor AprilTag condition. It does not establish universal UGV01 performance across all surfaces, speeds, and traction regimes.

![UGV01 asset instantiation](figures/{figs[5].name})

**Figure 6.** UGV01-specific calibration improves the same-window physical twin comparison.

## Consolidated Evidence Table

{md_table(["Research question", "Evidence", "Current conclusion", "Status"], [
    ["RQ1: How should DT fidelity be measured?", "multidimensional profile with ATE, heading, RPE, Dp, Dtheta, Dv, Domega", "one scalar trajectory score is insufficient", "supported/formalized"],
    ["RQ2: Why can local fidelity coexist with global divergence?", "parking01/parking02 plus mechanism analysis", "persistent orientation mismatch can accumulate into global divergence", "empirically supported"],
    ["RQ3: Does fidelity depend on operating condition?", "condition-dependent summaries across 10 sequences", "local and global degradation arise under different conditions", "supported"],
    ["RQ4: Can benign fidelity be characterized and validated?", "componentwise envelope plus LOSO envelope validation", "partially stable, sequence-sensitive for global dimensions", "supported with limitations"],
    ["RQ5: Can the framework instantiate on a real asset?", "UGV01 AprilTag same-window calibration comparison", "asset-specific binding improves fidelity under tested conditions", "supported for indoor low-speed run"],
])}

## What the Current Evidence Supports

- Digital-twin fidelity is multidimensional and should not be reduced to one trajectory metric.
- Short-horizon relative-motion fidelity can remain strong while global physical-virtual synchronization deteriorates.
- Persistent yaw mismatch is a measurable pathway for accumulated divergence.
- Fidelity varies with operating conditions, and local/global degradation are associated with different conditions.
- Benign physical-virtual divergence can be empirically characterized and partially generalized to unseen physical sequences.
- Asset-specific calibration improves the real UGV01 twin under the tested condition.

## Remaining Limitations

- UGV01 physical validation is limited to existing tested indoor conditions.
- Global envelope quantities are more sequence-sensitive than rate-domain quantities.
- The p95 envelope is descriptive rather than a universal trust threshold.

## Assessment of Research Readiness

The core DT-fidelity research questions are experimentally supported across frozen i2Nav evaluation, mechanism analysis, condition-dependent characterization, held-out benign-envelope validation, and real-robot UGV01 instantiation. Remaining work is primarily manuscript construction and external positioning rather than further development of the core fidelity framework.
"""
    out = out_dir / "DT_Fidelity_Consolidated_PreBenchmark_Results.md"
    out.write_text(text, encoding="utf-8")
    return out


def p(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    return f"<w:p><w:pPr>{style_xml}</w:pPr><w:r><w:t>{escape(text)}</w:t></w:r></w:p>"


def tbl(rows: list[list[str]]) -> str:
    out = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
    for row in rows:
        out.append("<w:tr>")
        for cell in row:
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="2400" w:type="dxa"/></w:tcPr><w:p><w:r><w:t>{escape(str(cell))}</w:t></w:r></w:p></w:tc>')
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def img_xml(img: Img) -> str:
    with Image.open(img.path) as im:
        w, h = im.size
    cx = int(img.width_in * 914400)
    cy = int(cx * h / w)
    name = escape(img.path.name)
    return f"""
<w:p><w:r><w:drawing>
<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{img.rel_id[3:]}" name="{name}"/>
<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:nvPicPr><pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{img.rel_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic>
</wp:inline></w:drawing></w:r></w:p>"""


def build_docx(out_dir: Path, data: dict, figs: list[Path]) -> Path:
    macro = data["macro"]
    mech = data["mech_seq"]
    assoc = data["assoc"]
    cond = data["cond"]
    loso = data["loso"]
    ugv = data["ugv"]

    imgs = [Img(path=fig, rel_id=f"rId{i+1}") for i, fig in enumerate(figs)]
    result_rows = [["Metric", "V1", "V2", "V2 change", "Improved", "95% CI"]] + [
        metric_row(macro, "ate_m", "ATE RMSE", "m"),
        metric_row(macro, "heading_mae_deg", "Heading MAE", "deg"),
        metric_row(macro, "rpe_1s_m", "RPE1", "m"),
        metric_row(macro, "rpe_5s_m", "RPE5", "m"),
        metric_row(macro, "rpe_10s_m", "RPE10", "m"),
    ]
    lg_rows = [["Sequence", "ATE", "RPE10", "Dp p95", "Dtheta p95"]]
    for r in mech[mech["sequence"].isin(["parking00", "parking01", "parking02"])].itertuples():
        lg_rows.append([r.sequence, f(r.ATE_m), f(r.RPE10_m), f(r.Dp_p95_m), f(r.Dtheta_p95_deg)])
    s1 = ugv[ugv["stage_id"] == "S1"].iloc[0]
    s2 = ugv[ugv["stage_id"] == "S2"].iloc[0]
    s3 = ugv[ugv["stage_id"] == "S3"].iloc[0]
    cond_rows = [["Component", "Condition contrast", "Median delta", "Degraded"]]
    for label, var, comp, met in [
        ("RPE1", "wheel_imu_disagreement", "high", "RPE1_m"),
        ("RPE5", "turning", "high", "RPE5_m"),
        ("RPE10", "turning", "high", "RPE10_m"),
        ("Dp p95", "acceleration", "high", "Dp_p95_m"),
        ("Dtheta p95", "curvature", "high", "Dtheta_p95_deg"),
    ]:
        row = cond_lookup(cond, var, comp, met)
        cond_rows.append([label, f"high vs low {var}", f(row.median_delta_vs_nominal), f"{int(row.n_sequences_degraded)}/10"])
    loso_rows = [["Component", "Conditioned p95 coverage", "Unconditional p95 coverage"]]
    for component, label in [("Dv_mps", "Dv"), ("Domega_radps", "Domega"), ("Dp_m", "Dp"), ("Dtheta_deg", "Dtheta")]:
        row = loso_all(loso, component)
        loso_rows.append([label, f(row.mean_conditioned_inside_p95_fraction), f(row.mean_unconditional_inside_p95_fraction)])

    body = [
        p("Sensor-Lightweight Digital Twin Fidelity for Mobile Robots - Consolidated Experimental Results", "Title"),
        p("Pre-Manuscript Research Review | 2026-08-20", "Subtitle"),
        p("Executive Summary", "Heading1"),
        p("This project studies the fidelity of the digital twin itself: how closely a sensor-lightweight computational replica remains synchronized with a physical mobile robot, how that synchronization degrades, and what asset-specific information is required before a generic model can reasonably be called a twin of a specific robot."),
        p("The core experiments are now coherent enough to support manuscript writing: frozen 10-sequence x 3-seed i2Nav evaluation, local/global fidelity analysis, mechanism analysis, condition-dependent fidelity analysis, empirical benign fidelity characterization, leave-one-sequence-out envelope validation, and UGV01 asset-specific physical instantiation."),
        p("Core Experimental Protocol", "Heading1"),
        p("The frozen i2Nav study uses 10 held-out physical sequences and three seeds per held-out sequence, giving 30 frozen Twin V2 runs. The hierarchy is timestamps subset seed run subset physical sequence subset dataset. Dataset-level claims use physical sequence as the primary unit, avoiding timestamp-level pseudo-replication."),
        p("Main Frozen V1 -> V2 Results", "Heading1"),
        tbl(result_rows),
        p("Heading and longer-horizon fidelity improve clearly. RPE1 is essentially preserved. ATE improvement is meaningful but not universal, so V2 should be described as reducing an important long-horizon failure mode rather than solving global fidelity everywhere."),
        p("Local Versus Global Fidelity", "Heading1"),
        tbl(lg_rows),
        img_xml(imgs[0]),
        p("Figure 1. Good local relative-motion fidelity does not necessarily imply good long-horizon digital-twin synchronization."),
        p("Mechanism of Accumulated Divergence", "Heading1"),
        tbl([
            ["Association", "Pearson r", "Spearman r"],
            ["persistent yaw mismatch -> accumulated yaw residual", f(assoc_value(assoc, "persistent yaw mismatch -> accumulated yaw residual", "pearson_r_sequence_level")), f(assoc_value(assoc, "persistent yaw mismatch -> accumulated yaw residual", "spearman_r_sequence_level"))],
            ["accumulated yaw residual -> heading divergence", f(assoc_value(assoc, "Iomega -> heading divergence", "pearson_r_sequence_level")), f(assoc_value(assoc, "Iomega -> heading divergence", "spearman_r_sequence_level"))],
            ["heading divergence -> position divergence", f(assoc_value(assoc, "heading divergence -> position divergence", "pearson_r_sequence_level")), f(assoc_value(assoc, "heading divergence -> position divergence", "spearman_r_sequence_level"))],
        ]),
        img_xml(imgs[1]),
        p("Figure 2. Persistent yaw mismatch is a measurable failure pathway, but not a universal monotonic causal law."),
        p("Condition-Dependent Fidelity", "Heading1"),
        tbl(cond_rows),
        img_xml(imgs[2]),
        p("Figure 3. Conditions associated with degraded local fidelity are not identical to conditions associated with degraded global synchronization."),
        p("Empirical Benign Fidelity Characterization", "Heading1"),
        p("The empirical object is D(t) | H0,c ~ P_benign(D | c): how much physical-virtual disagreement is normally observed under benign operating conditions. The p95 envelope is componentwise, descriptive, and not a universal failure or trust threshold."),
        p("For elapsed-time context, Dp p95 changes from 5.828 m in the early-run bin to 17.118 m in the late-run bin, showing why an unconditional envelope hides condition dependence."),
        img_xml(imgs[3]),
        p("Figure 4. Componentwise benign divergence envelopes vary by operating condition."),
        p("Held-Out Envelope Validation", "Heading1"),
        tbl(loso_rows),
        img_xml(imgs[4]),
        p("Figure 5. The envelope is partially stable: rate-domain disagreement generalizes strongly, while global position and heading are more sequence-sensitive."),
        p("UGV01 Asset-Specific Instantiation", "Heading1"),
        tbl([
            ["Metric", "Current UGV01 model", "Asset-specific fitted", "Change"],
            ["ATE RMSE", f"{f(s1.ate_rmse_m)} m", f"{f(s2.ate_rmse_m)} m", pct((s2.ate_rmse_m - s1.ate_rmse_m) / s1.ate_rmse_m * 100.0)],
            ["RPE1 RMSE", f"{f(s1.rpe1_rmse_m)} m", f"{f(s2.rpe1_rmse_m)} m", pct((s2.rpe1_rmse_m - s1.rpe1_rmse_m) / s1.rpe1_rmse_m * 100.0)],
            ["Heading MAE", f"{f(s1.heading_mae_deg)} deg", f"{f(s2.heading_mae_deg)} deg", pct((s2.heading_mae_deg - s1.heading_mae_deg) / s1.heading_mae_deg * 100.0)],
        ]),
        p(f"Current full repaired UGV01 headline metrics: ATE RMSE {f(s3.ate_rmse_m)} m, RPE1 {f(s3.rpe1_rmse_m)} m, RPE5 {f(s3.rpe5_rmse_m)} m, RPE10 {f(s3.rpe10_rmse_m)} m, heading MAE {f(s3.heading_mae_deg)} deg."),
        img_xml(imgs[5]),
        p("Figure 6. Asset-specific calibration materially improves physical-virtual fidelity under the validated low-speed indoor AprilTag condition."),
        p("Consolidated Evidence", "Heading1"),
        tbl([
            ["Research question", "Evidence", "Current conclusion", "Status"],
            ["How should DT fidelity be measured?", "multidimensional profile", "one scalar is insufficient", "supported"],
            ["Why can local fidelity coexist with global divergence?", "parking01/parking02 and mechanism analysis", "orientation mismatch can accumulate", "supported"],
            ["Does fidelity depend on condition?", "condition analysis", "local/global degradation differ", "supported"],
            ["Can benign fidelity be characterized?", "envelope plus LOSO validation", "partially stable", "supported with limitations"],
            ["Can the framework instantiate on a real asset?", "UGV01 AprilTag comparison", "calibration improves fidelity", "supported for tested condition"],
        ]),
        p("Assessment of Research Readiness", "Heading1"),
        p("The core DT-fidelity research questions are experimentally supported across frozen i2Nav evaluation, mechanism analysis, condition-dependent characterization, held-out benign-envelope validation, and real-robot UGV01 instantiation. Remaining work is primarily manuscript construction and external positioning rather than further development of the core fidelity framework."),
    ]

    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<w:body>{''.join(body)}<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr></w:body></w:document>'''
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Aptos"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:color w:val="173F5F"/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr><w:color w:val="555555"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:color w:val="173F5F"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdDoc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    doc_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">', '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>']
    for im in imgs:
        doc_rels.append(f'<Relationship Id="{im.rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{im.path.name}"/>')
    doc_rels.append("</Relationships>")

    out = out_dir / "DT_Fidelity_Consolidated_PreBenchmark_Results.docx"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/styles.xml", styles_xml)
        z.writestr("word/_rels/document.xml.rels", "\n".join(doc_rels))
        for im in imgs:
            z.write(im.path, f"word/media/{im.path.name}")
    return out


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = gather(args.results_root)
    figs = build_figures(args.results_root, args.output_dir)
    md = build_markdown(args.output_dir, data, figs)
    docx = build_docx(args.output_dir, data, figs)
    print(md)
    print(docx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/results_quality_review"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
