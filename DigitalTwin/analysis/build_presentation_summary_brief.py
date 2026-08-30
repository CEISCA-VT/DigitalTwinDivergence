"""Build a compact presentation-facing digital-twin fidelity results dossier."""

from __future__ import annotations

import argparse
import html
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


BLUE = "#1f77b4"
DARK = "#173f5f"
ORANGE = "#f28e2b"
GRAY = "#9aa5b1"
LIGHT = "#eef3f7"


@dataclass
class ImageRef:
    path: Path
    rel_id: str
    width_in: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def build_official_tradeoff(results_root: Path, fig_dir: Path) -> Path:
    df = pd.read_csv(results_root / "sensing_fidelity_comparison" / "sensing_fidelity_comparison.csv")
    direct = df[
        (df["comparability_status"] == "DIRECTLY_COMPARABLE")
        & df["official_ape_translation_rmse_m"].notna()
        & (df["official_ape_translation_rmse_m"] != "")
    ].copy()
    direct["official_ape_translation_rmse_m"] = pd.to_numeric(direct["official_ape_translation_rmse_m"])
    direct = direct.sort_values("official_ape_translation_rmse_m", ascending=True)
    colors = [BLUE if m == "Twin V2" else ORANGE if po == "yes" else GRAY for m, po in zip(direct["method"], direct["proprioceptive_only"])]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(direct["method"], direct["official_ape_translation_rmse_m"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Official ATE RMSE, sequence-RMS (m)")
    ax.set_title("Official i2Nav Positioning: Accuracy Versus Runtime Sensing")
    ax.grid(axis="x", alpha=0.22)
    ax.text(
        0.98,
        0.04,
        "Blue: Twin V2\nOrange: local fixed baseline\nGray: heavier published systems",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )
    for y, value in enumerate(direct["official_ape_translation_rmse_m"]):
        ax.text(value + 0.08, y, f"{value:.2f}", va="center", fontsize=9)
    out = fig_dir / "summary_fig1_official_sensing_tradeoff.png"
    save_fig(out)
    return out


def build_v1_v2_progress(fig_dir: Path) -> Path:
    metrics = ["ATE\nRMSE (m)", "Heading\nMAE (deg)", "RPE10\nRMSE (m)"]
    v1 = [2.8339329, 3.3358077, 0.2714466]
    v2 = [2.3979394, 2.5689349, 0.2532473]
    pct = [(b - a) / a * 100.0 for a, b in zip(v1, v2)]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.4))
    for ax, metric, a, b, p in zip(axes, metrics, v1, v2, pct):
        ax.bar(["V1", "V2"], [a, b], color=[GRAY, BLUE], width=0.62)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.22)
        ax.text(0.5, max(a, b) * 0.92, f"{p:.1f}%", ha="center", color=DARK, fontsize=11, fontweight="bold")
        for i, val in enumerate([a, b]):
            ax.text(i, val, f"{val:.3g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Frozen LOSO: V2 Improves the Main Fidelity Metrics", fontsize=14, fontweight="bold")
    out = fig_dir / "summary_fig2_v1_v2_progress.png"
    save_fig(out)
    return out


def build_local_global_scatter(results_root: Path, fig_dir: Path) -> Path:
    df = pd.read_csv(results_root / "i2nav_frozen_v2_fidelity_analysis" / "all_sequence_mechanism" / "per_sequence_mechanism.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = [ORANGE if s in {"parking01", "parking02"} else BLUE for s in df["sequence"]]
    ax.scatter(df["RPE10_m"], df["Dp_p95_m"], s=80, c=colors, edgecolor="white", linewidth=1.0)
    for _, row in df.iterrows():
        if row["sequence"] in {"parking01", "parking02", "playground00", "street00"}:
            ax.annotate(row["sequence"], (row["RPE10_m"], row["Dp_p95_m"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Short-horizon local error: RPE10 (m)")
    ax.set_ylabel("Global divergence: Dp p95 (m)")
    ax.set_title("Local Fidelity and Global Synchronization Are Different")
    ax.grid(alpha=0.25)
    ax.text(
        0.98,
        0.04,
        "parking02: low local error,\nhigh global divergence",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )
    out = fig_dir / "summary_fig3_local_vs_global.png"
    save_fig(out)
    return out


def build_ugv01_instantiation(results_root: Path, fig_dir: Path) -> Path:
    df = pd.read_csv(results_root / "ugv01_physical_instantiation" / "ugv01_fidelity_by_stage.csv")
    current = df[df["stage_id"] == "S1"].iloc[0]
    fitted = df[df["stage_id"] == "S2"].iloc[0]
    metrics = [
        ("ATE RMSE (m)", "ate_rmse_m"),
        ("RPE1 RMSE (m)", "rpe1_rmse_m"),
        ("Heading MAE (deg)", "heading_mae_deg"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.4))
    for ax, (title, col) in zip(axes, metrics):
        vals = [float(current[col]), float(fitted[col])]
        ax.bar(["Current", "Asset\nfitted"], vals, color=[GRAY, BLUE], width=0.62)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        pct = (vals[1] - vals[0]) / vals[0] * 100.0
        ax.text(0.5, max(vals) * 0.92, f"{pct:.0f}%", ha="center", color=DARK, fontsize=11, fontweight="bold")
        for i, val in enumerate(vals):
            ax.text(i, val, f"{val:.3g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("UGV01: Asset-Specific Calibration Improves the Physical Twin", fontsize=14, fontweight="bold")
    out = fig_dir / "summary_fig4_ugv01_instantiation.png"
    save_fig(out)
    return out


def copy_or_build_figures(results_root: Path, out_dir: Path) -> list[Path]:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    figures = [
        build_v1_v2_progress(fig_dir),
        build_local_global_scatter(results_root, fig_dir),
        build_ugv01_instantiation(results_root, fig_dir),
        build_official_tradeoff(results_root, fig_dir),
    ]
    return figures


def pct_change(old: float, new: float) -> str:
    return f"{((new - old) / old * 100.0):.1f}%"


def build_markdown(out_dir: Path, figures: list[Path]) -> Path:
    fig_rel = [f"figures/{p.name}" for p in figures]
    text = f"""# Digital Twin Fidelity: Presentation Results Brief

Date: 2026-08-20  
Scope: frozen Twin V2 i2Nav fidelity analysis, official benchmark positioning, and UGV01 asset-specific instantiation.

## One-Page Takeaway

The project is now best framed as a **sensor-lightweight digital-twin fidelity** study. The central result is not simply that one model has lower RMSE. The stronger point is that a mobile-robot twin must be evaluated by how well the virtual robot stays synchronized with the physical robot over time, across operating conditions, and after binding a generic template to a specific robot.

**Main message:** Twin V2 improves the frozen i2Nav fidelity result, exposes a clear local-versus-global fidelity distinction, and transfers the framework to the UGV01 through AprilTag-referenced asset calibration.

## Key Results To Show First

| Question | Result | Presentation-level interpretation |
|---|---:|---|
| Does V2 improve over V1? | ATE: 2.834 m -> 2.398 m; Heading: 3.336 deg -> 2.569 deg; RPE10: 0.271 m -> 0.253 m | V2 improves the main fidelity metrics without changing the frozen result after inspection. |
| Is short-horizon accuracy enough? | parking02 has RPE10 = 0.097 m but Dp p95 = 22.345 m | No. A twin can move correctly locally while drifting globally. |
| Is fidelity condition-dependent? | Turning and wheel-IMU disagreement degrade local RPE; acceleration/curvature affect global synchronization | Fidelity should be reported as a profile, not one scalar score. |
| Does asset-specific binding matter? | UGV01 same-window ATE improves 0.131 m -> 0.099 m; heading MAE improves 13.3 deg -> 5.6 deg | Calibration makes the generic representation more like this specific physical rover. |
| Is the method sensor-lightweight? | Runtime inputs are wheel/odometry + IMU only; no camera/LiDAR/radar/GNSS | This supports the sensor-lightweight digital-twin framing. |

## Figure 1. Frozen LOSO Progress

![Frozen LOSO progress]({fig_rel[0]})

Twin V2 improves the main frozen LOSO metrics compared with V1. The most useful paper claim is not "new odometry SOTA"; it is that V2 improves physical-virtual fidelity while preserving a frozen, sequence-aware evaluation.

## Figure 2. Local Versus Global Fidelity

![Local versus global fidelity]({fig_rel[1]})

This is the strongest conceptual figure. It shows why one aggregate number is not enough. parking02 has low short-horizon relative error but severe global divergence, which supports the paper's argument that digital-twin fidelity must be multidimensional.

## Figure 3. UGV01 Asset-Specific Instantiation

![UGV01 instantiation]({fig_rel[2]})

The UGV01 result shows the framework on the actual tracked rover. The correct wording is: asset-specific calibration improves fidelity under the tested low-speed indoor AprilTag condition. Do not claim universal UGV01 performance across all surfaces and speeds yet.

## Figure 4. Sensing-Fidelity Positioning

![Sensing-fidelity tradeoff]({fig_rel[3]})

Twin V2 ranks in the middle of directly comparable i2Nav ATE/ARE rows, while using only wheel/odometry and IMU at runtime. Strong LiDAR/visual systems are more accurate, so this should be presented as sensing-burden positioning, not a leaderboard victory.

## Publication-Safe Claims

1. A sensor-lightweight mobile-robot twin can maintain useful finite-horizon fidelity using wheel/odometry and IMU only.
2. Local relative fidelity and long-horizon physical-virtual synchronization are different properties.
3. Persistent yaw mismatch is a measurable pathway for global divergence.
4. Benign fidelity is condition-dependent and should be described componentwise.
5. UGV01 calibration shows how a generic template becomes a twin of a specific physical rover.

## What To Avoid Saying

- Do not claim odometry SOTA.
- Do not claim parking02 is solved.
- Do not call the p95 benign envelope an anomaly or attack threshold.
- Do not claim UGV01 performance across surfaces/speeds that were not validated.
- Do not claim Pareto optimality in sensing burden.

## Recommended Presentation Narrative

"Since the last version, I moved the work into a cleaner digital-twin fidelity framing. The core contribution is a framework for measuring when a sensor-lightweight computational twin remains synchronized with the physical robot, when it diverges, and why. The i2Nav results give the broad frozen validation; the UGV01 AprilTag result shows asset-specific instantiation on the actual rover."
"""
    out = out_dir / "Digital_Twin_Fidelity_Presentation_Brief.md"
    out.write_text(text, encoding="utf-8")
    return out


def paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    return f"<w:p><w:pPr>{style_xml}</w:pPr><w:r><w:t>{escape(text)}</w:t></w:r></w:p>"


def image_paragraph(img: ImageRef) -> str:
    with Image.open(img.path) as im:
        w, h = im.size
    cx = int(img.width_in * 914400)
    cy = int(cx * h / w)
    name = escape(img.path.name)
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{img.rel_id[3:]}" name="{name}"/>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr><pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{img.rel_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>"""


def table(rows: list[list[str]]) -> str:
    cells = []
    for row in rows:
        row_xml = "".join(
            f'<w:tc><w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr><w:p><w:r><w:t>{escape(cell)}</w:t></w:r></w:p></w:tc>'
            for cell in row
        )
        cells.append(f"<w:tr>{row_xml}</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblLook w:val="04A0"/></w:tblPr>'
        + "".join(cells)
        + "</w:tbl>"
    )


def build_docx(out_dir: Path, figures: list[Path]) -> Path:
    docx_path = out_dir / "Digital_Twin_Fidelity_Presentation_Brief.docx"
    image_refs = [ImageRef(path=p, rel_id=f"rId{i + 1}", width_in=5.9) for i, p in enumerate(figures)]

    body = [
        paragraph("Digital Twin Fidelity: Presentation Results Brief", "Title"),
        paragraph("Frozen Twin V2 i2Nav fidelity analysis, official benchmark positioning, and UGV01 asset-specific instantiation.", "Subtitle"),
        paragraph("One-Page Takeaway", "Heading1"),
        paragraph("The project is now best framed as a sensor-lightweight digital-twin fidelity study. The core contribution is measuring when the virtual robot remains synchronized with the physical robot, when it diverges, and why."),
        table(
            [
                ["Question", "Result", "Interpretation"],
                ["Does V2 improve over V1?", "ATE 2.834 -> 2.398 m; Heading 3.336 -> 2.569 deg; RPE10 0.271 -> 0.253 m", "V2 improves the main frozen fidelity metrics."],
                ["Is short-horizon accuracy enough?", "parking02: RPE10 0.097 m but Dp p95 22.345 m", "No. Local motion can look good while global synchronization drifts."],
                ["Does UGV01 calibration matter?", "ATE 0.131 -> 0.099 m; Heading 13.3 -> 5.6 deg", "Asset-specific binding improves this rover twin under tested conditions."],
                ["Is the method sensor-lightweight?", "Wheel/odometry + IMU only at runtime", "Supports sensing-fidelity framing, not odometry SOTA."],
            ]
        ),
        paragraph("1. Frozen LOSO Progress", "Heading1"),
        image_paragraph(image_refs[0]),
        paragraph("Twin V2 improves the main frozen LOSO metrics compared with V1. The clean claim is fidelity maintenance under a frozen, sequence-aware evaluation."),
        paragraph("2. Local Versus Global Fidelity", "Heading1"),
        image_paragraph(image_refs[1]),
        paragraph("This is the strongest conceptual result: short-horizon relative motion and long-horizon physical-virtual synchronization are different. parking02 is the clearest example."),
        paragraph("3. UGV01 Asset-Specific Instantiation", "Heading1"),
        image_paragraph(image_refs[2]),
        paragraph("The UGV01 result shows how asset-specific calibration turns a generic tracked-rover representation into a better twin of the actual rover under low-speed indoor AprilTag validation."),
        paragraph("4. Sensing-Fidelity Positioning", "Heading1"),
        image_paragraph(image_refs[3]),
        paragraph("Twin V2 is not the most accurate i2Nav method, but it is sensor-lightweight: no camera, LiDAR, radar, or GNSS at runtime. Use this as positioning, not a leaderboard claim."),
        paragraph("Publication-Safe Claims", "Heading1"),
        paragraph("A sensor-lightweight mobile-robot twin can maintain useful finite-horizon fidelity using wheel/odometry and IMU only."),
        paragraph("Local relative fidelity and long-horizon physical-virtual synchronization are different properties."),
        paragraph("Persistent yaw mismatch is a measurable pathway for global divergence."),
        paragraph("Benign fidelity is condition-dependent and should be described componentwise."),
        paragraph("UGV01 calibration shows how a generic template becomes a twin of a specific physical rover."),
        paragraph("What To Avoid Saying", "Heading1"),
        paragraph("Do not claim odometry SOTA, Pareto optimality, universal UGV01 performance, or that p95 benign envelopes are anomaly thresholds."),
    ]

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <w:body>
  {''.join(body)}
  <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr>
 </w:body>
</w:document>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Aptos"/><w:sz w:val="22"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:color w:val="173F5F"/><w:sz w:val="34"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr><w:color w:val="666666"/><w:sz w:val="22"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:color w:val="173F5F"/><w:sz w:val="26"/></w:rPr></w:style>
 <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""

    rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rIdDoc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>',
        "</Relationships>",
    ]
    doc_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    doc_rels.append('<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    for img in image_refs:
        doc_rels.append(f'<Relationship Id="{img.rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{img.path.name}"/>')
    doc_rels.append("</Relationships>")
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
 <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", "\n".join(rels))
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/styles.xml", styles_xml)
        z.writestr("word/_rels/document.xml.rels", "\n".join(doc_rels))
        for img in image_refs:
            z.write(img.path, f"word/media/{img.path.name}")
    return docx_path


def run(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    ensure_dir(out_dir)
    figures = copy_or_build_figures(args.results_root, out_dir)
    md = build_markdown(out_dir, figures)
    docx = build_docx(out_dir, figures)
    print(md)
    print(docx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/presentation_summary_brief"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
