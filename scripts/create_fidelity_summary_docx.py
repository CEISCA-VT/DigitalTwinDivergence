from __future__ import annotations

import html
import os
import struct
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "digital_twin_fidelity_results_summary.docx"


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return (1200, 800)
    return struct.unpack(">II", header[16:24])


def para(text: str = "", style: str | None = None, bold: bool = False) -> str:
    pstyle = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    b = "<w:b/>" if bold else ""
    return (
        "<w:p>"
        f"{pstyle}"
        "<w:r>"
        f"<w:rPr>{b}</w:rPr>"
        f"<w:t xml:space=\"preserve\">{esc(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def bullet(text: str) -> str:
    return (
        '<w:p><w:pPr><w:pStyle w:val="ListBullet"/></w:pPr>'
        f'<w:r><w:t xml:space="preserve">- {esc(text)}</w:t></w:r></w:p>'
    )


def table(rows: list[list[str]], highlight_first_cell: str | None = None) -> str:
    out = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/></w:tblPr>'
    ]
    for r, row in enumerate(rows):
        out.append("<w:tr>")
        highlight_row = bool(highlight_first_cell and row and row[0] == highlight_first_cell)
        for cell in row:
            if r == 0:
                shade = '<w:shd w:fill="1F4E79"/>'
            elif highlight_row:
                shade = '<w:shd w:fill="EAF2F8"/>'
            else:
                shade = '<w:shd w:fill="F7F7F7"/>' if r % 2 == 0 else ""
            color = '<w:color w:val="FFFFFF"/>' if r == 0 else ""
            out.append(
                "<w:tc><w:tcPr>"
                '<w:tcW w:w="2400" w:type="dxa"/>'
                f"{shade}"
                "</w:tcPr>"
                f"{para(cell, bold=(r == 0)).replace('<w:rPr>', f'<w:rPr>{color}', 1) if r == 0 else para(cell)}"
                "</w:tc>"
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def hyperlink_para(label: str, url: str, rid: str) -> str:
    return (
        "<w:p><w:r><w:t xml:space=\"preserve\">"
        f"{esc(label)} - "
        "</w:t></w:r>"
        f"<w:hyperlink r:id=\"{rid}\">"
        '<w:r><w:rPr><w:color w:val="0563C1"/><w:u w:val="single"/></w:rPr>'
        f"<w:t>{esc(url)}</w:t></w:r></w:hyperlink></w:p>"
    )


def baseline_comparison_table() -> tuple[str, list[tuple[str, str]]]:
    refs = [
        ("hId1", "https://doi.org/10.1109/JPROC.2016.2526658"),
        ("hId2", "https://doi.org/10.1109/TAC.1987.1104658"),
        ("hId3", "https://doi.org/10.1109/CCA.2016.7587875"),
        ("hId4", "https://doi.org/10.1016/j.measurement.2022.110962"),
        ("hId5", "https://doi.org/10.1016/j.cja.2024.103358"),
    ]
    rows = [
        [
            "Method",
            "Year",
            "Reference / DOI",
            "Simple idea",
            "Current result",
            "Takeaway",
        ],
        [
            "GPS Jump Detector",
            "2016",
            hyperlink_para("Psiaki & Humphreys, GNSS Spoofing and Detection", refs[0][1], refs[0][0]),
            "Detect implausibly large changes between consecutive GPS positions.",
            "Best abrupt-step baseline; epsilon_90 about 2.86-2.90 m. Weak on gradual drift.",
            "Strong simple guard for sudden spoofing, but not a slow-drift or uncertainty-adaptation solution.",
        ],
        [
            "Fixed NIS / Chi-Square EKF",
            "1987",
            hyperlink_para("Brumback & Srinath, A Chi-Square Test for Fault-Detection in Kalman Filters", refs[1][1], refs[1][0]),
            "Compare normalized EKF innovation with a fixed statistical threshold.",
            "Step epsilon_90 > 10 m; about 3.3% detection for the 0.05 m/s drift condition.",
            "Interpretable baseline, but weak against slowly accumulating attacks.",
        ],
        [
            "CUSUM Innovation Monitor",
            "2016",
            hyperlink_para("Murguia & Ruths, CUSUM and Chi-Squared Attack Detection of Compromised Sensors", refs[2][1], refs[2][0]),
            "Accumulate small residual evidence over time.",
            "Best simple gradual-drift detector here; about 10% detection at 0.05 m/s cross-track drift.",
            "Important sequential baseline and natural complement to a GPS-jump guard.",
        ],
        [
            "Residual-Adaptive EKF",
            "2022 / 2025",
            hyperlink_para("Liang et al.; Jin et al.", refs[3][1], refs[3][0])
            + hyperlink_para("Second adaptive-filtering context reference", refs[4][1], refs[4][0]),
            "Adapt filter uncertainty as conditions or residual behavior change; the project tests an intentionally vulnerable residual-to-Q variant.",
            "0% detection and 26.7% tolerance-exceeding paired divergence for 0.05 m/s cross-track drift.",
            "Key vulnerability baseline: GPS residuals should not be able to inflate the uncertainty used to judge GPS.",
        ],
        [
            "Ours: Evidence-Gated Adaptive Digital Twin",
            "This work",
            "This work",
            "Adapt Q only from GPS-independent motion, IMU, and protected timing/context evidence.",
            "10.0% tolerance-exceeding paired divergence versus 26.7% for naive residual adaptation; buffered benign alarms 0/4.",
            "Main contribution: preserve adaptive uncertainty while removing the attacked-GPS-residual to covariance-inflation pathway.",
        ],
    ]
    out = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/></w:tblPr>'
    ]
    widths = [1700, 850, 3300, 2500, 2600, 3100]
    for r, row in enumerate(rows):
        out.append("<w:tr>")
        highlight_row = r == len(rows) - 1
        for i, cell in enumerate(row):
            if r == 0:
                shade = '<w:shd w:fill="1F4E79"/>'
            elif highlight_row:
                shade = '<w:shd w:fill="EAF2F8"/>'
            else:
                shade = '<w:shd w:fill="F7F7F7"/>' if r % 2 == 0 else ""
            if cell.startswith("<w:p>"):
                content = cell
            else:
                color = '<w:color w:val="FFFFFF"/>' if r == 0 else ""
                content = para(cell, bold=(r == 0))
                if r == 0:
                    content = content.replace("<w:rPr>", f"<w:rPr>{color}", 1)
            out.append(
                "<w:tc><w:tcPr>"
                f'<w:tcW w:w="{widths[i]}" w:type="dxa"/>'
                f"{shade}"
                "</w:tcPr>"
                f"{content}"
                "</w:tc>"
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out), refs


def image_xml(rid: str, path: Path, caption: str, width_in: float = 5.7) -> str:
    px_w, px_h = png_size(path)
    emu_per_in = 914400
    cx = int(width_in * emu_per_in)
    cy = int(cx * px_h / max(px_w, 1))
    name = esc(path.name)
    return f"""
    <w:p>
      <w:r>
        <w:drawing>
          <wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
            <wp:extent cx="{cx}" cy="{cy}"/>
            <wp:docPr id="{rid[3:]}" name="{name}"/>
            <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                  <pic:nvPicPr><pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
                  <pic:blipFill><a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
                  <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
                </pic:pic>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing>
      </w:r>
    </w:p>
    {para(caption)}
    """


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def build_document() -> tuple[str, list[tuple[str, Path]], list[tuple[str, str]]]:
    images = [
        ("rId10", ROOT / "DigitalTwin" / "datasets" / "analysis" / "validation_carpet_142023_candidate" / "trajectory_fidelity.png"),
        ("rId11", ROOT / "results" / "i2nav_playground00" / "model_bakeoff" / "model_bakeoff_best_trajectory.png"),
        ("rId12", ROOT / "results" / "i2nav_playground00" / "model_bakeoff" / "model_bakeoff_ranking.png"),
        ("rId13", ROOT / "results" / "covariance_poisoning" / "covariance_poisoning_paired_effects.png"),
    ]
    existing_images = [(rid, p) for rid, p in images if p.exists()]

    body: list[str] = []
    body.append(para("Digital Twin Fidelity and Results Summary", "Title"))
    body.append(para("UGV01 rover project status document for advisor discussion", "Subtitle"))
    body.append(para("Prepared from the current repository results and generated artifacts."))

    body.append(para("Executive Summary", "Heading1"))
    for text in [
        "The project builds a security-aware digital twin for a tracked UGV01 rover. The twin combines encoder/IMU motion evidence, GPS measurements when available, and edge timing information to estimate rover state and decide when GPS should be trusted, down-weighted, or treated as suspicious.",
        "Current results are strongest as a systems-and-security validation: the telemetry pipeline, replay pipeline, alarm policies, covariance-poisoning analysis, and public-dataset digital-twin checks are implemented and reproducible.",
        "Physical localization fidelity on the UGV01 is promising but not yet final-publication grade because the cleanest AprilTag pilot did not include synchronized GPS and hardware sync. It should be treated as a pilot ground-truth result, not the final experimental standard.",
    ]:
        body.append(bullet(text))

    body.append(para("Key Current Metrics", "Heading1"))
    body.append(table([
        ["Category", "Current result", "Plain-language meaning"],
        ["UGV01 EKF-to-GPS agreement", "Pooled RMSE 1.227 m; median run RMSE 1.029 m", "The GPS-fused digital twin broadly follows the GPS stream, but this is sensor agreement, not true physical accuracy."],
        ["UGV01 AprilTag pilot", "Best pilot ATE RMSE 0.252 m; median 0.116 m; 1 s RPE 0.046 m", "Camera ground truth suggests the motion model can be close over selected indoor windows, but final sync/GPS validation is still needed."],
        ["Public i2Nav validation", "Playground EKF RMSE 0.593 m; street 1.201 m; parking about 2.39 m", "The same digital-twin style works best when GNSS/odometry conditions are clean and degrades under harder GNSS conditions."],
        ["Attack campaign scale", "20 runs, 13 variants, 24 attack profiles, 1,440 unique attack-run-start combinations, 18,720 detector-run evaluations", "The replay campaign is broad and reproducible, but still counterfactual/offline rather than live attacks."],
        ["Math diagnostics", "Score identity error 1.421e-14; residual-cover bound violation 0", "The revised math checks out numerically on the generated campaign artifacts."],
    ]))

    body.append(para("Digital Twin Fidelity", "Heading1"))
    body.append(para("Fidelity means how closely the digital twin’s estimated path matches an independent reference path. GPS alone is not enough indoors because a 1 m GPS error can be larger than the route itself. This is why AprilTag/video ground truth matters: it gives an external measurement of where the rover actually moved, independent of the GPS sensor that the security system is trying to judge."))
    body.append(para("The best current UGV01 AprilTag pilot gives 0.252 m absolute trajectory error RMSE and 0.046 m one-second relative pose error. That is a useful pilot result: the local motion shape is reasonable, while heading still needs work because heading MAE is about 21.4 degrees. The current UGV01 GPS-fused runs show about 1.227 m pooled EKF-to-GPS RMSE, but that number should be described as EKF-to-GPS agreement rather than physical localization accuracy."))
    if ("rId10", ROOT / "DigitalTwin" / "datasets" / "analysis" / "validation_carpet_142023_candidate" / "trajectory_fidelity.png") in existing_images:
        body.append(image_xml("rId10", ROOT / "DigitalTwin" / "datasets" / "analysis" / "validation_carpet_142023_candidate" / "trajectory_fidelity.png", "Figure 1. UGV01 AprilTag pilot trajectory fidelity. This is the clearest current rover-ground-truth visual, but it remains a pilot because GPS and hard synchronization were not included.", 5.4))

    body.append(para("External Digital Twin Check", "Heading1"))
    body.append(para("To avoid relying only on the small indoor UGV01 dataset, the project also tests the digital-twin/uncertainty idea on the public i2Nav-Robot dataset, which contains GNSS, IMU, odometry, and ground-truth trajectory. This is useful because it shows whether the method behaves sensibly on a larger independent robot dataset."))
    if ("rId11", ROOT / "results" / "i2nav_playground00" / "model_bakeoff" / "model_bakeoff_best_trajectory.png") in existing_images:
        body.append(image_xml("rId11", ROOT / "results" / "i2nav_playground00" / "model_bakeoff" / "model_bakeoff_best_trajectory.png", "Figure 2. Public i2Nav playground run: digital-twin trajectory compared with ground truth and GNSS. This is the best current external visual showing the twin following a complete robot path.", 5.4))
    body.append(para("Performance varies substantially across sequences: playground currently gives the lowest RMSE at about 0.593 m, street is intermediate at about 1.201 m, and parking is the most challenging of the evaluated sequences at about 2.39 m. The current evidence supports this sequence-level comparison, but not a definitive causal explanation for why each sequence behaves differently. The model-bakeoff also shows that learned process-uncertainty models can be compared systematically instead of chosen by intuition."))
    if ("rId12", ROOT / "results" / "i2nav_playground00" / "model_bakeoff" / "model_bakeoff_ranking.png") in existing_images:
        body.append(image_xml("rId12", ROOT / "results" / "i2nav_playground00" / "model_bakeoff" / "model_bakeoff_ranking.png", "Figure 3. Model bakeoff on i2Nav playground. Tree/boosting models slightly outperform the basic MLP in target error, while downstream EKF differences are smaller.", 5.2))

    body.append(para("Baseline Comparison", "Heading1"))
    body.append(para("The project compares against common GNSS/security and Kalman-filter monitoring baselines. The numbers below use the current replay campaign with benign-locked thresholds. The main conclusion is not that one detector wins everywhere: abrupt attacks, gradual drift, and unsafe adaptive uncertainty each require different defenses."))
    body.append(para("Important attribution note: Psiaki & Humphreys is used as a general GNSS spoofing/detection reference, not as the source of this exact consecutive-position jump implementation. Liang et al. and Jin et al. are used as adaptive-filtering and anti-spoofing context; the intentionally vulnerable residual-to-Q poisoning rule is this project's experimental vulnerability baseline, not an algorithm attributed to those papers."))
    baseline_xml, hyperlink_rels = baseline_comparison_table()
    body.append(baseline_xml)
    body.append(para("Slow-drift tolerance-exceeding paired divergence", "Heading1"))
    body.append(table([
        ["Replay-based comparison", "Value"],
        ["Naive residual-adaptive EKF", "26.7%"],
        ["Evidence-gated adaptive digital twin", "10.0%"],
    ], highlight_first_cell="Evidence-gated adaptive digital twin"))
    body.append(para("This is a replay-based paired-divergence result. It should not be described as final AprilTag-referenced physical localization error. The strongest final policy is a composite design: GPS jump guard for abrupt attacks, CUSUM/EWMA memory for slow drift, GPS-bias monitoring for persistent bias, and evidence-gated adaptive EKF for secure uncertainty handling."))

    body.append(para("Security Results", "Heading1"))
    body.append(para("The security experiment asks a different question from fidelity: can the digital twin detect or limit GPS manipulation? The current replay campaign tests GPS step, drift, replay/freeze-style effects, and multiple detector/model variants under benign-locked thresholds."))
    body.append(table([
        ["Finding", "What it says"],
        ["GPS jump detector wins abrupt GPS jumps", "Large sudden GPS steps are best caught by a simple consecutive-GPS-displacement rule; the project should be honest about this."],
        ["Slow drift is harder", "Most detectors have low detection probability on slow drift, which is realistic and motivates memory-based detectors such as CUSUM/EWMA."],
        ["Naive adaptive EKF is vulnerable", "When GPS residuals directly inflate uncertainty, the filter can become more tolerant of the attacker."],
        ["Evidence-gated adaptation is the main contribution", "GPS should not be allowed to increase process uncertainty unless independent IMU/timing/motion evidence supports that change."],
    ]))
    if ("rId13", ROOT / "results" / "covariance_poisoning" / "covariance_poisoning_paired_effects.png") in existing_images:
        body.append(image_xml("rId13", ROOT / "results" / "covariance_poisoning" / "covariance_poisoning_paired_effects.png", "Figure 4. Covariance-poisoning mechanism. The key security result is that residual-coupled adaptation can inflate uncertainty, suppress NIS scores, and increase undetected paired divergence.", 5.3))

    body.append(para("AprilTag Run Situation", "Heading1"))
    body.append(para("The AprilTag setup is the right direction for physical validation. The current pilot already shows that the phone/video pipeline can recover the rover tag and compare it with the digital-twin path. The missing final piece is a clean synchronized run where telemetry, GPS, and video are all recorded together with an obvious sync event at the start."))
    for text in [
        "Keep four fixed reference tags visible around the test area and one rover tag centered on top of the UGV01.",
        "Record 7-10 minutes on one controlled surface first, preferably carpet, with straight motion, reverse, 90-degree turns, gentle curves, S-curves, and short figure-eight-like motion.",
        "Add a sync event visible in video and telemetry, such as a brief lifted rotation or deliberate stop/start marker.",
        "Use the final output as t, x_GT, y_GT, theta_GT aligned to telemetry, then report position RMSE, heading MAE, route-shape error, NIS/NEES where valid, and loop-closure error.",
    ]:
        body.append(bullet(text))

    body.append(para("What Is Left", "Heading1"))
    for text in [
        "Finish independent UGV01 ground-truth validation using AprilTags with synchronized telemetry and GPS. This is the biggest credibility gap before a strong submission.",
        "Freeze a composite security policy that combines GPS jump guard, CUSUM/EWMA slow-drift memory, GPS-bias monitoring, and evidence-gated adaptive EKF.",
        "Use the current i2Nav public-dataset results as external validation and optionally add AIFARMS for rough-terrain support.",
        "Polish the manuscript around one clear thesis: secure adaptive digital twins should prevent attacker-controlled GPS from poisoning their own uncertainty estimates.",
    ]:
        body.append(bullet(text))
    body.append(para("Bottom line: the software and replay-analysis side is strong and reproducible; the remaining publication-critical work is the clean physical ground-truth validation and final policy framing."))

    sect = (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return "".join(body) + sect, existing_images, hyperlink_rels


def write_docx() -> None:
    body, images, hyperlink_rels = build_document()
    rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    ]
    rels.append(
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    for rid, path in images:
        rels.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{esc(path.name)}"/>'
        )
    for rid, url in hyperlink_rels:
        rels.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="{esc(url)}" TargetMode="External"/>'
        )
    rels.append("</Relationships>")

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
        "</Types>",
    ]
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        f"<w:body>{body}</w:body></w:document>"
    )
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>
      <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style>
      <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:rPr><w:i/><w:sz w:val="22"/><w:color w:val="555555"/></w:rPr></w:style>
      <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="26"/><w:color w:val="1F4E79"/></w:rPr></w:style>
      <w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="360" w:hanging="180"/></w:pPr></w:style>
      <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/><w:left w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/><w:right w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/></w:tblBorders></w:tblPr></w:style>
    </w:styles>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(content_types))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/_rels/document.xml.rels", "".join(rels))
        used_names: set[str] = set()
        for _, path in images:
            arc_name = path.name
            if arc_name in used_names:
                stem, suffix = path.stem, path.suffix
                arc_name = f"{stem}_{len(used_names)}{suffix}"
            used_names.add(arc_name)
            z.write(path, f"word/media/{arc_name}")


if __name__ == "__main__":
    write_docx()
    print(OUT)
