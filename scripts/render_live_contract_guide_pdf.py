"""Render the UGV01 live contract experiment guide as a PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path("docs/ugv01_live_contract_experiment_guide.pdf")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, style)) for item in items],
        bulletType="bullet",
        leftIndent=18,
    )


def numbered_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, style)) for item in items],
        bulletType="1",
        leftIndent=18,
    )


def build_pdf(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "GuideTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f2d4a"),
        spaceAfter=10,
    )
    h1 = ParagraphStyle(
        "GuideH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f2d4a"),
        spaceBefore=12,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "GuideH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0f2d4a"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "GuideBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        spaceAfter=5,
    )
    code = ParagraphStyle(
        "GuideCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.3,
        leading=10.5,
        backColor=colors.HexColor("#f5f5f5"),
        borderColor=colors.HexColor("#dddddd"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6,
    )
    callout = ParagraphStyle(
        "GuideCallout",
        parent=body,
        backColor=colors.HexColor("#eef5fb"),
        borderColor=colors.HexColor("#0f2d4a"),
        borderWidth=0.75,
        borderPadding=8,
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=8,
    )

    story = [
        p("UGV01 Live Contract Experiment Guide", title),
        p("<b>Updated:</b> August 29, 2026", body),
        p(
            "This guide explains how to run the <b>live contract experiment</b> "
            "with the current repository and UGV01 workflow.",
            body,
        ),
        p(
            "<b>Short answer:</b><br/>For the live contract dashboard, keep "
            "<b>GPS connected</b>.<br/>You do <b>not</b> need AprilTags for the "
            "live demo itself.<br/>AprilTags are only needed later for "
            "independent offline validation metrics.",
            callout,
        ),
        p("1. What This Experiment Is", h1),
        p(
            "The live contract experiment is the online digital-twin dashboard "
            "demonstration in which the rover is controlled from the interface, "
            "the twin is updated from onboard telemetry, service contracts are "
            "evaluated online, and the resource policy chooses 2 Hz, 5 Hz, or "
            "10 Hz behavior.",
            body,
        ),
        Preformatted(
            "python -m DigitalTwin.dashboard.server --mode live --rover-url "
            "http://192.168.4.1/js --host 127.0.0.1 --port 8765 "
            "--policy contract-aware --open",
            code,
        ),
        p("2. What Must Be Connected", h1),
        p("Required for the live contract experiment", h2),
        bullet_list(
            [
                "UGV01 rover powered on",
                "Wi-Fi connection to the rover or its station-mode IP",
                "Firmware streaming telemetry at <font name='Courier'>/js</font>",
                "Working IMU",
                "Working encoders",
            ],
            body,
        ),
        p("Strongly recommended", h2),
        bullet_list(["<b>GPS connected and functioning</b>"], body),
        p(
            "The current live dashboard uses GPS as the live operational "
            "reference for contract-style monitoring. Without GPS, the dashboard "
            "can still run, but the live fidelity and contract evidence becomes "
            "much weaker.",
            body,
        ),
        p("Not required for the live contract experiment", h2),
        bullet_list(
            ["AprilTags", "ChArUco board", "Overhead phone video"],
            body,
        ),
    ]

    table = Table(
        [
            ["Experiment", "GPS", "AprilTags"],
            ["Live contract dashboard", "Yes, recommended", "No"],
            ["Offline physical validation", "Optional", "Yes"],
        ],
        colWidths=[2.7 * inch, 1.7 * inch, 1.3 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8c8c8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 0.08 * inch),
            p("3. Rover Bring-Up", h1),
            p(
                "<b>Active firmware path:</b> "
                "<font name='Courier'>ugv01_gps_dev/General_Driver</font>",
                body,
            ),
            p("<b>BN220 wiring:</b>", body),
            Preformatted(
                "BN-220 white -> UGV01 RX\n"
                "BN-220 red   -> UGV01 5V\n"
                "BN-220 black -> UGV01 GND",
                code,
            ),
            numbered_list(
                [
                    "Power on the rover.",
                    "Connect your laptop to the rover Wi-Fi or station-mode Wi-Fi.",
                    "Open <font name='Courier'>http://192.168.4.1</font>.",
                    "Confirm the stock control page responds and telemetry updates.",
                    "Confirm GPS values update if GPS is attached.",
                ],
                body,
            ),
            p(
                "If the rover is on station-mode Wi-Fi, replace "
                "<font name='Courier'>192.168.4.1</font> with the IP shown on the rover display.",
                body,
            ),
            p("4. Recommended Experiment Order", h1),
            p("Step 1: Rehearse the dashboard without the rover", h2),
            Preformatted(
                "python -m DigitalTwin.dashboard.server --mode csv --csv "
                "raw_logs\\telemetry\\ugv_t147_bench_20260814_143729.csv "
                "--host 127.0.0.1 --port 8765 --policy contract-aware --open",
                code,
            ),
            p("Step 2: Run the actual live rover dashboard", h2),
            Preformatted(
                "python -m DigitalTwin.dashboard.server --mode live --rover-url "
                "http://192.168.4.1/js --host 127.0.0.1 --port 8765 "
                "--policy contract-aware --open",
                code,
            ),
            p("Optional comparison policies", h2),
            Preformatted(
                "python -m DigitalTwin.dashboard.server --mode live --rover-url "
                "http://192.168.4.1/js --host 127.0.0.1 --port 8765 "
                "--policy static-low --open\n"
                "python -m DigitalTwin.dashboard.server --mode live --rover-url "
                "http://192.168.4.1/js --host 127.0.0.1 --port 8765 "
                "--policy static-high --open\n"
                "python -m DigitalTwin.dashboard.server --mode live --rover-url "
                "http://192.168.4.1/js --host 127.0.0.1 --port 8765 "
                "--policy aoi-only --open",
                code,
            ),
            p("Step 3: Drive one controlled live session", h2),
            bullet_list(
                [
                    "5 s stationary start",
                    "straight forward",
                    "straight reverse",
                    "one or two gentle turns",
                    "a short turning-intensive segment",
                    "5 s stationary end",
                ],
                body,
            ),
            p("Step 4: Save the live log", h2),
            p(
                "The dashboard automatically writes JSONL logs to "
                "<font name='Courier'>raw_logs/live_validation/</font>.",
                body,
            ),
            Preformatted(
                "Get-ChildItem .\\raw_logs\\live_validation | Sort-Object "
                "LastWriteTime -Descending | Select-Object -First 10",
                code,
            ),
            p("5. What The Live Experiment Produces", h1),
            bullet_list(
                [
                    "dashboard visualization",
                    "screenshots or screen recording",
                    "live contract decisions",
                    "live policy decisions",
                    "JSONL session logs",
                ],
                body,
            ),
            p(
                "It does not by itself produce final AprilTag-based ATE, RPE, or heading metrics.",
                body,
            ),
            p("6. When AprilTags Are Needed", h1),
            p(
                "Use AprilTags only if you want independent physical-validation "
                "metrics: ATE, RPE, heading error, and physical-versus-virtual "
                "trajectory comparison. That is a separate paired experiment: "
                "collect telemetry CSV, record overhead video, then run the "
                "offline AprilTag analysis pipeline.",
                body,
            ),
            p("7. Should GPS Stay Connected?", h1),
            p("<b>Yes.</b> For the live contract experiment, GPS should stay connected.", body),
            bullet_list(
                [
                    "the dashboard uses live GPS-versus-twin agreement",
                    "it makes the contract panels meaningful in real time",
                    "the later follow-up work still needs GPS",
                ],
                body,
            ),
            p(
                "If GPS is disconnected, the twin can still propagate from IMU "
                "and encoders, but the live contract evidence becomes much less persuasive.",
                body,
            ),
            p("8. Important Limitation", h1),
            p(
                "Do not run multiple simultaneous aggressive telemetry pollers "
                "against the rover unless you intentionally want extra network "
                "load. Safest workflow: run the live dashboard experiment by "
                "itself, then run any offline telemetry/AprilTag validation as "
                "a separate trial.",
                body,
            ),
            p("9. Minimal Checklist", h1),
            numbered_list(
                [
                    "Keep <b>GPS connected</b>.",
                    "Do not worry about AprilTags for the live demo.",
                    "Power the rover and confirm <font name='Courier'>http://192.168.4.1</font>.",
                    "Run the live dashboard command.",
                    "Drive one controlled session.",
                    "Save screenshots and the JSONL log.",
                    "If you later want physical fidelity metrics, run a separate AprilTag session.",
                ],
                body,
            ),
            p("10. Related Repo Files", h1),
            bullet_list(
                [
                    "<font name='Courier'>docs/ugv01_live_validation_runbook.md</font>",
                    "<font name='Courier'>docs/ugv01_live_validation_dashboard.md</font>",
                    "<font name='Courier'>docs/ugv01_esp32_bringup.md</font>",
                    "<font name='Courier'>DigitalTwin/configs/ugv01_live_service_contracts.json</font>",
                ],
                body,
            ),
            Spacer(1, 0.08 * inch),
            HRFlowable(width="100%", color=colors.HexColor("#cccccc")),
            Spacer(1, 0.06 * inch),
            p(
                "Practical rule: for the live contract experiment, use GPS and "
                "the dashboard only. Use AprilTags later when you want independent offline metrics.",
                body,
            ),
        ]
    )

    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="UGV01 Live Contract Experiment Guide",
        author="Codex",
    )
    doc.build(story)


if __name__ == "__main__":
    build_pdf(OUTPUT)
    print(OUTPUT)
