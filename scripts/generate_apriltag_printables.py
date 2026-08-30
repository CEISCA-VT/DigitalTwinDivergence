"""Generate print-ready tag36h11 rover and world-reference tags."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


TAG_IDS = range(7)
TAG_SIZE_MM = 120.0
PIXELS_PER_MM = 20
ROVER_OPTION_TAG_SIZES_MM = (50.0, 60.0, 70.0, 80.0)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "printables"
ROVER_SIZE_SHEET_PDF_PATH = OUTPUT_DIR / "apriltag_rover_id0_50_60_70_80mm_letter.pdf"


def generate_tag_png(tag_id: int, tag_size_mm: float = TAG_SIZE_MM) -> Path:
    side_px = int(tag_size_mm * PIXELS_PER_MM)
    path = OUTPUT_DIR / f"apriltag_tag36h11_id{tag_id}_{int(tag_size_mm)}mm.png"

    if cv2 is None:
        source_path = OUTPUT_DIR / f"apriltag_tag36h11_id{tag_id}_120mm.png"
        if not source_path.exists():
            raise RuntimeError(
                "OpenCV is required to generate new AprilTags when no existing "
                f"source PNG is available: {source_path}"
            )
        image = Image.open(source_path).resize(
            (side_px, side_px),
            resample=Image.Resampling.NEAREST,
        )
    else:
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        image_array = cv2.aruco.generateImageMarker(
            dictionary,
            tag_id,
            side_px,
            borderBits=1,
        )
        image = Image.fromarray(image_array)

    image.save(
        path,
        dpi=(PIXELS_PER_MM * 25.4, PIXELS_PER_MM * 25.4),
    )
    return path


def add_pdf_page(
    pdf: canvas.Canvas,
    tag_id: int,
    png_path: Path,
    tag_size_mm: float = TAG_SIZE_MM,
) -> None:
    page_width, page_height = letter
    tag_size = tag_size_mm * mm
    tag_x = (page_width - tag_size) / 2.0
    tag_y = 75.0 * mm

    if tag_id == 0:
        role = "ROVER MOVING TAG"
    elif tag_id <= 4:
        role = "FIXED WORLD CALIBRATION REFERENCE"
    else:
        role = "FIXED HELD-OUT VALIDATION REFERENCE"
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawCentredString(
        page_width / 2.0,
        258.0 * mm,
        f"AprilTag tag36h11 - ID {tag_id}",
    )
    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(page_width / 2.0, 250.0 * mm, role)
    if tag_id == 0:
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawCentredString(
            page_width / 2.0,
            242.0 * mm,
            "ALIGN THE TOP EDGE OF THIS PAGE WITH THE ROVER FRONT",
        )

    pdf.drawImage(
        str(png_path),
        tag_x,
        tag_y,
        width=tag_size,
        height=tag_size,
        preserveAspectRatio=True,
        anchor="c",
    )

    scale_width = 100.0 * mm
    scale_x = (page_width - scale_width) / 2.0
    scale_y = 42.0 * mm
    pdf.setLineWidth(1.0)
    pdf.line(scale_x, scale_y, scale_x + scale_width, scale_y)
    pdf.line(scale_x, scale_y - 2.0 * mm, scale_x, scale_y + 2.0 * mm)
    pdf.line(
        scale_x + scale_width,
        scale_y - 2.0 * mm,
        scale_x + scale_width,
        scale_y + 2.0 * mm,
    )
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(
        page_width / 2.0,
        33.0 * mm,
        "100 mm scale check - print at Actual size / 100% and disable Fit to page",
    )
    pdf.drawCentredString(
        page_width / 2.0,
        27.0 * mm,
        f"Configured tag size is the {int(tag_size_mm)} mm outer black-square width",
    )
    pdf.showPage()


def write_pdf(
    path: Path,
    ids: list[int],
    png_paths: dict[int, Path],
    tag_size_mm: float = TAG_SIZE_MM,
) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setTitle(f"AprilTag tag36h11 at {int(tag_size_mm)} mm")
    pdf.setAuthor("DigitalTwinDivergence")
    pdf.setSubject("UGV01 rover and fixed world-reference tracking tags")
    for tag_id in ids:
        add_pdf_page(pdf, tag_id, png_paths[tag_id], tag_size_mm=tag_size_mm)
    pdf.save()


def write_rover_size_sheet(path: Path, png_paths: dict[float, Path]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setTitle("AprilTag tag36h11 rover ID0 size options")
    pdf.setAuthor("DigitalTwinDivergence")
    pdf.setSubject("UGV01 rover tag size options for fit testing")
    page_width, page_height = letter

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawCentredString(
        page_width / 2.0,
        258.0 * mm,
        "AprilTag tag36h11 - Rover ID 0 Size Options",
    )
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(
        page_width / 2.0,
        250.0 * mm,
        "Print at Actual size / 100%; mount exactly one ID 0 tag on the rover.",
    )

    placements = [
        (50.0, 25.0, 166.0),
        (60.0, 125.0, 156.0),
        (70.0, 25.0, 62.0),
        (80.0, 125.0, 52.0),
    ]
    for tag_size_mm, x_mm, y_mm in placements:
        tag_size = tag_size_mm * mm
        pdf.drawImage(
            str(png_paths[tag_size_mm]),
            x_mm * mm,
            y_mm * mm,
            width=tag_size,
            height=tag_size,
            preserveAspectRatio=True,
            anchor="c",
        )
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawCentredString(
            (x_mm + tag_size_mm / 2.0) * mm,
            (y_mm - 5.0) * mm,
            f"ID 0 - {int(tag_size_mm)} mm",
        )
        pdf.setFont("Helvetica", 7)
        pdf.drawCentredString(
            (x_mm + tag_size_mm / 2.0) * mm,
            (y_mm - 9.0) * mm,
            f"tag_size_m = {tag_size_mm / 1000.0:.3f}",
        )

    scale_width = 100.0 * mm
    scale_x = (page_width - scale_width) / 2.0
    scale_y = 22.0 * mm
    pdf.setLineWidth(1.0)
    pdf.line(scale_x, scale_y, scale_x + scale_width, scale_y)
    pdf.line(scale_x, scale_y - 2.0 * mm, scale_x, scale_y + 2.0 * mm)
    pdf.line(
        scale_x + scale_width,
        scale_y - 2.0 * mm,
        scale_x + scale_width,
        scale_y + 2.0 * mm,
    )
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(
        page_width / 2.0,
        14.0 * mm,
        "100 mm scale check - disable Fit to page / Shrink / Enlarge",
    )
    pdf.showPage()
    pdf.save()


def build_output_paths(tag_size_mm: float) -> dict[str, Path]:
    tag_size_token = f"{int(tag_size_mm)}mm"
    return {
        "combined_pdf": OUTPUT_DIR / f"apriltag_tag36h11_ids_0_to_6_{tag_size_token}_letter.pdf",
        "rover_pdf": OUTPUT_DIR / f"apriltag_rover_id0_{tag_size_token}_letter.pdf",
        "world_pdf": OUTPUT_DIR / f"apriltag_world_reference_ids_1_to_6_{tag_size_token}_letter.pdf",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate print-ready AprilTag PDFs for rover and fixed references."
    )
    parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=TAG_SIZE_MM,
        help="Outer black-square width for the main rover/world tag pack.",
    )
    parser.add_argument(
        "--skip-rover-size-sheet",
        action="store_true",
        help="Skip the 50/60/70/80 mm rover size comparison sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = build_output_paths(args.tag_size_mm)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_paths = {
        tag_id: generate_tag_png(tag_id, tag_size_mm=args.tag_size_mm)
        for tag_id in TAG_IDS
    }
    rover_option_pngs = {
        tag_size_mm: {0: generate_tag_png(0, tag_size_mm=tag_size_mm)}
        for tag_size_mm in ROVER_OPTION_TAG_SIZES_MM
    }

    write_pdf(
        output_paths["combined_pdf"],
        list(TAG_IDS),
        png_paths,
        tag_size_mm=args.tag_size_mm,
    )
    write_pdf(
        output_paths["rover_pdf"],
        [0],
        png_paths,
        tag_size_mm=args.tag_size_mm,
    )
    for tag_size_mm, rover_png in rover_option_pngs.items():
        rover_pdf_path = (
            OUTPUT_DIR / f"apriltag_rover_id0_{int(tag_size_mm)}mm_letter.pdf"
        )
        write_pdf(
            rover_pdf_path,
            [0],
            rover_png,
            tag_size_mm=tag_size_mm,
        )
    if not args.skip_rover_size_sheet:
        write_rover_size_sheet(
            ROVER_SIZE_SHEET_PDF_PATH,
            {tag_size_mm: rover_png[0] for tag_size_mm, rover_png in rover_option_pngs.items()},
        )
    write_pdf(
        output_paths["world_pdf"],
        list(range(1, 7)),
        png_paths,
        tag_size_mm=args.tag_size_mm,
    )

    print(output_paths["combined_pdf"])
    print(output_paths["rover_pdf"])
    for tag_size_mm in ROVER_OPTION_TAG_SIZES_MM:
        print(OUTPUT_DIR / f"apriltag_rover_id0_{int(tag_size_mm)}mm_letter.pdf")
    if not args.skip_rover_size_sheet:
        print(ROVER_SIZE_SHEET_PDF_PATH)
    print(output_paths["world_pdf"])
    for path in png_paths.values():
        print(path)


if __name__ == "__main__":
    main()
