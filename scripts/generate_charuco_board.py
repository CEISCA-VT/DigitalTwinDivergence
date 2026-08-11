"""Generate the project's print-ready US Letter ChArUco calibration board."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


SQUARES_X = 7
SQUARES_Y = 5
SQUARE_MM = 30.0
MARKER_MM = 22.0
BOARD_WIDTH_MM = SQUARES_X * SQUARE_MM
BOARD_HEIGHT_MM = SQUARES_Y * SQUARE_MM
PIXELS_PER_MM = 20

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "printables"
PNG_PATH = OUTPUT_DIR / "charuco_7x5_30mm_letter_landscape.png"
PDF_PATH = OUTPUT_DIR / "charuco_7x5_30mm_letter_landscape.pdf"


def generate_board_image() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_MM / 1000.0,
        MARKER_MM / 1000.0,
        dictionary,
    )
    width_px = int(BOARD_WIDTH_MM * PIXELS_PER_MM)
    height_px = int(BOARD_HEIGHT_MM * PIXELS_PER_MM)
    image = board.generateImage((width_px, height_px), marginSize=0, borderBits=1)

    # PNG is exactly 20 px/mm, or 508 dpi. The PDF remains the preferred print file.
    Image.fromarray(image).save(
        PNG_PATH,
        dpi=(PIXELS_PER_MM * 25.4, PIXELS_PER_MM * 25.4),
    )


def generate_letter_pdf() -> None:
    page_width, page_height = landscape(letter)
    pdf = canvas.Canvas(str(PDF_PATH), pagesize=(page_width, page_height))
    pdf.setTitle("ChArUco 7x5 30 mm - US Letter Landscape")
    pdf.setAuthor("DigitalTwinDivergence")
    pdf.setSubject("UGV01 phone-camera intrinsic calibration board")

    board_width = BOARD_WIDTH_MM * mm
    board_height = BOARD_HEIGHT_MM * mm
    board_x = (page_width - board_width) / 2.0
    board_y = 36.0 * mm

    pdf.drawImage(
        str(PNG_PATH),
        board_x,
        board_y,
        width=board_width,
        height=board_height,
        preserveAspectRatio=True,
        anchor="c",
    )

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(
        page_width / 2.0,
        202.0 * mm,
        "ChArUco 7 x 5 | DICT_5X5_100 | square 30 mm | marker 22 mm",
    )

    scale_width = 100.0 * mm
    scale_x = (page_width - scale_width) / 2.0
    scale_y = 16.0 * mm
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
        9.0 * mm,
        "100 mm scale check - print at Actual size / 100% and disable Fit to page",
    )
    pdf.showPage()
    pdf.save()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_board_image()
    generate_letter_pdf()
    print(PDF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
