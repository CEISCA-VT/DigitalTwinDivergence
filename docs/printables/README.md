# Printable Calibration Assets

Print `charuco_7x5_30mm_letter_landscape.pdf` on US Letter paper in landscape
orientation.

- Select **Actual size** or **100%**.
- Disable **Fit**, **Shrink oversized pages**, and borderless enlargement.
- After printing, confirm that the PDF's scale-check line is exactly `100 mm`.
- Confirm that several chessboard squares are exactly `30 mm`.
- Mount the sheet flat on foam board without covering the pattern.

Board definition:

- OpenCV dictionary: `DICT_5X5_100`
- chessboard: `7 x 5` squares
- square side: `30 mm`
- marker side: `22 mm`
- pattern area: `210 x 150 mm`

The PDF is the authoritative print file. The PNG is provided for inspection
and has embedded `508 dpi` metadata.

## AprilTag Tracking Pack

The recommended print files are:

- `apriltag_rover_id0_120mm_letter.pdf`: the moving rover tag
- `apriltag_rover_id0_50mm_letter.pdf`: smaller moving rover tag
- `apriltag_rover_id0_60mm_letter.pdf`: smaller moving rover tag
- `apriltag_rover_id0_70mm_letter.pdf`: smaller moving rover tag
- `apriltag_rover_id0_80mm_letter.pdf`: smaller moving rover tag
- `apriltag_rover_id0_50_60_70_80mm_letter.pdf`: one-page rover tag size test sheet
- `apriltag_world_reference_ids_1_to_6_120mm_letter.pdf`: six fixed tags
- `apriltag_tag36h11_ids_0_to_6_120mm_letter.pdf`: combined seven-page pack

Use IDs `1-4` to fit the camera-to-floor transformation. Keep IDs `5-6` out of
that fit and use their measured positions to quantify held-out transformation
error.

Print the rover PDF twice if a spare is wanted, but never show duplicate ID `0`
tags in the same camera view. The `120 mm` rover tag gives the strongest
detection, while the `50-80 mm` rover-tag PDFs are easier to mount flat on the
UGV01. Use `apriltag_rover_id0_50_60_70_80mm_letter.pdf` to print all four
smaller size options on one US Letter sheet. Every page includes a `100 mm`
scale-check line. Select **Actual size** or **100%**, disable all
fit/shrink/enlarge options, and verify both measurements after printing.

Mount each sheet separately on flat foam board. Do not cover or crop the white
area around the black tag. The fixed world-reference tags use:

```text
family = tag36h11
tag_size_m = 0.120
```

The moving rover tag must use the size of the specific PDF that is mounted:

```text
apriltag_rover_id0_50mm_letter.pdf  -> tag_size_m = 0.050
apriltag_rover_id0_60mm_letter.pdf  -> tag_size_m = 0.060
apriltag_rover_id0_70mm_letter.pdf  -> tag_size_m = 0.070
apriltag_rover_id0_80mm_letter.pdf  -> tag_size_m = 0.080
apriltag_rover_id0_120mm_letter.pdf -> tag_size_m = 0.120
```
