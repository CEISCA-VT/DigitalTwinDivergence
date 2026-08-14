"""Export AprilTag rover ground truth from a tracking summary JSON to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SUMMARY = Path(
    "DigitalTwin/datasets/analysis/apriltag_trapezoid_metric/"
    "apriltag_still_summary.json"
)
DEFAULT_OUTPUT = Path(
    "DigitalTwin/datasets/analysis/apriltag_trapezoid_metric/"
    "ground_truth_trajectory.csv"
)


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def export_ground_truth(summary_path: Path, output_path: Path) -> int:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = data.get("frame_summaries", [])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "video_time_s",
                "x_gt_m",
                "y_gt_m",
                "theta_gt_rad",
                "theta_gt_deg",
                "tracking_status",
                "transform_method",
                "reference_tags",
                "frames_since_rover_decode",
                "optical_flow_error_px",
            ],
        )
        writer.writeheader()
        for frame in frames:
            xy = frame.get("rover_xy_m")
            if not isinstance(xy, list) or len(xy) != 2:
                continue
            x = _as_float(xy[0])
            y = _as_float(xy[1])
            theta = _as_float(frame.get("rover_heading_rad"))
            if x is None or y is None or theta is None:
                continue
            writer.writerow(
                {
                    "frame_index": frame.get("frame_index"),
                    "video_time_s": frame.get("time_s"),
                    "x_gt_m": x,
                    "y_gt_m": y,
                    "theta_gt_rad": theta,
                    "theta_gt_deg": theta * 180.0 / 3.141592653589793,
                    "tracking_status": frame.get("rover_tracking_status", ""),
                    "transform_method": frame.get("transform_method", ""),
                    "reference_tags": " ".join(
                        str(tag) for tag in frame.get("reference_tags", [])
                    ),
                    "frames_since_rover_decode": frame.get(
                        "frames_since_rover_decode", ""
                    ),
                    "optical_flow_error_px": frame.get("optical_flow_error_px", ""),
                }
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    count = export_ground_truth(args.summary, args.output)
    print(args.output)
    print(f"exported_rows={count}")


if __name__ == "__main__":
    main()
