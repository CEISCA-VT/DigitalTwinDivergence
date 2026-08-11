"""Remap an existing tracked AprilTag summary into a revised measured layout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from DigitalTwin.analysis.analyze_apriltag_still_video import (
    TRAPEZOID_WORLD_TAGS_M,
    WORLD_LAYOUTS,
    _format_report,
)


DEFAULT_INPUT = Path(
    "DigitalTwin/datasets/analysis/apriltag_trapezoid_tracked/apriltag_still_summary.json"
)
DEFAULT_OUTPUT = Path("DigitalTwin/datasets/analysis/apriltag_trapezoid_metric")


def _map_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = transform @ np.array([point[0], point[1], 1.0])
    homogeneous /= homogeneous[2]
    return homogeneous[:2]


def remap_summary(
    input_path: Path,
    output_dir: Path,
    *,
    target_layout: dict[int, tuple[float, float]] = TRAPEZOID_WORLD_TAGS_M,
    target_layout_name: str = "trapezoid",
) -> dict[str, object]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    source_layout = {
        int(tag_id): tuple(map(float, xy))
        for tag_id, xy in payload["world_tags_m"].items()
    }
    common_ids = sorted(set(source_layout) & set(target_layout))
    if len(common_ids) < 4:
        raise RuntimeError("layout remapping requires four common fixed tags")
    transform, _ = cv2.findHomography(
        np.asarray([source_layout[tag_id] for tag_id in common_ids], dtype=float),
        np.asarray([target_layout[tag_id] for tag_id in common_ids], dtype=float),
        method=0,
    )
    if transform is None:
        raise RuntimeError("could not solve layout-to-layout homography")

    positions = []
    for row in payload["frame_summaries"]:
        xy = row.get("rover_xy_m")
        heading = row.get("rover_heading_rad")
        if xy is None:
            continue
        source_xy = np.asarray(xy, dtype=float)
        mapped_xy = _map_point(transform, source_xy)
        row["rover_xy_m"] = mapped_xy.tolist()
        positions.append(mapped_xy)
        if heading is not None:
            direction_point = source_xy + 1e-3 * np.array(
                [math.cos(float(heading)), math.sin(float(heading))]
            )
            mapped_direction = _map_point(transform, direction_point) - mapped_xy
            row["rover_heading_rad"] = float(
                math.atan2(mapped_direction[1], mapped_direction[0])
            )

    rover = np.asarray(positions, dtype=float)
    payload["world_tags_m"] = {
        str(tag_id): list(xy) for tag_id, xy in target_layout.items()
    }
    payload["rover_still_summary"] = {
        "samples": int(len(rover)),
        "x_median_m": float(np.median(rover[:, 0])),
        "y_median_m": float(np.median(rover[:, 1])),
        "x_span_m": float(np.ptp(rover[:, 0])),
        "y_span_m": float(np.ptp(rover[:, 1])),
    }
    payload["layout_remap"] = {
        "source_summary": str(input_path),
        "method": "four_correspondence_projective_remap",
        "target_layout": target_layout_name,
        "measured_sides_m": {
            f"{tag_a}-{tag_b}": float(
                np.linalg.norm(
                    np.asarray(target_layout[tag_b]) - np.asarray(target_layout[tag_a])
                )
            )
            for tag_a, tag_b in ((4, 1), (1, 2), (2, 3), (3, 4))
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "apriltag_still_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "apriltag_still_report.md").write_text(
        _format_report(payload), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--world-layout", choices=sorted(WORLD_LAYOUTS), default="trapezoid"
    )
    args = parser.parse_args()
    payload = remap_summary(
        args.input,
        args.output_dir,
        target_layout=WORLD_LAYOUTS[args.world_layout],
        target_layout_name=args.world_layout,
    )
    print(args.output_dir / "apriltag_still_summary.json")
    print("world_tags_m=", payload["world_tags_m"])
    print("rover=", payload["rover_still_summary"])


if __name__ == "__main__":
    main()
