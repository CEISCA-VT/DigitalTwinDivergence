"""Repair short AprilTag tracking gaps without changing measured detections.

This is a data-salvage utility for fixed-camera UGV01 AprilTag runs. It keeps
decoded and optical-flow rover poses intact, fills missing interior gaps by
time interpolation, and labels every repaired sample with a status/confidence
so downstream fidelity reports can separate measured from reconstructed poses.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INPUT = Path(
    "DigitalTwin/datasets/analysis/apriltag_carpet_2x1_full_1x_elevation_corrected/apriltag_still_summary.json"
)
DEFAULT_OUTPUT = Path(
    "DigitalTwin/datasets/analysis/apriltag_carpet_2x1_full_1x_continuity_repaired"
)


def _angle_lerp(start: float, end: float, alpha: float) -> float:
    delta = math.atan2(math.sin(end - start), math.cos(end - start))
    return float(math.atan2(math.sin(start + alpha * delta), math.cos(start + alpha * delta)))


def _gap_runs(valid: list[bool]) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    start = None
    for index, ok in enumerate(valid):
        if not ok and start is None:
            start = index
        elif ok and start is not None:
            gaps.append((start, index - 1))
            start = None
    if start is not None:
        gaps.append((start, len(valid) - 1))
    return gaps


def _status_confidence(status: str | None) -> float:
    if status == "decoded":
        return 1.0
    if status in {"optical_flow", "rejected_quad"}:
        return 0.85
    return 0.0


def _recompute_rover_summary(rows: list[dict[str, Any]]) -> dict[str, float | int] | None:
    positions = [row["rover_xy_m"] for row in rows if row.get("rover_xy_m") is not None]
    if not positions:
        return None
    rover = np.asarray(positions, dtype=float)
    return {
        "samples": int(len(rover)),
        "x_median_m": float(np.median(rover[:, 0])),
        "y_median_m": float(np.median(rover[:, 1])),
        "x_span_m": float(np.ptp(rover[:, 0])),
        "y_span_m": float(np.ptp(rover[:, 1])),
    }


def _format_continuity_report(payload: dict[str, Any]) -> str:
    repair = payload["continuity_repair"]
    lines = [
        "# AprilTag Continuity Repair",
        "",
        f"- Source summary: `{repair['source_summary']}`",
        f"- Output schema: `{payload['schema']}`",
        f"- Frames: `{repair['total_frames']}`",
        f"- Original rover-valid frames: `{repair['original_valid_frames']}` (`{repair['original_valid_fraction']:.3f}`)",
        f"- Repaired rover-valid frames: `{repair['repaired_valid_frames']}` (`{repair['repaired_valid_fraction']:.3f}`)",
        f"- Filled frames: `{repair['filled_frames']}`",
        f"- Remaining missing frames: `{repair['remaining_missing_frames']}`",
        "",
        "## Status Counts",
        "",
        "| Status | Frames | Meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "decoded": "ID 0 was directly decoded by OpenCV.",
        "optical_flow": "ID 0 was propagated by optical flow.",
        "rejected_quad": "OpenCV rejected-quad candidate matched the tracked rover tag.",
        "interpolated_short_gap": "Missing pose bridged between neighboring valid poses.",
        "interpolated_long_gap": "Longer bridge; usable for visualization, lower confidence.",
        "missing": "No interior neighbors or gap exceeded configured limit.",
    }
    for status, count in repair["post_repair_status_counts"].items():
        lines.append(f"| {status} | {count} | {meanings.get(status, '')} |")
    lines.extend(
        [
            "",
            "## Filled Gaps",
            "",
            "| Start s | End s | Duration s | Frames | Repair Status |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for gap in repair["filled_gaps"]:
        lines.append(
            f"| {gap['start_time_s']:.3f} | {gap['end_time_s']:.3f} | "
            f"{gap['duration_s']:.3f} | {gap['frames']} | {gap['status']} |"
        )
    if not repair["filled_gaps"]:
        lines.append("|  |  |  |  | No gaps filled |")
    lines.extend(
        [
            "",
            "## Use Guidance",
            "",
            "- Use `decoded`, `optical_flow`, and `rejected_quad` frames for the strictest ground-truth claims.",
            "- Include `interpolated_short_gap` frames for continuous trajectory plots and engineering fidelity checks.",
            "- Treat `interpolated_long_gap` as low-confidence visualization support, not final publication-grade ground truth unless independently justified.",
            "- Fixed reference tags are static by design; the output adds their known world positions to every frame so downstream tools have continuous anchor metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_basic_report(payload: dict[str, Any]) -> str:
    video = payload.get("video", {})
    sampling = payload.get("sampling", {})
    repair = payload["continuity_repair"]
    lines = [
        "# AprilTag Still-Video Check",
        "",
        f"- Video: `{payload.get('video_path')}`",
        f"- Calibration: `{payload.get('calibration_path')}`",
        f"- Resolution: `{video.get('image_size_px')}`",
        f"- FPS: `{float(video.get('fps') or 0.0):.3f}`",
        f"- Attempted sampled frames: `{sampling.get('attempted_frames', len(payload['frame_summaries']))}`",
        f"- Rover-valid frames after repair: `{repair['repaired_valid_frames']}`",
        f"- Rover ID 0 filled frames: `{repair['filled_frames']}`",
        "",
        "## Tag Detection Counts",
        "",
        "| Tag ID | Frames Detected | Role |",
        "|---:|---:|---|",
    ]
    roles = {"0": "rover"}
    for tag_id in sorted(payload.get("world_tags_m", {}), key=lambda item: int(item)):
        roles[str(tag_id)] = "fixed world reference"
    for tag_id, count in sorted(
        payload.get("tag_detection_counts", {}).items(), key=lambda item: int(item[0])
    ):
        lines.append(f"| {tag_id} | {count} | {roles.get(str(tag_id), '')} |")
    lines.extend(["", "## World Frame", ""])
    for tag_id, xy in payload.get("world_tags_m", {}).items():
        lines.append(f"- ID {tag_id}: `({float(xy[0]):.4f}, {float(xy[1]):.4f}) m`")
    lines.append("")
    return "\n".join(lines)


def repair_summary(
    input_path: Path,
    output_dir: Path,
    *,
    max_short_gap_s: float,
    max_long_gap_s: float,
) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload["frame_summaries"]
    if not rows:
        raise RuntimeError("summary contains no frame_summaries")

    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    original_valid = [row.get("rover_xy_m") is not None for row in rows]
    original_valid_count = int(sum(original_valid))
    filled_gaps: list[dict[str, Any]] = []

    for row in rows:
        status = row.get("rover_tracking_status")
        row["ground_truth_status"] = status if row.get("rover_xy_m") is not None else "missing"
        row["ground_truth_confidence"] = _status_confidence(status if isinstance(status, str) else None)

    for start, end in _gap_runs(original_valid):
        before = start - 1
        after = end + 1
        if before < 0 or after >= len(rows):
            continue
        if rows[before].get("rover_xy_m") is None or rows[after].get("rover_xy_m") is None:
            continue
        duration = float(times[after] - times[before])
        if duration > max_long_gap_s:
            continue
        status = "interpolated_short_gap" if duration <= max_short_gap_s else "interpolated_long_gap"
        confidence = 0.60 if status == "interpolated_short_gap" else 0.25
        xy0 = np.asarray(rows[before]["rover_xy_m"], dtype=float)
        xy1 = np.asarray(rows[after]["rover_xy_m"], dtype=float)
        h0 = rows[before].get("rover_heading_rad")
        h1 = rows[after].get("rover_heading_rad")
        for index in range(start, end + 1):
            alpha = float((times[index] - times[before]) / (times[after] - times[before]))
            rows[index]["rover_xy_m"] = ((1.0 - alpha) * xy0 + alpha * xy1).tolist()
            if h0 is not None and h1 is not None:
                rows[index]["rover_heading_rad"] = _angle_lerp(float(h0), float(h1), alpha)
            rows[index]["rover_tracking_status"] = status
            rows[index]["ground_truth_status"] = status
            rows[index]["ground_truth_confidence"] = confidence
        filled_gaps.append(
            {
                "start_frame_index": int(rows[start]["frame_index"]),
                "end_frame_index": int(rows[end]["frame_index"]),
                "start_time_s": float(times[start]),
                "end_time_s": float(times[end]),
                "duration_s": float(times[end] - times[start]),
                "frames": int(end - start + 1),
                "status": status,
            }
        )

    world_tags = {
        str(tag_id): list(xy) for tag_id, xy in payload.get("world_tags_m", {}).items()
    }
    for row in rows:
        tags = {
            tag_id: {
                "xy_m": xy,
                "status": "fixed_reference",
                "confidence": 1.0,
            }
            for tag_id, xy in world_tags.items()
        }
        if row.get("rover_xy_m") is not None:
            tags["0"] = {
                "xy_m": row["rover_xy_m"],
                "heading_rad": row.get("rover_heading_rad"),
                "status": row["ground_truth_status"],
                "confidence": row["ground_truth_confidence"],
            }
        row["continuous_tags_world_m"] = tags

    repaired_valid = [row.get("rover_xy_m") is not None for row in rows]
    status_counts = Counter(str(row.get("ground_truth_status", "missing")) for row in rows)
    payload["schema"] = "ugv01_apriltag_continuity_repaired_v1"
    payload["rover_still_summary"] = _recompute_rover_summary(rows)
    payload["continuity_repair"] = {
        "source_summary": str(input_path),
        "method": "preserve_detections_then_interpolate_interior_rover_gaps",
        "max_short_gap_s": float(max_short_gap_s),
        "max_long_gap_s": float(max_long_gap_s),
        "total_frames": int(len(rows)),
        "original_valid_frames": original_valid_count,
        "original_valid_fraction": float(original_valid_count / len(rows)),
        "repaired_valid_frames": int(sum(repaired_valid)),
        "repaired_valid_fraction": float(sum(repaired_valid) / len(rows)),
        "filled_frames": int(sum(repaired_valid) - original_valid_count),
        "remaining_missing_frames": int(len(rows) - sum(repaired_valid)),
        "filled_gaps": filled_gaps,
        "post_repair_status_counts": dict(sorted(status_counts.items())),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "apriltag_still_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "apriltag_still_report.md").write_text(
        _format_basic_report(payload), encoding="utf-8"
    )
    (output_dir / "apriltag_continuity_report.md").write_text(
        _format_continuity_report(payload), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-short-gap-s", type=float, default=6.0)
    parser.add_argument("--max-long-gap-s", type=float, default=30.0)
    args = parser.parse_args()
    payload = repair_summary(
        args.input,
        args.output_dir,
        max_short_gap_s=args.max_short_gap_s,
        max_long_gap_s=args.max_long_gap_s,
    )
    repair = payload["continuity_repair"]
    print(args.output_dir / "apriltag_still_summary.json")
    print(args.output_dir / "apriltag_continuity_report.md")
    print(
        "valid:",
        f"{repair['original_valid_frames']} -> {repair['repaired_valid_frames']}",
        "filled:",
        repair["filled_frames"],
        "remaining:",
        repair["remaining_missing_frames"],
    )


if __name__ == "__main__":
    main()
