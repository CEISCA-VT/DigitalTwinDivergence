"""Audit synchronized encoder turn counts against AprilTag heading changes."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from DigitalTwin.kinematics import DifferentialDriveGeometry


OUTPUT_DIR = Path("DigitalTwin/datasets/analysis/apriltag_turn_event_audit")
TURN_WIDTH_M = 0.192


@dataclass(frozen=True)
class RunSpec:
    name: str
    tracking: Path
    telemetry: Path
    video_minus_telemetry_offset_s: float


RUNS = (
    RunSpec(
        "trapezoid",
        Path(
            "DigitalTwin/datasets/analysis/apriltag_trapezoid_metric/"
            "apriltag_still_summary.json"
        ),
        Path("raw_logs/telemetry/ugv_t147_interactive_20260805_192736.csv"),
        -10.15,
    ),
    RunSpec(
        "trial1_square_1p5",
        Path(
            "DigitalTwin/datasets/analysis/apriltag_trial1_square_1p5_tracking/"
            "apriltag_still_summary.json"
        ),
        Path("raw_logs/telemetry/ugv_t147_interactive_20260805_174551.csv"),
        3.90,
    ),
)


def _telemetry_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row.get("cycle_ok") == "True"]


def _groups(mask: np.ndarray) -> list[tuple[int, int]]:
    groups = []
    start = None
    for index, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= 2:
                groups.append((start, index - 1))
            start = None
    return groups


def _audit_run(spec: RunSpec) -> list[dict[str, object]]:
    payload = json.loads(spec.tracking.read_text(encoding="utf-8"))
    fps = float(payload["video"]["fps"])
    frame_rows = payload["frame_summaries"]
    telemetry = _telemetry_rows(spec.telemetry)
    source_start = float(telemetry[0]["source_sample_time_s"])
    elapsed = np.asarray(
        [float(row["source_sample_time_s"]) - source_start for row in telemetry]
    )
    left_command = np.asarray([float(row["L"]) for row in telemetry])
    right_command = np.asarray([float(row["R"]) for row in telemetry])
    left_count = np.asarray([int(row["enc_left"]) for row in telemetry])
    right_count = np.asarray([int(row["enc_right"]) for row in telemetry])

    # Require opposing/pivot track motion. Straight segments with small track
    # imbalance are deliberately excluded from the event calibration.
    turning = (
        (np.abs(left_command - right_command) >= 0.06)
        & ((left_command * right_command) <= 0.001)
    )
    geometry = DifferentialDriveGeometry(effective_track_width_m=TURN_WIDTH_M)
    results = []
    for event_index, (first, last) in enumerate(_groups(turning), start=1):
        before_index = max(0, first - 1)
        after_index = min(len(telemetry) - 1, last + 1)
        video_start = (
            float(elapsed[before_index]) + spec.video_minus_telemetry_offset_s
        )
        video_end = float(elapsed[after_index]) + spec.video_minus_telemetry_offset_s
        padded_start = video_start - 1.0
        padded_end = video_end + 1.0
        selected = [
            row
            for row in frame_rows
            if padded_start <= float(row["time_s"]) <= padded_end
            and row.get("rover_heading_rad") is not None
        ]
        selected_times = np.asarray([float(row["time_s"]) for row in selected])
        expected_frames = max(1, int(round((padded_end - padded_start) * fps)))
        coverage = len(selected) / expected_frames
        maximum_gap = (
            float(np.max(np.diff(selected_times))) if len(selected_times) >= 2 else math.inf
        )
        quality = "valid"
        if coverage < 0.90 or maximum_gap > 0.50:
            quality = "invalid_tracking_gap"
        elif video_end - video_start > 20.0:
            quality = "valid_long_spin"

        observed_degrees = math.nan
        if selected:
            headings = np.unwrap(
                np.asarray([float(row["rover_heading_rad"]) for row in selected])
            )
            before = selected_times <= video_start + 0.25
            after = selected_times >= video_end - 0.25
            if before.any() and after.any():
                observed_degrees = math.degrees(
                    float(np.median(headings[after]) - np.median(headings[before]))
                )

        delta_left = int(left_count[after_index] - left_count[before_index])
        delta_right = int(right_count[after_index] - right_count[before_index])
        left_distance = (
            delta_left * geometry.meters_per_tick * geometry.left_tick_sign
        )
        right_distance = (
            delta_right * geometry.meters_per_tick * geometry.right_tick_sign
        )
        encoder_degrees = math.degrees(
            (right_distance - left_distance) / TURN_WIDTH_M
        )
        effective_width = (
            (right_distance - left_distance) / math.radians(observed_degrees)
            if math.isfinite(observed_degrees) and abs(observed_degrees) >= 5.0
            else math.nan
        )
        results.append(
            {
                "run": spec.name,
                "event": event_index,
                "quality": quality,
                "telemetry_start_s": float(elapsed[before_index]),
                "telemetry_end_s": float(elapsed[after_index]),
                "video_start_s": video_start,
                "video_end_s": video_end,
                "duration_s": video_end - video_start,
                "tracking_coverage_fraction": coverage,
                "maximum_tracking_gap_s": maximum_gap,
                "delta_left_counts": delta_left,
                "delta_right_counts": delta_right,
                "apriltag_turn_deg": observed_degrees,
                "encoder_turn_deg_at_0p192_m": encoder_degrees,
                "encoder_to_apriltag_ratio": (
                    encoder_degrees / observed_degrees
                    if math.isfinite(observed_degrees) and abs(observed_degrees) >= 5.0
                    else math.nan
                ),
                "event_effective_width_m": effective_width,
            }
        )
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row for spec in RUNS for row in _audit_run(spec)]
    with (OUTPUT_DIR / "turn_events.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid_short = [row for row in rows if row["quality"] == "valid"]
    ratios = np.asarray(
        [abs(float(row["encoder_to_apriltag_ratio"])) for row in valid_short]
    )
    widths = np.asarray(
        [abs(float(row["event_effective_width_m"])) for row in valid_short]
    )
    summary = {
        "schema": "ugv01_apriltag_turn_event_audit_v1",
        "model_effective_width_m": TURN_WIDTH_M,
        "events_total": len(rows),
        "valid_short_events": len(valid_short),
        "median_absolute_encoder_to_apriltag_ratio": float(np.median(ratios)),
        "median_event_effective_width_m": float(np.median(widths)),
        "event_effective_width_range_m": [float(np.min(widths)), float(np.max(widths))],
        "events": rows,
        "interpretation": (
            "This audit measures encoder-predicted angle against physical AprilTag "
            "angle. It does not recover the requested terminal angle because the "
            "interactive CSV records realized track motion, not the user's menu choice."
        ),
    }
    (OUTPUT_DIR / "turn_event_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# AprilTag Turn-Event Audit",
        "",
        "Each event compares the encoder counter change with the synchronized physical "
        "heading change measured by AprilTag ID 0. Requested menu angles are not present "
        "in the CSV and therefore are not inferred.",
        "",
        "| Run | Event | Quality | AprilTag turn | Encoder turn | Ratio | Effective width |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['run']} | {row['event']} | {row['quality']} | "
            f"{float(row['apriltag_turn_deg']):.1f} deg | "
            f"{float(row['encoder_turn_deg_at_0p192_m']):.1f} deg | "
            f"{float(row['encoder_to_apriltag_ratio']):.2f} | "
            f"{float(row['event_effective_width_m']):.3f} m |"
        )
    report.extend(
        [
            "",
            f"- Valid short-event median absolute ratio: **{np.median(ratios):.3f}**",
            f"- Valid short-event median effective width: **{np.median(widths):.3f} m**",
            f"- Valid short-event width range: **{np.min(widths):.3f}-{np.max(widths):.3f} m**",
            "",
            "A ratio above 1 means the encoder model overestimates the physical turn. "
            "A ratio below 1 means it underestimates it.",
            "",
        ]
    )
    (OUTPUT_DIR / "turn_event_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(OUTPUT_DIR / "turn_event_report.md")


if __name__ == "__main__":
    main()
