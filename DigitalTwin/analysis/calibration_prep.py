"""Preparation utilities for tracked-rover calibration once batteries arrive."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from DigitalTwin.telemetry import gps_to_local_xy

from .common import parse_float, parse_int


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json_md(out_prefix: Path, payload: dict[str, object], title: str, notes: list[str]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        lines.append(f"- `{key}`: `{value}`")
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def straight_encoder_scale(csv_path: Path, measured_distance_m: float, out_prefix: Path) -> None:
    rows = _read_rows(csv_path)
    left = [value for value in (parse_int(row.get("enc_left", ""), None) for row in rows) if value is not None]
    right = [value for value in (parse_int(row.get("enc_right", ""), None) for row in rows) if value is not None]
    left_delta = (left[-1] - left[0]) if len(left) >= 2 else 0
    right_delta = (right[-1] - right[0]) if len(right) >= 2 else 0
    payload = {
        "csv_path": str(csv_path),
        "measured_distance_m": measured_distance_m,
        "left_tick_delta": left_delta,
        "right_tick_delta": right_delta,
        "left_meters_per_tick": (measured_distance_m / left_delta) if left_delta else None,
        "right_meters_per_tick": (measured_distance_m / right_delta) if right_delta else None,
    }
    _write_json_md(
        out_prefix,
        payload,
        "Straight Encoder Scale Calibration",
        [
            "Use a slow straight run with tracks on the ground.",
            "Repeat 3 to 5 runs and average the left/right meters-per-tick values.",
        ],
    )


def turn_track_width(csv_path: Path, turn_angle_deg: float, left_distance_m: float, right_distance_m: float, out_prefix: Path) -> None:
    theta_rad = math.radians(turn_angle_deg)
    payload = {
        "csv_path": str(csv_path),
        "turn_angle_deg": turn_angle_deg,
        "turn_angle_rad": theta_rad,
        "left_distance_m": left_distance_m,
        "right_distance_m": right_distance_m,
        "effective_track_width_m": ((right_distance_m - left_distance_m) / theta_rad) if theta_rad else None,
    }
    _write_json_md(
        out_prefix,
        payload,
        "In-Place Turn Track Width Calibration",
        [
            "Positive turn angle should match the chosen heading sign convention.",
            "Use clockwise and counterclockwise trials to verify sign consistency.",
        ],
    )


def imu_alignment(csv_path: Path, expected_heading_change_deg: float, out_prefix: Path) -> None:
    rows = _read_rows(csv_path)
    yaw_values = [value for value in (parse_float(row.get("y", row.get("imu_y", "")), None) for row in rows) if value is not None]
    observed_change = (yaw_values[-1] - yaw_values[0]) if len(yaw_values) >= 2 else None
    heading_sign = None
    if observed_change is not None and expected_heading_change_deg != 0:
        heading_sign = 1 if observed_change * expected_heading_change_deg >= 0 else -1
    payload = {
        "csv_path": str(csv_path),
        "expected_heading_change_deg": expected_heading_change_deg,
        "observed_heading_change_deg": observed_change,
        "heading_sign": heading_sign,
        "yaw_samples": len(yaw_values),
    }
    _write_json_md(
        out_prefix,
        payload,
        "IMU Heading Sign Check",
        [
            "Run this on a known clockwise or counterclockwise turn.",
            "If heading_sign is -1, invert yaw sign in the downstream calibration config.",
        ],
    )


def gps_local_frame(csv_path: Path, out_prefix: Path) -> None:
    rows = _read_rows(csv_path)
    lat_lon = [
        (parse_float(row.get("lat", row.get("gps_lat", "")), None), parse_float(row.get("lon", row.get("gps_lon", "")), None))
        for row in rows
    ]
    valid = [(lat, lon) for lat, lon in lat_lon if lat is not None and lon is not None]
    local_points: list[tuple[float, float]] = []
    if valid:
        origin_lat, origin_lon = valid[0]
        for lat, lon in valid:
            assert origin_lat is not None and origin_lon is not None
            local_points.append(gps_to_local_xy(lat, lon, origin_lat, origin_lon))
    payload = {
        "csv_path": str(csv_path),
        "valid_fix_count": len(valid),
        "origin_lat_deg": valid[0][0] if valid else None,
        "origin_lon_deg": valid[0][1] if valid else None,
        "local_x_span_m": (max(point[0] for point in local_points) - min(point[0] for point in local_points)) if local_points else None,
        "local_y_span_m": (max(point[1] for point in local_points) - min(point[1] for point in local_points)) if local_points else None,
    }
    _write_json_md(
        out_prefix,
        payload,
        "GPS Local-Frame Validation",
        [
            "Use this first on stationary logs, then on straight and square routes.",
            "The same origin convention should be reused by route-reference comparison scripts.",
        ],
    )


def route_reference_template(out_prefix: Path) -> None:
    payload = {
        "route_name": "",
        "reference_type": "overhead_video_or_fiducial",
        "origin_marker_id": "",
        "time_alignment_method": "manual_or_sync_pulse",
        "fields_expected_in_join": [
            "time_s",
            "ekf_x_m",
            "ekf_y_m",
            "gps_x_m",
            "gps_y_m",
            "reference_x_m",
            "reference_y_m",
            "route_segment",
        ],
    }
    _write_json_md(
        out_prefix,
        payload,
        "Route Reference Alignment Template",
        [
            "Fill in the route metadata after the first overhead-video or fiducial-tracked capture.",
            "Keep route names aligned with the preregistered run naming convention.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    straight = subparsers.add_parser("straight")
    straight.add_argument("csv_path")
    straight.add_argument("--distance-m", type=float, required=True)
    straight.add_argument("--out-prefix", required=True)

    turn = subparsers.add_parser("turn")
    turn.add_argument("csv_path")
    turn.add_argument("--turn-angle-deg", type=float, required=True)
    turn.add_argument("--left-distance-m", type=float, required=True)
    turn.add_argument("--right-distance-m", type=float, required=True)
    turn.add_argument("--out-prefix", required=True)

    imu = subparsers.add_parser("imu")
    imu.add_argument("csv_path")
    imu.add_argument("--expected-heading-change-deg", type=float, required=True)
    imu.add_argument("--out-prefix", required=True)

    gps = subparsers.add_parser("gps")
    gps.add_argument("csv_path")
    gps.add_argument("--out-prefix", required=True)

    route = subparsers.add_parser("route-template")
    route.add_argument("--out-prefix", required=True)

    args = parser.parse_args()
    if args.command == "straight":
        straight_encoder_scale(Path(args.csv_path), args.distance_m, Path(args.out_prefix))
    elif args.command == "turn":
        turn_track_width(Path(args.csv_path), args.turn_angle_deg, args.left_distance_m, args.right_distance_m, Path(args.out_prefix))
    elif args.command == "imu":
        imu_alignment(Path(args.csv_path), args.expected_heading_change_deg, Path(args.out_prefix))
    elif args.command == "gps":
        gps_local_frame(Path(args.csv_path), Path(args.out_prefix))
    else:
        route_reference_template(Path(args.out_prefix))


if __name__ == "__main__":
    main()
