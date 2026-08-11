"""Render an AprilTag ground-truth video with metric layout annotations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SUMMARY = Path(
    "DigitalTwin/datasets/analysis/apriltag_trapezoid_metric/"
    "apriltag_still_summary.json"
)
DEFAULT_OUTPUT = Path(
    "DigitalTwin/datasets/analysis/apriltag_trapezoid_metric/"
    "apriltag_tracking_overlay.mp4"
)
SIDE_ORDER = (4, 1, 2, 3)


def _center(corners: np.ndarray) -> np.ndarray:
    return np.asarray(corners, dtype=np.float64).reshape(4, 2).mean(axis=0)


def _project(H_world_to_image: np.ndarray, point_m: np.ndarray) -> np.ndarray:
    point = H_world_to_image @ np.array([point_m[0], point_m[1], 1.0])
    point /= point[2]
    return point[:2]


def _label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def render_overlay(
    summary_path: Path,
    output_path: Path,
    *,
    start_s: float,
    duration_s: float,
) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    video_path = Path(payload["video_path"])
    rows = {int(row["frame_index"]): row for row in payload["frame_summaries"]}
    world_tags = {
        int(tag_id): np.asarray(xy, dtype=np.float64)
        for tag_id, xy in payload["world_tags_m"].items()
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(round(start_s * fps)))
    end_frame = start_frame + max(1, int(round(duration_s * fps)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not create output video: {output_path}")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # The phone is stationary. Lock the display transform from an early frame
    # containing all four references so the metric overlay remains visible when
    # a fixed tag later leaves the image.
    static_H_image_to_world = None
    for calibration_frame in range(min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 300)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame)
        ok, calibration_image = cap.read()
        if not ok:
            break
        calibration_gray = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
        calibration_corners, calibration_ids_array, _ = detector.detectMarkers(
            calibration_gray
        )
        if calibration_ids_array is None:
            continue
        calibration_ids = [int(value) for value in calibration_ids_array.reshape(-1)]
        calibration_detected = {
            tag_id: np.asarray(tag_corners, dtype=np.float64).reshape(4, 2)
            for tag_id, tag_corners in zip(calibration_ids, calibration_corners)
        }
        if all(tag_id in calibration_detected for tag_id in SIDE_ORDER):
            image_points = np.asarray(
                [_center(calibration_detected[tag_id]) for tag_id in SIDE_ORDER]
            )
            world_points = np.asarray([world_tags[tag_id] for tag_id in SIDE_ORDER])
            static_H_image_to_world, _ = cv2.findHomography(
                image_points, world_points, method=0
            )
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    static_H_world_to_image = (
        None
        if static_H_image_to_world is None
        else np.linalg.inv(static_H_image_to_world)
    )
    trail: list[np.ndarray] = []
    written = 0

    for frame_index in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        row = rows.get(frame_index)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids_array, _ = detector.detectMarkers(gray)
        ids = [] if ids_array is None else [int(value) for value in ids_array.reshape(-1)]
        detected = {
            tag_id: np.asarray(tag_corners, dtype=np.float64).reshape(4, 2)
            for tag_id, tag_corners in zip(ids, corners)
        }

        if ids:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids_array)
            for tag_id in ids:
                point = _center(detected[tag_id]).astype(int)
                cv2.circle(frame, tuple(point), 5, (20, 220, 255), -1, cv2.LINE_AA)

        fixed_ids = [tag_id for tag_id in SIDE_ORDER if tag_id in detected]
        H_world_to_image = static_H_world_to_image
        if H_world_to_image is None and len(fixed_ids) == 4:
            image_points = np.asarray([_center(detected[tag_id]) for tag_id in SIDE_ORDER])
            world_points = np.asarray([world_tags[tag_id] for tag_id in SIDE_ORDER])
            H_image_to_world, _ = cv2.findHomography(
                image_points, world_points, method=0
            )
            if H_image_to_world is not None:
                H_world_to_image = np.linalg.inv(H_image_to_world)

        if H_world_to_image is not None:
            for index, tag_a in enumerate(SIDE_ORDER):
                tag_b = SIDE_ORDER[(index + 1) % len(SIDE_ORDER)]
                point_a = _project(H_world_to_image, world_tags[tag_a]).astype(int)
                point_b = _project(H_world_to_image, world_tags[tag_b]).astype(int)
                cv2.line(frame, tuple(point_a), tuple(point_b), (40, 210, 40), 2, cv2.LINE_AA)
                midpoint = ((point_a + point_b) / 2).astype(int)
                distance = float(np.linalg.norm(world_tags[tag_b] - world_tags[tag_a]))
                _label(
                    frame,
                    f"{tag_a}-{tag_b}: {distance:.2f} m",
                    (int(midpoint[0]) + 4, int(midpoint[1]) - 5),
                    color=(80, 255, 80),
                )

        rover_xy = None if row is None else row.get("rover_xy_m")
        heading = None if row is None else row.get("rover_heading_rad")
        status = "unavailable" if row is None else str(row.get("rover_tracking_status"))
        if H_world_to_image is not None and rover_xy is not None:
            rover_world = np.asarray(rover_xy, dtype=np.float64)
            rover_px = _project(H_world_to_image, rover_world).astype(int)
            trail.append(rover_px)
            trail = trail[-int(max(fps * 4.0, 2)) :]
            if len(trail) >= 2:
                cv2.polylines(
                    frame,
                    [np.asarray(trail, dtype=np.int32)],
                    False,
                    (255, 180, 40),
                    2,
                    cv2.LINE_AA,
                )
            cv2.circle(frame, tuple(rover_px), 9, (0, 80, 255), 2, cv2.LINE_AA)
            if heading is not None:
                arrow_world = rover_world + 0.16 * np.array(
                    [math.cos(float(heading)), math.sin(float(heading))]
                )
                arrow_px = _project(H_world_to_image, arrow_world).astype(int)
                cv2.arrowedLine(
                    frame,
                    tuple(rover_px),
                    tuple(arrow_px),
                    (0, 80, 255),
                    3,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )

        cv2.rectangle(frame, (8, 8), (415, 82), (0, 0, 0), -1)
        time_s = frame_index / fps
        _label(frame, f"AprilTag metric ground truth | t={time_s:.2f} s", (18, 30), scale=0.55)
        if rover_xy is None:
            _label(frame, "ID 0 location unavailable", (18, 55), color=(80, 80, 255))
        else:
            _label(
                frame,
                f"ID 0: x={rover_xy[0]:.3f} m, y={rover_xy[1]:.3f} m from ID 4",
                (18, 55),
                color=(80, 220, 255),
            )
        _label(frame, f"tracking: {status}", (18, 76), scale=0.42, color=(210, 210, 210))
        writer.write(frame)
        written += 1

    writer.release()
    cap.release()
    return {
        "output_path": str(output_path),
        "source_video": str(video_path),
        "start_s": start_s,
        "duration_s": written / fps,
        "frames": written,
        "fps": fps,
        "resolution": [width, height],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=15.0)
    args = parser.parse_args()
    result = render_overlay(
        args.summary,
        args.output,
        start_s=args.start_s,
        duration_s=args.duration_s,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
