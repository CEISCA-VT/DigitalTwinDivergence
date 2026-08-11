"""Verify fixed AprilTag layout and rover tag visibility in a still video."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CALIBRATION = Path(
    "DigitalTwin/datasets/analysis/camera_calibration/camera_calibration_charuco.json"
)
DEFAULT_OUTPUT_DIR = Path("DigitalTwin/datasets/analysis/apriltag_still")
RECTANGLE_WORLD_TAGS_M = {
    4: (0.0, 0.0),
    3: (1.5, 0.0),
    2: (1.5, 1.0),
    1: (0.0, 1.0),
}
SQUARE_1P5_WORLD_TAGS_M = {
    4: (0.0, 0.0),
    3: (1.5, 0.0),
    2: (1.5, 1.5),
    1: (0.0, 1.5),
}
TRAPEZOID_WORLD_TAGS_M = {
    4: (0.0, 0.0),
    1: (0.0, 0.70),
    3: (0.83494275, 0.3619),
    2: (0.83494275, 1.5619),
}
WORLD_LAYOUTS = {
    "rectangle": RECTANGLE_WORLD_TAGS_M,
    "square_1p5": SQUARE_1P5_WORLD_TAGS_M,
    "trapezoid": TRAPEZOID_WORLD_TAGS_M,
}
WORLD_TAGS_M = RECTANGLE_WORLD_TAGS_M


def _load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration = payload["calibration"]
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def _marker_center(corners: np.ndarray) -> np.ndarray:
    return np.asarray(corners, dtype=np.float64).reshape(4, 2).mean(axis=0)


def _marker_heading_image(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    top_mid = 0.5 * (pts[0] + pts[1])
    bottom_mid = 0.5 * (pts[2] + pts[3])
    direction = top_mid - bottom_mid
    return float(math.atan2(direction[1], direction[0]))


def _detect_frame(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
) -> tuple[np.ndarray, list[int], dict[int, np.ndarray], list[np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None:
        return gray, [], {}, [np.asarray(x, dtype=np.float64).reshape(4, 2) for x in rejected]
    tag_corners = {
        int(tag_id): np.asarray(corner, dtype=np.float64).reshape(4, 2)
        for tag_id, corner in zip(ids.reshape(-1), corners)
    }
    return (
        gray,
        [int(x) for x in ids.reshape(-1)],
        tag_corners,
        [np.asarray(x, dtype=np.float64).reshape(4, 2) for x in rejected],
    )


def _quad_perimeter(corners: np.ndarray) -> float:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(
        sum(np.linalg.norm(points[(index + 1) % 4] - points[index]) for index in range(4))
    )


def _align_quad(candidate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    candidate = np.asarray(candidate, dtype=np.float64).reshape(4, 2)
    reference = np.asarray(reference, dtype=np.float64).reshape(4, 2)
    options = []
    for points in (candidate, candidate[::-1]):
        for shift in range(4):
            aligned = np.roll(points, shift, axis=0)
            error = float(np.mean(np.linalg.norm(aligned - reference, axis=1)))
            options.append((error, aligned))
    error, aligned = min(options, key=lambda item: item[0])
    return aligned, error


def _track_rover_corners(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    previous_corners: np.ndarray,
    rejected: list[np.ndarray],
) -> tuple[np.ndarray | None, str | None, float | None]:
    previous_points = np.asarray(previous_corners, dtype=np.float32).reshape(-1, 1, 2)
    predicted, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        previous_points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if predicted is None or status_forward is None or not np.all(status_forward):
        return None, None, None
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        predicted,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if backward is None or status_backward is None or not np.all(status_backward):
        return None, None, None
    forward_backward_error = float(
        np.max(np.linalg.norm(backward.reshape(4, 2) - previous_points.reshape(4, 2), axis=1))
    )
    predicted_quad = predicted.reshape(4, 2).astype(np.float64)
    if forward_backward_error > 2.5 or not cv2.isContourConvex(predicted_quad.astype(np.float32)):
        return None, None, forward_backward_error
    previous_perimeter = _quad_perimeter(previous_corners)
    predicted_perimeter = _quad_perimeter(predicted_quad)
    if (
        predicted_perimeter < 20.0
        or predicted_perimeter < 0.55 * previous_perimeter
        or predicted_perimeter > 1.8 * previous_perimeter
    ):
        return None, None, forward_backward_error

    best_candidate = None
    best_error = math.inf
    for candidate in rejected:
        perimeter = _quad_perimeter(candidate)
        if perimeter < 0.5 * predicted_perimeter or perimeter > 2.0 * predicted_perimeter:
            continue
        aligned, error = _align_quad(candidate, predicted_quad)
        if error < best_error:
            best_candidate = aligned
            best_error = error
    candidate_limit = max(5.0, 0.12 * predicted_perimeter)
    if best_candidate is not None and best_error <= candidate_limit:
        refined = 0.75 * best_candidate + 0.25 * predicted_quad
        return refined, "rejected_quad", forward_backward_error
    return predicted_quad, "optical_flow", forward_backward_error


def _world_transform_from_tags(
    tag_corners: dict[int, np.ndarray],
    world_tags_m: dict[int, tuple[float, float]],
) -> tuple[np.ndarray | None, str | None, list[int]]:
    image_points = []
    world_points = []
    for tag_id, xy in world_tags_m.items():
        if tag_id not in tag_corners:
            continue
        image_points.append(_marker_center(tag_corners[tag_id]))
        world_points.append(xy)
    reference_tags = [tag_id for tag_id in world_tags_m if tag_id in tag_corners]
    if len(image_points) >= 4:
        transform, _ = cv2.findHomography(
            np.asarray(image_points, dtype=np.float64),
            np.asarray(world_points, dtype=np.float64),
            method=0,
        )
        return transform, "four_tag_homography", reference_tags
    if len(image_points) == 3:
        affine = cv2.getAffineTransform(
            np.asarray(image_points, dtype=np.float32),
            np.asarray(world_points, dtype=np.float32),
        )
        transform = np.vstack([affine, np.array([0.0, 0.0, 1.0])])
        return transform, "three_tag_affine", reference_tags
    return None, None, reference_tags


def _apply_homography(H: np.ndarray, image_xy: np.ndarray) -> np.ndarray:
    point = np.array([image_xy[0], image_xy[1], 1.0], dtype=np.float64)
    mapped = H @ point
    mapped /= mapped[2]
    return mapped[:2]


def analyze_still_video(
    video_path: Path,
    output_dir: Path,
    calibration_path: Path,
    *,
    sample_stride: int,
    max_frames: int,
    preview_count: int,
    world_tags_m: dict[int, tuple[float, float]] = RECTANGLE_WORLD_TAGS_M,
) -> dict[str, object]:
    camera_matrix, dist_coeffs = _load_calibration(calibration_path)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    image_size = None
    frame_index = -1
    attempted = 0
    accepted = 0
    id_counts: Counter[int] = Counter()
    frame_summaries: list[dict[str, object]] = []
    rover_positions: list[list[float]] = []
    world_tag_frames = 0
    three_tag_frames = 0
    four_tag_frames = 0
    fixed_tag_centers: dict[int, list[np.ndarray]] = {
        tag_id: [] for tag_id in world_tags_m
    }
    previous_gray: np.ndarray | None = None
    previous_rover_corners: np.ndarray | None = None
    frames_since_rover_decode = 0
    rover_tracking_counts: Counter[str] = Counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % sample_stride != 0:
            continue
        attempted += 1
        if image_size is None:
            image_size = [int(frame.shape[1]), int(frame.shape[0])]

        gray, ids, tag_corners, rejected = _detect_frame(
            frame,
            detector,
            camera_matrix,
            dist_coeffs,
        )
        rover_tracking_status = None
        optical_flow_error_px = None
        if 0 in tag_corners:
            rover_tracking_status = "decoded"
            frames_since_rover_decode = 0
        elif (
            sample_stride == 1
            and previous_gray is not None
            and previous_rover_corners is not None
            and frames_since_rover_decode < 600
        ):
            tracked, rover_tracking_status, optical_flow_error_px = _track_rover_corners(
                previous_gray,
                gray,
                previous_rover_corners,
                rejected,
            )
            if tracked is not None:
                tag_corners[0] = tracked
                frames_since_rover_decode += 1
            else:
                previous_rover_corners = None
        if 0 in tag_corners:
            previous_rover_corners = tag_corners[0].copy()
            rover_tracking_counts[str(rover_tracking_status)] += 1
        previous_gray = gray
        for tag_id in ids:
            id_counts[tag_id] += 1
            if tag_id in fixed_tag_centers:
                fixed_tag_centers[tag_id].append(_marker_center(tag_corners[tag_id]))

        world_seen = sorted(tag_id for tag_id in world_tags_m if tag_id in tag_corners)
        H, transform_method, reference_tags = _world_transform_from_tags(
            tag_corners, world_tags_m
        )
        rover_xy = None
        rover_heading_rad = None
        if H is not None:
            world_tag_frames += 1
            if transform_method == "four_tag_homography":
                four_tag_frames += 1
            elif transform_method == "three_tag_affine":
                three_tag_frames += 1
            if 0 in tag_corners:
                rover_corners = tag_corners[0]
                rover_center = _marker_center(rover_corners)
                rover_xy_arr = _apply_homography(H, rover_center)
                rover_xy = [float(rover_xy_arr[0]), float(rover_xy_arr[1])]
                rover_positions.append(rover_xy)
                top_mid = 0.5 * (rover_corners[0] + rover_corners[1])
                top_world = _apply_homography(H, top_mid)
                direction = top_world - rover_xy_arr
                rover_heading_rad = float(math.atan2(direction[1], direction[0]))

        frame_summaries.append(
            {
                "frame_index": frame_index,
                "time_s": frame_index / fps if fps > 0 else None,
                "ids": sorted(ids),
                "world_tags_seen": world_seen,
                "rover_xy_m": rover_xy,
                "rover_heading_rad": rover_heading_rad,
                "transform_method": transform_method,
                "reference_tags": reference_tags,
                "rover_tracking_status": rover_tracking_status,
                "frames_since_rover_decode": frames_since_rover_decode,
                "optical_flow_error_px": optical_flow_error_px,
                "_rover_center_px": (
                    _marker_center(tag_corners[0]).tolist()
                    if 0 in tag_corners
                    else None
                ),
                "_rover_top_mid_px": (
                    (0.5 * (tag_corners[0][0] + tag_corners[0][1])).tolist()
                    if 0 in tag_corners
                    else None
                ),
            }
        )
        if ids:
            accepted += 1

        if accepted <= preview_count and ids:
            annotated = frame.copy()
            cv2.aruco.drawDetectedMarkers(
                annotated,
                [tag_corners[tag_id].reshape(1, 4, 2).astype(np.float32) for tag_id in ids],
                np.asarray(ids, dtype=np.int32).reshape(-1, 1),
            )
            cv2.imwrite(
                str(preview_dir / f"detected_frame_{accepted:02d}.jpg"),
                annotated,
            )

        if attempted >= max_frames:
            break

    cap.release()

    # The phone is fixed. A single projective map estimated from median fixed-tag
    # centers is more stable than a per-frame three-point affine fallback.
    static_image_points = []
    static_world_points = []
    for tag_id, world_xy in world_tags_m.items():
        samples = fixed_tag_centers[tag_id]
        if not samples:
            continue
        static_image_points.append(np.median(np.asarray(samples), axis=0))
        static_world_points.append(world_xy)
    static_H = None
    if len(static_image_points) >= 4:
        static_H, _ = cv2.findHomography(
            np.asarray(static_image_points, dtype=np.float64),
            np.asarray(static_world_points, dtype=np.float64),
            method=0,
        )
    if static_H is not None:
        rover_positions.clear()
        for summary in frame_summaries:
            center_px = summary.pop("_rover_center_px")
            top_mid_px = summary.pop("_rover_top_mid_px")
            if center_px is None:
                summary["rover_xy_m"] = None
                summary["rover_heading_rad"] = None
                continue
            rover_xy_arr = _apply_homography(
                static_H, np.asarray(center_px, dtype=np.float64)
            )
            top_world = _apply_homography(
                static_H, np.asarray(top_mid_px, dtype=np.float64)
            )
            direction = top_world - rover_xy_arr
            rover_xy = [float(rover_xy_arr[0]), float(rover_xy_arr[1])]
            summary["rover_xy_m"] = rover_xy
            summary["rover_heading_rad"] = float(
                math.atan2(direction[1], direction[0])
            )
            summary["transform_method"] = "static_four_tag_homography"
            summary["reference_tags"] = sorted(world_tags_m)
            rover_positions.append(rover_xy)
    else:
        for summary in frame_summaries:
            summary.pop("_rover_center_px")
            summary.pop("_rover_top_mid_px")

    rover_summary = None
    if rover_positions:
        rover = np.asarray(rover_positions, dtype=np.float64)
        rover_summary = {
            "samples": int(len(rover)),
            "x_median_m": float(np.median(rover[:, 0])),
            "y_median_m": float(np.median(rover[:, 1])),
            "x_span_m": float(np.max(rover[:, 0]) - np.min(rover[:, 0])),
            "y_span_m": float(np.max(rover[:, 1]) - np.min(rover[:, 1])),
        }

    payload: dict[str, object] = {
        "schema": "ugv01_apriltag_still_check_v1",
        "video_path": str(video_path),
        "calibration_path": str(calibration_path) if calibration_path.exists() else None,
        "world_tags_m": {str(k): v for k, v in world_tags_m.items()},
        "video": {
            "image_size_px": image_size,
            "fps": fps,
            "frame_count": frame_count,
        },
        "sampling": {
            "sample_stride": sample_stride,
            "attempted_frames": attempted,
            "frames_with_any_tags": accepted,
            "frames_with_world_transform": world_tag_frames,
            "frames_with_all_world_tags": four_tag_frames,
            "frames_with_three_tag_fallback": three_tag_frames,
        },
        "tag_detection_counts": dict(sorted(id_counts.items())),
        "rover_tracking_counts": dict(sorted(rover_tracking_counts.items())),
        "rover_still_summary": rover_summary,
        "frame_summaries": frame_summaries,
    }

    (output_dir / "apriltag_still_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "apriltag_still_report.md").write_text(
        _format_report(payload),
        encoding="utf-8",
    )
    return payload


def _format_report(payload: dict[str, object]) -> str:
    sampling = payload["sampling"]
    video = payload["video"]
    counts = payload["tag_detection_counts"]
    tracking = payload.get("rover_tracking_counts", {})
    rover = payload["rover_still_summary"]
    recovered = int(tracking.get("optical_flow", 0)) + int(
        tracking.get("rejected_quad", 0)
    )
    tracked = int(tracking.get("decoded", 0)) + recovered
    lines = [
        "# AprilTag Still-Video Check",
        "",
        f"- Video: `{payload['video_path']}`",
        f"- Calibration: `{payload['calibration_path']}`",
        f"- Resolution: `{video['image_size_px']}`",
        f"- FPS: `{float(video['fps']):.3f}`",
        f"- Attempted sampled frames: `{sampling['attempted_frames']}`",
        f"- Frames with any tags: `{sampling['frames_with_any_tags']}`",
        f"- Frames with a world transform: `{sampling['frames_with_world_transform']}`",
        f"- Frames with all world tags: `{sampling['frames_with_all_world_tags']}`",
        f"- Frames using three-tag fallback: `{sampling['frames_with_three_tag_fallback']}`",
        f"- Rover ID 0 decoded frames: `{tracking.get('decoded', 0)}`",
        f"- Rover ID 0 temporally recovered frames: `{recovered}`",
        f"- Rover ID 0 total tracked frames: `{tracked}`",
        "",
        "## Tag Detection Counts",
        "",
        "| Tag ID | Frames Detected | Role |",
        "|---:|---:|---|",
    ]
    roles = {
        0: "rover",
        1: "top-left world",
        2: "top-right world",
        3: "bottom-right world",
        4: "bottom-left world",
    }
    for tag_id, count in counts.items():
        lines.append(f"| {tag_id} | {count} | {roles.get(int(tag_id), '')} |")
    lines.extend(["", "## Rover Still Position", ""])
    if rover:
        lines.extend(
            [
                f"- Samples: `{rover['samples']}`",
                f"- Median position: `({rover['x_median_m']:.3f}, {rover['y_median_m']:.3f}) m`",
                f"- Still jitter span: `x={rover['x_span_m']:.3f} m`, `y={rover['y_span_m']:.3f} m`",
            ]
        )
    else:
        lines.append("- Rover tag ID `0` was not mapped into the world frame.")
    lines.extend(["", "## World Frame", ""])
    for tag_id, xy in payload["world_tags_m"].items():
        lines.append(f"- ID {tag_id}: `({float(xy[0]):.4f}, {float(xy[1]):.4f}) m`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--sample-stride", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument(
        "--world-layout", choices=sorted(WORLD_LAYOUTS), default="rectangle"
    )
    args = parser.parse_args()

    payload = analyze_still_video(
        args.video_path,
        args.output_dir,
        args.calibration,
        sample_stride=args.sample_stride,
        max_frames=args.max_frames,
        preview_count=args.preview_count,
        world_tags_m=WORLD_LAYOUTS[args.world_layout],
    )
    print(args.output_dir / "apriltag_still_summary.json")
    print(args.output_dir / "apriltag_still_report.md")
    print("tag_counts=", payload["tag_detection_counts"])
    print("rover=", payload["rover_still_summary"])


if __name__ == "__main__":
    main()
