"""Calibrate phone-camera intrinsics from a ChArUco calibration video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.022
DEFAULT_OUTPUT_DIR = Path("DigitalTwin/datasets/analysis/camera_calibration")


def _make_board() -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )
    return board, dictionary


def _detector_params() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def _detect_charuco(
    frame: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
    detector_params: cv2.aruco.DetectorParameters,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "CharucoDetector"):
        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_detector.setDetectorParameters(detector_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            charuco_detector.detectBoard(gray)
        )
        marker_count = 0 if marker_ids is None else len(marker_ids)
    else:
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        marker_count = 0 if marker_ids is None else len(marker_ids)
        if marker_ids is None or marker_count == 0:
            return None, None, 0

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
    if charuco_ids is None or charuco_corners is None:
        return None, None, marker_count
    if len(charuco_ids) < 4:
        return None, None, marker_count
    return charuco_corners, charuco_ids, marker_count


def calibrate_from_video(
    video_path: Path,
    output_dir: Path,
    *,
    sample_stride: int,
    max_frames: int,
    min_corners: int,
    preview_count: int,
) -> dict[str, object]:
    board, dictionary = _make_board()
    detector_params = _detector_params()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    accepted_corners: list[np.ndarray] = []
    accepted_ids: list[np.ndarray] = []
    accepted_frames: list[dict[str, object]] = []
    attempted = 0
    frame_index = -1
    image_size: tuple[int, int] | None = None
    marker_counts: list[int] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % sample_stride != 0:
            continue
        attempted += 1
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        charuco_corners, charuco_ids, marker_count = _detect_charuco(
            frame,
            board,
            dictionary,
            detector_params,
        )
        marker_counts.append(marker_count)
        if charuco_ids is None or charuco_corners is None:
            continue
        corner_count = int(len(charuco_ids))
        if corner_count < min_corners:
            continue

        accepted_corners.append(charuco_corners)
        accepted_ids.append(charuco_ids)
        accepted_frames.append(
            {
                "frame_index": frame_index,
                "time_s": frame_index / fps if fps > 0 else None,
                "marker_count": marker_count,
                "charuco_corner_count": corner_count,
            }
        )

        if len(accepted_frames) <= preview_count:
            annotated = frame.copy()
            try:
                cv2.aruco.drawDetectedCornersCharuco(
                    annotated,
                    np.asarray(charuco_corners, dtype=np.float32),
                    np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1),
                    (0, 255, 0),
                )
            except cv2.error:
                for corner in np.asarray(charuco_corners).reshape(-1, 2):
                    cv2.circle(
                        annotated,
                        (int(corner[0]), int(corner[1])),
                        4,
                        (0, 255, 0),
                        -1,
                    )
            cv2.imwrite(
                str(preview_dir / f"accepted_frame_{len(accepted_frames):02d}.jpg"),
                annotated,
            )

        if len(accepted_frames) >= max_frames:
            break

    cap.release()

    if image_size is None:
        raise RuntimeError("video did not contain readable frames")
    if len(accepted_frames) < 6:
        raise RuntimeError(
            "not enough usable ChArUco views for calibration: "
            f"{len(accepted_frames)} accepted"
        )

    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
            cv2.aruco.calibrateCameraCharuco(
                accepted_corners,
                accepted_ids,
                board,
                image_size,
                None,
                None,
            )
        )
    else:
        chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)
        object_points = []
        image_points = []
        for corners, ids in zip(accepted_corners, accepted_ids):
            flat_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
            object_points.append(chessboard_corners[flat_ids].reshape(-1, 1, 3))
            image_points.append(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2))
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )

    focal_x = float(camera_matrix[0, 0])
    focal_y = float(camera_matrix[1, 1])
    principal_x = float(camera_matrix[0, 2])
    principal_y = float(camera_matrix[1, 2])
    fov_x_deg = float(2.0 * math.degrees(math.atan(image_size[0] / (2.0 * focal_x))))
    fov_y_deg = float(2.0 * math.degrees(math.atan(image_size[1] / (2.0 * focal_y))))

    payload: dict[str, object] = {
        "schema": "ugv01_charuco_camera_calibration_v1",
        "video_path": str(video_path),
        "board": {
            "dictionary": "DICT_5X5_100",
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length_m": SQUARE_LENGTH_M,
            "marker_length_m": MARKER_LENGTH_M,
        },
        "video": {
            "image_width_px": image_size[0],
            "image_height_px": image_size[1],
            "fps": fps,
            "frame_count": total_video_frames,
        },
        "sampling": {
            "sample_stride": sample_stride,
            "attempted_frames": attempted,
            "accepted_frames": len(accepted_frames),
            "min_corners": min_corners,
            "max_frames": max_frames,
            "marker_count_median": float(np.median(marker_counts)) if marker_counts else 0.0,
        },
        "calibration": {
            "rms_reprojection_error_px": float(rms),
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
            "focal_length_px": [focal_x, focal_y],
            "principal_point_px": [principal_x, principal_y],
            "field_of_view_deg": [fov_x_deg, fov_y_deg],
            "distortion_model": "OpenCV pinhole/radtan",
        },
        "accepted_frames": accepted_frames,
    }

    json_path = output_dir / "camera_calibration_charuco.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = output_dir / "camera_calibration_report.md"
    report_path.write_text(_format_report(payload), encoding="utf-8")
    return payload


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_report(payload: dict[str, object]) -> str:
    video = payload["video"]
    sampling = payload["sampling"]
    calibration = payload["calibration"]
    assert isinstance(video, dict)
    assert isinstance(sampling, dict)
    assert isinstance(calibration, dict)
    return "\n".join(
        [
            "# ChArUco Phone-Camera Calibration",
            "",
            "This report summarizes the camera intrinsic calibration from the ChArUco footage.",
            "",
            "## Input",
            "",
            f"- Video: `{payload['video_path']}`",
            f"- Resolution: `{video['image_width_px']} x {video['image_height_px']}` px",
            f"- FPS: `{_fmt(video['fps'])}`",
            f"- Video frames: `{video['frame_count']}`",
            "",
            "## Detection",
            "",
            f"- Sample stride: `{sampling['sample_stride']}`",
            f"- Attempted sampled frames: `{sampling['attempted_frames']}`",
            f"- Accepted ChArUco views: `{sampling['accepted_frames']}`",
            f"- Minimum accepted corners per view: `{sampling['min_corners']}`",
            f"- Median detected ArUco markers per sampled frame: `{_fmt(sampling['marker_count_median'])}`",
            "",
            "## Calibration",
            "",
            f"- RMS reprojection error: `{_fmt(calibration['rms_reprojection_error_px'])}` px",
            f"- Focal length: `{_fmt(calibration['focal_length_px'][0])}, {_fmt(calibration['focal_length_px'][1])}` px",
            f"- Principal point: `{_fmt(calibration['principal_point_px'][0])}, {_fmt(calibration['principal_point_px'][1])}` px",
            f"- Field of view: `{_fmt(calibration['field_of_view_deg'][0])}, {_fmt(calibration['field_of_view_deg'][1])}` deg",
            "",
            "## Next Step",
            "",
            "Record a fixed-phone AprilTag route video with world tags visible, then use this calibration JSON to undistort frames and map the rover tag into the measured floor frame.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-stride", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--min-corners", type=int, default=10)
    parser.add_argument("--preview-count", type=int, default=8)
    args = parser.parse_args()

    payload = calibrate_from_video(
        args.video_path,
        args.output_dir,
        sample_stride=args.sample_stride,
        max_frames=args.max_frames,
        min_corners=args.min_corners,
        preview_count=args.preview_count,
    )
    print(args.output_dir / "camera_calibration_charuco.json")
    print(args.output_dir / "camera_calibration_report.md")
    print(
        "accepted_frames=",
        payload["sampling"]["accepted_frames"],
        "rms_px=",
        payload["calibration"]["rms_reprojection_error_px"],
    )


if __name__ == "__main__":
    main()
