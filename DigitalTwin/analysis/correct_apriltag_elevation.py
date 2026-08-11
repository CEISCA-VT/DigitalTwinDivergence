"""Correct an elevated horizontal rover tag mapped through a floor homography."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from DigitalTwin.analysis.analyze_apriltag_still_video import _format_report


DEFAULT_CALIBRATION = Path(
    "DigitalTwin/datasets/analysis/camera_calibration/camera_calibration_charuco.json"
)


def _landscape_camera_model(
    calibration_payload: dict[str, object], width: int, height: int
) -> tuple[np.ndarray, np.ndarray, str]:
    portrait = np.asarray(
        calibration_payload["calibration"]["camera_matrix"], dtype=float
    )
    portrait_width = float(calibration_payload["video"]["image_width_px"])
    portrait_height = float(calibration_payload["video"]["image_height_px"])
    distortion = np.asarray(
        calibration_payload["calibration"]["distortion_coefficients"], dtype=float
    )
    if int(portrait_width) == width and int(portrait_height) == height:
        return portrait, distortion, "native_matching_resolution"
    scale_x = width / portrait_height
    scale_y = height / portrait_width
    # Tracking footage is the clockwise landscape rotation of the portrait
    # camera stream, followed by resizing to the encoded resolution.
    matrix = np.asarray(
        [
            [portrait[1, 1] * scale_x, 0.0, (portrait_height - 1.0 - portrait[1, 2]) * scale_x],
            [0.0, portrait[0, 0] * scale_y, portrait[0, 2] * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # Rotating tangential distortion coefficients requires an explicit image
    # coordinate transform. Keep this legacy fallback conservative; native
    # landscape calibration is preferred and uses the full distortion model.
    return matrix, np.zeros(5), "legacy_rotated_without_distortion"


def correct_summary(
    input_path: Path,
    output_dir: Path,
    calibration_path: Path,
    *,
    tag_size_m: float,
    tag_height_m: float,
    maximum_reprojection_error_px: float,
) -> dict[str, object]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    video_path = Path(payload["video_path"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters(),
    )
    fixed_ids = sorted(int(tag_id) for tag_id in payload["world_tags_m"])
    image_points = None
    for frame_index in range(300):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids_array, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        if ids_array is None:
            continue
        detected = {
            int(tag_id): np.asarray(marker, dtype=float).reshape(4, 2).mean(axis=0)
            for tag_id, marker in zip(ids_array.reshape(-1), corners)
        }
        if all(tag_id in detected for tag_id in fixed_ids):
            image_points = np.asarray([detected[tag_id] for tag_id in fixed_ids])
            break
    cap.release()
    if image_points is None:
        raise RuntimeError("could not find one frame containing all fixed tags")

    world_points = np.asarray(
        [
            [*payload["world_tags_m"][str(tag_id)], 0.0]
            for tag_id in fixed_ids
        ],
        dtype=float,
    )
    camera_matrix, distortion, camera_model_source = _landscape_camera_model(
        calibration, width, height
    )
    solved, rotation_vector, translation_vector = cv2.solvePnP(
        world_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise RuntimeError("could not solve the fixed-camera pose")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera_center = (-rotation.T @ translation_vector).reshape(3)
    projected, _ = cv2.projectPoints(
        world_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    reprojection_errors = np.linalg.norm(
        projected.reshape(-1, 2) - image_points, axis=1
    )
    mean_reprojection_error = float(np.mean(reprojection_errors))
    if mean_reprojection_error > maximum_reprojection_error_px:
        raise RuntimeError(
            "reference geometry is inconsistent with the camera model: "
            f"mean reprojection error {mean_reprojection_error:.2f} px exceeds "
            f"{maximum_reprojection_error_px:.2f} px"
        )
    if camera_center[2] <= tag_height_m:
        raise RuntimeError("estimated camera height is below the rover tag")

    perspective_scale = 1.0 - tag_height_m / float(camera_center[2])
    camera_xy = camera_center[:2]
    corrected_positions = []
    for row in payload["frame_summaries"]:
        xy = row.get("rover_xy_m")
        if xy is None:
            continue
        floor_intersection = np.asarray(xy, dtype=float)
        corrected = camera_xy + perspective_scale * (
            floor_intersection - camera_xy
        )
        row["rover_xy_m"] = corrected.tolist()
        corrected_positions.append(corrected)

    positions = np.asarray(corrected_positions, dtype=float)
    payload["rover_still_summary"] = {
        "samples": int(len(positions)),
        "x_median_m": float(np.median(positions[:, 0])),
        "y_median_m": float(np.median(positions[:, 1])),
        "x_span_m": float(np.ptp(positions[:, 0])),
        "y_span_m": float(np.ptp(positions[:, 1])),
    }
    payload["elevation_correction"] = {
        "schema": "ugv01_apriltag_elevation_correction_v1",
        "source_summary": str(input_path),
        "tag_size_m": tag_size_m,
        "tag_center_height_m": tag_height_m,
        "tag_center_offset_from_rover_center_m": [0.0, 0.0],
        "tag_tilt_deg": 0.0,
        "camera_center_world_m": camera_center.tolist(),
        "camera_height_m": float(camera_center[2]),
        "perspective_scale": perspective_scale,
        "mean_reference_reprojection_error_px": mean_reprojection_error,
        "maximum_reference_reprojection_error_px": float(
            np.max(reprojection_errors)
        ),
        "camera_matrix_landscape": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "camera_model_source": camera_model_source,
        "calibration_path": str(calibration_path),
        "heading_changed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "apriltag_still_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = _format_report(payload)
    report += (
        "\n## Elevation Correction\n\n"
        f"- Tag size: `{tag_size_m:.3f} m` square\n"
        f"- Tag-center height: `{tag_height_m:.5f} m`\n"
        f"- Camera height: `{camera_center[2]:.3f} m`\n"
        f"- Perspective scale: `{perspective_scale:.6f}`\n"
        f"- Mean reference reprojection error: `{mean_reprojection_error:.3f} px`\n"
        "- Tag offset and tilt: `0 m`, `0 deg`\n"
    )
    (output_dir / "apriltag_still_report.md").write_text(
        report, encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--tag-size-m", type=float, default=0.08)
    parser.add_argument("--tag-height-m", type=float, default=0.08112)
    parser.add_argument("--maximum-reprojection-error-px", type=float, default=5.0)
    args = parser.parse_args()
    payload = correct_summary(
        args.input,
        args.output_dir,
        args.calibration,
        tag_size_m=args.tag_size_m,
        tag_height_m=args.tag_height_m,
        maximum_reprojection_error_px=args.maximum_reprojection_error_px,
    )
    print(args.output_dir / "apriltag_still_summary.json")
    print(json.dumps(payload["elevation_correction"], indent=2))


if __name__ == "__main__":
    main()
