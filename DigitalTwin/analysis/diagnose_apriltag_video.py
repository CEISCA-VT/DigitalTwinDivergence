"""Diagnose missing rover AprilTag detections in UGV01 videos.

The normal tracker intentionally assumes a tag36h11 rover marker with ID 0.
This diagnostic keeps that scientific convention fixed while testing likely
failure mechanisms: wrong AprilTag family, duplicate IDs, motion blur or poor
contrast, calibration/resolution mismatch, undistortion damage, and tags close
to the image boundary.  It never repairs or fabricates a ground-truth path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


DEFAULT_CALIBRATION = Path(
    "DigitalTwin/datasets/analysis/camera_calibration_landscape/"
    "camera_calibration_charuco.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "DigitalTwin/datasets/analysis/apriltag_video_diagnostic"
)
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def available_apriltag_families() -> dict[str, int]:
    """Return OpenCV AprilTag dictionaries available in this installation."""

    names = {
        "tag16h5": "DICT_APRILTAG_16h5",
        "tag25h9": "DICT_APRILTAG_25h9",
        "tag36h10": "DICT_APRILTAG_36h10",
        "tag36h11": "DICT_APRILTAG_36h11",
    }
    return {
        name: int(getattr(cv2.aruco, constant))
        for name, constant in names.items()
        if hasattr(cv2.aruco, constant)
    }


def make_detector(dictionary_id: int) -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary_id), parameters
    )


def detect_markers(
    image: np.ndarray, detector: cv2.aruco.ArucoDetector
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """Detect markers and normalize OpenCV's optional outputs."""

    corners, ids_array, rejected = detector.detectMarkers(image)
    ids = [] if ids_array is None else [int(value) for value in ids_array.reshape(-1)]
    normalized_corners = [
        np.asarray(marker, dtype=np.float64).reshape(4, 2) for marker in corners
    ]
    normalized_rejected = [
        np.asarray(marker, dtype=np.float64).reshape(4, 2) for marker in rejected
    ]
    return ids, normalized_corners, normalized_rejected


def load_calibration(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration = payload.get("calibration", {})
    if "camera_matrix" not in calibration or "distortion_coefficients" not in calibration:
        raise ValueError(f"{path} does not contain a usable camera calibration")
    return payload


def calibration_diagnostic(
    payload: dict[str, Any] | None, frame_width: int, frame_height: int
) -> dict[str, Any]:
    if payload is None:
        return {
            "available": False,
            "video_resolution_px": [frame_width, frame_height],
            "calibration_resolution_px": None,
            "resolution_match": None,
        }
    video = payload.get("video", {})
    calibration_width = int(video.get("image_width_px", 0) or 0)
    calibration_height = int(video.get("image_height_px", 0) or 0)
    known_resolution = calibration_width > 0 and calibration_height > 0
    return {
        "available": True,
        "video_resolution_px": [frame_width, frame_height],
        "calibration_resolution_px": (
            [calibration_width, calibration_height] if known_resolution else None
        ),
        "resolution_match": (
            calibration_width == frame_width and calibration_height == frame_height
            if known_resolution
            else None
        ),
        "rms_reprojection_error_px": payload.get("calibration", {}).get(
            "rms_reprojection_error_px"
        ),
        "source_video": payload.get("video_path"),
    }


def undistort_frame(frame: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    calibration = payload["calibration"]
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        calibration["distortion_coefficients"], dtype=np.float64
    )
    return cv2.undistort(frame, camera_matrix, distortion)


def preprocess_variants(gray: np.ndarray) -> dict[str, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0.0)
    return {"raw": gray, "clahe": clahe, "sharpened": sharpened}


def frame_quality(gray: np.ndarray) -> dict[str, float]:
    return {
        "mean_luma": float(np.mean(gray)),
        "luma_std": float(np.std(gray)),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_fraction": float(np.mean(gray <= 15)),
        "bright_fraction": float(np.mean(gray >= 240)),
    }


def marker_geometry(
    marker: np.ndarray, frame_width: int, frame_height: int
) -> dict[str, float]:
    marker = np.asarray(marker, dtype=np.float64).reshape(4, 2)
    area = abs(float(cv2.contourArea(marker.astype(np.float32))))
    perimeter = float(cv2.arcLength(marker.astype(np.float32), True))
    minimum_edge_margin = float(
        min(
            np.min(marker[:, 0]),
            np.min(marker[:, 1]),
            frame_width - 1 - np.max(marker[:, 0]),
            frame_height - 1 - np.max(marker[:, 1]),
        )
    )
    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "minimum_edge_margin_px": minimum_edge_margin,
    }


def _count_json(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _rate(count: int, attempted: int) -> float:
    return float(count / attempted) if attempted else 0.0


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _annotate_frame(
    frame: np.ndarray,
    ids: list[int],
    corners: list[np.ndarray],
    text: list[str],
) -> np.ndarray:
    annotated = frame.copy()
    if ids:
        cv2.aruco.drawDetectedMarkers(
            annotated,
            [marker.reshape(1, 4, 2).astype(np.float32) for marker in corners],
            np.asarray(ids, dtype=np.int32).reshape(-1, 1),
        )
    y = 28
    for line in text:
        cv2.putText(
            annotated,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 25
    return annotated


def infer_findings(summary: dict[str, Any], target_id: int) -> list[dict[str, str]]:
    """Generate conservative, evidence-linked diagnostic findings."""

    findings: list[dict[str, str]] = []
    attempted = int(summary["sampling"]["attempted_frames"])
    raw_count = int(summary["expected_family"]["raw_target_detections"])
    undistorted_count = int(
        summary["expected_family"].get("undistorted_target_detections", 0)
    )
    duplicate_frames = int(summary["expected_family"]["duplicate_target_frames"])
    calibration = summary["calibration"]

    if calibration.get("resolution_match") is False:
        findings.append(
            {
                "severity": "error",
                "code": "calibration_resolution_mismatch",
                "message": (
                    "The ChArUco calibration resolution differs from the video. "
                    "Do not use this calibration without recalibrating in the same video mode."
                ),
            }
        )
    if duplicate_frames:
        findings.append(
            {
                "severity": "error",
                "code": "duplicate_target_id",
                "message": (
                    f"Multiple ID {target_id} markers were decoded in {duplicate_frames} "
                    "frames. The current tracker keeps only one marker per ID, so the rover "
                    "pose can jump between physical tags."
                ),
            }
        )

    alternative_hits = {
        family: int(values["target_detections"])
        for family, values in summary["alternative_families"].items()
    }
    best_alternative = max(alternative_hits, key=alternative_hits.get, default=None)
    best_alternative_count = alternative_hits.get(best_alternative, 0)
    if best_alternative_count >= max(5, 2 * max(raw_count, 1)):
        findings.append(
            {
                "severity": "error",
                "code": "possible_wrong_apriltag_family",
                "message": (
                    f"ID {target_id} was decoded {best_alternative_count} times as "
                    f"{best_alternative}, versus {raw_count} times as tag36h11. "
                    "Verify that the rover print came from the repository tag36h11 asset."
                ),
            }
        )

    variant_hits = summary["expected_family"]["preprocessing_target_detections"]
    best_variant = max(variant_hits, key=variant_hits.get, default="raw")
    best_variant_count = int(variant_hits.get(best_variant, 0))
    if best_variant != "raw" and best_variant_count >= max(5, 2 * max(raw_count, 1)):
        findings.append(
            {
                "severity": "warning",
                "code": "contrast_or_sharpness_sensitive",
                "message": (
                    f"{best_variant} preprocessing recovered ID {target_id} in "
                    f"{best_variant_count} frames versus {raw_count} raw frames. "
                    "Lighting, focus, glare, or motion blur is likely limiting detection."
                ),
            }
        )

    if calibration.get("available") and raw_count > 0 and undistorted_count < 0.6 * raw_count:
        findings.append(
            {
                "severity": "warning",
                "code": "undistortion_reduces_detection",
                "message": (
                    f"Raw frames decoded ID {target_id} {raw_count} times, but calibrated "
                    f"frames decoded it only {undistorted_count} times. Inspect calibration "
                    "resolution, lens/video mode, and reprojection error."
                ),
            }
        )

    fixed_counts = {
        int(key): int(value)
        for key, value in summary["expected_family"]["raw_id_counts"].items()
        if int(key) != target_id
    }
    strong_fixed = sum(count >= 0.8 * attempted for count in fixed_counts.values())
    if raw_count == 0 and strong_fixed >= 3 and best_alternative_count < 5:
        findings.append(
            {
                "severity": "error",
                "code": "target_print_mount_or_visibility_failure",
                "message": (
                    f"At least {strong_fixed} fixed tag36h11 markers decode reliably while "
                    f"ID {target_id} never decodes. This localizes the problem to the rover "
                    "tag print, white border, flatness, orientation to camera, occlusion, or "
                    "visibility—not the detector family or the complete video."
                ),
            }
        )
    elif attempted and _rate(raw_count, attempted) < 0.8:
        findings.append(
            {
                "severity": "error",
                "code": "target_coverage_too_low",
                "message": (
                    f"ID {target_id} direct detection coverage is "
                    f"{100.0 * _rate(raw_count, attempted):.1f}%, below the recommended "
                    "80% pilot acceptance level. Do not use this video as final ground truth."
                ),
            }
        )

    edge_margin = summary["expected_family"].get("target_edge_margin_px_median")
    if edge_margin is not None and edge_margin < 12.0:
        findings.append(
            {
                "severity": "warning",
                "code": "target_near_frame_boundary",
                "message": (
                    f"The median decoded target margin is only {edge_margin:.1f} px. "
                    "The rover tag is frequently cropped or too close to the image boundary."
                ),
            }
        )

    if not findings:
        findings.append(
            {
                "severity": "info",
                "code": "no_decisive_failure_identified",
                "message": (
                    "No single failure mechanism dominated. Inspect the annotated frames, "
                    "per-frame quality CSV, print geometry, and camera placement."
                ),
            }
        )
    return findings


def diagnose_video(
    video_path: Path,
    output_dir: Path,
    *,
    calibration_path: Path | None,
    target_id: int,
    sample_stride: int,
    max_frames: int,
    preview_count: int,
) -> dict[str, Any]:
    families = available_apriltag_families()
    if "tag36h11" not in families:
        raise RuntimeError("OpenCV installation does not provide DICT_APRILTAG_36h11")
    detectors = {name: make_detector(value) for name, value in families.items()}
    expected_detector = detectors["tag36h11"]
    calibration_payload = load_calibration(calibration_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "annotated_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)

    raw_counts: Counter[int] = Counter()
    undistorted_counts: Counter[int] = Counter()
    family_counts: dict[str, Counter[int]] = {
        family: Counter() for family in families if family != "tag36h11"
    }
    preprocessing_target_counts: Counter[str] = Counter()
    duplicate_target_frames = 0
    target_areas: list[float] = []
    target_perimeters: list[float] = []
    target_edge_margins: list[float] = []
    frame_rows: list[dict[str, Any]] = []
    attempted = 0
    frame_index = -1
    saved_previews = 0

    while attempted < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % sample_stride:
            continue
        attempted += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quality = frame_quality(gray)
        variants = preprocess_variants(gray)

        raw_ids, raw_corners, rejected = detect_markers(gray, expected_detector)
        raw_counts.update(raw_ids)
        target_occurrences = raw_ids.count(target_id)
        if target_occurrences > 1:
            duplicate_target_frames += 1
        target_geometry = None
        for marker_id, marker in zip(raw_ids, raw_corners):
            if marker_id != target_id:
                continue
            geometry = marker_geometry(marker, frame_width, frame_height)
            target_areas.append(geometry["area_px2"])
            target_perimeters.append(geometry["perimeter_px"])
            target_edge_margins.append(geometry["minimum_edge_margin_px"])
            target_geometry = geometry

        variant_target_hits: dict[str, int] = {}
        for variant_name, variant in variants.items():
            ids, _, _ = detect_markers(variant, expected_detector)
            hits = ids.count(target_id)
            preprocessing_target_counts[variant_name] += hits
            variant_target_hits[variant_name] = hits

        alternate_hits: dict[str, int] = {}
        for family, detector in detectors.items():
            if family == "tag36h11":
                continue
            ids, _, _ = detect_markers(gray, detector)
            family_counts[family].update(ids)
            alternate_hits[family] = ids.count(target_id)

        undistorted_target_hits = 0
        if calibration_payload is not None:
            calibrated_frame = undistort_frame(frame, calibration_payload)
            calibrated_gray = cv2.cvtColor(calibrated_frame, cv2.COLOR_BGR2GRAY)
            calibrated_ids, _, _ = detect_markers(calibrated_gray, expected_detector)
            undistorted_counts.update(calibrated_ids)
            undistorted_target_hits = calibrated_ids.count(target_id)

        row: dict[str, Any] = {
            "frame_index": frame_index,
            "time_s": frame_index / fps if fps else None,
            "raw_ids": ";".join(map(str, raw_ids)),
            "raw_target_hits": target_occurrences,
            "undistorted_target_hits": undistorted_target_hits,
            "rejected_quad_count": len(rejected),
            **quality,
            "target_area_px2": target_geometry["area_px2"] if target_geometry else None,
            "target_perimeter_px": (
                target_geometry["perimeter_px"] if target_geometry else None
            ),
            "target_edge_margin_px": (
                target_geometry["minimum_edge_margin_px"] if target_geometry else None
            ),
        }
        for name, hits in variant_target_hits.items():
            row[f"{name}_target_hits"] = hits
        for family, hits in alternate_hits.items():
            row[f"{family}_target_hits"] = hits
        frame_rows.append(row)

        interesting = target_occurrences > 0 or any(alternate_hits.values())
        if saved_previews < preview_count and (interesting or saved_previews < 3):
            annotated = _annotate_frame(
                frame,
                raw_ids,
                raw_corners,
                [
                    f"frame={frame_index} time={row['time_s']:.2f}s",
                    f"tag36h11 IDs={raw_ids or 'none'} rejected={len(rejected)}",
                    f"blur={quality['laplacian_variance']:.1f} luma={quality['mean_luma']:.1f}",
                ],
            )
            cv2.imwrite(
                str(preview_dir / f"frame_{frame_index:07d}.jpg"), annotated
            )
            saved_previews += 1

    capture.release()

    calibration_result = calibration_diagnostic(
        calibration_payload, frame_width, frame_height
    )
    summary: dict[str, Any] = {
        "schema": "ugv01_apriltag_video_diagnostic_v1",
        "video": {
            "path": str(video_path),
            "fps": fps,
            "frame_count": total_frames,
            "resolution_px": [frame_width, frame_height],
        },
        "sampling": {
            "sample_stride": sample_stride,
            "max_frames": max_frames,
            "attempted_frames": attempted,
        },
        "target": {"family": "tag36h11", "id": target_id},
        "calibration": calibration_result,
        "expected_family": {
            "raw_id_counts": _count_json(raw_counts),
            "raw_target_detections": int(raw_counts[target_id]),
            "raw_target_frame_coverage": _rate(
                sum(int(row["raw_target_hits"] > 0) for row in frame_rows), attempted
            ),
            "undistorted_id_counts": _count_json(undistorted_counts),
            "undistorted_target_detections": int(undistorted_counts[target_id]),
            "preprocessing_target_detections": {
                key: int(value) for key, value in sorted(preprocessing_target_counts.items())
            },
            "duplicate_target_frames": duplicate_target_frames,
            "target_area_px2_median": _median(target_areas),
            "target_perimeter_px_median": _median(target_perimeters),
            "target_edge_margin_px_median": _median(target_edge_margins),
        },
        "alternative_families": {
            family: {
                "id_counts": _count_json(counts),
                "target_detections": int(counts[target_id]),
            }
            for family, counts in family_counts.items()
        },
        "frame_quality": {
            "laplacian_variance_median": _median(
                [float(row["laplacian_variance"]) for row in frame_rows]
            ),
            "mean_luma_median": _median([float(row["mean_luma"]) for row in frame_rows]),
            "luma_std_median": _median([float(row["luma_std"]) for row in frame_rows]),
        },
        "artifacts": {
            "frame_csv": str(output_dir / "apriltag_video_frame_diagnostics.csv"),
            "annotated_frames": str(preview_dir),
        },
    }
    summary["findings"] = infer_findings(summary, target_id)

    if frame_rows:
        with (output_dir / "apriltag_video_frame_diagnostics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)

    (output_dir / "apriltag_video_diagnostic.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "apriltag_video_diagnostic_report.md").write_text(
        format_report(summary), encoding="utf-8"
    )
    return summary


def format_report(summary: dict[str, Any]) -> str:
    expected = summary["expected_family"]
    calibration = summary["calibration"]
    attempted = summary["sampling"]["attempted_frames"]
    lines = [
        "# UGV01 AprilTag Video Diagnostic",
        "",
        "This diagnostic evaluates detection evidence only. It does not repair missing poses or certify ground truth.",
        "",
        "## Input",
        "",
        f"- Video: `{summary['video']['path']}`",
        f"- Resolution: `{summary['video']['resolution_px'][0]} x {summary['video']['resolution_px'][1]}` px",
        f"- Sampled frames: `{attempted}`",
        f"- Intended rover marker: `tag36h11 ID {summary['target']['id']}`",
        "",
        "## Direct Detection",
        "",
        f"- Raw ID counts: `{expected['raw_id_counts']}`",
        f"- Rover detections: `{expected['raw_target_detections']}`",
        f"- Rover frame coverage: `{100.0 * expected['raw_target_frame_coverage']:.2f}%`",
        f"- Duplicate rover-ID frames: `{expected['duplicate_target_frames']}`",
        f"- Preprocessing counts: `{expected['preprocessing_target_detections']}`",
        "",
        "## Calibration Check",
        "",
        f"- Calibration available: `{calibration['available']}`",
        f"- Video resolution: `{calibration['video_resolution_px']}`",
        f"- Calibration resolution: `{calibration['calibration_resolution_px']}`",
        f"- Resolution match: `{calibration['resolution_match']}`",
        f"- Raw/undistorted rover detections: `{expected['raw_target_detections']} / {expected['undistorted_target_detections']}`",
        "",
        "## Alternate-Family Test",
        "",
        "| Family | ID 0 detections | All decoded IDs |",
        "|---|---:|---|",
    ]
    for family, result in summary["alternative_families"].items():
        lines.append(
            f"| {family} | {result['target_detections']} | `{result['id_counts']}` |"
        )
    lines.extend(["", "## Findings", ""])
    for finding in summary["findings"]:
        lines.append(
            f"- **{finding['severity'].upper()} — {finding['code']}:** {finding['message']}"
        )
    lines.extend(
        [
            "",
            "## Evidence Files",
            "",
            "- `apriltag_video_frame_diagnostics.csv`: per-frame quality and detection results.",
            "- `annotated_frames/`: decoded IDs drawn on representative raw frames.",
            "- `apriltag_video_diagnostic.json`: complete machine-readable result.",
            "",
        ]
    )
    return "\n".join(lines)


def discover_videos(input_path: Path) -> list[Path]:
    """Resolve a video file or recursively discover videos below a directory."""

    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    return sorted(
        (
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def _video_output_name(video_path: Path, input_dir: Path) -> str:
    relative = video_path.relative_to(input_dir)
    # Include the extension to avoid collisions such as run01.mp4 and run01.mov.
    name = "__".join(relative.parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "video"


def _format_directory_report(batch: dict[str, Any]) -> str:
    lines = [
        "# UGV01 AprilTag Directory Diagnostic",
        "",
        f"- Input directory: `{batch['input_directory']}`",
        f"- Videos discovered: `{batch['video_count']}`",
        f"- Successfully analyzed: `{batch['successful_count']}`",
        f"- Failed to analyze: `{batch['failed_count']}`",
        "",
        "## Per-video summary",
        "",
        "| Video | Status | Sampled frames | ID 0 coverage | Direct ID 0 detections | Main finding codes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in batch["results"]:
        if result["status"] == "failed":
            lines.append(
                f"| `{result['relative_video']}` | failed | — | — | — | "
                f"`{result['error']}` |"
            )
            continue
        codes = ", ".join(result["finding_codes"]) or "none"
        lines.append(
            f"| `{result['relative_video']}` | ok | {result['attempted_frames']} | "
            f"{100.0 * result['target_frame_coverage']:.2f}% | "
            f"{result['raw_target_detections']} | `{codes}` |"
        )
    lines.extend(
        [
            "",
            "Each successful video's folder contains its complete Markdown, JSON, CSV, "
            "and annotated-frame evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def diagnose_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    calibration_path: Path | None,
    target_id: int,
    sample_stride: int,
    max_frames: int,
    preview_count: int,
) -> dict[str, Any]:
    """Diagnose every supported video below input_dir and aggregate results."""

    videos = discover_videos(input_dir)
    if not videos:
        extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"no supported videos found below {input_dir}; expected: {extensions}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for video_path in videos:
        relative_video = str(video_path.relative_to(input_dir))
        video_output = output_dir / "videos" / _video_output_name(video_path, input_dir)
        try:
            summary = diagnose_video(
                video_path,
                video_output,
                calibration_path=calibration_path,
                target_id=target_id,
                sample_stride=sample_stride,
                max_frames=max_frames,
                preview_count=preview_count,
            )
            expected = summary["expected_family"]
            results.append(
                {
                    "relative_video": relative_video,
                    "status": "ok",
                    "output_directory": str(video_output),
                    "attempted_frames": summary["sampling"]["attempted_frames"],
                    "raw_target_detections": expected["raw_target_detections"],
                    "target_frame_coverage": expected["raw_target_frame_coverage"],
                    "duplicate_target_frames": expected["duplicate_target_frames"],
                    "resolution_match": summary["calibration"]["resolution_match"],
                    "finding_codes": [item["code"] for item in summary["findings"]],
                }
            )
        except Exception as exc:  # Keep a bad/corrupt video from aborting the batch.
            results.append(
                {
                    "relative_video": relative_video,
                    "status": "failed",
                    "output_directory": str(video_output),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    successful = sum(result["status"] == "ok" for result in results)
    batch: dict[str, Any] = {
        "schema": "ugv01_apriltag_directory_diagnostic_v1",
        "input_directory": str(input_dir),
        "video_count": len(videos),
        "successful_count": successful,
        "failed_count": len(videos) - successful,
        "results": results,
    }
    with (output_dir / "apriltag_directory_diagnostic.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "relative_video",
            "status",
            "output_directory",
            "attempted_frames",
            "raw_target_detections",
            "target_frame_coverage",
            "duplicate_target_frames",
            "resolution_match",
            "finding_codes",
            "error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["finding_codes"] = ";".join(row.get("finding_codes", []))
            writer.writerow({key: row.get(key) for key in fieldnames})
    (output_dir / "apriltag_directory_diagnostic.json").write_text(
        json.dumps(batch, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "apriltag_directory_diagnostic_report.md").write_text(
        _format_directory_report(batch), encoding="utf-8"
    )
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="One video file, or a directory recursively containing videos.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--preview-count", type=int, default=16)
    args = parser.parse_args()
    if args.sample_stride < 1 or args.max_frames < 1 or args.preview_count < 0:
        raise SystemExit("sample stride/max frames must be positive; preview count cannot be negative")

    if args.input_path.is_dir():
        batch = diagnose_directory(
            args.input_path,
            args.output_dir,
            calibration_path=args.calibration,
            target_id=args.target_id,
            sample_stride=args.sample_stride,
            max_frames=args.max_frames,
            preview_count=args.preview_count,
        )
        print(args.output_dir / "apriltag_directory_diagnostic_report.md")
        print(args.output_dir / "apriltag_directory_diagnostic.json")
        print(
            "videos=",
            batch["video_count"],
            "successful=",
            batch["successful_count"],
            "failed=",
            batch["failed_count"],
        )
    else:
        summary = diagnose_video(
            args.input_path,
            args.output_dir,
            calibration_path=args.calibration,
            target_id=args.target_id,
            sample_stride=args.sample_stride,
            max_frames=args.max_frames,
            preview_count=args.preview_count,
        )
        print(args.output_dir / "apriltag_video_diagnostic_report.md")
        print(args.output_dir / "apriltag_video_diagnostic.json")
        print("raw_id_counts=", summary["expected_family"]["raw_id_counts"])
        print("findings=", [item["code"] for item in summary["findings"]])


if __name__ == "__main__":
    main()
