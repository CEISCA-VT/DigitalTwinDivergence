#!/usr/bin/env python3
"""Fast batch audit for UGV01 AprilTag validation videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


DEFAULTS: dict[str, Any] = {
    "moving_tag_ids": [0],
    "reference_tag_ids": [],
    "expected_runs": 5,
    "minimum_run_seconds": 90.0,
    "sample_hz": 4.0,
    "minimum_moving_tag_coverage": 0.85,
    "warning_moving_tag_coverage": 0.65,
    "maximum_gap_seconds": 0.5,
    "warning_gap_seconds": 1.0,
    "minimum_tag_side_pixels": 24.0,
    "minimum_sharp_fraction": 0.70,
    "blur_laplacian_threshold": 45.0,
    "minimum_median_brightness": 35.0,
    "maximum_median_brightness": 220.0,
}


@dataclass
class VideoResult:
    path: str
    status: str = "FAIL"
    duration_s: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    sampled_frames: int = 0
    decoded_frames: int = 0
    decode_fraction: float = 0.0
    moving_tag_frames: int = 0
    moving_tag_coverage: float = 0.0
    any_tag_coverage: float = 0.0
    reference_tag_coverage: float | None = None
    longest_moving_tag_gap_s: float = 0.0
    median_moving_tag_side_px: float = 0.0
    p10_moving_tag_side_px: float = 0.0
    duplicate_moving_id_frames: int = 0
    median_brightness: float = 0.0
    sharp_frame_fraction: float = 0.0
    median_laplacian: float = 0.0
    detected_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    error: str | None = None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - position) + ordered[hi] * (position - lo))


def _detector(cv2: Any) -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def _longest_false_run(flags: list[bool], sample_period: float) -> float:
    longest = current = 0
    for present in flags:
        current = 0 if present else current + 1
        longest = max(longest, current)
    return longest * sample_period


def _audit_one(path_text: str, cfg: dict[str, Any]) -> VideoResult:
    result = VideoResult(path=path_text)
    try:
        import cv2
        import numpy as np

        cv2.setNumThreads(1)
        path = Path(path_text)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            result.failures.append("video could not be opened")
            return result

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result.fps = fps if math.isfinite(fps) else 0.0
        result.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if result.fps > 0 and frame_count > 0:
            result.duration_s = frame_count / result.fps

        sample_hz = float(cfg["sample_hz"])
        step = max(1, int(round(result.fps / sample_hz))) if result.fps > 0 else 1
        effective_hz = result.fps / step if result.fps > 0 else sample_hz
        sample_period = 1.0 / effective_hz
        detector = _detector(cv2)
        moving_ids = set(int(x) for x in cfg["moving_tag_ids"])
        reference_ids = set(int(x) for x in cfg["reference_tag_ids"])

        brightness: list[float] = []
        laplacians: list[float] = []
        moving_sides: list[float] = []
        moving_present: list[bool] = []
        any_present = 0
        reference_present = 0
        all_ids: set[int] = set()
        index = 0

        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            if index % step != 0:
                index += 1
                continue
            result.sampled_frames += 1
            ok, frame = cap.retrieve()
            index += 1
            if not ok or frame is None:
                moving_present.append(False)
                continue
            result.decoded_frames += 1
            h, w = frame.shape[:2]
            if w > 960:
                new_h = max(2, int(round(h * 960 / w)))
                frame = cv2.resize(frame, (960, new_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(np.median(gray)))
            laplacians.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            corners, ids, _ = detector.detectMarkers(gray)
            ids_flat = [] if ids is None else [int(x) for x in ids.flatten()]
            all_ids.update(ids_flat)
            if ids_flat:
                any_present += 1
            moving_indices = [i for i, tag_id in enumerate(ids_flat) if tag_id in moving_ids]
            has_moving = bool(moving_indices)
            moving_present.append(has_moving)
            if has_moving:
                result.moving_tag_frames += 1
                counts: dict[int, int] = {}
                for i in moving_indices:
                    tag_id = ids_flat[i]
                    counts[tag_id] = counts.get(tag_id, 0) + 1
                    pts = corners[i].reshape(4, 2)
                    sides = [float(np.linalg.norm(pts[j] - pts[(j + 1) % 4])) for j in range(4)]
                    moving_sides.append(sum(sides) / 4.0)
                if any(count > 1 for count in counts.values()):
                    result.duplicate_moving_id_frames += 1
            if reference_ids and reference_ids.issubset(set(ids_flat)):
                reference_present += 1

        cap.release()
        result.decode_fraction = result.decoded_frames / max(1, result.sampled_frames)
        result.moving_tag_coverage = result.moving_tag_frames / max(1, result.decoded_frames)
        result.any_tag_coverage = any_present / max(1, result.decoded_frames)
        if reference_ids:
            result.reference_tag_coverage = reference_present / max(1, result.decoded_frames)
        result.longest_moving_tag_gap_s = _longest_false_run(moving_present, sample_period)
        result.median_moving_tag_side_px = median(moving_sides) if moving_sides else 0.0
        result.p10_moving_tag_side_px = _percentile(moving_sides, 0.10)
        result.median_brightness = median(brightness) if brightness else 0.0
        result.median_laplacian = median(laplacians) if laplacians else 0.0
        result.sharp_frame_fraction = (
            sum(x >= float(cfg["blur_laplacian_threshold"]) for x in laplacians) / max(1, len(laplacians))
        )
        result.detected_ids = sorted(all_ids)

        if result.duration_s < float(cfg["minimum_run_seconds"]):
            result.failures.append(f"duration {result.duration_s:.1f}s is below the configured run minimum")
        if result.sampled_frames < 20 or result.decode_fraction < 0.98:
            result.failures.append(f"sampled-frame decode fraction is {result.decode_fraction:.1%}")
        if result.width < 854 or result.height < 480:
            result.failures.append(f"resolution {result.width}x{result.height} is too low")
        elif result.width < 1280 or result.height < 720:
            result.warnings.append(f"resolution {result.width}x{result.height} is below 720p")

        coverage = result.moving_tag_coverage
        if coverage < float(cfg["warning_moving_tag_coverage"]):
            result.failures.append(f"moving-tag coverage is only {coverage:.1%}")
        elif coverage < float(cfg["minimum_moving_tag_coverage"]):
            result.warnings.append(f"moving-tag coverage is {coverage:.1%}, below preferred coverage")
        if result.longest_moving_tag_gap_s > float(cfg["warning_gap_seconds"]):
            result.failures.append(f"longest moving-tag loss is {result.longest_moving_tag_gap_s:.2f}s")
        elif result.longest_moving_tag_gap_s > float(cfg["maximum_gap_seconds"]):
            result.warnings.append(f"longest moving-tag loss is {result.longest_moving_tag_gap_s:.2f}s")
        if result.p10_moving_tag_side_px < float(cfg["minimum_tag_side_pixels"]):
            result.failures.append(f"moving tag is too small in weak views (p10 side {result.p10_moving_tag_side_px:.1f}px)")
        if result.sharp_frame_fraction < float(cfg["minimum_sharp_fraction"]):
            result.warnings.append(f"only {result.sharp_frame_fraction:.1%} of sampled frames pass the sharpness screen")
        if not (float(cfg["minimum_median_brightness"]) <= result.median_brightness <= float(cfg["maximum_median_brightness"])):
            result.warnings.append(f"median brightness {result.median_brightness:.1f} is outside the preferred range")
        if result.duplicate_moving_id_frames:
            result.warnings.append(
                f"duplicate moving ID detected in {result.duplicate_moving_id_frames} sampled frames; pose identity is ambiguous"
            )

        result.status = "FAIL" if result.failures else ("WARN" if result.warnings else "PASS")
        return result
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.failures.append("unexpected processing error")
        return result


def _discover(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def _format_seconds(value: float) -> str:
    minutes, seconds = divmod(int(round(value)), 60)
    return f"{minutes}:{seconds:02d}"


def _write_outputs(output: Path, results: list[VideoResult], cfg: dict[str, Any]) -> str:
    output.mkdir(parents=True, exist_ok=True)
    passed = [r for r in results if r.status == "PASS"]
    usable = [r for r in results if r.status in {"PASS", "WARN"}]
    expected = int(cfg["expected_runs"])
    if len(passed) >= expected:
        decision = "READY_FOR_ANALYSIS"
    elif len(usable) >= expected:
        decision = "USABLE_WITH_WARNINGS"
    else:
        decision = "NOT_READY"

    summary = {
        "decision": decision,
        "decision_scope": "video/AprilTag technical screening only",
        "videos_found": len(results),
        "passing_runs": len(passed),
        "usable_runs_including_warnings": len(usable),
        "expected_independent_runs": expected,
        "total_duration_s": sum(r.duration_s for r in results),
        "usable_duration_s": sum(r.duration_s for r in usable),
        "configuration": cfg,
        "videos": [asdict(r) for r in results],
        "scientific_requirements_not_proven_by_video": [
            "each file is a genuinely independent held-out physical run",
            "camera intrinsics/extrinsics and tag geometry are calibrated and frozen",
            "video-to-telemetry synchronization uncertainty is acceptable",
            "matching raw UGV01 telemetry and twin traces exist",
            "resource-policy and wireless-condition factors follow the prospective matrix",
        ],
    }
    (output / "video_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = [
        "path", "status", "duration_s", "fps", "width", "height", "sampled_frames",
        "decode_fraction", "moving_tag_coverage", "any_tag_coverage", "reference_tag_coverage",
        "longest_moving_tag_gap_s", "median_moving_tag_side_px", "p10_moving_tag_side_px",
        "duplicate_moving_id_frames", "median_brightness", "sharp_frame_fraction",
        "median_laplacian", "detected_ids", "warnings", "failures", "error",
    ]
    with (output / "video_audit_videos.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["detected_ids"] = ";".join(map(str, r.detected_ids))
            row["warnings"] = " | ".join(r.warnings)
            row["failures"] = " | ".join(r.failures)
            writer.writerow({key: row.get(key) for key in fields})

    lines = [
        "# UGV01 video audit", "", f"**Decision: {decision}**", "",
        f"Found {len(results)} video files totaling {_format_seconds(summary['total_duration_s'])}. "
        f"{len(passed)} pass cleanly and {len(usable)} are usable including warnings; "
        f"the configured target is {expected} independent runs.", "",
        "| Video | Result | Duration | Moving tag | Longest loss | Tag p10 | Sharp |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {Path(r.path).name} | {r.status} | {_format_seconds(r.duration_s)} | "
            f"{r.moving_tag_coverage:.1%} | {r.longest_moving_tag_gap_s:.2f}s | "
            f"{r.p10_moving_tag_side_px:.1f}px | {r.sharp_frame_fraction:.1%} |"
        )
    lines += ["", "## Problems and warnings", ""]
    for r in results:
        issues = [f"FAIL: {x}" for x in r.failures] + [f"WARN: {x}" for x in r.warnings]
        if issues:
            lines.append(f"- **{Path(r.path).name}:** " + "; ".join(issues))
    if all(not r.failures and not r.warnings for r in results):
        lines.append("- None detected.")
    lines += [
        "", "## Scope", "",
        "This decision screens the video and AprilTag visibility. It does not by itself prove run independence, "
        "camera/world calibration, video-telemetry synchronization, availability of matching telemetry/twin logs, "
        "or completion of the wireless/resource-policy experimental matrix.", "",
    ]
    (output / "video_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video_directory", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/ugv01_video_audit"))
    p.add_argument("--config", type=Path)
    p.add_argument("--moving-tag-ids", type=int, nargs="+")
    p.add_argument("--reference-tag-ids", type=int, nargs="*")
    p.add_argument("--expected-runs", type=int)
    p.add_argument("--minimum-run-seconds", type=float)
    p.add_argument("--sample-hz", type=float)
    p.add_argument("--workers", type=int, help="0/omitted chooses up to four workers")
    p.add_argument("--no-recursive", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(json.loads(args.config.read_text(encoding="utf-8")))
    for key in ("moving_tag_ids", "reference_tag_ids", "expected_runs", "minimum_run_seconds", "sample_hz"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    root = args.video_directory.resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2
    videos = _discover(root, not args.no_recursive)
    if not videos:
        print(f"error: no supported video files found under {root}", file=sys.stderr)
        return 2
    workers = args.workers or min(4, len(videos), max(1, (os.cpu_count() or 2) // 2))
    print(f"Auditing {len(videos)} video(s) at {cfg['sample_hz']} sampled frame(s)/s with {workers} worker(s)...")
    results: list[VideoResult] = []
    if workers == 1:
        for index, video in enumerate(videos, 1):
            result = _audit_one(str(video), cfg)
            results.append(result)
            print(f"[{index}/{len(videos)}] {video.name}: {result.status}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(_audit_one, str(video), cfg): video for video in videos}
            for index, future in enumerate(as_completed(jobs), 1):
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(videos)}] {jobs[future].name}: {result.status}")
    results.sort(key=lambda r: r.path.lower())
    decision = _write_outputs(args.output.resolve(), results, cfg)
    print(f"Decision: {decision}")
    print(f"Report: {(args.output.resolve() / 'video_audit_report.md')}")
    return 0 if decision == "READY_FOR_ANALYSIS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

