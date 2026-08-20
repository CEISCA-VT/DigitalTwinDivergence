#!/usr/bin/env python3
"""
Download the minimal NCLT + RELLIS-3D data needed for the
sensor-lightweight digital-twin fidelity project.

What this script intentionally downloads
----------------------------------------

NCLT
  - ALL 27 sessions by default, but ONLY:
      * sensor archive (sen.tar.gz)
      * ground-truth pose CSV
  - NO Ladybug images
  - NO Velodyne
  - NO Hokuyo

  This preserves the lightweight signals needed for this project:
  wheel odometry, IMU, KVH gyro, GPS/RTK-related sensor data, plus GT.

RELLIS-3D
  - By default processes ALL five official sequences, one at a time in
    smallest-first order (00004, 00003, 00001, 00000, 00002).
  - RELLIS does not publish wheel/IMU/VectorNav streams as separate tiny
    downloads; those topics are bundled inside the "Full-stack Merged" ZIPs.
  - Therefore the script must temporarily download each selected merged ZIP.
  - It then extracts only the ROS topics useful to this project into CSV:
      /warthog_velocity_controller/odom
      /warthog_velocity_controller/cmd_vel
      /imu/data
      /imu/data_raw
      /vectornav/IMU
      /vectornav/Odom
      /vectornav/GPS
      /tf
      /tf_static

  For each sequence the script:
      1. downloads one official Full-stack Merged ZIP,
      2. extracts its ROS bag(s),
      3. extracts only the selected lightweight topics to CSV,
      4. validates row counts, time spans, and sensor/reference overlap,
      5. writes a validation manifest,
      6. deletes the large ZIP/BAG only if validation PASSES,
      7. then moves to the next sequence.

  If validation fails, raw data is preserved and the script stops before
  downloading further sequences. Use --keep-rellis-raw if you intentionally
  want to retain the large source archives after successful validation.

Default output
--------------
<repo>/public_datasets/
    nclt/
    rellis3d/

Dependencies
------------
    python -m pip install gdown rosbags

Recommended first run
---------------------
    python scripts/download_minimal_nclt_rellis.py

This now performs a self-validating RELLIS pilot automatically:
00004 is downloaded first; only after its extraction validates does the
script delete its raw data and continue through the remaining sequences.

Manual single-sequence pilot (optional)
---------------------------------------
    python scripts/download_minimal_nclt_rellis.py --rellis-sequences 00004

NCLT only
---------
    python scripts/download_minimal_nclt_rellis.py --skip-rellis

RELLIS only
-----------
    python scripts/download_minimal_nclt_rellis.py --skip-nclt --rellis-sequences 00003,00004

Keep RELLIS raw ZIP/BAG files instead of auto-cleaning
-------------------------------------------------------
    python scripts/download_minimal_nclt_rellis.py --keep-rellis-raw

Notes
-----
- Downloads are resumable where supported.
- Existing files are not redownloaded.
- The script writes JSON manifests so later analysis can record provenance.
- NCLT's ~100 Hz ground-truth trajectory was generated from a SLAM graph and
  interpolated using odometry. Treat that reference dependency carefully in
  publication claims; do not silently call it fully independent wheel-odometry
  ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Official dataset endpoints
# ---------------------------------------------------------------------------

NCLT_BASE = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"

NCLT_SESSIONS = [
    "2012-01-08",
    "2012-01-15",
    "2012-01-22",
    "2012-02-02",
    "2012-02-04",
    "2012-02-05",
    "2012-02-12",
    "2012-02-18",
    "2012-02-19",
    "2012-03-17",
    "2012-03-25",
    "2012-03-31",
    "2012-04-29",
    "2012-05-11",
    "2012-05-26",
    "2012-06-15",
    "2012-08-04",
    "2012-08-20",
    "2012-09-28",
    "2012-10-28",
    "2012-11-04",
    "2012-11-16",
    "2012-11-17",
    "2012-12-01",
    "2013-01-10",
    "2013-02-23",
    "2013-04-05",
]

# Official RELLIS-3D "Full-stack Merged" Google Drive IDs from the
# unmannedlab/RELLIS-3D repository.
RELLIS_MERGED_GDRIVE_IDS = {
    "00000": "1grcYRvtAijiA0Kzu-AV_9K4k2C1Kc3Tn",  # ~23 GB
    "00001": "1geoU45pPavnabQ0arm4ILeHSsG3cU6ti",  # ~16 GB
    "00002": "1h0CVg62jTXiJ91LnR6md-WrUBDxT543n",  # ~28 GB
    "00003": "1glJzgnTYLIB_ar3CgHpc_MBp5AafQpy9",  # ~15 GB
    "00004": "1AuEjX0do3jGZhGKPszSEUNoj85YswNya",  # ~14 GB
}

# Smallest-first is deliberate: sequence 00004 becomes an automatic smoke
# test of the download -> extraction -> validation -> cleanup pipeline.
RELLIS_DEFAULT_ORDER = ["00004", "00003", "00001", "00000", "00002"]

RELLIS_REQUIRED_MIN_ROWS = 100
RELLIS_REQUIRED_MIN_DURATION_S = 30.0
RELLIS_REQUIRED_MIN_OVERLAP_S = 10.0

RELLIS_SELECTED_TOPICS = {
    "/warthog_velocity_controller/odom",
    "/warthog_velocity_controller/cmd_vel",
    "/imu/data",
    "/imu/data_raw",
    "/vectornav/IMU",
    "/vectornav/Odom",
    "/vectornav/GPS",
    "/tf",
    "/tf_static",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def repo_root_from_cwd() -> Path:
    """
    Find a plausible repository root by walking upward from cwd.
    Falls back to cwd if no obvious repo marker exists.
    """
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "DigitalTwin").exists():
            return p
    return here


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def free_space(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def download_http_resumable(url: str, dst: Path, retries: int = 5) -> None:
    """
    Simple resumable HTTP downloader using Range requests.
    Safe for the NCLT S3 files.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        existing = dst.stat().st_size if dst.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                status = getattr(r, "status", None)
                # If Range was ignored, restart instead of appending duplicate bytes.
                append = existing > 0 and status == 206
                mode = "ab" if append else "wb"
                if existing and not append:
                    existing = 0

                total_header = r.headers.get("Content-Length")
                total = int(total_header) + existing if total_header else None

                with dst.open(mode) as f:
                    downloaded = existing
                    last_print = time.time()
                    while True:
                        block = r.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)
                        now = time.time()
                        if now - last_print >= 1.0:
                            if total:
                                pct = 100.0 * downloaded / total
                                print(
                                    f"\r  {dst.name}: {human_bytes(downloaded)} / "
                                    f"{human_bytes(total)} ({pct:5.1f}%)",
                                    end="",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\r  {dst.name}: {human_bytes(downloaded)}",
                                    end="",
                                    flush=True,
                                )
                            last_print = now

                print()
                return

        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"\n  download attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(min(30, 2 ** attempt))


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (out_dir / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tf.extractall(out_dir)


def safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        zf.extractall(out_dir)


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# NCLT
# ---------------------------------------------------------------------------

def nclt_urls(session: str) -> tuple[str, str]:
    sensor_url = f"{NCLT_BASE}/sensor_data/{session}_sen.tar.gz"
    gt_url = f"{NCLT_BASE}/ground_truth/groundtruth_{session}.csv"
    return sensor_url, gt_url


def download_nclt(
    root: Path,
    sessions: Iterable[str],
    *,
    keep_archives: bool,
) -> None:
    dataset_root = root / "nclt"
    dataset_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": "NCLT",
        "official_page": "https://robots.engin.umich.edu/nclt/index.html",
        "selection_reason": (
            "Only sensor archives + ground-truth pose CSVs are downloaded. "
            "Images, Velodyne, and Hokuyo are intentionally omitted."
        ),
        "ground_truth_caution": (
            "NCLT states that the ~100 Hz ground-truth trajectory was generated "
            "from a SLAM graph and interpolated using odometry. Account for this "
            "dependency in publication claims."
        ),
        "sessions": [],
    }

    print("\n" + "=" * 80)
    print("NCLT: lightweight sensor + ground-truth download")
    print("=" * 80)

    for idx, session in enumerate(sessions, start=1):
        print(f"\n[{idx}] NCLT session {session}")
        session_dir = dataset_root / session
        raw_dir = session_dir / "raw"
        sensor_dir = session_dir / "sensors"
        session_dir.mkdir(parents=True, exist_ok=True)

        sensor_url, gt_url = nclt_urls(session)
        sensor_archive = raw_dir / f"{session}_sen.tar.gz"
        gt_path = session_dir / f"groundtruth_{session}.csv"

        # GT
        if gt_path.exists() and gt_path.stat().st_size > 0:
            print(f"  ground truth exists: {gt_path}")
        else:
            print(f"  downloading GT: {gt_url}")
            download_http_resumable(gt_url, gt_path)

        # Sensor archive and extraction
        extraction_marker = sensor_dir / ".EXTRACTED_OK"
        if extraction_marker.exists():
            print(f"  sensors already extracted: {sensor_dir}")
        else:
            if not sensor_archive.exists() or sensor_archive.stat().st_size == 0:
                print(f"  downloading sensors: {sensor_url}")
                download_http_resumable(sensor_url, sensor_archive)
            else:
                print(f"  sensor archive exists: {sensor_archive}")

            print(f"  extracting sensors -> {sensor_dir}")
            safe_extract_tar(sensor_archive, sensor_dir)
            extraction_marker.write_text("ok\n", encoding="utf-8")

        if not keep_archives and sensor_archive.exists():
            print(f"  deleting sensor archive after verified extraction: {sensor_archive}")
            sensor_archive.unlink()
            try:
                raw_dir.rmdir()
            except OSError:
                pass

        entry = {
            "session": session,
            "sensor_url": sensor_url,
            "groundtruth_url": gt_url,
            "groundtruth_path": str(gt_path.relative_to(dataset_root)),
            "sensor_dir": str(sensor_dir.relative_to(dataset_root)),
            "groundtruth_size_bytes": gt_path.stat().st_size if gt_path.exists() else None,
            "groundtruth_sha256": sha256_file(gt_path) if gt_path.exists() else None,
        }
        manifest["sessions"].append(entry)
        write_json(dataset_root / "download_manifest.json", manifest)

    print(f"\nNCLT complete: {dataset_root}")


# ---------------------------------------------------------------------------
# RELLIS-3D ROS extraction
# ---------------------------------------------------------------------------

def require_rellis_deps() -> tuple[Any, Any]:
    try:
        import gdown  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "RELLIS download requires gdown.\n"
            "Install with:\n"
            "  python -m pip install gdown rosbags"
        ) from e

    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "RELLIS topic extraction requires rosbags.\n"
            "Install with:\n"
            "  python -m pip install gdown rosbags"
        ) from e

    return gdown, AnyReader


def _stamp_seconds(msg: Any, fallback_ns: int) -> float:
    try:
        stamp = msg.header.stamp
        sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
        nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", getattr(stamp, "nsecs", 0)))
        return float(sec) + float(nsec) * 1e-9
    except Exception:
        return float(fallback_ns) * 1e-9


def _odom_row(msg: Any, bag_ns: int) -> dict[str, Any]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    lv = msg.twist.twist.linear
    av = msg.twist.twist.angular
    return {
        "bag_time_s": bag_ns * 1e-9,
        "msg_time_s": _stamp_seconds(msg, bag_ns),
        "frame_id": getattr(msg.header, "frame_id", ""),
        "child_frame_id": getattr(msg, "child_frame_id", ""),
        "px_m": p.x,
        "py_m": p.y,
        "pz_m": p.z,
        "qx": q.x,
        "qy": q.y,
        "qz": q.z,
        "qw": q.w,
        "linear_x_mps": lv.x,
        "linear_y_mps": lv.y,
        "linear_z_mps": lv.z,
        "angular_x_radps": av.x,
        "angular_y_radps": av.y,
        "angular_z_radps": av.z,
    }


def _imu_row(msg: Any, bag_ns: int) -> dict[str, Any]:
    q = msg.orientation
    w = msg.angular_velocity
    a = msg.linear_acceleration
    return {
        "bag_time_s": bag_ns * 1e-9,
        "msg_time_s": _stamp_seconds(msg, bag_ns),
        "frame_id": getattr(msg.header, "frame_id", ""),
        "qx": q.x,
        "qy": q.y,
        "qz": q.z,
        "qw": q.w,
        "angular_x_radps": w.x,
        "angular_y_radps": w.y,
        "angular_z_radps": w.z,
        "linear_accel_x_mps2": a.x,
        "linear_accel_y_mps2": a.y,
        "linear_accel_z_mps2": a.z,
    }


def _gps_row(msg: Any, bag_ns: int) -> dict[str, Any]:
    return {
        "bag_time_s": bag_ns * 1e-9,
        "msg_time_s": _stamp_seconds(msg, bag_ns),
        "frame_id": getattr(msg.header, "frame_id", ""),
        "status": getattr(getattr(msg, "status", None), "status", None),
        "service": getattr(getattr(msg, "status", None), "service", None),
        "latitude_deg": msg.latitude,
        "longitude_deg": msg.longitude,
        "altitude_m": msg.altitude,
    }


def _twist_row(msg: Any, bag_ns: int) -> dict[str, Any]:
    return {
        "bag_time_s": bag_ns * 1e-9,
        "linear_x_mps": msg.linear.x,
        "linear_y_mps": msg.linear.y,
        "linear_z_mps": msg.linear.z,
        "angular_x_radps": msg.angular.x,
        "angular_y_radps": msg.angular.y,
        "angular_z_radps": msg.angular.z,
    }


def _tf_rows(msg: Any, bag_ns: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in msg.transforms:
        tr = t.transform.translation
        q = t.transform.rotation
        rows.append(
            {
                "bag_time_s": bag_ns * 1e-9,
                "msg_time_s": _stamp_seconds(t, bag_ns),
                "frame_id": getattr(t.header, "frame_id", ""),
                "child_frame_id": t.child_frame_id,
                "tx_m": tr.x,
                "ty_m": tr.y,
                "tz_m": tr.z,
                "qx": q.x,
                "qy": q.y,
                "qz": q.z,
                "qw": q.w,
            }
        )
    return rows


def _topic_filename(topic: str) -> str:
    return topic.strip("/").replace("/", "__") + ".csv"


def extract_rellis_bag(bag_path: Path, out_dir: Path, AnyReader: Any) -> dict[str, Any]:
    """
    Extract only the project-relevant standard ROS topics to CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    per_topic_rows: dict[str, list[dict[str, Any]]] = {
        topic: [] for topic in RELLIS_SELECTED_TOPICS
    }

    print(f"  reading ROS bag: {bag_path}")

    with AnyReader([bag_path]) as reader:
        connections = [
            c for c in reader.connections if c.topic in RELLIS_SELECTED_TOPICS
        ]

        available = sorted({c.topic for c in connections})
        print("  selected topics present:")
        for topic in available:
            print(f"    {topic}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            if topic in {
                "/warthog_velocity_controller/odom",
                "/vectornav/Odom",
            }:
                per_topic_rows[topic].append(_odom_row(msg, timestamp))
            elif topic in {
                "/imu/data",
                "/imu/data_raw",
                "/vectornav/IMU",
            }:
                per_topic_rows[topic].append(_imu_row(msg, timestamp))
            elif topic == "/vectornav/GPS":
                per_topic_rows[topic].append(_gps_row(msg, timestamp))
            elif topic == "/warthog_velocity_controller/cmd_vel":
                per_topic_rows[topic].append(_twist_row(msg, timestamp))
            elif topic in {"/tf", "/tf_static"}:
                per_topic_rows[topic].extend(_tf_rows(msg, timestamp))

    summary: dict[str, Any] = {
        "bag": str(bag_path),
        "topics": {},
    }

    for topic, rows in per_topic_rows.items():
        if not rows:
            summary["topics"][topic] = {"rows": 0, "csv": None}
            continue

        csv_path = out_dir / _topic_filename(topic)
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        time_values = []
        for row in rows:
            value = row.get("msg_time_s", row.get("bag_time_s"))
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value == value and abs(value) != float("inf"):
                time_values.append(value)

        t_min = min(time_values) if time_values else None
        t_max = max(time_values) if time_values else None

        summary["topics"][topic] = {
            "rows": len(rows),
            "csv": csv_path.name,
            "size_bytes": csv_path.stat().st_size,
            "sha256": sha256_file(csv_path),
            "time_min_s": t_min,
            "time_max_s": t_max,
            "duration_s": None if t_min is None or t_max is None else (t_max - t_min),
        }

    return summary



def validate_rellis_sequence(bag_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Validate that the extracted lightweight RELLIS evidence is scientifically
    usable before any large raw ZIP/BAG is deleted.

    Required evidence:
      - Warthog wheel/vehicle odometry
      - at least one Warthog IMU stream (/imu/data preferred, raw accepted)
      - VectorNav odometry as the independent high-grade reference candidate
      - at least modest common time overlap

    GPS, cmd_vel and TF are useful but are not hard requirements for allowing
    the extraction pipeline to advance.
    """
    aggregate: dict[str, dict[str, Any]] = {}

    for topic in RELLIS_SELECTED_TOPICS:
        infos = [
            b["topics"][topic]
            for b in bag_summaries
            if topic in b.get("topics", {}) and b["topics"][topic].get("rows", 0) > 0
        ]
        rows = sum(int(info.get("rows", 0)) for info in infos)
        starts = [
            float(info["time_min_s"])
            for info in infos
            if info.get("time_min_s") is not None
        ]
        ends = [
            float(info["time_max_s"])
            for info in infos
            if info.get("time_max_s") is not None
        ]
        aggregate[topic] = {
            "rows": rows,
            "time_min_s": min(starts) if starts else None,
            "time_max_s": max(ends) if ends else None,
        }
        if starts and ends:
            aggregate[topic]["duration_s"] = (
                aggregate[topic]["time_max_s"] - aggregate[topic]["time_min_s"]
            )
        else:
            aggregate[topic]["duration_s"] = None

    wheel = aggregate["/warthog_velocity_controller/odom"]
    vnav = aggregate["/vectornav/Odom"]

    imu_candidates = [
        aggregate["/imu/data"],
        aggregate["/imu/data_raw"],
    ]
    imu_topic_names = ["/imu/data", "/imu/data_raw"]
    usable_imu = None
    usable_imu_name = None
    for name, info in zip(imu_topic_names, imu_candidates):
        if (
            int(info["rows"]) >= RELLIS_REQUIRED_MIN_ROWS
            and info.get("duration_s") is not None
            and float(info["duration_s"]) >= RELLIS_REQUIRED_MIN_DURATION_S
        ):
            usable_imu = info
            usable_imu_name = name
            break

    checks: dict[str, Any] = {}

    checks["wheel_odom_rows"] = {
        "pass": int(wheel["rows"]) >= RELLIS_REQUIRED_MIN_ROWS,
        "value": int(wheel["rows"]),
        "minimum": RELLIS_REQUIRED_MIN_ROWS,
    }
    checks["wheel_odom_duration"] = {
        "pass": (
            wheel.get("duration_s") is not None
            and float(wheel["duration_s"]) >= RELLIS_REQUIRED_MIN_DURATION_S
        ),
        "value_s": wheel.get("duration_s"),
        "minimum_s": RELLIS_REQUIRED_MIN_DURATION_S,
    }
    checks["warthog_imu_available"] = {
        "pass": usable_imu is not None,
        "selected_topic": usable_imu_name,
        "rows": None if usable_imu is None else int(usable_imu["rows"]),
        "duration_s": None if usable_imu is None else usable_imu.get("duration_s"),
    }
    checks["vectornav_odom_rows"] = {
        "pass": int(vnav["rows"]) >= RELLIS_REQUIRED_MIN_ROWS,
        "value": int(vnav["rows"]),
        "minimum": RELLIS_REQUIRED_MIN_ROWS,
    }
    checks["vectornav_odom_duration"] = {
        "pass": (
            vnav.get("duration_s") is not None
            and float(vnav["duration_s"]) >= RELLIS_REQUIRED_MIN_DURATION_S
        ),
        "value_s": vnav.get("duration_s"),
        "minimum_s": RELLIS_REQUIRED_MIN_DURATION_S,
    }

    overlap_s = None
    if usable_imu is not None:
        required_intervals = [wheel, usable_imu, vnav]
        if all(
            x.get("time_min_s") is not None and x.get("time_max_s") is not None
            for x in required_intervals
        ):
            overlap_start = max(float(x["time_min_s"]) for x in required_intervals)
            overlap_end = min(float(x["time_max_s"]) for x in required_intervals)
            overlap_s = max(0.0, overlap_end - overlap_start)

    checks["sensor_reference_time_overlap"] = {
        "pass": overlap_s is not None and overlap_s >= RELLIS_REQUIRED_MIN_OVERLAP_S,
        "value_s": overlap_s,
        "minimum_s": RELLIS_REQUIRED_MIN_OVERLAP_S,
    }

    passed = all(bool(c["pass"]) for c in checks.values())

    return {
        "schema": "rellis3d_lightweight_extraction_validation_v1",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "requirements": {
            "min_rows": RELLIS_REQUIRED_MIN_ROWS,
            "min_duration_s": RELLIS_REQUIRED_MIN_DURATION_S,
            "min_overlap_s": RELLIS_REQUIRED_MIN_OVERLAP_S,
        },
        "checks": checks,
        "aggregate_topics": aggregate,
        "optional_topic_presence": {
            topic: int(aggregate[topic]["rows"]) > 0
            for topic in [
                "/warthog_velocity_controller/cmd_vel",
                "/vectornav/GPS",
                "/vectornav/IMU",
                "/tf",
                "/tf_static",
            ]
        },
    }



def download_rellis(
    root: Path,
    sequences: Iterable[str],
    *,
    keep_raw: bool,
) -> None:
    gdown, AnyReader = require_rellis_deps()

    dataset_root = root / "rellis3d"
    dataset_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": "RELLIS-3D",
        "official_repo": "https://github.com/unmannedlab/RELLIS-3D",
        "download_variant": "Full-stack Merged",
        "selection_reason": (
            "Merged bags are required because wheel odometry, Warthog IMU, "
            "VectorNav GNSS/INS and TF are bundled with the other sensors. "
            "All selected official sequences are processed sequentially; only "
            "project-relevant topics are retained in compact CSV form after "
            "validation."
        ),
        "selected_topics": sorted(RELLIS_SELECTED_TOPICS),
        "sequences": [],
    }

    print("\n" + "=" * 80)
    print("RELLIS-3D: selected full-stack merged sequences -> compact CSV")
    print("=" * 80)

    for idx, seq in enumerate(sequences, start=1):
        if seq not in RELLIS_MERGED_GDRIVE_IDS:
            raise ValueError(
                f"Unknown RELLIS sequence {seq!r}. "
                f"Choose from {sorted(RELLIS_MERGED_GDRIVE_IDS)}"
            )

        print(f"\n[{idx}] RELLIS sequence {seq}")
        seq_dir = dataset_root / seq
        raw_dir = seq_dir / "raw"
        extracted_dir = raw_dir / "unzipped"
        csv_dir = seq_dir / "selected_topics_csv"
        seq_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        zip_path = raw_dir / f"{seq}_20210224.zip"
        drive_id = RELLIS_MERGED_GDRIVE_IDS[seq]

        # If CSV extraction is already complete, skip expensive download.
        done_marker = csv_dir / ".EXTRACTED_SELECTED_TOPICS_OK"
        if done_marker.exists():
            print(f"  selected-topic extraction already validated: {csv_dir}")
            validation_path = seq_dir / "validation_manifest.json"
            if validation_path.exists():
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                if not validation.get("passed", False):
                    raise RuntimeError(
                        f"Existing validation manifest for RELLIS {seq} is not PASS: "
                        f"{validation_path}"
                    )
            if not keep_raw and raw_dir.exists():
                print(f"  validated CSVs already exist; deleting stale raw dir: {raw_dir}")
                shutil.rmtree(raw_dir, ignore_errors=True)
            continue

        if not zip_path.exists() or zip_path.stat().st_size == 0:
            print(
                "  downloading official Full-stack Merged ZIP from Google Drive.\n"
                "  This is large because RELLIS does not publish these ROS topics separately."
            )
            url = f"https://drive.google.com/uc?id={drive_id}"
            result = gdown.download(
                url=url,
                output=str(zip_path),
                quiet=False,
                resume=True,
            )
            if not result or not zip_path.exists():
                raise RuntimeError(f"gdown failed for RELLIS {seq}")
        else:
            print(f"  ZIP already exists: {zip_path}")

        # Extract ZIP only if needed.
        bag_paths = list(extracted_dir.rglob("*.bag")) if extracted_dir.exists() else []
        if not bag_paths:
            print(f"  extracting ZIP -> {extracted_dir}")
            safe_extract_zip(zip_path, extracted_dir)
            bag_paths = list(extracted_dir.rglob("*.bag"))

        if not bag_paths:
            raise RuntimeError(
                f"No .bag files found after extracting {zip_path}. "
                f"Inspect {extracted_dir} manually."
            )

        csv_dir.mkdir(parents=True, exist_ok=True)

        bag_summaries = []
        for bag_path in sorted(bag_paths):
            bag_out = csv_dir / bag_path.stem
            bag_summary = extract_rellis_bag(bag_path, bag_out, AnyReader)
            bag_summaries.append(bag_summary)

        # Validate BEFORE deleting any raw evidence or advancing to another
        # expensive sequence download.
        validation = validate_rellis_sequence(bag_summaries)
        write_json(seq_dir / "validation_manifest.json", validation)

        print("\n  validation checks:")
        for name, check in validation["checks"].items():
            mark = "PASS" if check["pass"] else "FAIL"
            print(f"    {mark:4s}  {name}: {check}")

        if not validation["passed"]:
            print(
                "\n  VALIDATION FAILED. Raw ZIP/BAG is preserved and the downloader "
                "will stop before fetching another sequence."
            )
            raise RuntimeError(
                f"RELLIS {seq} selected-topic extraction failed validation. "
                f"Inspect {seq_dir / 'validation_manifest.json'}"
            )

        total_rows = {
            topic: int(info["rows"])
            for topic, info in validation["aggregate_topics"].items()
        }

        sequence_summary = {
            "sequence": seq,
            "google_drive_file_id": drive_id,
            "zip_path": str(zip_path.relative_to(dataset_root)),
            "selected_csv_root": str(csv_dir.relative_to(dataset_root)),
            "bag_summaries": bag_summaries,
            "total_rows_by_topic": total_rows,
            "validation": validation,
        }
        write_json(seq_dir / "extraction_manifest.json", sequence_summary)

        # Marker is written only after scientific minimum checks pass.
        done_marker.write_text("ok\n", encoding="utf-8")

        manifest["sequences"] = [
            x for x in manifest["sequences"] if x.get("sequence") != seq
        ]
        manifest["sequences"].append(sequence_summary)
        write_json(dataset_root / "download_manifest.json", manifest)

        if not keep_raw:
            print(
                "  validation PASS -> deleting large raw ZIP/BAG before "
                "downloading the next sequence"
            )
            shutil.rmtree(raw_dir, ignore_errors=True)
        else:
            print("  validation PASS -> --keep-rellis-raw set, preserving raw data")

    print(f"\nRELLIS-3D complete: {dataset_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download only the NCLT and RELLIS-3D data needed for the "
            "sensor-lightweight digital-twin fidelity project."
        )
    )
    p.add_argument(
        "--public-datasets-root",
        type=Path,
        default=None,
        help=(
            "Destination root. Default: <detected repo root>/public_datasets"
        ),
    )
    p.add_argument("--skip-nclt", action="store_true")
    p.add_argument("--skip-rellis", action="store_true")
    p.add_argument(
        "--nclt-sessions",
        default="all",
        help=(
            'Comma-separated NCLT dates, or "all". '
            "Default downloads all 27 sessions but only sensor+GT."
        ),
    )
    p.add_argument(
        "--keep-nclt-archives",
        action="store_true",
        help="Keep NCLT sen.tar.gz files after successful extraction.",
    )
    p.add_argument(
        "--rellis-sequences",
        default="all",
        help=(
            'Comma-separated RELLIS sequence IDs, or "all". '
            "Default: all five sequences in smallest-first validation order."
        ),
    )
    p.add_argument(
        "--keep-rellis-raw",
        action="store_true",
        help=(
            "Preserve the large RELLIS ZIP/BAG files even after the selected "
            "topic extraction passes validation. Default behavior is to delete "
            "raw data only after PASS, then continue to the next sequence."
        ),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    repo_root = repo_root_from_cwd()
    public_root = (
        args.public_datasets_root.resolve()
        if args.public_datasets_root is not None
        else (repo_root / "public_datasets").resolve()
    )
    public_root.mkdir(parents=True, exist_ok=True)

    print("Repository root :", repo_root)
    print("Dataset root    :", public_root)
    print("Free disk       :", human_bytes(free_space(public_root)))

    if not args.skip_nclt:
        if args.nclt_sessions.strip().lower() == "all":
            nclt_sessions = list(NCLT_SESSIONS)
        else:
            nclt_sessions = parse_csv_list(args.nclt_sessions)
            unknown = sorted(set(nclt_sessions) - set(NCLT_SESSIONS))
            if unknown:
                raise ValueError(f"Unknown NCLT sessions: {unknown}")

        print(
            f"\nNCLT selection: {len(nclt_sessions)} sessions; "
            "sensor archives + ground truth only."
        )
        download_nclt(
            public_root,
            nclt_sessions,
            keep_archives=args.keep_nclt_archives,
        )

    if not args.skip_rellis:
        if args.rellis_sequences.strip().lower() == "all":
            rellis_sequences = list(RELLIS_DEFAULT_ORDER)
        else:
            rellis_sequences = parse_csv_list(args.rellis_sequences)

        print(
            f"\nRELLIS selection/order: {rellis_sequences}; "
            "each Full-stack Merged sequence is processed, validated, and "
            "auto-cleaned before the next download."
        )

        unknown = sorted(set(rellis_sequences) - set(RELLIS_MERGED_GDRIVE_IDS))
        if unknown:
            raise ValueError(f"Unknown RELLIS sequences: {unknown}")

        # Largest official merged ZIP is ~28 GB; extraction temporarily needs
        # both ZIP + uncompressed bag. Warn if peak scratch space is tight.
        free = free_space(public_root)
        if free < 70 * 1024**3:
            print(
                "\nWARNING: less than 70 GB free space is available. "
                "The largest RELLIS sequence may require substantial temporary "
                "ZIP + extracted BAG scratch space. Because processing is "
                "sequential, this is a peak-space warning rather than a "
                "96-GB-retained-data requirement.\n"
            )

        download_rellis(
            public_root,
            rellis_sequences,
            keep_raw=args.keep_rellis_raw,
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"NCLT root    : {public_root / 'nclt'}")
    print(f"RELLIS root  : {public_root / 'rellis3d'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
