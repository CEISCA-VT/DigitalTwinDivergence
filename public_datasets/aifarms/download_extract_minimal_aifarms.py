#!/usr/bin/env python3
"""
Download -> extract -> verify -> delete for the minimal AI Farms / TerraSentia subset.

Put this file directly at:
    public_datasets/aifarms/download_extract_minimal.py

Run from the repository root:
    python public_datasets/aifarms/download_extract_minimal.py

Dependencies:
    pip install requests rosbags

What is retained
----------------
For each selected ROS1 bag, this script keeps only compact CSV streams needed
for the digital-twin fidelity / terrain / slip study:

    imu.csv              /terrasentia/imu
    motors.csv           /terrasentia/motors (wheel/motor measurements)
    gps.csv              /terrasentia/full_gps
    reference_ekf.csv    /terrasentia/ekf (dataset reference pose)
    motion_command.csv   /terrasentia/motion_command, if present
    mhe_output.csv       /terrasentia/mhe_output, if present
    topic_inventory.json
    extraction_manifest.json

The large .bag is deleted ONLY after:
    1. all four required streams are found,
    2. each required CSV has enough rows,
    3. expected semantic fields are present,
    4. output hashes are written successfully.

If extraction or verification fails, the .bag is KEPT for inspection.

Ground-truth caveat
-------------------
TerraSentia's published reference pose is itself an estimated EKF/MHE-based pose,
not motion-capture ground truth. Use this dataset primarily for terrain/slip/
generalization experiments. Do not treat its reference as more authoritative
than i2Nav's dedicated benchmark ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'requests'. Install with:\n"
        "  pip install requests rosbags"
    ) from exc

try:
    from rosbags.highlevel import AnyReader
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'rosbags'. Install with:\n"
        "  pip install requests rosbags"
    ) from exc


# ---------------------------------------------------------------------------
# Minimal, pre-declared benchmark subset.
#
# All five are listed by the TerraSentia maintainers among 2022 sequences with
# reliable RTK corrections (<~2 cm according to their RTK_data_sequences.txt).
# ---------------------------------------------------------------------------

SEQUENCES = [
    {
        "stem": "ts_2022_06_09_13h16m39s_one_row",
        "role": "early-season simple/straight-row baseline",
        "group": "cornfield1",
        "url": "https://uofi.box.com/shared/static/ea426awnzglqb86vdahn1smeocilhsz0.bag",
    },
    {
        "stem": "ts_2022_06_15_11h48m34s_four_rows",
        "role": "early-season repeated-turn / multi-row baseline",
        "group": "cornfield1",
        "url": "https://uofi.box.com/shared/static/r0sg8conr2n3abhje0ljr3qn9gskkiyc.bag",
    },
    {
        "stem": "ts_2022_09_01_11h20m00s_two_random",
        "role": "late-season irregular/random trajectory",
        "group": "cornfield1",
        "url": "https://uofi.box.com/shared/static/jf2ol5vmmgyauens3d8cufao7f3hlg63.bag",
    },
    {
        "stem": "ts_2022_09_01_12h32m56s_double_loop_corridor",
        "role": "repeated loops / accumulated turn and drift stress",
        "group": "others",
        "url": "https://uofi.box.com/shared/static/xg07b214zf20qkk2880dnjeeah1zbpj4.bag",
    },
    {
        "stem": "ts_2022_09_06_12h37m11s_four_rows",
        "role": "very-late-season repeated-turn / rough-terrain stress",
        "group": "cornfield1",
        "url": "https://uofi.box.com/shared/static/ln3584pumoc35z09xtnn0gc3gua715sy.bag",
    },
]

METADATA_URLS = {
    "sensor_parameters.txt":
        "https://raw.githubusercontent.com/jrcuaranv/terrasentia-dataset/main/sensor_parameters.txt",
    "RTK_data_sequences.txt":
        "https://raw.githubusercontent.com/jrcuaranv/terrasentia-dataset/main/RTK_data_sequences.txt",
}

# Exact 2022 topic names we want first. Fallback matching is used only if an
# exact topic is absent.
STREAM_SPECS = {
    "imu": {
        "preferred": ["/terrasentia/imu"],
        "filename": "imu.csv",
        "required": True,
        "min_rows": 20,
        "fallback_tokens": ("imu",),
        "exclude_tokens": ("image", "camera_info", "zed_node/imu/data"),
        "required_field_tokens": ("angular_velocity", "linear_acceleration"),
    },
    "motors": {
        "preferred": ["/terrasentia/motors"],
        "filename": "motors.csv",
        "required": True,
        "min_rows": 20,
        "fallback_tokens": ("motor", "wheel", "encoder"),
        "exclude_tokens": ("motion_command",),
        "required_field_tokens": (),
    },
    "gps": {
        "preferred": ["/terrasentia/full_gps"],
        "filename": "gps.csv",
        "required": True,
        "min_rows": 3,
        "fallback_tokens": ("gps", "gnss"),
        "exclude_tokens": (),
        "required_field_tokens": ("latitude", "longitude"),
    },
    "reference_ekf": {
        "preferred": ["/terrasentia/ekf"],
        "filename": "reference_ekf.csv",
        "required": True,
        "min_rows": 20,
        "fallback_tokens": ("ekf", "groundtruth", "ground_truth", "reference"),
        "exclude_tokens": ("zed", "path"),
        "required_field_tokens": ("position", "orientation"),
    },
    "motion_command": {
        "preferred": ["/terrasentia/motion_command"],
        "filename": "motion_command.csv",
        "required": False,
        "min_rows": 1,
        "fallback_tokens": ("motion_command", "cmd_vel", "command"),
        "exclude_tokens": (),
        "required_field_tokens": (),
    },
    "mhe_output": {
        "preferred": ["/terrasentia/mhe_output"],
        "filename": "mhe_output.csv",
        "required": False,
        "min_rows": 1,
        "fallback_tokens": ("mhe",),
        "exclude_tokens": (),
        "required_field_tokens": (),
    },
}


def human_bytes(n: int | float | None) -> str:
    if n is None or not math.isfinite(float(n)):
        return "unknown"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def message_time_ns(msg: Any, bag_timestamp_ns: int) -> int:
    """Prefer header timestamp; fall back to rosbag receipt timestamp."""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is not None:
        # rosbags normally exposes ROS1 time using ROS2-style sec/nanosec names.
        sec = getattr(stamp, "sec", getattr(stamp, "secs", None))
        nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", None))
        if sec is not None and nsec is not None:
            candidate = int(sec) * 1_000_000_000 + int(nsec)
            if candidate > 0:
                return candidate
    return int(bag_timestamp_ns)


def scalar_or_json(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, bytes, bool, int, float)):
        if isinstance(value, bytes):
            return value.hex()
        return value
    # numpy scalar-like
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return json.dumps([scalar_or_json(x) for x in value], separators=(",", ":"))
    if hasattr(value, "tolist"):
        try:
            return json.dumps(value.tolist(), separators=(",", ":"))
        except Exception:
            pass
    return str(value)


def flatten_message(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested rosbags-generated dataclass messages.

    Arrays stay as compact JSON strings so custom fpn_msgs can be preserved
    without knowing their schema in advance.
    """
    out: dict[str, Any] = {}

    if obj is None:
        if prefix:
            out[prefix] = ""
        return out

    if isinstance(obj, (str, bytes, bool, int, float)):
        if prefix:
            out[prefix] = scalar_or_json(obj)
        return out

    # numpy arrays and list/tuple values should remain one CSV cell.
    if isinstance(obj, (list, tuple)) or hasattr(obj, "shape"):
        if prefix:
            out[prefix] = scalar_or_json(obj)
        return out

    if is_dataclass(obj):
        for f in fields(obj):
            key = f"{prefix}.{f.name}" if prefix else f.name
            value = getattr(obj, f.name)
            if is_dataclass(value):
                out.update(flatten_message(value, key))
            elif isinstance(value, (list, tuple)) or hasattr(value, "shape"):
                out[key] = scalar_or_json(value)
            elif isinstance(value, (str, bytes, bool, int, float)) or value is None:
                out[key] = scalar_or_json(value)
            else:
                nested = flatten_message(value, key)
                if nested:
                    out.update(nested)
                else:
                    out[key] = scalar_or_json(value)
        return out

    slots = getattr(obj, "__slots__", None)
    if slots:
        for name in slots:
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(value, (str, bytes, bool, int, float)) or value is None:
                out[key] = scalar_or_json(value)
            elif isinstance(value, (list, tuple)) or hasattr(value, "shape"):
                out[key] = scalar_or_json(value)
            else:
                out.update(flatten_message(value, key))
        return out

    if prefix:
        out[prefix] = scalar_or_json(obj)
    return out


def topic_inventory(reader: Any) -> list[dict[str, Any]]:
    inventory = []
    for c in reader.connections:
        inventory.append(
            {
                "topic": c.topic,
                "msgtype": c.msgtype,
                "msgcount": getattr(c, "msgcount", None),
            }
        )
    inventory.sort(key=lambda x: x["topic"])
    return inventory


def choose_topic(inventory: list[dict[str, Any]], spec: dict[str, Any]) -> str | None:
    names = [str(x["topic"]) for x in inventory]

    for preferred in spec["preferred"]:
        if preferred in names:
            return preferred

    best: tuple[int, str] | None = None
    for name in names:
        low = name.lower()
        if any(token in low for token in spec["exclude_tokens"]):
            continue
        score = sum(10 for token in spec["fallback_tokens"] if token in low)
        if low.startswith("/terrasentia/"):
            score += 2
        if score <= 0:
            continue
        candidate = (score, name)
        if best is None or candidate > best:
            best = candidate

    return best[1] if best else None


def write_stream_csv(
    reader: Any,
    topic: str,
    out_path: Path,
) -> tuple[int, list[str], str]:
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        raise RuntimeError(f"No connection found for topic {topic}")

    # One topic should have one schema. If multiple connections exist, read all.
    row_count = 0
    first_fields: list[str] | None = None
    msgtype = conns[0].msgtype
    writer = None
    file_handle = None

    try:
        for conn, bag_ts_ns, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            flat = flatten_message(msg)
            ts_ns = message_time_ns(msg, bag_ts_ns)

            row = {
                "timestamp_ns": ts_ns,
                "timestamp_s": f"{ts_ns / 1e9:.9f}",
                "bag_timestamp_ns": int(bag_ts_ns),
                "topic": topic,
                "msgtype": conn.msgtype,
            }
            row.update(flat)

            if writer is None:
                first_fields = list(row.keys())
                out_path.parent.mkdir(parents=True, exist_ok=True)
                file_handle = out_path.open("w", newline="", encoding="utf-8")
                writer = csv.DictWriter(
                    file_handle,
                    fieldnames=first_fields,
                    extrasaction="ignore",
                )
                writer.writeheader()

            writer.writerow(row)
            row_count += 1
    finally:
        if file_handle is not None:
            file_handle.close()

    return row_count, (first_fields or []), msgtype


def verify_stream(
    logical_name: str,
    out_path: Path,
    row_count: int,
    fieldnames: list[str],
    spec: dict[str, Any],
) -> tuple[bool, str]:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False, "output missing or empty"
    if row_count < int(spec["min_rows"]):
        return False, f"only {row_count} rows; need >= {spec['min_rows']}"

    lower_fields = " ".join(x.lower() for x in fieldnames)
    for token in spec["required_field_tokens"]:
        if token.lower() not in lower_fields:
            return False, f"expected field token '{token}' not found"

    # Extra semantic checks.
    if logical_name == "motors":
        payload_fields = [
            x for x in fieldnames
            if x not in {
                "timestamp_ns", "timestamp_s", "bag_timestamp_ns", "topic", "msgtype"
            }
        ]
        if len(payload_fields) < 1:
            return False, "motor stream had no payload fields"

    return True, "ok"


def remote_size(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        value = r.headers.get("Content-Length")
        return int(value) if value and value.isdigit() else None
    except Exception:
        return None


def download_with_resume(
    session: requests.Session,
    url: str,
    final_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """
    Download to *.part with HTTP range resume when the server supports it.
    Rename to .bag only after transfer completion.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    part = final_path.with_suffix(final_path.suffix + ".part")

    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    with session.get(
        url,
        stream=True,
        allow_redirects=True,
        headers=headers,
        timeout=(30, 180),
    ) as r:
        if existing > 0 and r.status_code != 206:
            # Server ignored Range; restart cleanly.
            existing = 0
            headers = {}
            r.close()
            if part.exists():
                part.unlink()
            return download_with_resume(session, url, final_path, chunk_size)

        r.raise_for_status()
        content_len = r.headers.get("Content-Length")
        incoming = int(content_len) if content_len and content_len.isdigit() else None
        total = existing + incoming if incoming is not None else None

        mode = "ab" if existing > 0 else "wb"
        downloaded = existing
        last_report = time.monotonic()

        with part.open(mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_report >= 2.0:
                    if total:
                        pct = 100.0 * downloaded / total
                        print(
                            f"\r    {human_bytes(downloaded)} / {human_bytes(total)} "
                            f"({pct:5.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r    {human_bytes(downloaded)} downloaded",
                            end="",
                            flush=True,
                        )
                    last_report = now

        print()

    if total is not None and part.stat().st_size != total:
        raise RuntimeError(
            f"download size mismatch: got {part.stat().st_size}, expected {total}"
        )

    part.replace(final_path)
    return final_path


def download_metadata(session: requests.Session, root: Path) -> None:
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name, url in METADATA_URLS.items():
        dest = meta_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        r = session.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)


def extraction_already_verified(seq_dir: Path) -> bool:
    manifest = seq_dir / "extraction_manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not data.get("verified", False):
        return False

    required_files = [
        seq_dir / STREAM_SPECS[name]["filename"]
        for name in ("imu", "motors", "gps", "reference_ekf")
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required_files)


def extract_and_verify(
    bag_path: Path,
    seq_dir: Path,
    sequence: dict[str, str],
) -> dict[str, Any]:
    seq_dir.mkdir(parents=True, exist_ok=True)

    with AnyReader([bag_path]) as reader:
        inventory = topic_inventory(reader)
        (seq_dir / "topic_inventory.json").write_text(
            json.dumps(inventory, indent=2),
            encoding="utf-8",
        )

        chosen: dict[str, str | None] = {}
        for name, spec in STREAM_SPECS.items():
            chosen[name] = choose_topic(inventory, spec)

        missing = [
            name for name, spec in STREAM_SPECS.items()
            if spec["required"] and chosen[name] is None
        ]
        if missing:
            raise RuntimeError(
                "required topic category/categories not found: "
                + ", ".join(missing)
            )

        stream_results: dict[str, Any] = {}
        all_required_ok = True

        for name, spec in STREAM_SPECS.items():
            topic = chosen[name]
            if topic is None:
                stream_results[name] = {
                    "topic": None,
                    "required": spec["required"],
                    "extracted": False,
                    "verified": not spec["required"],
                    "reason": "topic not present",
                }
                continue

            out_path = seq_dir / spec["filename"]
            print(f"    extracting {name:<15} <- {topic}")

            try:
                count, fieldnames, msgtype = write_stream_csv(reader, topic, out_path)
                ok, reason = verify_stream(
                    name, out_path, count, fieldnames, spec
                )
            except Exception as exc:
                count = 0
                fieldnames = []
                msgtype = ""
                ok = False
                reason = f"{type(exc).__name__}: {exc}"

            if spec["required"] and not ok:
                all_required_ok = False

            stream_results[name] = {
                "topic": topic,
                "msgtype": msgtype,
                "required": bool(spec["required"]),
                "extracted": out_path.exists() and out_path.stat().st_size > 0,
                "verified": bool(ok),
                "reason": reason,
                "rows": count,
                "file": spec["filename"],
                "size_bytes": out_path.stat().st_size if out_path.exists() else 0,
                "sha256": sha256_file(out_path) if ok and out_path.exists() else None,
                "fieldnames": fieldnames,
            }

    if not all_required_ok:
        failed = [
            f"{name}: {info['reason']}"
            for name, info in stream_results.items()
            if STREAM_SPECS[name]["required"] and not info["verified"]
        ]
        raise RuntimeError("verification failed: " + "; ".join(failed))

    manifest = {
        "schema": "terrasentia_minimal_numeric_v1",
        "sequence": sequence["stem"],
        "group": sequence["group"],
        "role": sequence["role"],
        "source_url": sequence["url"],
        "source_bag_filename": bag_path.name,
        "source_bag_size_bytes": bag_path.stat().st_size,
        "source_bag_sha256": sha256_file(bag_path),
        "verified": True,
        "reference_note": (
            "/terrasentia/ekf is the dataset's fused/estimated pose reference; "
            "it is not independent motion-capture ground truth."
        ),
        "timestamp_note": (
            "CSV timestamp_ns prefers each ROS message header stamp when present; "
            "bag_timestamp_ns is retained separately for auditing."
        ),
        "streams": stream_results,
    }

    # Write the manifest LAST. This acts as the commit marker that allows deletion.
    manifest_path = seq_dir / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def clean_failed_output(seq_dir: Path) -> None:
    """
    Keep useful inventory/partial CSVs for diagnosis. Only remove a stale manifest
    so a failed extraction can never be mistaken for a verified one.
    """
    manifest = seq_dir / "extraction_manifest.json"
    if manifest.exists():
        manifest.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sequentially download, extract numerical TerraSentia streams, "
            "verify them, and delete each large bag after success."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Output root. Default: directory containing this script "
            "(recommended public_datasets/aifarms)."
        ),
    )
    p.add_argument(
        "--keep-bags",
        action="store_true",
        help="Do not delete successfully extracted bag files.",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional sequence stems to process.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List the selected minimal sequences and exit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = (args.root or script_dir).resolve()

    selected = SEQUENCES
    if args.only:
        wanted = set(args.only)
        selected = [s for s in SEQUENCES if s["stem"] in wanted]
        unknown = wanted - {s["stem"] for s in selected}
        if unknown:
            print("Unknown sequence(s):", ", ".join(sorted(unknown)), file=sys.stderr)
            return 2

    if args.list:
        for s in selected:
            print(f"{s['stem']}: {s['role']}")
        return 0

    processed_root = root / "processed"
    cache_root = root / ".download_cache"
    processed_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("TerraSentia minimal numeric extractor")
    print(f"Root: {root}")
    print(f"Selected sequences: {len(selected)}")
    print("Deletion policy: " + ("KEEP bags" if args.keep_bags else "DELETE after verified extraction"))
    print("=" * 76)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "DigitalTwinDivergence-TerraSentiaExtractor/1.0 "
                "(research dataset downloader)"
            )
        }
    )

    try:
        download_metadata(session, root)
    except Exception as exc:
        print(f"[warning] metadata download failed: {exc}")

    failures: list[dict[str, str]] = []
    completions: list[dict[str, Any]] = []

    for idx, seq in enumerate(selected, start=1):
        stem = seq["stem"]
        seq_dir = processed_root / stem
        bag_path = cache_root / f"{stem}.bag"

        print()
        print(f"[{idx}/{len(selected)}] {stem}")
        print(f"  role: {seq['role']}")

        if extraction_already_verified(seq_dir):
            print("  [skip] verified extracted dataset already exists")
            # Remove a leftover cached bag from a previously interrupted cleanup.
            if bag_path.exists() and not args.keep_bags:
                print(f"  [cleanup] deleting leftover bag {human_bytes(bag_path.stat().st_size)}")
                bag_path.unlink()
            continue

        clean_failed_output(seq_dir)

        try:
            size = remote_size(session, seq["url"])
            if size is not None:
                usage = shutil.disk_usage(root)
                safety = 2 * 1024**3
                need = size + safety
                print(f"  remote bag size: {human_bytes(size)}")
                print(f"  free disk:       {human_bytes(usage.free)}")
                if usage.free < need:
                    raise RuntimeError(
                        f"not enough free disk for one bag plus 2 GB safety margin "
                        f"(need about {human_bytes(need)}, have {human_bytes(usage.free)})"
                    )
            else:
                print("  remote bag size: unavailable before download")

            print("  [download] starting/resuming one bag only")
            download_with_resume(session, seq["url"], bag_path)
            print(f"  [download] complete: {human_bytes(bag_path.stat().st_size)}")

            print("  [extract] reading only numeric topics needed by the project")
            manifest = extract_and_verify(bag_path, seq_dir, seq)
            completions.append(manifest)

            retained = sum(
                int(info.get("size_bytes", 0))
                for info in manifest["streams"].values()
                if info.get("extracted")
            )
            print(f"  [verify] SUCCESS; retained CSV payload: {human_bytes(retained)}")

            if args.keep_bags:
                print("  [keep] --keep-bags specified; source bag retained")
            else:
                bag_size = bag_path.stat().st_size
                bag_path.unlink()
                print(f"  [delete] verified source bag deleted ({human_bytes(bag_size)})")

        except KeyboardInterrupt:
            print("\nInterrupted. Partial .bag is kept for resume.", file=sys.stderr)
            return 130
        except Exception as exc:
            failures.append({"sequence": stem, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
            if bag_path.exists():
                print(
                    f"  [safety] bag KEPT for inspection/resume: {bag_path} "
                    f"({human_bytes(bag_path.stat().st_size)})",
                    file=sys.stderr,
                )

    summary = {
        "schema": "terrasentia_minimal_download_run_v1",
        "root": str(root),
        "selected_sequences": [s["stem"] for s in selected],
        "successful_this_run": [m["sequence"] for m in completions],
        "failures": failures,
        "keep_bags": bool(args.keep_bags),
    }
    (root / "download_run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    if failures:
        print(f"Finished with {len(failures)} failure(s).")
        print("Failed bags were intentionally retained.")
        for f in failures:
            print(f"  - {f['sequence']}: {f['error']}")
        return 1

    print("All selected sequences are available as verified compact CSV datasets.")
    if not args.keep_bags:
        print("No successfully processed .bag files were retained.")
    print(f"Processed data: {processed_root}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
