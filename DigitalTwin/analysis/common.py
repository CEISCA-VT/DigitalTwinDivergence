"""Shared CSV and filename helpers for analysis scripts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re
from statistics import median
from typing import Iterable


RUN_RE = re.compile(
    r"speed-(?P<speed>[0-9.]+)_terrain-(?P<terrain>[0-9.]+)_latency-(?P<latency>[0-9]+)"
    r"_attack-(?P<attack>[a-z_]+)(?:_eps-(?P<epsilon>[0-9.]+))?_trial-(?P<trial>[0-9]+)"
)

HARDWARE_RUN_RE = re.compile(
    r"speed-(?P<speed>low|medium)_surface-(?P<surface>.+?)"
    r"_latency-(?P<latency>.+?)_route-(?P<route>.+?)"
    r"_attack-(?P<attack>.+?)_trial-(?P<trial>[0-9]+)"
    r"(?:_(?P<timestamp>[0-9]{8}_[0-9]{6}))?$"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_run_name(path: Path) -> dict[str, str]:
    match = RUN_RE.search(path.stem)
    if match:
        data = match.groupdict()
        data["epsilon"] = data.get("epsilon") or ""
        data.update({"surface": data.get("terrain", ""), "route": "", "timestamp": ""})
        return data

    match = HARDWARE_RUN_RE.search(path.stem)
    if match:
        data = match.groupdict()
        data.update({"terrain": data.get("surface", ""), "epsilon": ""})
        data["timestamp"] = data.get("timestamp") or ""
        return data

    return {
        "speed": "",
        "surface": "",
        "terrain": "",
        "latency": "",
        "route": "",
        "attack": "",
        "epsilon": "",
        "trial": "",
        "timestamp": "",
    }


def is_attack_row(row: dict[str, str]) -> bool:
    label = row.get("attack_label", "none")
    return label not in {"", "none", "replay-warmup"}


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: object, default: float | None = None) -> float | None:
    text = str(value).strip()
    if text in {"", "None", "null", "nan"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: object, default: int | None = None) -> int | None:
    parsed = parse_float(value, None)
    if parsed is None:
        return default
    return int(round(parsed))


def first_present(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row:
            return row[key]
    return ""


def cleaned_floats(values: Iterable[object]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        parsed = parse_float(value, None)
        if parsed is not None and math.isfinite(parsed):
            cleaned.append(parsed)
    return cleaned


def quantile(values: Iterable[object], q: float) -> float | None:
    cleaned = sorted(cleaned_floats(values))
    if not cleaned:
        return None
    if q <= 0:
        return cleaned[0]
    if q >= 1:
        return cleaned[-1]
    index = (len(cleaned) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return cleaned[low]
    weight = index - low
    return cleaned[low] * (1.0 - weight) + cleaned[high] * weight


def stats_dict(values: Iterable[object], prefix: str) -> dict[str, float | int | None]:
    cleaned = cleaned_floats(values)
    if not cleaned:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_mean": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_count": len(cleaned),
        f"{prefix}_min": min(cleaned),
        f"{prefix}_median": median(cleaned),
        f"{prefix}_mean": sum(cleaned) / len(cleaned),
        f"{prefix}_p95": quantile(cleaned, 0.95),
        f"{prefix}_p99": quantile(cleaned, 0.99),
        f"{prefix}_max": max(cleaned),
    }
