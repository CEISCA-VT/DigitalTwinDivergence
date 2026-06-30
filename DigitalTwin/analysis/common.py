"""Shared CSV and filename helpers for analysis scripts."""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Iterable


RUN_RE = re.compile(
    r"speed-(?P<speed>[0-9.]+)_terrain-(?P<terrain>[0-9.]+)_latency-(?P<latency>[0-9]+)"
    r"_attack-(?P<attack>[a-z_]+)(?:_eps-(?P<epsilon>[0-9.]+))?_trial-(?P<trial>[0-9]+)"
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
    if not match:
        return {
            "speed": "",
            "terrain": "",
            "latency": "",
            "attack": "",
            "epsilon": "",
            "trial": "",
        }
    data = match.groupdict()
    data["epsilon"] = data.get("epsilon") or ""
    return data


def is_attack_row(row: dict[str, str]) -> bool:
    label = row.get("attack_label", "none")
    return label not in {"", "none", "replay-warmup"}
