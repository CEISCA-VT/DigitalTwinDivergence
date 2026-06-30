"""Lock a Mahalanobis threshold from benign nominal CSVs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .common import parse_run_name, read_rows


def empirical_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot lock threshold from zero samples")
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(quantile * len(values)) - 1))
    return float(values[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="DigitalTwin/configs/locked_threshold.json")
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.inputs:
        matches = sorted(Path().glob(item)) if any(ch in item for ch in "*?[") else [Path(item)]
        paths.extend(path for path in matches if path.exists())

    nominal_paths = [path for path in paths if parse_run_name(path).get("attack") in {"none", ""}]
    values: list[float] = []
    for path in nominal_paths:
        for row in read_rows(path):
            values.append(float(row["mahalanobis"]))

    threshold = empirical_quantile(values, 1.0 - args.alpha)
    false_alarm_count = sum(value > threshold for value in values)
    payload = {
        "threshold": threshold,
        "alpha": args.alpha,
        "samples": len(values),
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": false_alarm_count / len(values) if values else None,
        "source_files": [str(path) for path in nominal_paths],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
