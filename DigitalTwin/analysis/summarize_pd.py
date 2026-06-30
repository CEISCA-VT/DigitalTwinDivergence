"""Summarize empirical detection probability from experiment CSVs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .common import is_attack_row, parse_run_name, read_rows, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    parser.add_argument("--out", default="DigitalTwin/datasets/analysis/pd_summary.csv")
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.inputs:
        matches = sorted(Path().glob(item)) if any(ch in item for ch in "*?[") else [Path(item)]
        paths.extend(path for path in matches if path.exists())

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        meta = parse_run_name(path)
        rows = read_rows(path)
        attacked_rows = [row for row in rows if is_attack_row(row)]
        considered_rows = attacked_rows if attacked_rows else rows
        detections = sum(int(float(row["detected"])) for row in considered_rows)
        trial_detected = int(detections > 0)
        first_detection_s = ""
        for row in considered_rows:
            if int(float(row["detected"])) > 0:
                first_detection_s = row["time_s"]
                break
        key = (meta["speed"], meta["terrain"], meta["latency"], meta["attack"], meta["epsilon"])
        grouped[key].append(
            {
                "trial": meta["trial"],
                "rows": len(considered_rows),
                "detections": detections,
                "trial_detected": trial_detected,
                "first_detection_s": first_detection_s,
                "mean_epsilon_min_m": sum(float(row["epsilon_min_m"]) for row in considered_rows) / len(considered_rows),
            }
        )

    summary_rows: list[dict[str, object]] = []
    for key, trials in sorted(grouped.items()):
        speed, terrain, latency, attack, epsilon = key
        sample_rows = sum(int(trial["rows"]) for trial in trials)
        sample_detections = sum(int(trial["detections"]) for trial in trials)
        trial_detections = sum(int(trial["trial_detected"]) for trial in trials)
        first_times = [float(trial["first_detection_s"]) for trial in trials if trial["first_detection_s"] != ""]
        summary_rows.append(
            {
                "speed": speed,
                "terrain": terrain,
                "latency": latency,
                "attack": attack,
                "epsilon_m": epsilon,
                "trials": len(trials),
                "trial_pd": trial_detections / len(trials),
                "sample_detection_rate": sample_detections / sample_rows if sample_rows else 0.0,
                "mean_first_detection_s": sum(first_times) / len(first_times) if first_times else "",
                "mean_epsilon_min_m": sum(float(trial["mean_epsilon_min_m"]) for trial in trials) / len(trials),
            }
        )

    fieldnames = [
        "speed",
        "terrain",
        "latency",
        "attack",
        "epsilon_m",
        "trials",
        "trial_pd",
        "sample_detection_rate",
        "mean_first_detection_s",
        "mean_epsilon_min_m",
    ]
    write_rows(Path(args.out), summary_rows, fieldnames)
    print(Path(args.out))


if __name__ == "__main__":
    main()
