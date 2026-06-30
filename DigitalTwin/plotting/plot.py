"""Plot experiment CSVs.

Example:
    python -m DigitalTwin.plotting.plot DigitalTwin/datasets/run.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_columns(path: Path) -> dict[str, list[float | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    columns: dict[str, list[float | str]] = {}
    for key in rows[0].keys():
        values: list[float | str] = []
        for row in rows:
            value = row[key]
            try:
                values.append(float(value))
            except ValueError:
                values.append(value)
        columns[key] = values
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/plots")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = read_columns(csv_path)
    stem = csv_path.stem

    plt.figure()
    plt.plot(c["truth_x_m"], c["truth_y_m"], label="truth")
    plt.plot(c["gps_x_m"], c["gps_y_m"], ".", markersize=2, label="gps")
    plt.plot(c["ekf_x_m"], c["ekf_y_m"], label="ekf")
    plt.axis("equal")
    plt.xlabel("east (m)")
    plt.ylabel("north (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_trajectory.png", dpi=160)
    plt.close()

    for y_name, title in [
        ("mahalanobis", "Mahalanobis distance"),
        ("epsilon_min_m", "epsilon_min"),
        ("epsilon_stealth_max_m", "epsilon stealth bound"),
        ("confidence", "confidence"),
        ("q_xx", "Q estimate"),
    ]:
        plt.figure()
        plt.plot(c["time_s"], c[y_name], label=y_name)
        if y_name == "mahalanobis":
            plt.plot(c["time_s"], c["threshold"], label="threshold")
        plt.xlabel("time (s)")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_{y_name}.png", dpi=160)
        plt.close()

    region_map = {"safe": 2.0, "warning": 1.0, "blind": 0.0}
    if "envelope_region" in c:
        plt.figure()
        plt.plot(c["time_s"], [region_map.get(str(v), -1.0) for v in c["envelope_region"]])
        plt.xlabel("time (s)")
        plt.ylabel("envelope")
        plt.yticks([0, 1, 2], ["blind", "warning", "safe"])
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_envelope.png", dpi=160)
        plt.close()

    plt.figure()
    plt.plot(c["time_s"], c["detected"], label="detected")
    plt.xlabel("time (s)")
    plt.ylabel("detection")
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_detection.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
