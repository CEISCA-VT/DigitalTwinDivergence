#!/usr/bin/env python3
"""Plot the corrected capacity-contract simulation summary."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ("static-low", "static-high", "aoi-only", "contract-aware")
COLORS = {
    "static-low": "#d62728",
    "static-high": "#2ca02c",
    "aoi-only": "#1f77b4",
    "contract-aware": "#ff7f0e",
}
MARKERS = {
    "static-low": "o",
    "static-high": "s",
    "aoi-only": "^",
    "contract-aware": "D",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service", default="global_state_tracking")
    args = parser.parse_args()

    rows = [row for row in read_rows(args.input) if row["service_id"] == args.service]
    if not rows:
        services = sorted({row["service_id"] for row in read_rows(args.input)})
        parser.error(f"service {args.service!r} not found; available: {', '.join(services)}")

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6), constrained_layout=True)
    panels = (
        ("transport_deliveries_per_s_mean", "Transport deliveries/s", None),
        ("p95_delivery_aoi_s_mean", "p95 delivery AoI (s)", None),
        (
            "satisfaction_given_observable_time_weighted_mean",
            "Time-weighted satisfaction",
            (0.0, 1.02),
        ),
        ("repeated_source_fraction_mean", "Repeated-source fraction", (0.0, 1.02)),
    )

    for policy in POLICIES:
        policy_rows = sorted(
            (row for row in rows if row["policy"] == policy),
            key=lambda row: number(row, "capacity_hz"),
        )
        capacities = [number(row, "capacity_hz") for row in policy_rows]
        for axis, (metric, ylabel, limits) in zip(axes, panels):
            values = [number(row, metric) for row in policy_rows]
            axis.plot(
                capacities,
                values,
                marker=MARKERS[policy],
                linewidth=2,
                markersize=6,
                color=COLORS[policy],
                label=policy,
            )
            axis.set_xlabel("Transport capacity (updates/s)")
            axis.set_ylabel(ylabel)
            if limits is not None:
                axis.set_ylim(*limits)
            axis.grid(alpha=0.25)

    axes[0].legend(fontsize=9)
    fig.suptitle(f"Trace-driven capacity replay: {args.service.replace('_', ' ')}", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
