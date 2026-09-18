"""Replay frozen rate policies through measured UGV01 response times.

The trace supplies source arrivals, response sizes, and contract states. A
single outstanding HTTP request is allowed, matching the dashboard client.
No sensor measurements are interpolated or synthesized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DigitalTwin.dashboard.contracts import ResourcePolicy, load_contract_config


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "public_datasets" / "ugv01_live_contract"
OUTPUT = ROOT / "results" / "ugv01_live_contract_trace_replay"
POLICIES = ("static-low", "static-high", "aoi-only", "contract-aware")


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def read_trace(path: Path) -> list[dict]:
    rows = [json.loads(line)["point"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError(f"Trace too short: {path}")
    return rows


def replay(points: list[dict], policy_name: str, config: dict) -> dict:
    """Run an on-demand cached-source model with one serial request at a time."""
    policy = ResourcePolicy(policy_name, config)
    source_clock = [float(point.get("source_time_s", math.nan)) for point in points]
    if all(math.isfinite(value) for value in source_clock) and all(
        right > left for left, right in zip(source_clock, source_clock[1:])
    ):
        source_times = [value - source_clock[0] for value in source_clock]
    else:
        t0 = float(points[0]["t"])
        source_times = [float(point["t"]) - t0 for point in points]
    horizon = source_times[-1]
    send = 0.0
    index = 0
    deliveries = []
    ages = []
    bytes_total = 0
    repeated = 0
    previous_index = -1
    modes = Counter()
    switches = 0

    while send <= horizon:
        while index + 1 < len(points) and source_times[index + 1] <= send + 1e-9:
            index += 1
        source = points[index]
        latency = max(0.0, float(source.get("latency_ms") or 0.0) / 1000.0)
        complete = send + latency
        if complete > horizon:
            break
        if index == previous_index:
            repeated += 1
        previous_index = index
        bytes_total += int(source.get("payload_bytes") or 0)
        age = complete - source_times[index]
        ages.append(age)
        deliveries.append(complete)
        modes[policy.mode] += 1

        # Contract states are recorded evidence, not recalculated under the
        # counterfactual schedule. This only drives policy decisions.
        contracts = source.get("contracts") or []
        if policy.update(complete, age, contracts) is not None:
            switches += 1
        period = 1.0 / policy.update_rate_hz
        send = max(complete, send + period)

    return {
        "policy": policy_name,
        "source_duration_s": horizon,
        "requests": len(deliveries),
        "requests_per_s": len(deliveries) / horizon,
        "response_bytes": bytes_total,
        "response_bytes_per_s": bytes_total / horizon,
        "repeated_source_fraction": repeated / len(deliveries) if deliveries else math.nan,
        "aoi_at_delivery_median_s": median(ages) if ages else math.nan,
        "aoi_at_delivery_p95_s": percentile(ages, 0.95),
        "rate_switches": switches,
        "economy_fraction": modes["economy"] / len(deliveries) if deliveries else math.nan,
        "normal_fraction": modes["normal"] / len(deliveries) if deliveries else math.nan,
        "high_fraction": modes["high"] / len(deliveries) if deliveries else math.nan,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(dataset: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "DigitalTwin" / "configs" / "ugv01_live_service_contracts.json"
    config = load_contract_config(config_path)
    run_rows = []
    for trial in range(1, 6):
        path = dataset / "trials" / "contract_aware" / f"trial_{trial:02d}" / "live_contract.jsonl"
        points = read_trace(path)
        for policy in POLICIES:
            row = replay(points, policy, config)
            row["trial"] = trial
            row["source_log"] = str(path.relative_to(dataset)).replace("\\", "/")
            run_rows.append(row)
    write_csv(output / "counterfactual_per_trial.csv", run_rows)

    summary = []
    for policy in POLICIES:
        rows = [row for row in run_rows if row["policy"] == policy]
        result = {"policy": policy, "trials": len(rows)}
        for key in ("requests_per_s", "response_bytes_per_s", "repeated_source_fraction", "aoi_at_delivery_p95_s", "rate_switches"):
            values = [float(row[key]) for row in rows]
            result[f"{key}_mean"] = mean(values)
            result[f"{key}_min"] = min(values)
            result[f"{key}_max"] = max(values)
        summary.append(result)
    write_csv(output / "counterfactual_policy_summary.csv", summary)

    # Each policy uses the same physical trace within a trial; differences
    # are therefore paired at trial level, not at timestamp level.
    paired = []
    for trial in range(1, 6):
        by_policy = {row["policy"]: row for row in run_rows if row["trial"] == trial}
        baseline = by_policy["static-high"]
        for policy in POLICIES:
            row = by_policy[policy]
            paired.append({
                "trial": trial,
                "policy": policy,
                "response_byte_saving_vs_static_high_fraction": 1.0 - row["response_bytes"] / baseline["response_bytes"],
                "request_saving_vs_static_high_fraction": 1.0 - row["requests"] / baseline["requests"],
                "aoi_p95_delta_vs_static_high_s": row["aoi_at_delivery_p95_s"] - baseline["aoi_at_delivery_p95_s"],
            })
    write_csv(output / "paired_policy_differences.csv", paired)

    # Fluid capacity sensitivity. This is a scheduler-demand calculation, not
    # fresh-sensor replay: offered demand comes from the measured policy logs
    # and delivered demand is capped by an explicit transport capacity.
    packet_sizes = [
        float(point.get("payload_bytes") or 0.0)
        for trial in range(1, 6)
        for point in read_trace(dataset / "trials" / "contract_aware" / f"trial_{trial:02d}" / "live_contract.jsonl")
    ]
    mean_packet_bytes = mean(packet_sizes)
    observed_demand = {
        "static-low": 2.0,
        "static-high": 10.0,
        "aoi-only": mean(
            float(point.get("requested_update_rate_hz") or 0.0)
            for trial in range(1, 6)
            for point in read_trace(dataset / "trials" / "aoi_only" / f"trial_{trial:02d}" / "live_contract.jsonl")
        ),
        "contract-aware": mean(
            float(point.get("requested_update_rate_hz") or 0.0)
            for trial in range(1, 6)
            for point in read_trace(dataset / "trials" / "contract_aware" / f"trial_{trial:02d}" / "live_contract.jsonl")
        ),
    }
    capacity_rows = []
    for capacity in (1.5, 2.0, 5.0, 10.0):
        high_delivered = min(observed_demand["static-high"], capacity)
        for policy in POLICIES:
            demand = observed_demand[policy]
            delivered = min(demand, capacity)
            capacity_rows.append({
                "capacity_hz": capacity,
                "policy": policy,
                "offered_rate_hz": demand,
                "capacity_limited_rate_hz": delivered,
                "response_payload_bytes_per_s": delivered * mean_packet_bytes,
                "payload_saving_vs_static_high_fraction": 1.0 - delivered / high_delivered,
                "capacity_saturated": demand > capacity,
            })
    write_csv(output / "capacity_sensitivity.csv", capacity_rows)

    labels = [item["policy"] for item in summary]
    rates = [item["requests_per_s_mean"] for item in summary]
    bytes_rate = [item["response_bytes_per_s_mean"] for item in summary]
    ages = [item["aoi_at_delivery_p95_s_mean"] for item in summary]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    for ax, values, title, ylabel in zip(
        axes,
        (rates, bytes_rate, ages),
        ("Delivered requests", "Response payload", "AoI at delivery"),
        ("requests/s", "bytes/s", "p95 seconds"),
    ):
        ax.bar(labels, values, color=["#8aa6b8", "#b5a173", "#71939c", "#327a69"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=32)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "trace_replay_policy_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    styles = {"static-low": "o-", "static-high": "s-", "aoi-only": "^-", "contract-aware": "D-"}
    for policy in POLICIES:
        rows = [row for row in capacity_rows if row["policy"] == policy]
        ax.plot([row["capacity_hz"] for row in rows], [row["capacity_limited_rate_hz"] for row in rows], styles[policy], label=policy)
    ax.axvline(1.5, color="#555555", linestyle="--", linewidth=1, label="measured capacity")
    ax.set_xlabel("Provisioned transport capacity (updates/s)")
    ax.set_ylabel("Capacity-limited delivered rate (updates/s)")
    ax.set_title("Policy demand becomes distinguishable only above the bottleneck")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output / "transport_capacity_sensitivity.png", dpi=220)
    plt.close(fig)

    metadata = {
        "schema": "ugv01_live_contract_trace_replay_v1",
        "input_dataset": str(dataset),
        "input_trials": [1, 2, 3, 4, 5],
        "input_policy_trace": "contract-aware",
        "counterfactual_policies": list(POLICIES),
        "transport": "one outstanding on-demand request; no parallel responses",
        "source": "latest recorded source sample at request send time; no synthesized updates",
        "latency": "recorded HTTP latency of selected source sample",
        "bytes": "recorded JSON response payload only; HTTP headers and request bytes excluded",
        "policy_inputs": "recorded contract statuses reused as exogenous decisions; service outcomes not recomputed",
        "statistical_unit": "physical trial; five paired traces",
        "limits": [
            "The original stream contains only about 1.5 new samples/s.",
            "The replay cannot establish 5/10 Hz fresh-sensor fidelity or service satisfaction.",
            "Latency is sampled from one measured condition; no independent wireless-stress factor is claimed.",
            "Capacity sensitivity is a fluid demand bound; it does not estimate service satisfaction.",
        ],
    }
    (output / "simulation_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# UGV01 Trace-Driven Communication Replay",
        "",
        "Five measured contract-aware UGV01 traces were replayed under all four frozen policies. Every policy in a trial shares the same physical source trace. The simulation uses measured response times and payload sizes, a serial HTTP client, and the latest recorded sample available at request time. No new sensor samples are fabricated.",
        "",
        "| Policy | Requests/s | Response bytes/s | Repeated source fraction | p95 delivery AoI (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['policy']} | {row['requests_per_s_mean']:.3f} | {row['response_bytes_per_s_mean']:.1f} | {row['repeated_source_fraction_mean']:.3f} | {row['aoi_at_delivery_p95_s_mean']:.3f} |")
    lines.extend([
        "",
        "These are simulated request and response-payload costs under the recorded service time. Policy inputs use recorded contract states, so counterfactual contract satisfaction is not estimated. A request to a cached source may return a repeated sample. The paper must distinguish these replayed resource quantities from directly measured live throughput.",
        "",
        "The low-speed GPS course limitation in the observed live experiment remains. No policy-level claim about preservation of 1/5/10-s service qualification follows from this replay.",
        "",
        "## Capacity sensitivity",
        "",
        f"Using the measured mean response size ({mean_packet_bytes:.1f} B), a separate fluid calculation caps each policy's observed offered demand at 1.5, 2, 5, and 10 updates/s. At the measured 1.5-update/s capacity every policy saturates and no resource separation is possible. At 10 updates/s, the nominal payload-demand reductions relative to static-high are 80.0% for static-low, {100.0 * (1.0 - observed_demand['aoi-only'] / 10.0):.1f}% for AoI-only, and {100.0 * (1.0 - observed_demand['contract-aware'] / 10.0):.1f}% for contract-aware. These are provisioned-capacity demand bounds, not measured savings or service-preservation results.",
    ])
    (output / "trace_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "trace_replay_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
