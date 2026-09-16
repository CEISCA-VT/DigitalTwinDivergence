"""Offline position-only service evaluation on recorded UGV01 live traces.

This is a retrospective service definition. It never rewrites the online
heading-dependent decisions and never synthesizes higher-rate sensor samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DigitalTwin.dashboard.contracts import ResourcePolicy, load_contract_config


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "public_datasets" / "ugv01_live_contract"
OUTPUT = ROOT / "results" / "ugv01_live_position_contract"
POLICIES = ("static-low", "static-high", "aoi-only", "contract-aware")
STATES = ("pass", "fail", "unobservable")


def read_points(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["point"] for line in handle if line.strip()]


def finite(value: object) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_position(history: list[dict], spec: dict, quality: dict) -> dict:
    now = history[-1]
    result = {"service_id": spec["service_id"], "state": "unobservable", "reason": "reference unavailable",
              "position_error_m": None, "aoi_s": now.get("aoi_s")}
    if not now.get("gps_valid") or not all(finite(now.get(key)) for key in ("gps_x", "gps_y", "twin_x", "twin_y")):
        return result
    if int(now.get("satellites") or 0) < quality["minimum_satellites"] or not finite(now.get("hdop")) or float(now["hdop"]) > quality["maximum_hdop"]:
        result["reason"] = "GPS quality"
        return result
    if not finite(now.get("gps_age_s")) or float(now["gps_age_s"]) > quality["maximum_gps_age_s"]:
        result["reason"] = "GPS age"
        return result
    if not finite(now.get("aoi_s")):
        result["reason"] = "source age unavailable"
        return result
    start = None
    horizon = float(spec["horizon_s"])
    if horizon:
        target = float(now["t"]) - horizon
        candidates = [point for point in history[:-1] if point.get("gps_valid") and
                      all(finite(point.get(key)) for key in ("gps_x", "gps_y", "twin_x", "twin_y")) and
                      int(point.get("satellites") or 0) >= quality["minimum_satellites"] and
                      finite(point.get("hdop")) and float(point["hdop"]) <= quality["maximum_hdop"] and
                      finite(point.get("gps_age_s")) and float(point["gps_age_s"]) <= quality["maximum_gps_age_s"]]
        if candidates:
            start = min(candidates, key=lambda point: abs(float(point["t"]) - target))
        if start is None or abs(float(start["t"]) - target) > quality["maximum_horizon_match_error_s"]:
            result["reason"] = "no synchronized position window"
            return result
    if start is None:
        error = math.hypot(float(now["twin_x"]) - float(now["gps_x"]),
                           float(now["twin_y"]) - float(now["gps_y"]))
    else:
        error = math.hypot(
            (float(now["twin_x"]) - float(start["twin_x"])) - (float(now["gps_x"]) - float(start["gps_x"])),
            (float(now["twin_y"]) - float(start["twin_y"])) - (float(now["gps_y"]) - float(start["gps_y"])),
        )
    result["position_error_m"] = error
    result["state"] = "pass" if error <= spec["position_tolerance_m"] and float(now["aoi_s"]) <= spec["maximum_aoi_s"] else "fail"
    result["reason"] = "within contract" if result["state"] == "pass" else "position or AoI limit"
    return result


def evaluate_trace(points: list[dict], config: dict) -> list[dict]:
    return [
        {"time_s": point["t"], **evaluate_position(points[:index + 1], spec, config["reference_quality"])}
        for index, point in enumerate(points)
        for spec in config["services"]
    ]


def summarize(evaluations: list[dict], service_id: str) -> dict:
    states = Counter(row["state"] for row in evaluations if row["service_id"] == service_id)
    total = sum(states.values())
    observed = states["pass"] + states["fail"]
    return {"service_id": service_id, "windows": total, "pass": states["pass"], "fail": states["fail"],
            "unobservable": states["unobservable"], "observable_fraction": observed / total if total else math.nan,
            "satisfaction_given_observable": states["pass"] / observed if observed else math.nan}


def replay(points: list[dict], policy_name: str, config: dict) -> tuple[dict, list[dict]]:
    policy = ResourcePolicy(policy_name, config)
    source_times = [float(point["t"]) - float(points[0]["t"]) for point in points]
    horizon = source_times[-1]
    send = 0.0
    index = 0
    previous = -1
    history: list[dict] = []
    evaluations: list[dict] = []
    delivery_ages: list[float] = []
    bytes_total = repeated = switches = 0
    while send <= horizon:
        while index + 1 < len(points) and source_times[index + 1] <= send + 1e-9:
            index += 1
        source = points[index]
        complete = send + max(0.0, float(source.get("latency_ms") or 0) / 1000)
        if complete > horizon:
            break
        repeated += index == previous
        previous = index
        bytes_total += int(source.get("payload_bytes") or 0)
        age = complete - source_times[index]
        delivery_ages.append(age)
        sample = dict(source)
        sample["t"] = complete
        sample["aoi_s"] = age
        history.append(sample)
        current = [evaluate_position(history, spec, config["reference_quality"]) for spec in config["services"]]
        evaluations.extend(current)
        contracts = [{"service_id": item["service_id"], "status":
                     {"pass": "qualified", "fail": "withdrawn", "unobservable": "unobservable"}[item["state"]]}
                     for item in current]
        switches += policy.update(complete, age, contracts) is not None
        send = max(complete, send + 1.0 / policy.update_rate_hz)
    ages = sorted(delivery_ages)
    p95 = ages[int(math.ceil(0.95 * len(ages))) - 1] if ages else math.nan
    return ({"policy": policy_name, "requests": len(ages), "requests_per_s": len(ages) / horizon,
             "response_bytes_per_s": bytes_total / horizon, "p95_delivery_age_s": p95,
             "repeated_source_fraction": repeated / len(ages) if ages else math.nan,
             "rate_switches": switches}, evaluations)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(dataset: Path = DATASET, output: Path = OUTPUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = load_contract_config()
    measured: list[dict] = []
    for policy in POLICIES:
        for trial in range(1, 6):
            path = dataset / "trials" / policy.replace("-", "_") / f"trial_{trial:02d}" / "live_contract.jsonl"
            points = read_points(path)
            evaluation = evaluate_trace(points, config)
            for spec in config["services"]:
                measured.append({"policy": policy, "trial": trial, **summarize(evaluation, spec["service_id"])})
    write_csv(output / "measured_run_position_contracts.csv", measured)

    replay_rows: list[dict] = []
    for trial in range(1, 6):
        points = read_points(dataset / "trials" / "contract_aware" / f"trial_{trial:02d}" / "live_contract.jsonl")
        for policy in POLICIES:
            transport, evaluation = replay(points, policy, config)
            for spec in config["services"]:
                replay_rows.append({"trial": trial, **transport, **summarize(evaluation, spec["service_id"])})
    write_csv(output / "paired_replay_position_contracts.csv", replay_rows)
    policy_rows = []
    for policy in POLICIES:
        for spec in config["services"]:
            rows = [row for row in replay_rows if row["policy"] == policy and row["service_id"] == spec["service_id"]]
            policy_rows.append({"policy": policy, "service_id": spec["service_id"], "physical_traces": len(rows),
                                "mean_observable_fraction": mean(row["observable_fraction"] for row in rows),
                                "mean_satisfaction_given_observable": mean(row["satisfaction_given_observable"] for row in rows if finite(row["satisfaction_given_observable"])) if any(finite(row["satisfaction_given_observable"]) for row in rows) else math.nan,
                                "mean_requests_per_s": mean(row["requests_per_s"] for row in rows),
                                "mean_bytes_per_s": mean(row["response_bytes_per_s"] for row in rows)})
    write_csv(output / "paired_replay_policy_summary.csv", policy_rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    global_rows = [row for row in policy_rows if row["service_id"] == "global_state_tracking"]
    labels = [row["policy"] for row in global_rows]
    axes[0].bar(labels, [row["mean_observable_fraction"] for row in global_rows], color="#3c7d89")
    axes[0].set_ylabel("Observable fraction")
    axes[0].set_title("Position-only global service")
    axes[1].bar(labels, [row["mean_requests_per_s"] for row in global_rows], color="#b07842")
    axes[1].set_ylabel("Delivered requests/s")
    axes[1].set_title("Recorded source + serial HTTP")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "position_only_replay.png", dpi=200)
    plt.close(fig)

    manifest = {"schema": "ugv01_position_only_offline_v1", "dataset": str(dataset),
                "contract_source": str(config["provenance"]), "source_records": 2702,
                "states": list(STATES), "denominator": "observable windows only for satisfaction; observability separately",
                "local_position_error": "difference of world-frame displacement vectors; no heading required",
                "replay": "five recorded contract-aware source traces; four policies; one outstanding HTTP request",
                "limitations": ["Retrospective position-only service, not original online policy result",
                               "GPS is operational reference, not AprilTag ground truth",
                               "Recorded approximately 1.5 Hz source cannot establish fresh 5/10 Hz fidelity",
                               "Capacity sweeps in the separate transport analysis are hypothetical bounds"]}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Offline UGV01 Position-Only Contracts", "",
             "This is a retrospective service redefinition. It does not change the original online contracts or measured policies.",
             "GPS position is an operational reference, not independent physical ground truth. Local displacement-vector comparison is in a common world frame and does not require course.",
             "Pass and fail are counted only after the quality, history, and synchronization gates. Satisfaction uses observable windows; observability uses all windows.",
             "", "## Paired replay on five physical source traces", "",
             "| Policy | Global observable | Global satisfaction when observable | Requests/s | Bytes/s |", "|---|---:|---:|---:|---:|"]
    for row in global_rows:
        lines.append(f"| {row['policy']} | {row['mean_observable_fraction']:.3f} | {row['mean_satisfaction_given_observable']:.3f} | {row['mean_requests_per_s']:.3f} | {row['mean_bytes_per_s']:.1f} |")
    lines += ["", "The replay recomputes position-only outcomes at delivered samples; it does not create fresh high-rate measurements. The physical trace is the comparison unit."]
    (output / "position_only_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.dataset, args.output)


if __name__ == "__main__":
    main()
