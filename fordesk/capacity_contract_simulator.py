#!/usr/bin/env python3
"""Capacity-controlled, trace-driven replay for UGV01 position contracts.

Run from any directory by passing --repo-root. The replay never fabricates a
new physical observation: cached responses retain their original source time,
and only newly delivered source observations enter contract history.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


POLICIES = ("static-low", "static-high", "aoi-only", "contract-aware")
STATES = ("qualified", "at_risk", "withdrawn", "unobservable")


def finite(value: object) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def read_points(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        points = [json.loads(line)["point"] for line in handle if line.strip()]
    if len(points) < 2:
        raise ValueError(f"Trace contains fewer than two points: {path}")
    return points


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PositionLifecycle:
    """Apply the repository warning and recovery rules to position-only results."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.states = {
            str(spec["service_id"]): "unobservable" for spec in config["services"]
        }
        self.recovery_since: dict[str, float | None] = {
            str(spec["service_id"]): None for spec in config["services"]
        }

    def update(
        self,
        result: dict[str, Any],
        spec: dict[str, Any],
        now_s: float,
    ) -> dict[str, str]:
        service_id = str(spec["service_id"])
        raw_state = str(result["state"])
        previous = self.states[service_id]

        if raw_state == "unobservable":
            status = "unobservable"
            self.recovery_since[service_id] = None
        elif raw_state == "fail":
            status = "withdrawn"
            self.recovery_since[service_id] = None
        else:
            position_error = float(result["position_error_m"])
            aoi = float(result["aoi_s"])
            position_tolerance = float(spec["position_tolerance_m"])
            maximum_aoi = float(spec["maximum_aoi_s"])
            normalized_margin = min(
                (position_tolerance - position_error) / position_tolerance,
                (maximum_aoi - aoi) / maximum_aoi,
            )
            warning = float(
                self.config["state_machine"]["warning_margin_fraction"]
            )
            candidate = "at_risk" if normalized_margin <= warning else "qualified"

            if previous == "withdrawn":
                if self.recovery_since[service_id] is None:
                    self.recovery_since[service_id] = now_s
                recovery = float(
                    self.config["state_machine"]["recovery_interval_s"]
                )
                if now_s - float(self.recovery_since[service_id]) < recovery:
                    status = "withdrawn"
                else:
                    status = candidate
                    self.recovery_since[service_id] = None
            else:
                status = candidate
                self.recovery_since[service_id] = None

        self.states[service_id] = status
        return {"service_id": service_id, "status": status}


def time_weighted_service_metrics(
    events: list[tuple[float, dict[str, str]]],
    service_id: str,
    horizon_s: float,
) -> dict[str, float]:
    durations = Counter({state: 0.0 for state in STATES})
    if not events or horizon_s <= 0.0:
        return {
            "observable_fraction_time_weighted": math.nan,
            "satisfaction_given_observable_time_weighted": math.nan,
        }

    for event_index, (event_time, statuses) in enumerate(events):
        next_time = (
            events[event_index + 1][0]
            if event_index + 1 < len(events)
            else horizon_s
        )
        duration = max(0.0, next_time - event_time)
        durations[statuses[service_id]] += duration

    covered = sum(durations.values())
    observable = (
        durations["qualified"]
        + durations["at_risk"]
        + durations["withdrawn"]
    )
    satisfied = durations["qualified"] + durations["at_risk"]
    return {
        "observable_fraction_time_weighted": observable / covered if covered else math.nan,
        "satisfaction_given_observable_time_weighted": (
            satisfied / observable if observable else math.nan
        ),
    }


def simulate_trial(
    points: list[dict[str, Any]],
    trial: int,
    policy_name: str,
    capacity_hz: float,
    config: dict[str, Any],
    resource_policy_class: type,
    evaluate_position: Any,
) -> list[dict[str, Any]]:
    if capacity_hz <= 0.0:
        raise ValueError("Transport capacity must be positive")

    policy = resource_policy_class(policy_name, config)
    lifecycle = PositionLifecycle(config)
    t0 = float(points[0]["t"])
    source_times = [float(point["t"]) - t0 for point in points]
    horizon = source_times[-1]

    send = 0.0
    source_index = 0
    previous_source_index = -1
    unique_history: list[dict[str, Any]] = []
    delivery_ages: list[float] = []
    contract_events: list[tuple[float, dict[str, str]]] = []
    request_state_counts: dict[str, Counter[str]] = {
        str(spec["service_id"]): Counter() for spec in config["services"]
    }
    mode_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    requests = repeated = bytes_total = switches = 0

    while send <= horizon + 1e-12:
        # The HTTP response is a snapshot of the latest source sample available
        # when the request is sent.
        while (
            source_index + 1 < len(points)
            and source_times[source_index + 1] <= send + 1e-9
        ):
            source_index += 1

        source = points[source_index]
        source_time = source_times[source_index]
        complete = send + 1.0 / capacity_hz
        if complete > horizon + 1e-12:
            break

        requests += 1
        bytes_total += int(source.get("payload_bytes") or 0)
        age = max(0.0, complete - source_time)
        delivery_ages.append(age)

        # Only a genuinely new source observation enters kinematic history.
        # Its timestamp remains the source timestamp, not the delivery time.
        if source_index != previous_source_index:
            historical = dict(source)
            historical["t"] = source_time
            unique_history.append(historical)
            previous_source_index = source_index
        else:
            repeated += 1

        current = dict(unique_history[-1])
        current["t"] = source_time
        current["aoi_s"] = age
        if finite(source.get("gps_age_s")):
            current["gps_age_s"] = float(source["gps_age_s"]) + age

        evaluation_history = unique_history[:-1] + [current]
        contracts: list[dict[str, str]] = []
        statuses: dict[str, str] = {}
        for spec in config["services"]:
            result = evaluate_position(
                evaluation_history, spec, config["reference_quality"]
            )
            contract = lifecycle.update(result, spec, complete)
            service_id = str(spec["service_id"])
            status = contract["status"]
            contracts.append(contract)
            statuses[service_id] = status
            request_state_counts[service_id][status] += 1
            reason_counts[f"{service_id}:{result['reason']}"] += 1

        contract_events.append((complete, statuses))
        mode_counts[policy.mode] += 1
        if policy.update(complete, age, contracts) is not None:
            switches += 1

        # One outstanding request. Capacity is represented by deterministic
        # service time; the requested mode may be slower than that capacity.
        requested_period = 1.0 / policy.update_rate_hz
        send = max(complete, send + requested_period)

    transport = {
        "trial": trial,
        "capacity_hz": capacity_hz,
        "policy": policy_name,
        "source_duration_s": horizon,
        "source_observations": len(points),
        "source_observation_rate_hz": (len(points) - 1) / horizon,
        "transport_deliveries": requests,
        "transport_deliveries_per_s": requests / horizon,
        "unique_delivered_sources": len(unique_history),
        "unique_source_delivery_rate_hz": len(unique_history) / horizon,
        "repeated_source_fraction": repeated / requests if requests else math.nan,
        "response_bytes_per_s": bytes_total / horizon,
        "p95_delivery_aoi_s": percentile(delivery_ages, 0.95),
        "mean_delivery_aoi_s": mean(delivery_ages) if delivery_ages else math.nan,
        "rate_switches": switches,
        "economy_fraction": mode_counts["economy"] / requests if requests else math.nan,
        "normal_fraction": mode_counts["normal"] / requests if requests else math.nan,
        "high_fraction": mode_counts["high"] / requests if requests else math.nan,
    }

    rows: list[dict[str, Any]] = []
    for spec in config["services"]:
        service_id = str(spec["service_id"])
        counts = request_state_counts[service_id]
        observable = counts["qualified"] + counts["at_risk"] + counts["withdrawn"]
        satisfied = counts["qualified"] + counts["at_risk"]
        row = {
            **transport,
            "service_id": service_id,
            "qualified_deliveries": counts["qualified"],
            "at_risk_deliveries": counts["at_risk"],
            "withdrawn_deliveries": counts["withdrawn"],
            "unobservable_deliveries": counts["unobservable"],
            "observable_fraction_request_weighted": (
                observable / requests if requests else math.nan
            ),
            "satisfaction_given_observable_request_weighted": (
                satisfied / observable if observable else math.nan
            ),
            **time_weighted_service_metrics(contract_events, service_id, horizon),
        }
        rows.append(row)

    return rows


def aggregate(per_trial: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_trial:
        grouped[(float(row["capacity_hz"]), str(row["policy"]), str(row["service_id"]))].append(row)

    metrics = (
        "source_observation_rate_hz",
        "transport_deliveries_per_s",
        "unique_source_delivery_rate_hz",
        "repeated_source_fraction",
        "response_bytes_per_s",
        "p95_delivery_aoi_s",
        "mean_delivery_aoi_s",
        "rate_switches",
        "economy_fraction",
        "normal_fraction",
        "high_fraction",
        "observable_fraction_request_weighted",
        "satisfaction_given_observable_request_weighted",
        "observable_fraction_time_weighted",
        "satisfaction_given_observable_time_weighted",
    )
    summary: list[dict[str, Any]] = []
    for (capacity, policy, service_id), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "capacity_hz": capacity,
            "policy": policy,
            "service_id": service_id,
            "trials": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if finite(row[metric])]
            result[f"{metric}_mean"] = mean(values) if values else math.nan
            result[f"{metric}_min"] = min(values) if values else math.nan
            result[f"{metric}_max"] = max(values) if values else math.nan
        summary.append(result)
    return summary


def parse_capacities(text: str) -> list[float]:
    capacities = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not capacities or any(value <= 0.0 for value in capacities):
        raise argparse.ArgumentTypeError("capacities must be positive comma-separated numbers")
    return capacities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--capacities", type=parse_capacities, default=parse_capacities("1.5,2,5,10,20"))
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    if not (repo_root / "DigitalTwin").is_dir():
        parser.error(f"--repo-root does not contain DigitalTwin/: {repo_root}")
    sys.path.insert(0, str(repo_root))

    from DigitalTwin.analysis.live_position_contract_study import evaluate_position
    from DigitalTwin.dashboard.contracts import ResourcePolicy, load_contract_config

    dataset = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else repo_root / "public_datasets" / "ugv01_live_contract"
    )
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_contract_config(
        repo_root / "DigitalTwin" / "configs" / "ugv01_live_service_contracts.json"
    )
    rows: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        trace = (
            dataset
            / "trials"
            / "contract_aware"
            / f"trial_{trial:02d}"
            / "live_contract.jsonl"
        )
        points = read_points(trace)
        for capacity in args.capacities:
            for policy_name in POLICIES:
                rows.extend(
                    simulate_trial(
                        points,
                        trial,
                        policy_name,
                        capacity,
                        config,
                        ResourcePolicy,
                        evaluate_position,
                    )
                )

    summary = aggregate(rows)
    write_csv(output_dir / "capacity_contract_per_trial.csv", rows)
    write_csv(output_dir / "capacity_contract_summary.csv", summary)

    manifest = {
        "schema": "ugv01_capacity_contract_simulation_v2",
        "repo_root": str(repo_root),
        "dataset": str(dataset),
        "capacities_hz": args.capacities,
        "policies": list(POLICIES),
        "trials": args.trials,
        "source_selection": "latest source observation available at request send time",
        "transport": "one outstanding request; deterministic service time 1/capacity",
        "history_clock": "original source timestamps; cached responses never become new observations",
        "freshness": "delivery AoI = completion time - source time; GPS age is advanced by delivery AoI",
        "contract": "retrospective position-only evaluation with warning and recovery lifecycle",
        "satisfaction": "both request-weighted and wall-clock time-weighted metrics are reported",
        "limitation": "the measured approximately 1.5 Hz source is not synthesized into fresh 2/5/10/20 Hz observations",
    }
    (output_dir / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output_dir / "capacity_contract_summary.csv")


if __name__ == "__main__":
    main()
