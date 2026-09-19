"""AoI-tight and burst-outage replay to test whether freshness binds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DigitalTwin.analysis import deterministic_delivery_replay as delivery
from DigitalTwin.analysis import service_timing_budget_study as base


DEFAULT_OUT = base.ROOT / "results/freshness_binding_sweep"

AOI_LIMITS_S = (None, 0.5, 0.3, 0.2)
CONDITIONS = (
    {"name": "ideal_10hz", "rate_hz": 10.0, "delay_ms": 0.0, "jitter_sd_ms": 0.0, "loss_probability": 0.0,
     "burst_period_s": None, "burst_duration_s": 0.0},
    {"name": "nominal_5hz", "rate_hz": 5.0, "delay_ms": 50.0, "jitter_sd_ms": 20.0, "loss_probability": 0.01,
     "burst_period_s": None, "burst_duration_s": 0.0},
    {"name": "degraded_2hz", "rate_hz": 2.0, "delay_ms": 200.0, "jitter_sd_ms": 50.0, "loss_probability": 0.05,
     "burst_period_s": None, "burst_duration_s": 0.0},
    {"name": "stress_1hz", "rate_hz": 1.0, "delay_ms": 300.0, "jitter_sd_ms": 100.0, "loss_probability": 0.10,
     "burst_period_s": None, "burst_duration_s": 0.0},
    {"name": "burst_5hz_1s_every_8s", "rate_hz": 5.0, "delay_ms": 50.0, "jitter_sd_ms": 20.0,
     "loss_probability": 0.01, "burst_period_s": 8.0, "burst_duration_s": 1.0},
    {"name": "burst_2hz_2s_every_10s", "rate_hz": 2.0, "delay_ms": 200.0, "jitter_sd_ms": 50.0,
     "loss_probability": 0.05, "burst_period_s": 10.0, "burst_duration_s": 2.0},
)


def stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "little")


def burst_delivery(t: np.ndarray, condition: dict, seed: int) -> tuple[np.ndarray, dict]:
    """Sample delivery and drop packets whose source time falls inside burst windows."""
    dt = float(np.median(np.diff(t)))
    stride = max(1, int(round((1.0 / condition["rate_hz"]) / dt)))
    source = np.arange(0, len(t), stride)
    rng = np.random.default_rng(seed)
    random_lost = rng.random(len(source)) < condition["loss_probability"]
    burst_period = condition.get("burst_period_s")
    burst_duration = float(condition.get("burst_duration_s") or 0.0)
    if burst_period and burst_duration > 0:
        phase = np.mod(t[source] - t[0], float(burst_period))
        burst_lost = phase < burst_duration
    else:
        burst_lost = np.zeros(len(source), dtype=bool)
    lost = random_lost | burst_lost
    jitter = rng.normal(0.0, condition["jitter_sd_ms"], len(source))
    delay_ms = np.maximum(0.0, condition["delay_ms"] + jitter)
    keep_source = source[~lost]
    keep_delay = delay_ms[~lost]
    arrivals = t[keep_source] + keep_delay / 1000.0
    order = np.argsort(arrivals, kind="stable")
    arrivals, keep_source, keep_delay = arrivals[order], keep_source[order], keep_delay[order]
    delivered = np.full(len(t), -1, dtype=int)
    newest = -1
    cursor = 0
    for index, now in enumerate(t):
        while cursor < len(arrivals) and arrivals[cursor] <= now + 1e-12:
            newest = max(newest, int(keep_source[cursor]))
            cursor += 1
        delivered[index] = newest
    duration = max(float(t[-1] - t[0]), dt)
    arrived_in_run = arrivals <= t[-1] + 1e-12
    stats = {
        "requested_rate_hz": condition["rate_hz"],
        "realized_arrival_rate_hz": float(arrived_in_run.sum() / duration),
        "requested_packets": int(len(source)),
        "delivered_packets": int((~lost).sum()),
        "arrived_within_run_packets": int(arrived_in_run.sum()),
        "random_lost_packets": int(random_lost.sum()),
        "burst_lost_packets": int(burst_lost.sum()),
        "lost_packets": int(lost.sum()),
        "realized_loss_fraction": float(lost.mean()) if len(lost) else np.nan,
        "burst_loss_fraction": float(burst_lost.mean()) if len(burst_lost) else 0.0,
        "delay_p50_ms": float(np.quantile(keep_delay, .50)) if len(keep_delay) else np.nan,
        "delay_p95_ms": float(np.quantile(keep_delay, .95)) if len(keep_delay) else np.nan,
        "delay_max_ms": float(np.max(keep_delay)) if len(keep_delay) else np.nan,
    }
    return delivered, stats


def service_with_aoi(service: dict, aoi_limit_s: float | None) -> dict:
    copy = dict(service)
    if aoi_limit_s is not None:
        copy["aoi_limit_s"] = float(aoi_limit_s)
    return copy


def aoi_label(aoi_limit_s: float | None) -> str:
    return "original" if aoi_limit_s is None else f"aoi_{aoi_limit_s:.1f}s"


def build_ledger(per_sequence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (aoi_setting, sequence, service), group in per_sequence.groupby(["aoi_setting", "sequence", "service"]):
        indexed = group.set_index("condition")
        ideal = indexed.loc["ideal_10hz"]
        for condition, row in indexed.iterrows():
            classification = "already_qualified" if bool(row.qualified) else (
                "persistent_under_ideal" if not bool(ideal.qualified) else "delivery_limited"
            )
            rows.append({
                "aoi_setting": aoi_setting,
                "sequence": sequence,
                "service": service,
                "condition": condition,
                "qualified": int(row.qualified),
                "ideal_qualified": int(ideal.qualified),
                "classification": classification,
                "failure_component": delivery.failure_component(row),
                "coverage": row.coverage,
                "physical_satisfaction": row.physical_satisfaction,
                "freshness_satisfaction": row.freshness_satisfaction,
                "joint_satisfaction": row.joint_satisfaction,
                "mean_aoi_s": row.mean_aoi_s,
                "max_aoi_s": row.max_aoi_s,
            })
    return pd.DataFrame(rows)


def write_report(output: Path, component_summary: pd.DataFrame, transport: pd.DataFrame) -> None:
    tight = component_summary[
        component_summary["aoi_setting"].isin(["aoi_0.2s", "aoi_0.3s"])
        & component_summary["failure_component"].isin(["freshness_only", "physical_and_freshness"])
    ]
    fresh_cases = int(tight["cases"].sum()) if not tight.empty else 0
    component_pivot = component_summary.pivot_table(
        index=["aoi_setting", "condition"], columns="failure_component", values="cases", fill_value=0, aggfunc="sum"
    ).reset_index()
    lines = [
        "# Freshness-binding sweep",
        "",
        "This analysis replays the frozen 10-Hz V2 states under tighter AoI limits and bursty outages.",
        "It is a sensitivity study over saved trajectories, not a new model-training or live-network result.",
        "",
        "## Main answer",
        "",
    ]
    if fresh_cases:
        lines.append(f"- Freshness binds under the tight/bursty sweep: **{fresh_cases}** sequence-service-condition cases have `freshness_only` or `physical_and_freshness` failures at AoI 0.2-0.3 s.")
    else:
        lines.append("- Freshness still does **not** bind under AoI 0.2-0.3 s in this replay; failures remain physical-discrepancy dominated.")
    lines += [
        "- `freshness_only` means physical satisfaction reaches the 0.80 requirement but freshness does not.",
        "- `physical_and_freshness` means both physical discrepancy and freshness fail.",
        "",
        "## Component counts",
        "",
        markdown_table(component_pivot),
        "",
        "## Transport summary",
        "",
    ]
    transport_summary = transport.groupby("condition", as_index=False).agg(
        requested_rate_hz=("requested_rate_hz", "first"),
        realized_rate_hz=("realized_arrival_rate_hz", "mean"),
        loss_fraction=("realized_loss_fraction", "mean"),
        burst_loss_fraction=("burst_loss_fraction", "mean"),
        delay_p95_ms=("delay_p95_ms", "mean"),
    )
    lines.append(markdown_table(transport_summary))
    (output / "freshness_binding_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame.to_dict("records"):
        cells = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def plot_components(output: Path, component_summary: pd.DataFrame) -> None:
    plot_data = component_summary[
        component_summary["aoi_setting"].isin(["original", "aoi_0.3s", "aoi_0.2s"])
        & component_summary["condition"].isin(["degraded_2hz", "stress_1hz", "burst_5hz_1s_every_8s", "burst_2hz_2s_every_10s"])
    ]
    pivot = plot_data.pivot_table(
        index=["aoi_setting", "condition"], columns="failure_component", values="cases", fill_value=0, aggfunc="sum"
    )
    for column in ("none", "physical_only", "freshness_only", "physical_and_freshness", "unobservable"):
        if column not in pivot:
            pivot[column] = 0
    order = ["none", "physical_only", "freshness_only", "physical_and_freshness", "unobservable"]
    labels = [f"{a}\n{c.replace('_', ' ')}" for a, c in pivot.index]
    colors = {
        "none": "#4C78A8",
        "physical_only": "#F58518",
        "freshness_only": "#54A24B",
        "physical_and_freshness": "#B279A2",
        "unobservable": "#8C8C8C",
    }
    fig, ax = plt.subplots(figsize=(12, 5.8))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for column in order:
        values = pivot[column].to_numpy(float)
        ax.bar(x, values, bottom=bottom, label=column.replace("_", " "), color=colors[column])
        bottom += values
    ax.set_ylabel("Sequence-service cases")
    ax.set_title("Freshness-binding sweep failure components")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "freshness_failure_components.png", dpi=200)
    plt.close(fig)


def run(output: Path) -> None:
    rows: list[dict] = []
    transport_rows: list[dict] = []
    for path in base.discover():
        data = base.load(path)
        sequence = path.parent.name.split("_", 2)[-1]
        replicate = path.parent.parent.name
        for condition in CONDITIONS:
            seed = stable_seed(sequence, replicate, condition["name"])
            delivered, stats = burst_delivery(data["time_s"], condition, seed)
            transport_rows.append({"sequence": sequence, "replicate": replicate, "condition": condition["name"],
                                   "random_seed": seed, **stats})
            for aoi_limit_s in AOI_LIMITS_S:
                for service in base.SERVICES:
                    candidate = service_with_aoi(service, aoi_limit_s)
                    rows.append({
                        "sequence": sequence,
                        "replicate": replicate,
                        "condition": condition["name"],
                        "aoi_setting": aoi_label(aoi_limit_s),
                        "service": candidate["service"],
                        "original_aoi_limit_s": service["aoi_limit_s"],
                        "aoi_limit_s": candidate["aoi_limit_s"],
                        **{key: value for key, value in condition.items() if key != "name"},
                        **delivery.evaluate_indices(data, candidate, delivered),
                    })
    output.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(rows)
    transport = pd.DataFrame(transport_rows)
    metrics = ["coverage", "physical_satisfaction", "freshness_satisfaction", "joint_satisfaction",
               "mean_aoi_s", "max_aoi_s"]
    per_sequence = per_run.groupby(["sequence", "condition", "aoi_setting", "service"], as_index=False)[metrics].mean()
    per_sequence["qualified"] = (
        (per_sequence.coverage >= base.MIN_COVERAGE)
        & (per_sequence.physical_satisfaction >= base.TARGET)
        & (per_sequence.freshness_satisfaction >= base.TARGET)
        & (per_sequence.joint_satisfaction >= base.TARGET)
    ).astype(int)
    ledger = build_ledger(per_sequence)
    component_summary = ledger.groupby(
        ["aoi_setting", "condition", "classification", "failure_component"], as_index=False
    ).size().rename(columns={"size": "cases"})
    per_run.to_csv(output / "per_run_freshness_binding.csv", index=False)
    per_sequence.to_csv(output / "per_sequence_freshness_binding.csv", index=False)
    transport.to_csv(output / "transport_statistics.csv", index=False)
    ledger.to_csv(output / "freshness_binding_case_ledger.csv", index=False)
    component_summary.to_csv(output / "freshness_failure_component_summary.csv", index=False)
    manifest = {
        "schema": "freshness_binding_sweep_v1",
        "conditions": CONDITIONS,
        "aoi_limits_s": AOI_LIMITS_S,
        "target_satisfaction": base.TARGET,
        "minimum_coverage": base.MIN_COVERAGE,
        "status": "sensitivity analysis over frozen V2 saved states",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plot_components(output, component_summary)
    write_report(output, component_summary, transport)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
