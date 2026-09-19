"""Deterministic sampled delay, jitter, and loss replay over frozen states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from DigitalTwin.analysis import service_timing_budget_study as base


DEFAULT_OUT = base.ROOT / "results/deterministic_delivery_replay"
CONDITIONS = (
    {"name": "ideal", "rate_hz": 10.0, "delay_ms": 0.0, "jitter_sd_ms": 0.0, "loss_probability": 0.0},
    {"name": "practical", "rate_hz": 5.0, "delay_ms": 50.0, "jitter_sd_ms": 20.0, "loss_probability": 0.01},
    {"name": "degraded", "rate_hz": 2.0, "delay_ms": 200.0, "jitter_sd_ms": 50.0, "loss_probability": 0.05},
    {"name": "stress", "rate_hz": 1.0, "delay_ms": 300.0, "jitter_sd_ms": 100.0, "loss_probability": 0.10},
)


def stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "little")


def sampled_delivery(t: np.ndarray, condition: dict, seed: int) -> tuple[np.ndarray, dict]:
    dt = float(np.median(np.diff(t)))
    stride = max(1, int(round((1.0 / condition["rate_hz"]) / dt)))
    source = np.arange(0, len(t), stride)
    rng = np.random.default_rng(seed)
    lost = rng.random(len(source)) < condition["loss_probability"]
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
        "requested_packets": int(len(source)), "delivered_packets": int((~lost).sum()),
        "arrived_within_run_packets": int(arrived_in_run.sum()), "lost_packets": int(lost.sum()),
        "realized_loss_fraction": float(lost.mean()) if len(lost) else np.nan,
        "delay_p50_ms": float(np.quantile(keep_delay, .50)) if len(keep_delay) else np.nan,
        "delay_p95_ms": float(np.quantile(keep_delay, .95)) if len(keep_delay) else np.nan,
        "delay_max_ms": float(np.max(keep_delay)) if len(keep_delay) else np.nan,
    }
    return delivered, stats


def evaluate_indices(data: dict, service: dict, delivered: np.ndarray) -> dict:
    t = data["time_s"]
    horizon = int(round(service["horizon_s"] * 10))
    eligible = np.arange(len(t)) >= int(round(base.PREFIX_S * 10)) + horizon
    observable = eligible & (delivered >= 0)
    if horizon:
        observable &= np.r_[np.zeros(horizon, bool), delivered[:-horizon] >= 0]
    use = np.flatnonzero(observable)
    position = np.full(len(t), np.nan)
    heading = np.full(len(t), np.nan)
    if service["family"] == "global":
        source = delivered[use]
        position[use] = np.hypot(data["estimate_east_m"][source] - data["gt_east_m"][use],
                                 data["estimate_north_m"][source] - data["gt_north_m"][use])
        heading[use] = np.abs(np.rad2deg(base.wrap(data["estimate_heading_rad"][source] - data["gt_heading_rad"][use])))
    else:
        initial = use - horizon
        source_initial, source_final = delivered[initial], delivered[use]
        truth = base.relative(data["gt_east_m"], data["gt_north_m"], data["gt_heading_rad"], initial, use)
        estimate = base.relative(data["estimate_east_m"], data["estimate_north_m"], data["estimate_heading_rad"],
                                 source_initial, source_final)
        position[use] = np.hypot(estimate[0] - truth[0], estimate[1] - truth[1])
        heading[use] = np.abs(np.rad2deg(base.wrap(estimate[2] - truth[2])))
    age = np.full(len(t), np.inf)
    age[use] = t[use] - t[delivered[use]]
    physical = observable & (position <= service["pos_tol_m"]) & (heading <= service["heading_tol_deg"])
    fresh = observable & (age <= service["aoi_limit_s"])
    denominator = int(observable.sum())
    eligible_n = int(eligible.sum())
    metrics = {
        "coverage": denominator / max(1, eligible_n),
        "physical_satisfaction": float(physical.sum() / max(1, denominator)),
        "freshness_satisfaction": float(fresh.sum() / max(1, denominator)),
        "joint_satisfaction": float((physical & fresh).sum() / max(1, denominator)),
        "eligible_samples": eligible_n, "observable_samples": denominator,
        "mean_aoi_s": float(np.mean(age[observable])) if denominator else np.nan,
        "max_aoi_s": float(np.max(age[observable])) if denominator else np.nan,
    }
    metrics["qualified"] = int(metrics["coverage"] >= base.MIN_COVERAGE and
                               metrics["physical_satisfaction"] >= base.TARGET and
                               metrics["freshness_satisfaction"] >= base.TARGET and
                               metrics["joint_satisfaction"] >= base.TARGET)
    return metrics


def failure_component(row) -> str:
    if row.coverage < base.MIN_COVERAGE:
        return "unobservable"
    physical = row.physical_satisfaction >= base.TARGET
    fresh = row.freshness_satisfaction >= base.TARGET
    if physical and fresh:
        return "none"
    if not physical and fresh:
        return "physical_only"
    if physical and not fresh:
        return "freshness_only"
    return "physical_and_freshness"


def build_ledger(per_sequence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sequence, service), group in per_sequence.groupby(["sequence", "service"]):
        values = group.set_index("condition")
        ideal, practical, degraded = values.loc["ideal"], values.loc["practical"], values.loc["degraded"]
        if not bool(ideal.qualified):
            classification = "persistent_under_ideal"
        elif bool(degraded.qualified):
            classification = "already_qualified_degraded"
        else:
            classification = "delivery_remediable"
        row = {"sequence": sequence, "service": service, "classification": classification,
               "ideal_qualified": int(ideal.qualified), "practical_qualified": int(practical.qualified),
               "degraded_qualified": int(degraded.qualified),
               "practical_restores": int(not degraded.qualified and practical.qualified),
               "ideal_failure_component": failure_component(ideal),
               "practical_failure_component": failure_component(practical),
               "degraded_failure_component": failure_component(degraded)}
        rows.append(row)
    return pd.DataFrame(rows)


def run(output: Path) -> None:
    rows, transport_rows = [], []
    paths = base.discover()
    for path in paths:
        data = base.load(path)
        sequence = path.parent.name.split("_", 2)[-1]
        replicate = path.parent.parent.name
        for condition in CONDITIONS:
            seed = stable_seed(sequence, replicate, condition["name"])
            delivered, stats = sampled_delivery(data["time_s"], condition, seed)
            transport_rows.append({"sequence": sequence, "replicate": replicate,
                                   "condition": condition["name"], "random_seed": seed, **stats})
            for service in base.SERVICES:
                rows.append({"sequence": sequence, "replicate": replicate, "condition": condition["name"],
                             "service": service["service"], **condition, **evaluate_indices(data, service, delivered)})
    per_run = pd.DataFrame(rows)
    transport = pd.DataFrame(transport_rows)
    metrics = ["coverage", "physical_satisfaction", "freshness_satisfaction", "joint_satisfaction",
               "mean_aoi_s", "max_aoi_s"]
    per_sequence = per_run.groupby(["sequence", "condition", "service"], as_index=False)[metrics].mean()
    per_sequence["qualified"] = ((per_sequence.coverage >= base.MIN_COVERAGE) &
                                  (per_sequence.physical_satisfaction >= base.TARGET) &
                                  (per_sequence.freshness_satisfaction >= base.TARGET) &
                                  (per_sequence.joint_satisfaction >= base.TARGET)).astype(int)
    ledger = build_ledger(per_sequence)
    output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output / "per_run_contract_replay.csv", index=False)
    per_sequence.to_csv(output / "per_sequence_contract_replay.csv", index=False)
    transport.to_csv(output / "realized_delivery_statistics.csv", index=False)
    ledger.to_csv(output / "emulated_delivery_case_ledger.csv", index=False)
    ledger.groupby("classification").size().rename("cases").reset_index().to_csv(
        output / "emulated_delivery_ledger_summary.csv", index=False
    )
    definitions = {"schema": "deterministic_delivery_replay_v1", "conditions": CONDITIONS,
                   "randomization": "SHA-256-derived seed per physical sequence, model seed, and condition",
                   "receiver": "newest source timestamp among packets arrived by each 10-Hz evaluation time",
                   "jitter": "zero-mean Gaussian added to nominal delay then truncated at zero",
                   "inference_unit": "physical sequence; model seeds aggregated within sequence",
                   "status": "descriptive emulation over precomputed states; not live network measurement"}
    (output / "delivery_protocol.json").write_text(json.dumps(definitions, indent=2) + "\n", encoding="utf-8")
    transport_summary = transport.groupby("condition", as_index=False).agg(
        requested_rate_hz=("requested_rate_hz", "first"),
        realized_rate_hz=("realized_arrival_rate_hz", "mean"),
        loss_fraction=("realized_loss_fraction", "mean"),
        delay_p95_ms=("delay_p95_ms", "mean"),
    )
    counts = ledger.classification.value_counts()
    report = ["# Deterministic delivery replay", "",
              "This is deterministic emulation over saved model states. It is not a live transport experiment.", "",
              "| Condition | Requested Hz | Realized Hz | Loss | Mean p95 delay (ms) |",
              "|---|---:|---:|---:|---:|"]
    for row in transport_summary.itertuples():
        report.append(f"| {row.condition} | {row.requested_rate_hz:.1f} | {row.realized_rate_hz:.3f} | {row.loss_fraction:.3%} | {row.delay_p95_ms:.1f} |")
    report += ["", "## Emulated remediability ledger", "",
               f"- Delivery-remediable: **{int(counts.get('delivery_remediable', 0))}/40**",
               f"- Persistent under ideal delivery: **{int(counts.get('persistent_under_ideal', 0))}/40**",
               f"- Already qualified under degraded delivery: **{int(counts.get('already_qualified_degraded', 0))}/40**",
               f"- Restored by the practical emulated condition: **{int(ledger.practical_restores.sum())}** cases"]
    (output / "deterministic_delivery_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
