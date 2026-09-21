"""Position-only delivery/remediability ledger for the UGV01 physical runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "docs/protocols/ugv01_position_ledger_protocol.json"
DEFAULT_OUTPUT = ROOT / "results/ugv01_position_ledger"


def delivered_indices(t: np.ndarray, rate_hz: float, delay_s: float) -> np.ndarray:
    source_period = float(np.median(np.diff(t)))
    stride = max(1, int(round((1.0 / rate_hz) / source_period)))
    source = np.arange(0, len(t), stride)
    arrival = t[source] + delay_s
    pointer = np.searchsorted(arrival, t, side="right") - 1
    result = np.full(len(t), -1, dtype=int)
    valid = pointer >= 0
    result[valid] = source[pointer[valid]]
    return result


def interval_has_large_gap(t: np.ndarray, start: np.ndarray, end: np.ndarray, limit_s: float) -> np.ndarray:
    bad = np.diff(t) > limit_s
    cumulative = np.r_[0, np.cumsum(bad.astype(int))]
    lo = np.minimum(start, end); hi = np.maximum(start, end)
    return (cumulative[hi] - cumulative[lo]) > 0


def evaluate_case(frame: pd.DataFrame, service: dict[str, Any], rate_hz: float,
                  delay_ms: int, protocol: dict[str, Any]) -> dict[str, Any]:
    t = frame.time_s.to_numpy(float)
    dt = float(np.median(np.diff(t)))
    delivered = delivered_indices(t, rate_hz, delay_ms / 1000.0)
    horizon_n = int(round(float(service["horizon_s"]) / dt))
    eligible = np.arange(len(t)) >= horizon_n
    observable = eligible & (delivered >= 0)
    if horizon_n:
        observable &= np.r_[np.zeros(horizon_n, dtype=bool), delivered[:-horizon_n] >= 0]
    use = np.flatnonzero(observable)
    max_gap = float(protocol["reference"]["maximum_permitted_reference_gap_s"])
    if horizon_n and len(use):
        start = use - horizon_n
        source_start = delivered[start]; source_end = delivered[use]
        valid_pair = ~interval_has_large_gap(t, start, use, max_gap)
        valid_pair &= ~interval_has_large_gap(t, source_start, source_end, max_gap)
        observable[use[~valid_pair]] = False
        use = np.flatnonzero(observable)

    discrepancy = np.full(len(t), np.nan)
    gt_x = frame.gt_east_m.to_numpy(float); gt_y = frame.gt_north_m.to_numpy(float)
    twin_x = frame.twin_east_m.to_numpy(float); twin_y = frame.twin_north_m.to_numpy(float)
    if float(service["horizon_s"]) == 0.0:
        source = delivered[use]
        discrepancy[use] = np.hypot(twin_x[source] - gt_x[use], twin_y[source] - gt_y[use])
    else:
        start = use - horizon_n
        source_start = delivered[start]; source_end = delivered[use]
        gt_dx, gt_dy = gt_x[use] - gt_x[start], gt_y[use] - gt_y[start]
        twin_dx = twin_x[source_end] - twin_x[source_start]
        twin_dy = twin_y[source_end] - twin_y[source_start]
        discrepancy[use] = np.hypot(twin_dx - gt_dx, twin_dy - gt_dy)

    age = np.full(len(t), np.inf)
    age[use] = t[use] - t[delivered[use]]
    physical = observable & (discrepancy <= float(service["position_tolerance_m"]))
    fresh = observable & (age <= float(service["aoi_limit_s"]))
    joint = physical & fresh
    eligible_n = int(eligible.sum()); observable_n = int(observable.sum())
    denominator = max(1, observable_n)
    q = protocol["qualification"]
    coverage = observable_n / max(1, eligible_n)
    physical_rate = float(physical.sum() / denominator)
    freshness_rate = float(fresh.sum() / denominator)
    joint_rate = float(joint.sum() / denominator)
    if coverage < float(q["minimum_observable_fraction"]):
        state = "unobservable"
    elif (physical_rate >= float(q["minimum_physical_satisfaction"]) and
          freshness_rate >= float(q["minimum_freshness_satisfaction"]) and
          joint_rate >= float(q["minimum_joint_satisfaction"])):
        state = "qualified"
    else:
        state = "not_qualified"
    return {
        "service": service["service"], "horizon_s": service["horizon_s"],
        "position_tolerance_m": service["position_tolerance_m"],
        "aoi_limit_s": service["aoi_limit_s"], "rate_hz": rate_hz, "delay_ms": delay_ms,
        "eligible_samples": eligible_n, "observable_samples": observable_n,
        "observable_fraction": coverage, "physical_satisfaction": physical_rate,
        "freshness_satisfaction": freshness_rate, "joint_satisfaction": joint_rate,
        "position_discrepancy_p95_m": float(np.nanpercentile(discrepancy[observable], 95)) if observable_n else np.nan,
        "aoi_p95_s": float(np.nanpercentile(age[observable], 95)) if observable_n else np.nan,
        "physical_failure": physical_rate < float(q["minimum_physical_satisfaction"]),
        "freshness_failure": freshness_rate < float(q["minimum_freshness_satisfaction"]),
        "state": state,
    }


def run(protocol_path: Path, output: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_LEDGER_COMPUTATION":
        raise RuntimeError("protocol is not frozen")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_id in protocol["runs"]:
        path = ROOT / protocol["dataset_root"] / run_id / "aligned_samples.csv"
        frame = pd.read_csv(path)
        rate = 1.0 / float(np.median(np.diff(frame.time_s.to_numpy(float))))
        settings = [(r, d) for r in protocol["delivery_replay"]["common_supported_rates_hz"]
                    for d in protocol["delivery_replay"]["common_delays_ms"]]
        if protocol["delivery_replay"]["run1_diagnostic_only"] and rate >= 9.5:
            settings += [(r, d) for r in protocol["delivery_replay"]["run1_diagnostic_rates_hz"]
                         for d in protocol["delivery_replay"]["common_delays_ms"]]
        for service in protocol["services"]:
            for replay_rate, delay in settings:
                rows.append({"run_id": run_id, "source_rate_hz": rate,
                             "high_rate_run1_diagnostic": replay_rate > 2.0,
                             **evaluate_case(frame, service, replay_rate, delay, protocol)})
    cases = pd.DataFrame(rows)
    cases.to_csv(output / "ugv01_position_contract_cases.csv", index=False)

    degraded_cfg = protocol["delivery_replay"]["primary_degraded_setting"]
    ideal_cfg = protocol["delivery_replay"]["primary_ideal_setting"]
    common = cases[~cases.high_rate_run1_diagnostic]
    ledger_rows = []
    for (run_id, service), group in common.groupby(["run_id", "service"]):
        degraded = group[(group.rate_hz == degraded_cfg["rate_hz"]) &
                         (group.delay_ms == degraded_cfg["delay_ms"])].iloc[0]
        ideal = group[(group.rate_hz == ideal_cfg["rate_hz"]) &
                      (group.delay_ms == ideal_cfg["delay_ms"])].iloc[0]
        if degraded.state == "unobservable" or ideal.state == "unobservable":
            category = "unobservable"
        elif degraded.state == "qualified" and ideal.state == "qualified":
            category = "already_qualified"
        elif degraded.state != "qualified" and ideal.state == "qualified":
            category = "delivery_remediable"
        elif degraded.state == "qualified" and ideal.state != "qualified":
            category = "degradation_induced"
        else:
            category = "persistent"
        ledger_rows.append({
            "run_id": run_id, "service": service,
            "degraded_state": degraded.state, "ideal_state": ideal.state,
            "degraded_physical_failure": bool(degraded.physical_failure),
            "degraded_freshness_failure": bool(degraded.freshness_failure),
            "ideal_physical_failure": bool(ideal.physical_failure),
            "ideal_freshness_failure": bool(ideal.freshness_failure),
            "degraded_joint_satisfaction": degraded.joint_satisfaction,
            "ideal_joint_satisfaction": ideal.joint_satisfaction,
            "category": category,
        })
    ledger = pd.DataFrame(ledger_rows)
    ledger.to_csv(output / "ugv01_position_remediability_ledger.csv", index=False)
    counts = (ledger.groupby("category").size().rename("count").reset_index())
    counts.to_csv(output / "ugv01_position_ledger_counts.csv", index=False)

    order = ["already_qualified", "delivery_remediable", "persistent", "unobservable", "degradation_induced"]
    values = [int((ledger.category == name).sum()) for name in order]
    shown = [(name, value) for name, value in zip(order, values) if value > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = {"already_qualified": "#287271", "delivery_remediable": "#2a9d8f", "persistent": "#e76f51"}
    axes[0].bar([x.replace("_", "\n") for x, _ in shown], [v for _, v in shown],
                color=[colors.get(x, "#888888") for x, _ in shown])
    axes[0].set_ylabel("Run-service cases"); axes[0].set_title("UGV01 position-only remediability")
    selected = common[((common.rate_hz == degraded_cfg["rate_hz"]) & (common.delay_ms == degraded_cfg["delay_ms"])) |
                      ((common.rate_hz == ideal_cfg["rate_hz"]) & (common.delay_ms == ideal_cfg["delay_ms"]))].copy()
    selected["delivery"] = np.where(selected.rate_hz == ideal_cfg["rate_hz"], "2 Hz / 0 ms", "1 Hz / 200 ms")
    plot = selected.groupby(["service", "delivery"], as_index=False).joint_satisfaction.mean()
    service_order = ["local_1s", "local_5s", "local_10s", "global"]
    for label, group in plot.groupby("delivery"):
        group = group.assign(service=pd.Categorical(group.service, service_order, ordered=True)).sort_values("service")
        axes[1].plot(group.service.astype(str), group.joint_satisfaction, marker="o", label=label)
    axes[1].set_ylim(0, 1.02); axes[1].set_ylabel("Mean joint satisfaction")
    axes[1].set_title("Delivery effect across three runs"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / "ugv01_position_ledger.png", dpi=180); plt.close(fig)

    count_text = ", ".join(f"{r.category}={int(r['count'])}" for _, r in counts.iterrows())
    lines = [
        "# UGV01 Position-Only Remediability Ledger", "",
        f"The frozen ledger contains {len(ledger)} run-service cases across three physical runs and four services.",
        f"Partition: **{count_text}**.", "",
        "The primary comparison is 1 Hz/200 ms degraded delivery versus 2 Hz/0 ms ideal delivery. "
        "The 10/5 Hz rows for run 1 are diagnostic only and do not enter the ledger.", "",
        "## Interpretation", "",
        "This small retrospective study tests whether the taxonomy computes on UGV01; it does not estimate population-level category rates. "
        "AprilTag position is the qualification reference. Heading is excluded, and windows crossing the frozen reference-gap limit are removed from the observable denominator.", "",
        "## Limitations", "",
        "- Three physical runs are available, and run 1 is an asset-calibration run rather than an untouched holdout.",
        "- The common source-rate ceiling is 2 Hz because the two holdout exports cannot support 10/5 Hz replay.",
        "- Thresholds are platform-scaled and frozen before this ledger computation, but informed by prior UGV01 characterization rather than prospectively blinded.",
        "- This is offline delivery replay over saved physical trajectories, not an online communication-savings experiment.",
    ]
    (output / "ugv01_position_ledger_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "ugv01_position_remediability_ledger_v1",
        "protocol": str(protocol_path.relative_to(ROOT)), "protocol_status": protocol["status"],
        "n_runs": int(ledger.run_id.nunique()), "n_services": int(ledger.service.nunique()),
        "n_cases": len(ledger), "category_counts": dict(zip(counts.category, counts["count"].astype(int))),
        "common_source_rate_ceiling_hz": 2.0, "high_rate_rows_excluded_from_ledger": True,
    }
    (output / "ugv01_position_ledger_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
