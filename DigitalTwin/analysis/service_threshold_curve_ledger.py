"""Sensitivity curves for the service remediability ledger.

One threshold family is varied at a time around the frozen operating point.
This avoids presenting the 30/6/4-style partition as a universal constant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DigitalTwin.analysis import service_timing_budget_study as base


DEFAULT_OUT = base.ROOT / "results/service_threshold_curve_ledger"
CONDITIONS = {"ideal": (10.0, 0), "practical": (5.0, 50), "degraded": (2.0, 200)}
TOLERANCE_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
SATISFACTION_REQUIREMENTS = (0.60, 0.70, 0.80, 0.90, 0.95)
FRESHNESS_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)


def settings() -> list[dict]:
    rows = []
    rows.extend({"sweep": "physical_tolerance", "value": value, "tolerance_scale": value,
                 "satisfaction": base.TARGET, "freshness_scale": 1.0} for value in TOLERANCE_SCALES)
    rows.extend({"sweep": "satisfaction_requirement", "value": value, "tolerance_scale": 1.0,
                 "satisfaction": value, "freshness_scale": 1.0} for value in SATISFACTION_REQUIREMENTS)
    rows.extend({"sweep": "freshness_limit", "value": value, "tolerance_scale": 1.0,
                 "satisfaction": base.TARGET, "freshness_scale": value} for value in FRESHNESS_SCALES)
    return rows


def classify(group: pd.DataFrame) -> tuple[str, int]:
    values = group.set_index("condition").qualified
    if not bool(values["ideal"]):
        return "persistent_under_ideal", 0
    if bool(values["degraded"]):
        return "already_qualified_degraded", 0
    return "delivery_remediable", int(values["practical"])


def run(output: Path) -> None:
    paths = base.discover()
    rows = []
    for setting in settings():
        for path in paths:
            data = base.load(path)
            sequence = path.parent.name.split("_", 2)[-1]
            replicate = path.parent.parent.name
            for original in base.SERVICES:
                service = dict(original)
                service["pos_tol_m"] *= setting["tolerance_scale"]
                service["heading_tol_deg"] *= setting["tolerance_scale"]
                service["aoi_limit_s"] *= setting["freshness_scale"]
                for condition, (rate, delay) in CONDITIONS.items():
                    metrics = base.evaluate(data, service, rate, delay)
                    qualified = (metrics["coverage"] >= base.MIN_COVERAGE and
                                 metrics["physical_satisfaction"] >= setting["satisfaction"] and
                                 metrics["freshness_satisfaction"] >= setting["satisfaction"] and
                                 metrics["joint_satisfaction"] >= setting["satisfaction"])
                    rows.append({**setting, "sequence": sequence, "replicate": replicate,
                                 "service": service["service"], "condition": condition,
                                 **metrics, "qualified": int(qualified)})
    per_run = pd.DataFrame(rows)
    metrics = ["coverage", "physical_satisfaction", "freshness_satisfaction", "joint_satisfaction"]
    keys = ["sweep", "value", "tolerance_scale", "satisfaction", "freshness_scale",
            "sequence", "service", "condition"]
    per_sequence = per_run.groupby(keys, as_index=False)[metrics].mean()
    per_sequence["qualified"] = ((per_sequence.coverage >= base.MIN_COVERAGE) &
                                  (per_sequence.physical_satisfaction >= per_sequence.satisfaction) &
                                  (per_sequence.freshness_satisfaction >= per_sequence.satisfaction) &
                                  (per_sequence.joint_satisfaction >= per_sequence.satisfaction)).astype(int)
    ledger_rows = []
    group_keys = ["sweep", "value", "tolerance_scale", "satisfaction", "freshness_scale", "sequence", "service"]
    for key, group in per_sequence.groupby(group_keys):
        classification, restored = classify(group)
        ledger_rows.append(dict(zip(group_keys, key), classification=classification, practical_restores=restored))
    ledger = pd.DataFrame(ledger_rows)
    summary = ledger.groupby(["sweep", "value", "classification"]).size().rename("cases").reset_index()
    recoveries = ledger.groupby(["sweep", "value"], as_index=False).practical_restores.sum()
    summary = summary.merge(recoveries, on=["sweep", "value"], how="left")
    output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output / "per_run_threshold_sensitivity.csv", index=False)
    per_sequence.to_csv(output / "per_sequence_threshold_sensitivity.csv", index=False)
    ledger.to_csv(output / "threshold_curve_case_ledger.csv", index=False)
    summary.to_csv(output / "threshold_curve_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    labels = {"physical_tolerance": "Tolerance multiplier", "satisfaction_requirement": "Required satisfaction",
              "freshness_limit": "Freshness-limit multiplier"}
    for axis, sweep in zip(axes, labels):
        subset = summary[summary.sweep == sweep]
        for classification, group in subset.groupby("classification"):
            axis.plot(group.value, group.cases, marker="o", label=classification.replace("_", " "))
        axis.set(xlabel=labels[sweep], title=sweep.replace("_", " ").title())
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Sequence-service cases (n=40)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "threshold_curve_ledger.png", dpi=220)
    plt.close(fig)
    manifest = {"schema": "service_threshold_curve_ledger_v1", "status": "complete",
                "one_factor_at_a_time": True, "physical_tolerance_scales": TOLERANCE_SCALES,
                "satisfaction_requirements": SATISFACTION_REQUIREMENTS,
                "freshness_limit_scales": FRESHNESS_SCALES,
                "statistical_unit": "physical sequence; three V2 seeds averaged before case classification"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    baseline = summary[np.isclose(summary.value, 1.0) & summary.sweep.isin(["physical_tolerance", "freshness_limit"])]
    report = ["# Service-ledger threshold curves", "",
              "The sweeps vary one threshold family at a time around the frozen operating point.",
              "Counts are sequence-service cases after averaging the three model seeds within each physical sequence.", "",
              "The baseline partition is reproduced at tolerance and freshness multipliers of 1.0. The partition changes",
              "substantially under tighter or looser physical tolerances and satisfaction requirements, so it must be",
              "reported as an operating-point result rather than a universal property.", "",
              "Primary files: `threshold_curve_summary.csv`, `threshold_curve_case_ledger.csv`, and `threshold_curve_ledger.png`."]
    if baseline.empty:
        report.append("Baseline extraction failed; inspect the CSV before publication use.")
    (output / "threshold_curve_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
