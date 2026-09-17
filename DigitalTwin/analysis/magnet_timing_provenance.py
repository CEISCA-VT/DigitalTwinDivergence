"""Read-only provenance gate for a possible MAGNET causal delivery replay.

The released single-file forecast has target times but no verified issue times.
This audit intentionally does not turn forecast-window starts into issue times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/magnet_tfp/data"
OUT = ROOT / "results/magnet_timing_provenance"
CHANNELS = [f"Heat Pipe TC-{i:02d}" for i in range(1, 11)]
PHYSICAL = "MAGNET_Heat_Pipe_2022-03-30.csv"
FORECAST = "ML_MAGNET_2022-03-30.csv"
TARGET_COLUMN = "Time (s)"
ISSUE_CANDIDATES = ("issue_time_s", "forecast_issue_time_s", "prediction_time_s", "generated_at_s")
ARRIVAL_CANDIDATES = ("arrival_time_s", "delivery_time_s", "received_at_s")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def timestamp_semantics(columns):
    names = set(columns)
    return {"target_time": TARGET_COLUMN if TARGET_COLUMN in names else None,
            "issue_time": next((x for x in ISSUE_CANDIDATES if x in names), None),
            "arrival_time": next((x for x in ARRIVAL_CANDIDATES if x in names), None)}


def require_causal_forecast_times(columns):
    semantics = timestamp_semantics(columns)
    if semantics["target_time"] is None or semantics["issue_time"] is None:
        raise ValueError("Causal forecast replay blocked: target and independently recorded issue times are required")
    return semantics


def run(data_dir: Path = DATA, output_dir: Path = OUT):
    physical_path, forecast_path = data_dir / PHYSICAL, data_dir / FORECAST
    physical = pd.read_csv(physical_path)
    forecast = pd.read_csv(forecast_path)
    missing = [c for c in [TARGET_COLUMN, *CHANNELS] if c not in physical or c not in forecast]
    if missing:
        raise ValueError(f"Missing paired columns: {missing}")
    ptime = physical[TARGET_COLUMN].to_numpy(float)
    ftime = forecast[TARGET_COLUMN].to_numpy(float)
    if not np.all(np.diff(ptime) > 0):
        raise ValueError("Physical clock is not strictly increasing")
    if len(forecast) % 600:
        raise ValueError("Cannot audit documented 600-row forecast windows")
    windows = []
    for wid in range(len(forecast) // 600):
        part = forecast.iloc[wid * 600:(wid + 1) * 600]
        target = part[TARGET_COLUMN].to_numpy(float)
        windows.append({"window_id": wid, "first_target_s": target[0], "last_target_s": target[-1],
                        "rows": len(part), "target_strictly_increasing": bool(np.all(np.diff(target) > 0)),
                        "channel_missing_fraction_max": float(part[CHANNELS].isna().mean().max()),
                        "issue_time_s": np.nan, "arrival_time_s": np.nan})
    window_table = pd.DataFrame(windows)
    semantics = timestamp_semantics(forecast.columns)
    status = "blocked_missing_issue_time" if semantics["issue_time"] is None else "requires_issue_time_validation"
    manifest = {"schema": "magnet_causal_timing_provenance_v1", "status": status,
                "physical": {"path": str(physical_path.relative_to(ROOT)), "sha256": sha256(physical_path),
                             "rows": len(physical), "columns": list(physical.columns),
                             "time_first_s": float(ptime[0]), "time_last_s": float(ptime[-1]),
                             "median_step_s": float(np.median(np.diff(ptime)))},
                "forecast": {"path": str(forecast_path.relative_to(ROOT)), "sha256": sha256(forecast_path),
                             "rows": len(forecast), "columns": list(forecast.columns),
                             "windows": len(windows), "unique_target_times": int(np.unique(ftime).size),
                             "time_first_s": float(ftime.min()), "time_last_s": float(ftime.max()),
                             "missing_by_channel": {c: int(forecast[c].isna().sum()) for c in CHANNELS}},
                "semantics": semantics, "channel_names": CHANNELS,
                "source_urls": {
                    "physical": "https://github.com/IdahoLabResearch/MAGNET-Heat-Pipe-Data/tree/main/Experiment/Single_File",
                    "forecast": "https://github.com/IdahoLabResearch/MAGNET-Heat-Pipe-Data/tree/main/Machine_Learning/Single_File",
                    "documentation": "https://github.com/IdahoLabResearch/MAGNET-Heat-Pipe-Data"},
                "prior_artifacts": ["results/magnet_tfp/results/provenance.json",
                                    "results/magnet_tfp/results/eligibility_audit.csv",
                                    "results/cross_domain_contract_generalization/analysis_manifest.json"],
                "causal_replay_allowed": False,
                "reason": "Target/valid times are present, but forecast issue and delivery times are not independently recorded in the local paired archive."}
    output_dir.mkdir(parents=True, exist_ok=True)
    window_table.to_csv(output_dir / "forecast_window_audit.csv", index=False)
    (output_dir / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    protocol = {"schema": "magnet_thermal_timing_protocol_gate_v1", "status": "not_frozen_causal_replay_blocked",
                "source_manifest": "provenance_manifest.json",
                "physical_quantities": CHANNELS, "physical_unit": "degrees Celsius (prior MAGNET analysis convention)",
                "prior_descriptive_forecast_horizons_s": [60, 300, 600],
                "prior_descriptive_fixed_persistence_threshold_c": 5.0,
                "delivery_rates": None, "added_delays": None, "freshness_rule": None,
                "qualification_thresholds": None,
                "blocked_fields": ["independently recorded forecast issue time",
                                   "documented relation between issue time and first target time",
                                   "availability/delivery time or documented assumption allowing a replay arrival model"],
                "claim_boundary": "Existing forecast-target fidelity remains descriptive; causal delivery/remediability and freshness claims are not identifiable."}
    (output_dir / "protocol_manifest.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir)
    print(result["status"])
