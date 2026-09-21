"""Five-fold TerraSentia calibration and position-contract transfer study.

The three deterministic physics parameters are fitted on four physical
sequences and evaluated on the untouched fifth sequence.  RTK position is the
primary evaluation reference.  Dataset fused heading is used only to set the
initial propagation heading and for secondary diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from DigitalTwin.analysis.i2nav_fidelity_evaluator import evaluate_fidelity_frames


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/terrasentia_external_validation"
DEFAULT_OUTPUT = ROOT / "results/terrasentia_loso_calibrated_physics"
SEQUENCES = (
    "ts_2022_06_09_13h16m39s_one_row",
    "ts_2022_06_15_11h48m34s_four_rows",
    "ts_2022_09_01_11h20m00s_two_random",
    "ts_2022_09_01_12h32m56s_double_loop_corridor",
    "ts_2022_09_06_12h37m11s_four_rows",
)
SERVICES = (
    {"service": "local_1s", "horizon_s": 1.0, "pos_tol_m": 0.10, "aoi_limit_s": 0.60},
    {"service": "local_5s", "horizon_s": 5.0, "pos_tol_m": 0.20, "aoi_limit_s": 1.00},
    {"service": "local_10s", "horizon_s": 10.0, "pos_tol_m": 0.50, "aoi_limit_s": 1.50},
    {"service": "global", "horizon_s": 0.0, "pos_tol_m": 1.00, "aoi_limit_s": 1.00},
)
PARAMETER_BOUNDS = ((0.80, 1.20), (0.80, 1.20), (-0.02, 0.02))


def wrap(a: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def load_sequences(root: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    required = {
        "time_s", "rtk_east_m", "rtk_north_m", "reference_heading_rad",
        "forward_motor_speed_mps", "imu_yaw_rate_radps",
    }
    for sequence in SEQUENCES:
        path = root / sequence / "aligned_terrasentia_v2_inputs.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        values = frame[list(required)].to_numpy(float)
        if not np.isfinite(values).all() or not np.all(np.diff(frame.time_s.to_numpy(float)) > 0):
            raise ValueError(f"invalid numeric data or clock in {path}")
        data[sequence] = frame
    return data


def propagate(frame: pd.DataFrame, parameters: np.ndarray, label: str) -> pd.DataFrame:
    speed_scale, yaw_scale, yaw_bias = map(float, parameters)
    t = frame.time_s.to_numpy(float)
    dt = np.diff(t, prepend=t[0])
    v = speed_scale * frame.forward_motor_speed_mps.to_numpy(float)
    omega = yaw_scale * frame.imu_yaw_rate_radps.to_numpy(float) + yaw_bias
    theta = np.empty(len(frame), dtype=float)
    theta[0] = float(frame.reference_heading_rad.iloc[0])
    if len(frame) > 1:
        theta[1:] = wrap(theta[0] + np.cumsum(omega[1:] * dt[1:]))
    x = np.empty(len(frame), dtype=float); y = np.empty(len(frame), dtype=float)
    x[0] = float(frame.rtk_east_m.iloc[0]); y[0] = float(frame.rtk_north_m.iloc[0])
    if len(frame) > 1:
        x[1:] = x[0] + np.cumsum(v[1:] * np.cos(theta[:-1]) * dt[1:])
        y[1:] = y[0] + np.cumsum(v[1:] * np.sin(theta[:-1]) * dt[1:])
    return pd.DataFrame({
        "time_s": t, "method": label, "pred_v_T_mps": v,
        "pred_omega_T_radps": omega, "x_T_m": x, "y_T_m": y,
        "theta_T_rad": theta,
    })


def path_length(frame: pd.DataFrame) -> float:
    xy = frame[["rtk_east_m", "rtk_north_m"]].to_numpy(float)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def calibration_objective(parameters: np.ndarray, training: dict[str, pd.DataFrame]) -> float:
    """Sequence-balanced robust position loss; no held-out samples enter here."""
    losses = []
    for frame in training.values():
        trace = propagate(frame, parameters, "candidate")
        error = np.hypot(
            trace.x_T_m.to_numpy(float) - frame.rtk_east_m.to_numpy(float),
            trace.y_T_m.to_numpy(float) - frame.rtk_north_m.to_numpy(float),
        )
        scale = max(path_length(frame), 10.0)
        normalized = error / scale
        delta = 0.10
        huber = np.where(normalized <= delta, 0.5 * normalized**2,
                         delta * (normalized - 0.5 * delta))
        losses.append(float(np.mean(huber)))
    return float(np.mean(losses))


def fit_parameters(training: dict[str, pd.DataFrame]) -> tuple[np.ndarray, dict[str, Any]]:
    initial = np.array([1.0, 1.0, 0.0], dtype=float)
    result = minimize(
        calibration_objective, initial, args=(training,), method="Powell",
        bounds=PARAMETER_BOUNDS,
        options={"xtol": 1e-5, "ftol": 1e-8, "maxiter": 250},
    )
    if not result.success:
        raise RuntimeError(f"calibration failed: {result.message}")
    return result.x, {
        "optimizer": "scipy.optimize.minimize/Powell",
        "objective": "mean sequence-balanced Huber loss of RTK position error normalized by max(path_length,10m)",
        "initial": initial.tolist(), "bounds": [list(v) for v in PARAMETER_BOUNDS],
        "objective_value": float(result.fun), "evaluations": int(result.nfev),
    }


def evaluate(frame: pd.DataFrame, trace: pd.DataFrame, sequence: str, model: str) -> tuple[dict[str, Any], pd.DataFrame]:
    trajectory = pd.DataFrame({
        "time_s": frame.time_s,
        "gt_east_m": frame.rtk_east_m, "gt_north_m": frame.rtk_north_m,
        "gt_heading_rad": frame.reference_heading_rad,
        "estimate_east_m": trace.x_T_m, "estimate_north_m": trace.y_T_m,
        "estimate_heading_rad": trace.theta_T_rad,
    })
    profile, series = evaluate_fidelity_frames(trajectory, model=model, sequence=sequence)
    profile["heading_metrics_role"] = "secondary_dataset_fused_reference"
    return profile, series


def delivered_indices(t: np.ndarray, rate_hz: float, delay_s: float) -> np.ndarray:
    dt = float(np.median(np.diff(t)))
    stride = max(1, int(round(1.0 / (rate_hz * dt))))
    source = np.arange(0, len(t), stride)
    arrivals = t[source] + delay_s
    pointer = np.searchsorted(arrivals, t, side="right") - 1
    result = np.full(len(t), -1, dtype=int)
    valid = pointer >= 0
    result[valid] = source[pointer[valid]]
    return result


def relative_xy(x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = x[j] - x[i], y[j] - y[i]
    c, s = np.cos(theta[i]), np.sin(theta[i])
    return c * dx + s * dy, -s * dx + c * dy


def contract_result(frame: pd.DataFrame, trace: pd.DataFrame, service: dict[str, float | str],
                    rate_hz: float, delay_ms: int) -> dict[str, Any]:
    t = frame.time_s.to_numpy(float)
    dt = float(np.median(np.diff(t)))
    delivered = delivered_indices(t, rate_hz, delay_ms / 1000.0)
    h = int(round(float(service["horizon_s"]) / dt))
    eligible = np.arange(len(t)) >= max(h, int(round(30.0 / dt)))
    observable = eligible & (delivered >= 0)
    if h:
        observable &= np.r_[np.zeros(h, dtype=bool), delivered[:-h] >= 0]
    use = np.flatnonzero(observable)
    error = np.full(len(t), np.nan)
    if service["service"] == "global":
        j = delivered[use]
        error[use] = np.hypot(
            trace.x_T_m.to_numpy(float)[j] - frame.rtk_east_m.to_numpy(float)[use],
            trace.y_T_m.to_numpy(float)[j] - frame.rtk_north_m.to_numpy(float)[use],
        )
    else:
        start = use - h; di, dj = delivered[start], delivered[use]
        gx, gy = relative_xy(frame.rtk_east_m.to_numpy(float), frame.rtk_north_m.to_numpy(float),
                             frame.reference_heading_rad.to_numpy(float), start, use)
        ex, ey = relative_xy(trace.x_T_m.to_numpy(float), trace.y_T_m.to_numpy(float),
                             trace.theta_T_rad.to_numpy(float), di, dj)
        error[use] = np.hypot(ex - gx, ey - gy)
    age = np.full(len(t), np.inf)
    age[use] = t[use] - t[delivered[use]]
    physical = observable & (error <= float(service["pos_tol_m"]))
    fresh = observable & (age <= float(service["aoi_limit_s"]))
    denom = max(1, int(observable.sum()))
    return {
        "service": service["service"], "rate_hz": rate_hz, "delay_ms": delay_ms,
        "observable_samples": int(observable.sum()),
        "physical_satisfaction": float(physical.sum() / denom),
        "freshness_satisfaction": float(fresh.sum() / denom),
        "joint_satisfaction": float((physical & fresh).sum() / denom),
        "p95_position_discrepancy_m": float(np.nanpercentile(error[observable], 95)) if observable.any() else np.nan,
        "qualified": bool(observable.any() and (physical & fresh).sum() / denom >= 0.80),
        "physical_failure": bool(observable.any() and physical.sum() / denom < 0.80),
        "freshness_failure": bool(observable.any() and fresh.sum() / denom < 0.80),
    }


def run(input_root: Path, output: Path, only_sequence: str | None = None) -> None:
    data = load_sequences(input_root)
    held_out = [only_sequence] if only_sequence else list(SEQUENCES)
    if only_sequence and only_sequence not in data:
        raise ValueError(f"unknown sequence {only_sequence}")
    output.mkdir(parents=True, exist_ok=True)
    parameter_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    traces: dict[str, dict[str, pd.DataFrame]] = {}
    for sequence in held_out:
        training = {k: v for k, v in data.items() if k != sequence}
        parameters, audit = fit_parameters(training)
        fold_dir = output / sequence; fold_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "held_out_sequence": sequence, "training_sequences": ";".join(training),
            "held_out_excluded_from_fit": True,
            "speed_scale": parameters[0], "yaw_scale": parameters[1], "yaw_bias_radps": parameters[2],
            **audit,
        }
        parameter_rows.append(row)
        frame = data[sequence]
        physics = propagate(frame, np.array([1.0, 1.0, 0.0]), "nominal_physics")
        calibrated = propagate(frame, parameters, "loso_calibrated_physics")
        traces[sequence] = {"nominal_physics": physics, "loso_calibrated_physics": calibrated}
        physics.to_csv(fold_dir / "nominal_physics_trace.csv", index=False)
        calibrated.to_csv(fold_dir / "loso_calibrated_physics_trace.csv", index=False)
        for model, trace in traces[sequence].items():
            profile, _ = evaluate(frame, trace, sequence, model)
            metric_rows.append(profile)
            for service in SERVICES:
                for rate, delay in ((10.0, 0), (2.0, 200)):
                    contract_rows.append({"sequence": sequence, "model": model,
                                          **contract_result(frame, trace, service, rate, delay)})
        (fold_dir / "fit_manifest.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

    parameters = pd.DataFrame(parameter_rows)
    metrics = pd.DataFrame(metric_rows)
    contracts = pd.DataFrame(contract_rows)
    parameters.to_csv(output / "loso_calibration_parameters.csv", index=False)
    metrics.to_csv(output / "loso_fidelity_metrics.csv", index=False)
    contracts.to_csv(output / "position_contract_delivery_results.csv", index=False)

    ledger_rows = []
    for (sequence, model, service), group in contracts.groupby(["sequence", "model", "service"]):
        degraded = group[(group.rate_hz == 2.0) & (group.delay_ms == 200)].iloc[0]
        ideal = group[(group.rate_hz == 10.0) & (group.delay_ms == 0)].iloc[0]
        if degraded.qualified:
            category = "already_qualified"
        elif ideal.qualified:
            category = "delivery_remediable"
        else:
            category = "persistent"
        ledger_rows.append({
            "sequence": sequence, "model": model, "service": service,
            "degraded_qualified": bool(degraded.qualified), "ideal_qualified": bool(ideal.qualified),
            "degraded_physical_failure": bool(degraded.physical_failure),
            "degraded_freshness_failure": bool(degraded.freshness_failure), "category": category,
        })
    ledger = pd.DataFrame(ledger_rows)
    ledger.to_csv(output / "position_contract_remediability_ledger.csv", index=False)

    if len(held_out) == len(SEQUENCES):
        pivot = metrics.pivot(index="sequence", columns="model", values=["ATE_m", "RPEp_1s_m", "RPEp_5s_m", "RPEp_10s_m"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].scatter(pivot[("RPEp_1s_m", "loso_calibrated_physics")], pivot[("ATE_m", "loso_calibrated_physics")])
        for seq in pivot.index:
            axes[0].annotate(seq.split("_")[-2], (pivot.loc[seq, ("RPEp_1s_m", "loso_calibrated_physics")], pivot.loc[seq, ("ATE_m", "loso_calibrated_physics")]), fontsize=7)
        axes[0].set(xlabel="RPE1 RMSE (m)", ylabel="ATE RMSE (m)", title="Held-out local vs global fidelity")
        mean = metrics.groupby("model")[["ATE_m", "RPEp_1s_m", "RPEp_5s_m", "RPEp_10s_m"]].mean()
        mean.T.plot.bar(ax=axes[1]); axes[1].set_ylabel("Sequence-mean error (m)"); axes[1].set_title("Nominal vs LOSO-calibrated physics")
        fig.tight_layout(); fig.savefig(output / "terrasentia_loso_fidelity.png", dpi=180); plt.close(fig)

    summary = [
        "# TerraSentia LOSO-Calibrated Physics Study", "",
        f"- Completed held-out folds: {len(held_out)}/{len(SEQUENCES)}.",
        "- Each fold fits three bounded deterministic parameters on the other four physical sequences only.",
        "- Primary evaluation is untouched held-out RTK position; fused-EKF heading remains secondary.",
        "- The position-only ledger compares 2 Hz/200 ms held delivery with 10 Hz/0 ms ideal delivery.",
        "- This is platform-specific deterministic calibration, not learned-V2 retraining.", "",
    ]
    if len(metrics):
        for model, group in metrics.groupby("model"):
            summary.append(f"- {model}: sequence-mean ATE {group.ATE_m.mean():.3f} m; RPE1 {group.RPEp_1s_m.mean():.3f} m; RPE5 {group.RPEp_5s_m.mean():.3f} m; RPE10 {group.RPEp_10s_m.mean():.3f} m.")
    if len(ledger):
        for model, group in ledger.groupby("model"):
            counts = group.category.value_counts().to_dict()
            summary.append(f"- {model} ledger: {counts}.")
    summary += ["", "## Claim boundary", "", "Five physical sequences support a preliminary platform-specific LOSO study, not a population-wide TerraSentia claim. Motor/IMU semantic provenance remains unresolved and is not corrected using held-out RTK performance."]
    (output / "terrasentia_loso_calibrated_physics_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    manifest = {
        "schema": "terrasentia_loso_calibrated_physics_v1", "input_root": str(input_root),
        "output": str(output), "held_out_sequences": held_out, "all_sequences": list(SEQUENCES),
        "parameter_bounds": [list(v) for v in PARAMETER_BOUNDS],
        "primary_reference": "RTK position", "secondary_reference": "dataset fused-EKF heading",
        "no_held_out_calibration": True, "no_gpu_required": True,
    }
    (output / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only-sequence", choices=SEQUENCES)
    args = parser.parse_args()
    run(args.input_root, args.output_dir, args.only_sequence)


if __name__ == "__main__":
    main()
