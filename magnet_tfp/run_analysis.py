from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SENSORS = [f"Heat Pipe TC-{i:02d}" for i in range(1, 11)]
HOT_SENSORS = [f"Heat Pipe TC-{i:02d}" for i in range(6, 11)]
TIME_COL = "Time (s)"
WINDOW_ROWS_DEFAULT = 600
HORIZON_TARGETS = (60, 300, 599)


@dataclass
class Config:
    forecast_rows: int = WINDOW_ROWS_DEFAULT
    strict_sensor_fraction: float = 0.95
    partial_min_sensors: int = 8
    early_envelope_seconds: int = 30
    envelope_mad_multiplier: float = 3.0
    envelope_floor_c: float = 1.0
    fixed_persistence_threshold_c: float = 5.0
    steady_slope_threshold_c_per_min: float = 0.75
    normalization_floor_c: float = 1.0
    random_seed: int = 20260821
    bootstrap_samples: int = 3000


def read_config(path: Path | None) -> Config:
    cfg = Config()
    if path is None or not path.exists():
        return cfg
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def split_forecast_windows(ml: pd.DataFrame, window_rows: int) -> List[pd.DataFrame]:
    if len(ml) % window_rows != 0:
        raise ValueError(
            f"Forecast CSV has {len(ml)} rows, not divisible by configured window size {window_rows}. "
            "The archived MAGNET single forecast file is expected to concatenate fixed 600-row horizons."
        )
    windows = []
    for idx in range(len(ml) // window_rows):
        w = ml.iloc[idx * window_rows : (idx + 1) * window_rows].copy().reset_index(drop=True)
        w["window_id"] = idx
        windows.append(w)
    return windows


def align_window(w: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    pred = w[[TIME_COL] + SENSORS].copy()
    pred = pred.rename(columns={c: f"pred::{c}" for c in SENSORS})
    pred["forecast_time"] = pd.to_numeric(pred[TIME_COL], errors="coerce").astype(float)
    pred = pred.drop(columns=[TIME_COL]).sort_values("forecast_time")

    phys = physical[[TIME_COL] + SENSORS + (["Heater Temp"] if "Heater Temp" in physical.columns else [])].copy()
    phys["physical_time"] = pd.to_numeric(phys[TIME_COL], errors="coerce").astype(float)
    phys = phys.drop(columns=[TIME_COL]).sort_values("physical_time")
    phys = phys.rename(columns={c: f"phys::{c}" for c in SENSORS})

    aligned = pd.merge_asof(
        pred,
        phys,
        left_on="forecast_time",
        right_on="physical_time",
        direction="nearest",
        tolerance=0.25,
    )
    if aligned.empty:
        return aligned
    start = float(aligned["forecast_time"].iloc[0])
    aligned["horizon_s"] = aligned["forecast_time"] - start
    aligned["time_offset_s"] = aligned["forecast_time"] - aligned["physical_time"]
    return aligned


def robust_scale(values: np.ndarray, floor: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor
    q05, q95 = np.quantile(values, [0.05, 0.95])
    return float(max(q95 - q05, floor))


def mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def classify_condition(aligned: pd.DataFrame, cfg: Config) -> Tuple[str, float]:
    cols = [f"phys::{s}" for s in HOT_SENSORS if f"phys::{s}" in aligned.columns]
    if not cols:
        return "unknown", float("nan")
    hot = aligned[cols].mean(axis=1, skipna=True).to_numpy(float)
    horizon_min = aligned["horizon_s"].to_numpy(float) / 60.0
    mask = np.isfinite(hot) & np.isfinite(horizon_min)
    if mask.sum() < 20:
        return "unknown", float("nan")
    slope = float(np.polyfit(horizon_min[mask], hot[mask], 1)[0])
    th = cfg.steady_slope_threshold_c_per_min
    if slope > th:
        return "heating", slope
    if slope < -th:
        return "cooling", slope
    return "steady", slope


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> Tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.mean(values))
    if values.size == 1:
        return estimate, estimate, estimate
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def greedy_nonoverlap(window_df: pd.DataFrame) -> pd.DataFrame:
    rows = window_df[window_df["strict_eligible"]].sort_values("forecast_start_s")
    keep = []
    last_end = -np.inf
    for _, r in rows.iterrows():
        if r["forecast_start_s"] > last_end:
            keep.append(int(r["window_id"]))
            last_end = float(r["forecast_end_s"])
    out = rows[rows["window_id"].isin(keep)].copy()
    out["nonoverlap_selected"] = True
    return out


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    xr = x[mask].rank(method="average").to_numpy(float)
    yr = y[mask].rank(method="average").to_numpy(float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def find_diagnostic_pairs(window_metrics: pd.DataFrame) -> pd.DataFrame:
    d = window_metrics[window_metrics["strict_eligible"]].reset_index(drop=True)
    rows = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            a, b = d.iloc[i], d.iloc[j]
            mean_rmse = max((float(a.rmse_c) + float(b.rmse_c)) / 2.0, 1e-9)
            rmse_rel_diff = abs(float(a.rmse_c) - float(b.rmse_c)) / mean_rmse
            if rmse_rel_diff > 0.10:
                continue
            p99_lo = max(min(float(a.p99_abs_c), float(b.p99_abs_c)), 1e-9)
            p99_ratio = max(float(a.p99_abs_c), float(b.p99_abs_c)) / p99_lo
            pers_diff = abs(float(a.persistence_envelope_frac) - float(b.persistence_envelope_frac))
            long_lo = max(min(float(a.band_301_599_rmse_c), float(b.band_301_599_rmse_c)), 1e-9)
            long_ratio = max(float(a.band_301_599_rmse_c), float(b.band_301_599_rmse_c)) / long_lo
            if p99_ratio >= 2.0 or pers_diff >= 0.25 or long_ratio >= 2.0:
                score = max(p99_ratio / 2.0, pers_diff / 0.25, long_ratio / 2.0)
                rows.append(
                    {
                        "window_a": int(a.window_id),
                        "window_b": int(b.window_id),
                        "rmse_a_c": float(a.rmse_c),
                        "rmse_b_c": float(b.rmse_c),
                        "rmse_relative_difference": rmse_rel_diff,
                        "p99_a_c": float(a.p99_abs_c),
                        "p99_b_c": float(b.p99_abs_c),
                        "p99_ratio": p99_ratio,
                        "persistence_a": float(a.persistence_envelope_frac),
                        "persistence_b": float(b.persistence_envelope_frac),
                        "persistence_difference": pers_diff,
                        "long_horizon_rmse_a_c": float(a.band_301_599_rmse_c),
                        "long_horizon_rmse_b_c": float(b.band_301_599_rmse_c),
                        "long_horizon_ratio": long_ratio,
                        "diagnostic_separation_score": score,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("diagnostic_separation_score", ascending=False).reset_index(drop=True)


def make_plots(
    out_dir: Path,
    strict_long: pd.DataFrame,
    component_metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
    diagnostic_pairs: pd.DataFrame,
    aligned_cache: Dict[int, pd.DataFrame],
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Horizon profile: window-level mean absolute error across available sensors.
    prof = strict_long.groupby(["window_id", "horizon_s"], as_index=False)["abs_error_c"].mean()
    q = prof.groupby("horizon_s")["abs_error_c"].quantile([0.25, 0.5, 0.75]).unstack()
    plt.figure(figsize=(8, 5))
    plt.plot(q.index, q[0.5], label="Median")
    plt.fill_between(q.index, q[0.25], q[0.75], alpha=0.25, label="IQR")
    plt.xlabel("Forecast horizon (s)")
    plt.ylabel("Mean absolute physical–virtual discrepancy (°C)")
    plt.title("MAGNET physical–virtual discrepancy vs forecast horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "horizon_error_profile.png", dpi=180)
    plt.close()

    # Component × horizon-bin heatmap.
    tmp = strict_long.copy()
    tmp["horizon_bin_min"] = (tmp["horizon_s"] // 60).clip(upper=9).astype(int)
    hm = tmp.groupby(["sensor", "horizon_bin_min"])["abs_error_c"].median().unstack()
    plt.figure(figsize=(10, 5.5))
    img = plt.imshow(hm.to_numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar(img, label="Median absolute discrepancy (°C)")
    plt.yticks(np.arange(len(hm.index)), [s.replace("Heat Pipe ", "") for s in hm.index])
    plt.xticks(np.arange(len(hm.columns)), [f"{int(x)}–{int(x)+1}" for x in hm.columns])
    plt.xlabel("Forecast horizon bin (min)")
    plt.ylabel("Thermowell")
    plt.title("Component-specific fidelity across forecast horizon")
    plt.tight_layout()
    plt.savefig(fig_dir / "component_horizon_heatmap.png", dpi=180)
    plt.close()

    d = window_metrics[window_metrics["strict_eligible"]]
    plt.figure(figsize=(7, 5))
    plt.scatter(d["rmse_c"], d["p99_abs_c"], alpha=0.75)
    plt.xlabel("Conventional RMSE (°C)")
    plt.ylabel("p99 absolute discrepancy (°C)")
    plt.title("Aggregate RMSE vs tail severity")
    plt.tight_layout()
    plt.savefig(fig_dir / "rmse_vs_tail.png", dpi=180)
    plt.close()

    cond_order = [c for c in ["heating", "steady", "cooling"] if c in set(d["condition"])]
    if cond_order:
        groups = [d.loc[d["condition"] == c, "rmse_c"].dropna().to_numpy() for c in cond_order]
        if any(len(g) for g in groups):
            plt.figure(figsize=(7, 5))
            plt.boxplot(groups, showfliers=True)
            plt.xticks(np.arange(1, len(cond_order) + 1), cond_order)
            plt.ylabel("Window RMSE (°C)")
            plt.title("Fidelity by physical operating regime")
            plt.tight_layout()
            plt.savefig(fig_dir / "condition_rmse_boxplot.png", dpi=180)
            plt.close()

    # Representative trace: select best diagnostic pair if available; otherwise worst-p99 window.
    selected = None
    if not diagnostic_pairs.empty:
        selected = int(diagnostic_pairs.iloc[0]["window_a"])
    elif not d.empty:
        selected = int(d.sort_values("p99_abs_c", ascending=False).iloc[0]["window_id"])
    if selected is not None and selected in aligned_cache:
        a = aligned_cache[selected]
        sensor_rows = component_metrics[component_metrics["window_id"] == selected]
        if not sensor_rows.empty:
            sensor = str(sensor_rows.sort_values("p99_abs_c", ascending=False).iloc[0]["sensor"])
            plt.figure(figsize=(8, 5))
            plt.plot(a["horizon_s"], a[f"phys::{sensor}"], label="Physical")
            plt.plot(a["horizon_s"], a[f"pred::{sensor}"], label="Digital-twin forecast")
            plt.xlabel("Forecast horizon (s)")
            plt.ylabel("Temperature (°C)")
            plt.title(f"Representative divergence: window {selected}, {sensor}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "representative_divergence_trace.png", dpi=180)
            plt.close()


def evidence_assessment(
    window_metrics: pd.DataFrame,
    nonoverlap: pd.DataFrame,
    diagnostic_pairs: pd.DataFrame,
    condition_summary: pd.DataFrame,
) -> Dict[str, object]:
    d = window_metrics[window_metrics["strict_eligible"]].copy()
    signals: List[str] = []
    score = 0

    n_strict = len(d)
    if n_strict >= 50:
        score += 1
        signals.append(f"Many usable forecast windows ({n_strict} strict-eligible).")

    if len(nonoverlap) >= 10:
        score += 1
        signals.append(f"At least 10 non-overlapping windows ({len(nonoverlap)}) support conservative uncertainty summaries.")

    short_med = float(d["band_0_60_rmse_c"].median()) if not d.empty else float("nan")
    long_med = float(d["band_301_599_rmse_c"].median()) if not d.empty else float("nan")
    growth = long_med / max(short_med, 1e-9) if np.isfinite(short_med) and np.isfinite(long_med) else float("nan")
    if np.isfinite(growth) and growth >= 1.5:
        score += 1
        signals.append(f"Long-horizon RMSE is materially larger than short-horizon RMSE (median ratio {growth:.2f}×).")

    if not diagnostic_pairs.empty and len(diagnostic_pairs) >= 3:
        score += 1
        signals.append(f"Found {len(diagnostic_pairs)} pairs with similar aggregate RMSE but strongly different tail/persistence/long-horizon behavior.")

    cond_effect = float("nan")
    if not condition_summary.empty:
        valid = condition_summary[condition_summary["n_windows"] >= 3]
        if len(valid) >= 2:
            vals = valid["median_rmse_c"].to_numpy(float)
            cond_effect = float(np.nanmax(vals) / max(np.nanmin(vals), 1e-9))
            if cond_effect >= 1.25:
                score += 1
                signals.append(f"Operating regime changes fidelity materially (max/min median RMSE ratio {cond_effect:.2f}×).")

    score = min(score, 5)
    if score >= 5:
        recommendation = "STRONG: worth a concise main-paper cross-domain transfer section."
    elif score >= 3:
        recommendation = "USEFUL: worth including as a compact transfer study; keep claims modest."
    else:
        recommendation = "WEAK: likely better as supplementary material or omit from the main submission."

    return {
        "heuristic_score_out_of_5": score,
        "recommendation": recommendation,
        "n_strict_eligible": n_strict,
        "n_nonoverlap": len(nonoverlap),
        "median_short_horizon_rmse_c": short_med,
        "median_long_horizon_rmse_c": long_med,
        "long_to_short_rmse_ratio": growth,
        "n_diagnostic_pairs": len(diagnostic_pairs),
        "condition_effect_ratio": cond_effect,
        "signals": signals,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-domain TFP-style fidelity audit for the INL MAGNET heat-pipe digital twin dataset.")
    ap.add_argument("--physical", type=Path, required=True, help="MAGNET_Heat_Pipe_2022-03-30.csv")
    ap.add_argument("--forecast", type=Path, required=True, help="ML_MAGNET_2022-03-30.csv")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    cfg = read_config(args.config)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    physical = pd.read_csv(args.physical)
    ml = pd.read_csv(args.forecast)
    ensure_columns(physical, [TIME_COL] + SENSORS, "Physical CSV")
    ensure_columns(ml, [TIME_COL] + SENSORS, "Forecast CSV")

    windows = split_forecast_windows(ml, int(cfg.forecast_rows))

    eligibility_rows = []
    aligned_cache: Dict[int, pd.DataFrame] = {}
    for wid, w in enumerate(windows):
        per_sensor_fraction = w[SENSORS].notna().mean()
        n_good = int((per_sensor_fraction >= cfg.strict_sensor_fraction).sum())
        strict_pred = n_good == len(SENSORS)
        partial_pred = n_good >= cfg.partial_min_sensors
        aligned = align_window(w, physical)
        physical_match_fraction = float(aligned["physical_time"].notna().mean()) if not aligned.empty else 0.0
        strict = bool(strict_pred and physical_match_fraction >= 0.99)
        partial = bool(partial_pred and physical_match_fraction >= 0.99)
        reasons = []
        if not strict_pred:
            missing = [s for s in SENSORS if per_sensor_fraction[s] < cfg.strict_sensor_fraction]
            reasons.append("forecast_missing:" + ",".join(missing))
        if physical_match_fraction < 0.99:
            reasons.append(f"physical_alignment={physical_match_fraction:.3f}")
        if not reasons:
            reasons.append("ok")
        eligibility_rows.append(
            {
                "window_id": wid,
                "forecast_start_s": float(w[TIME_COL].iloc[0]),
                "forecast_end_s": float(w[TIME_COL].iloc[-1]),
                "n_rows": len(w),
                "n_sensors_ge_95pct_present": n_good,
                "physical_match_fraction": physical_match_fraction,
                "strict_eligible": strict,
                "partial_eligible": partial,
                "reason": ";".join(reasons),
            }
        )
        if partial:
            aligned_cache[wid] = aligned

    eligibility = pd.DataFrame(eligibility_rows)
    eligibility.to_csv(out / "eligibility_audit.csv", index=False)

    strict_ids = eligibility.loc[eligibility["strict_eligible"], "window_id"].astype(int).tolist()
    if not strict_ids:
        raise RuntimeError("No strict-eligible MAGNET forecast windows were found. See eligibility_audit.csv.")

    # Scales are computed from physical measurements only over times used by strict-eligible forecasts.
    physical_values: Dict[str, List[np.ndarray]] = {s: [] for s in SENSORS}
    for wid in strict_ids:
        a = aligned_cache[wid]
        for s in SENSORS:
            physical_values[s].append(a[f"phys::{s}"].to_numpy(float))
    scales = {s: robust_scale(np.concatenate(physical_values[s]), cfg.normalization_floor_c) for s in SENSORS}
    pd.DataFrame({"sensor": list(scales.keys()), "robust_p95_minus_p05_scale_c": list(scales.values())}).to_csv(
        out / "normalization_scales.csv", index=False
    )

    # Early-horizon calibration envelope per component.
    envelope_rows = []
    thresholds: Dict[str, float] = {}
    for s in SENSORS:
        errs = []
        for wid in strict_ids:
            a = aligned_cache[wid]
            mask = a["horizon_s"] <= cfg.early_envelope_seconds
            e = (a.loc[mask, f"pred::{s}"] - a.loc[mask, f"phys::{s}"]).abs().to_numpy(float)
            errs.append(e)
        vals = np.concatenate(errs)
        med = float(np.nanmedian(vals))
        mad_raw = mad(vals)
        sigma_robust = 1.4826 * mad_raw if np.isfinite(mad_raw) else float("nan")
        threshold = max(med + cfg.envelope_mad_multiplier * sigma_robust, cfg.envelope_floor_c)
        thresholds[s] = float(threshold)
        envelope_rows.append(
            {
                "sensor": s,
                "early_horizon_s": cfg.early_envelope_seconds,
                "median_abs_error_c": med,
                "mad_abs_error_c": mad_raw,
                "robust_sigma_c": sigma_robust,
                "envelope_threshold_c": threshold,
            }
        )
    pd.DataFrame(envelope_rows).to_csv(out / "envelope_thresholds.csv", index=False)

    long_rows = []
    window_rows = []
    component_rows = []
    horizon_rows = []

    for _, er in eligibility.iterrows():
        wid = int(er.window_id)
        if not bool(er.partial_eligible):
            base = er.to_dict()
            base.update({"condition": "unavailable", "thermal_slope_c_per_min": np.nan})
            window_rows.append(base)
            continue
        a = aligned_cache[wid].copy()
        condition, slope = classify_condition(a, cfg)
        all_signed = []
        all_abs = []
        all_norm_abs = []
        all_exceed_env = []
        all_exceed_5 = []
        valid_sensors = []

        for s in SENSORS:
            pred_col, phys_col = f"pred::{s}", f"phys::{s}"
            valid = a[pred_col].notna() & a[phys_col].notna()
            if valid.mean() < cfg.strict_sensor_fraction:
                continue
            valid_sensors.append(s)
            signed = (a.loc[valid, pred_col] - a.loc[valid, phys_col]).to_numpy(float)
            abs_e = np.abs(signed)
            norm_abs = abs_e / scales[s]
            horizon_v = a.loc[valid, "horizon_s"].to_numpy(float)
            exceed_env = abs_e > thresholds[s]
            exceed_5 = abs_e > cfg.fixed_persistence_threshold_c

            for h, se, ae, ne, ee, e5 in zip(horizon_v, signed, abs_e, norm_abs, exceed_env, exceed_5):
                long_rows.append(
                    {
                        "window_id": wid,
                        "horizon_s": h,
                        "sensor": s,
                        "signed_error_c": se,
                        "abs_error_c": ae,
                        "normalized_abs_error": ne,
                        "exceeds_envelope": bool(ee),
                        "exceeds_5c": bool(e5),
                        "condition": condition,
                    }
                )

            component_rows.append(
                {
                    "window_id": wid,
                    "sensor": s,
                    "condition": condition,
                    "n": len(abs_e),
                    "mae_c": float(np.mean(abs_e)),
                    "rmse_c": float(np.sqrt(np.mean(signed**2))),
                    "bias_c": float(np.mean(signed)),
                    "p95_abs_c": float(np.quantile(abs_e, 0.95)),
                    "p99_abs_c": float(np.quantile(abs_e, 0.99)),
                    "max_abs_c": float(np.max(abs_e)),
                    "normalized_mae": float(np.mean(norm_abs)),
                    "persistence_envelope_frac": float(np.mean(exceed_env)),
                    "persistence_gt5c_frac": float(np.mean(exceed_5)),
                }
            )
            all_signed.append(signed)
            all_abs.append(abs_e)
            all_norm_abs.append(norm_abs)
            all_exceed_env.append(exceed_env.astype(float))
            all_exceed_5.append(exceed_5.astype(float))

        base = er.to_dict()
        base.update({"condition": condition, "thermal_slope_c_per_min": slope, "n_valid_sensors": len(valid_sensors)})
        if all_signed:
            signed_all = np.concatenate(all_signed)
            abs_all = np.concatenate(all_abs)
            norm_all = np.concatenate(all_norm_abs)
            env_all = np.concatenate(all_exceed_env)
            e5_all = np.concatenate(all_exceed_5)
            base.update(
                {
                    "mae_c": float(np.mean(abs_all)),
                    "rmse_c": float(np.sqrt(np.mean(signed_all**2))),
                    "bias_c": float(np.mean(signed_all)),
                    "p95_abs_c": float(np.quantile(abs_all, 0.95)),
                    "p99_abs_c": float(np.quantile(abs_all, 0.99)),
                    "max_abs_c": float(np.max(abs_all)),
                    "normalized_mae": float(np.mean(norm_all)),
                    "normalized_p95": float(np.quantile(norm_all, 0.95)),
                    "persistence_envelope_frac": float(np.mean(env_all)),
                    "persistence_gt5c_frac": float(np.mean(e5_all)),
                }
            )

            # Band RMSEs use all valid sensor-point residuals in each time range.
            bands = {
                "band_0_60_rmse_c": (0, 60),
                "band_61_300_rmse_c": (61, 300),
                "band_301_599_rmse_c": (301, 599.5),
            }
            for name, (lo, hi) in bands.items():
                vals = []
                for s in valid_sensors:
                    m = (a["horizon_s"] >= lo) & (a["horizon_s"] <= hi) & a[f"pred::{s}"].notna() & a[f"phys::{s}"].notna()
                    vals.append((a.loc[m, f"pred::{s}"] - a.loc[m, f"phys::{s}"]).to_numpy(float))
                v = np.concatenate(vals) if vals else np.array([])
                base[name] = float(np.sqrt(np.mean(v**2))) if len(v) else np.nan

            for target in HORIZON_TARGETS:
                # ±1 s band makes the output robust to fractional timestamps.
                vals = []
                for s in valid_sensors:
                    m = (a["horizon_s"] - target).abs() <= 1.0
                    e = (a.loc[m, f"pred::{s}"] - a.loc[m, f"phys::{s}"]).abs().to_numpy(float)
                    vals.extend(e[np.isfinite(e)].tolist())
                horizon_rows.append(
                    {
                        "window_id": wid,
                        "condition": condition,
                        "target_horizon_s": target,
                        "mean_abs_error_c": float(np.mean(vals)) if vals else np.nan,
                        "median_abs_error_c": float(np.median(vals)) if vals else np.nan,
                        "max_abs_error_c": float(np.max(vals)) if vals else np.nan,
                    }
                )
        window_rows.append(base)

    long_df = pd.DataFrame(long_rows)
    window_metrics = pd.DataFrame(window_rows)
    component_metrics = pd.DataFrame(component_rows)
    horizon_metrics = pd.DataFrame(horizon_rows)

    # Ensure expected diagnostic columns exist even for edge cases.
    for c in ["rmse_c", "mae_c", "p99_abs_c", "persistence_envelope_frac", "band_0_60_rmse_c", "band_301_599_rmse_c"]:
        if c not in window_metrics.columns:
            window_metrics[c] = np.nan

    window_metrics.to_csv(out / "window_metrics.csv", index=False)
    component_metrics.to_csv(out / "component_metrics.csv", index=False)
    horizon_metrics.to_csv(out / "horizon_metrics.csv", index=False)
    long_df.to_csv(out / "aligned_error_long.csv", index=False)

    strict = window_metrics[window_metrics["strict_eligible"]].copy()
    condition_summary = (
        strict.groupby("condition", as_index=False)
        .agg(
            n_windows=("window_id", "count"),
            median_rmse_c=("rmse_c", "median"),
            mean_rmse_c=("rmse_c", "mean"),
            median_p99_c=("p99_abs_c", "median"),
            median_persistence_envelope=("persistence_envelope_frac", "median"),
            median_long_horizon_rmse_c=("band_301_599_rmse_c", "median"),
        )
    )
    condition_summary.to_csv(out / "condition_summary.csv", index=False)

    nonoverlap = greedy_nonoverlap(window_metrics)
    nonoverlap.to_csv(out / "nonoverlap_windows.csv", index=False)

    rng = np.random.default_rng(cfg.random_seed)
    summary_rows = []
    for metric in ["mae_c", "rmse_c", "p95_abs_c", "p99_abs_c", "persistence_envelope_frac", "band_0_60_rmse_c", "band_301_599_rmse_c"]:
        vals = pd.to_numeric(nonoverlap[metric], errors="coerce").to_numpy(float) if metric in nonoverlap else np.array([])
        est, lo, hi = bootstrap_ci(vals, rng, cfg.bootstrap_samples)
        summary_rows.append({"metric": metric, "mean_nonoverlap": est, "bootstrap_95ci_low": lo, "bootstrap_95ci_high": hi, "n_nonoverlap": np.isfinite(vals).sum()})
    pd.DataFrame(summary_rows).to_csv(out / "nonoverlap_bootstrap_summary.csv", index=False)

    correlations = []
    for metric in ["p99_abs_c", "persistence_envelope_frac", "band_301_599_rmse_c", "normalized_p95"]:
        correlations.append({"metric_vs_rmse": metric, "spearman_rank_corr": rank_corr(strict["rmse_c"], strict[metric])})
    pd.DataFrame(correlations).to_csv(out / "metric_redundancy_vs_rmse.csv", index=False)

    diagnostic_pairs = find_diagnostic_pairs(window_metrics)
    diagnostic_pairs.to_csv(out / "diagnostic_pairs_similar_rmse.csv", index=False)

    make_plots(out, long_df[long_df["window_id"].isin(strict_ids)], component_metrics, window_metrics, diagnostic_pairs, aligned_cache)

    assessment = evidence_assessment(window_metrics, nonoverlap, diagnostic_pairs, condition_summary)
    (out / "evidence_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    # Machine-readable provenance/audit.
    provenance = {
        "dataset": "INL MAGNET Heat Pipe Digital Twin, March 30 2022",
        "physical_file": str(args.physical.resolve()),
        "forecast_file": str(args.forecast.resolve()),
        "physical_rows": len(physical),
        "forecast_rows": len(ml),
        "forecast_window_rows": cfg.forecast_rows,
        "n_windows_total": len(windows),
        "n_windows_strict_eligible": int(eligibility["strict_eligible"].sum()),
        "n_windows_partial_eligible": int(eligibility["partial_eligible"].sum()),
        "n_nonoverlap_strict": len(nonoverlap),
        "normalization": "Per-sensor physical p95-p05 range over strict-eligible forecast times, floored at config normalization_floor_c.",
        "persistence_envelope": "Per-sensor early-horizon absolute-error median + 3*1.4826*MAD, floored at 1 C by default.",
        "fixed_persistence_threshold_c": cfg.fixed_persistence_threshold_c,
        "condition_definition": f"Slope of physical TC06-TC10 mean over each forecast window; steady if |slope| <= {cfg.steady_slope_threshold_c_per_min} C/min.",
        "important_note": "The heuristic publication-worthiness score is an exploratory decision aid, not a statistical test or publication metric.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Human-readable summary.
    score = assessment["heuristic_score_out_of_5"]
    rec = assessment["recommendation"]
    lines = [
        "# MAGNET cross-domain fidelity audit",
        "",
        f"**Exploratory screening decision: {rec}**",
        "",
        "This score is deliberately heuristic. Use the actual tables/figures for the paper; do not report the score itself as a scientific result.",
        "",
        "## Dataset audit",
        f"- Total forecast windows reconstructed from the released single forecast CSV: **{len(windows)}**",
        f"- Strict-eligible 10-sensor windows: **{int(eligibility['strict_eligible'].sum())}**",
        f"- Partial-eligible windows (>= {cfg.partial_min_sensors} sensors): **{int(eligibility['partial_eligible'].sum())}**",
        f"- Greedy non-overlapping strict windows: **{len(nonoverlap)}**",
        "",
        "## Signals relevant to a stronger publication",
    ]
    if assessment["signals"]:
        lines.extend([f"- {x}" for x in assessment["signals"]])
    else:
        lines.append("- No strong complementary-fidelity signals passed the conservative exploratory criteria.")
    lines.extend(
        [
            "",
            "## Key descriptive results",
            f"- Median short-horizon (0–60 s) window RMSE: **{assessment['median_short_horizon_rmse_c']:.3f} °C**",
            f"- Median long-horizon (301–599 s) window RMSE: **{assessment['median_long_horizon_rmse_c']:.3f} °C**",
            f"- Long/short median RMSE ratio: **{assessment['long_to_short_rmse_ratio']:.2f}×**",
            f"- Similar-RMSE but diagnostically different window pairs: **{assessment['n_diagnostic_pairs']}**",
            "",
            "## Interpretation rule",
            "MAGNET is most valuable if aggregate RMSE alone hides materially different horizon, component, persistence, tail, or operating-condition behavior. "
            "That supports a cross-domain claim that the fidelity decomposition adds diagnostic structure rather than merely renaming trajectory metrics.",
            "",
            "## Files to inspect first",
            "1. `figures/horizon_error_profile.png`",
            "2. `figures/component_horizon_heatmap.png`",
            "3. `figures/rmse_vs_tail.png`",
            "4. `diagnostic_pairs_similar_rmse.csv`",
            "5. `condition_summary.csv`",
            "6. `eligibility_audit.csv`",
            "",
            "## Method cautions",
            "- Forecast windows overlap heavily; inferential summaries therefore use a greedy non-overlapping subset.",
            "- The released single forecast CSV contains missing forecast blocks; these are reported, not silently removed.",
            "- The early-horizon envelope is an exploratory operational threshold, not a physics-certified allowable-error bound.",
            "- Do not claim universal digital-twin validity from one thermal transfer case. The defensible claim is cross-domain transfer beyond mobile-robot state variables.",
        ]
    )
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("MAGNET fidelity audit complete.")
    print(f"Total windows: {len(windows)} | strict eligible: {int(eligibility['strict_eligible'].sum())} | non-overlap: {len(nonoverlap)}")
    print(rec)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
