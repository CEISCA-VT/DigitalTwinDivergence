#!/usr/bin/env python3
"""Shared utilities for multi-method digital-twin fidelity evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import math
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import (
    angular_rate,
    forward_speed,
    load_pose_trajectory,
    sequence_id,
    seed_id,
    wrap_angle,
)


def load_manifest(path: str | Path, repo_root: str | Path = ".") -> pd.DataFrame:
    root = Path(repo_root).resolve()
    m = pd.read_csv(path)
    req = ["method", "sequence", "seed", "trajectory"]
    miss = [c for c in req if c not in m.columns]
    if miss:
        raise ValueError(f"Manifest missing {miss}")
    abs_paths = []
    exists = []
    for p in m["trajectory"].astype(str):
        q = Path(p)
        if not q.is_absolute():
            q = root / q
        abs_paths.append(str(q.resolve()))
        exists.append(q.exists())
    m = m.copy(); m["trajectory_abs"] = abs_paths; m["exists"] = exists
    bad = m[~m.exists]
    if len(bad):
        sample = "\n".join(bad.trajectory_abs.head(20))
        raise FileNotFoundError(f"Manifest contains {len(bad)} missing trajectory files. First paths:\n{sample}")
    return m


def relative_translation(x0, y0, h0, x1, y1):
    dx = x1 - x0; dy = y1 - y0
    c = np.cos(h0); s = np.sin(h0)
    return c * dx + s * dy, -s * dx + c * dy


def rpe_translation(t, gx, gy, gh, ex, ey, eh, horizon_s: float) -> float:
    t = np.asarray(t, float)
    j = np.searchsorted(t, t + horizon_s, side="left")
    i = np.arange(len(t))
    ok = j < len(t); i = i[ok]; j = j[ok]
    if not len(i):
        return np.nan
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    tol = max(0.15, 2.5 * float(np.median(dt))) if len(dt) else 0.25
    ok = np.abs(t[j] - (t[i] + horizon_s)) <= tol
    i = i[ok]; j = j[ok]
    if not len(i):
        return np.nan
    gdx, gdy = relative_translation(gx[i], gy[i], gh[i], gx[j], gy[j])
    edx, edy = relative_translation(ex[i], ey[i], eh[i], ex[j], ey[j])
    return float(np.sqrt(np.mean((edx - gdx) ** 2 + (edy - gdy) ** 2)))


def trajectory_tfp_metrics(path: str | Path) -> dict:
    d = load_pose_trajectory(path)
    t = d["time_s"].to_numpy(float)
    t = t - t[0]
    gx = d["gt_east_m"].to_numpy(float); gy = d["gt_north_m"].to_numpy(float); gh = d["gt_heading_rad"].to_numpy(float)
    ex = d["estimate_east_m"].to_numpy(float); ey = d["estimate_north_m"].to_numpy(float); eh = d["estimate_heading_rad"].to_numpy(float)
    dp = np.hypot(gx - ex, gy - ey)
    signed_dtheta = wrap_angle(gh - eh)
    dtheta = np.abs(signed_dtheta)

    gv = forward_speed(t, gx, gy, gh)
    ev = forward_speed(t, ex, ey, eh)
    gw = angular_rate(t, gh)
    ew = angular_rate(t, eh)
    rv = gv - ev
    rw = gw - ew
    dv = np.abs(rv); dw = np.abs(rw)
    # trapezoidal cumulative signed yaw-rate residual, starting at zero
    acc = np.zeros(len(t), float)
    if len(t) > 1:
        inc = 0.5 * (rw[1:] + rw[:-1]) * np.diff(t)
        acc[1:] = np.cumsum(inc)

    def q95(a): return float(np.nanquantile(a, 0.95))
    def mx(a): return float(np.nanmax(a))
    def mean(a): return float(np.nanmean(a))

    return {
        "n_samples": len(t),
        "duration_s": float(t[-1]),
        "ate_m": float(np.sqrt(np.mean(dp ** 2))),
        "heading_mae_deg": float(np.degrees(np.mean(dtheta))),
        "rpe1_m": rpe_translation(t, gx, gy, gh, ex, ey, eh, 1.0),
        "rpe5_m": rpe_translation(t, gx, gy, gh, ex, ey, eh, 5.0),
        "rpe10_m": rpe_translation(t, gx, gy, gh, ex, ey, eh, 10.0),
        "dp_mean_m": mean(dp),
        "dp_p95_m": q95(dp),
        "dp_max_m": mx(dp),
        "dtheta_mean_deg": float(np.degrees(mean(dtheta))),
        "dtheta_p95_deg": float(np.degrees(q95(dtheta))),
        "dtheta_max_deg": float(np.degrees(mx(dtheta))),
        "signed_heading_error_mean_deg": float(np.degrees(mean(signed_dtheta))),
        "dv_pose_mae_mps": mean(dv),
        "dv_pose_p95_mps": q95(dv),
        "dv_pose_max_mps": mx(dv),
        "signed_speed_residual_mean_mps": mean(rv),
        "domega_pose_mae_radps": mean(dw),
        "domega_pose_p95_radps": q95(dw),
        "domega_pose_max_radps": mx(dw),
        "domega_pose_mae_degps": float(np.degrees(mean(dw))),
        "domega_pose_p95_degps": float(np.degrees(q95(dw))),
        "signed_yaw_residual_mean_radps": mean(rw),
        "signed_yaw_residual_mean_degps": float(np.degrees(mean(rw))),
        "accum_yaw_residual_final_deg": float(np.degrees(acc[-1])),
        "accum_yaw_residual_absmax_deg": float(np.degrees(np.max(np.abs(acc)))),
    }


def sequence_mean(run: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    use = [c for c in metrics if c in run.columns]
    agg = run.groupby(["method", "sequence"], dropna=False)[use].mean().reset_index()
    return agg


def percentile_bootstrap_mean(values: np.ndarray, n_boot: int = 5000, seed: int = 2026) -> Tuple[float, float, float]:
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    center = float(np.mean(x))
    if len(x) == 1:
        return center, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return center, float(lo), float(hi)


def paired_bootstrap_difference(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str,
    n_boot: int = 5000,
    seed: int = 2026,
) -> dict:
    z = a[["sequence", metric]].merge(b[["sequence", metric]], on="sequence", suffixes=("_a", "_b")).dropna()
    if not len(z):
        return {"n_sequences": 0, "mean_difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "relative_change_pct": np.nan}
    va = z[f"{metric}_a"].to_numpy(float); vb = z[f"{metric}_b"].to_numpy(float)
    diff = va - vb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff))) if len(diff) > 1 else np.zeros((1, 1), int)
    boot = np.mean(diff[idx], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975]) if len(diff) > 1 else (np.nan, np.nan)
    denom = float(np.mean(vb))
    rel = 100.0 * float(np.mean(diff)) / denom if abs(denom) > 1e-12 else np.nan
    return {
        "n_sequences": len(z),
        "mean_difference": float(np.mean(diff)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "relative_change_pct": float(rel),
    }


def spearman(x, y) -> float:
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 3:
        return np.nan
    rx = z.x.rank().to_numpy(float); ry = z.y.rank().to_numpy(float)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])
