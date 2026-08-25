#!/usr/bin/env python3
"""Service-relative operational-fidelity audit for frozen i2Nav Twin V2.

Purpose
-------
This analysis asks a narrower question than ordinary trajectory scoring:

    For which *services* (local relative-motion use vs global synchronized-state
    use) does a frozen sensor-lightweight twin satisfy a stated tolerance, and
    can a prevalidated envelope based only on online-observable operating
    context identify where that service is supported?

Scientific safeguards
---------------------
* Twin V2 is read-only. This script NEVER trains or tunes it.
* Ground truth is used only to construct evaluation outcomes and offline
  calibration envelopes. It is never passed as an online feature.
* Service tolerances are swept over a broad predeclared grid; representative
  examples are illustrative, not standards or safety limits.
* The physical sequence is the independent experimental unit. Seeds are
  averaged within each physical sequence/window before cross-sequence analysis.
* A retrospective aggregate baseline is included only as a scalar comparator;
  it is explicitly NOT described as deployable online.
* The conditioned envelope is falsifiable: if it does not reduce unsafe support
  without an unacceptable coverage cost, the report says so.

Expected frozen input
---------------------
results/i2nav_v2_full_loso/**/v2_evaluated_trajectory.csv
and, where available, adjacent v2_prediction_trace.csv files.

Outputs
-------
results/service_relative_fidelity/
    input_audit.json
    frozen_signature_verification.csv
    parking00_vs_parking02_verification.csv/.md
    physical_windows_seed_averaged.csv
    service_pass_rates_per_sequence.csv
    loso_monitor_decisions.csv
    loso_monitor_macro.csv
    condition_bin_definitions.csv
    service_relative_fidelity_report.md
    analysis_manifest.json
    figures/*.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

EXPECTED_SEQUENCES = [
    "building00", "building01", "building02", "parking00", "parking01",
    "parking02", "playground00", "street00", "street01", "street02",
]
EXPECTED_BASE_SEEDS = [42, 1042, 2042]
SOURCE_FULL_LOSO_COMMIT = "6540c01f90f3c1074de0d8dae9964a5276fbbc91"
V2_SCHEMA = "i2nav_twin_v2_slow_additive_sensor_consistency_v1"

# Frozen three-seed means previously audited from full_loso_per_sequence.csv.
# The pre-flight check is intentionally narrow: it protects against pointing the
# analysis at the wrong experiment tree. Tolerances are relative because tiny
# numerical differences can arise from CSV round trips / recomputation details.
FROZEN_SIGNATURE = {
    "parking00": {
        "ate_m": 2.106805,
        "heading_mae_deg": 0.613552,
        "rpe1_m": 0.132211,
        "rpe5_m": 0.318032,
        "rpe10_m": 0.453027,
        "dp_p95_m": 3.085755,
        "dtheta_p95_deg": 2.395108,
    },
    "parking02": {
        "ate_m": 11.350361,
        "heading_mae_deg": 16.719753,
        "rpe1_m": 0.019314,
        "rpe5_m": 0.054970,
        "rpe10_m": 0.097139,
        "dp_p95_m": 22.344767,
        "dtheta_p95_deg": 30.415317,
    },
}

DEFAULT_CONFIG = {
    "horizons_s": [1.0, 5.0, 10.0],
    "window_stride_mode": "nonoverlap_horizon",
    "envelope_quantile": 0.95,
    "min_bin_windows": 20,
    "condition_quantiles": [0.50, 0.80],
    "online_features": [
        "abs_speed_mps",
        "abs_yaw_rate_radps",
        "abs_accel_mps2",
        "abs_wheel_imu_disagreement_radps",
        "curvature_abs_radpm",
        "elapsed_s",
    ],
    "local_position_tolerances_m": [0.05, 0.10, 0.20, 0.50, 1.00],
    "local_heading_tolerances_deg": [1.0, 2.0, 5.0, 10.0, 20.0],
    "global_position_tolerances_m": [0.50, 1.0, 2.0, 5.0, 10.0, 20.0],
    "global_heading_tolerances_deg": [2.0, 5.0, 10.0, 20.0, 30.0, 45.0],
    "representative_services": [
        {
            "service_id": "local_1s_tight",
            "family": "local_relative_motion",
            "horizon_s": 1.0,
            "position_tolerance_m": 0.10,
            "heading_tolerance_deg": 2.0,
        },
        {
            "service_id": "local_5s_moderate",
            "family": "local_relative_motion",
            "horizon_s": 5.0,
            "position_tolerance_m": 0.25,
            "heading_tolerance_deg": 5.0,
        },
        {
            "service_id": "local_10s_preview",
            "family": "local_relative_motion",
            "horizon_s": 10.0,
            "position_tolerance_m": 0.50,
            "heading_tolerance_deg": 10.0,
        },
        {
            "service_id": "global_state_tracking",
            "family": "global_synchronization",
            "horizon_s": 0.0,
            "position_tolerance_m": 1.0,
            "heading_tolerance_deg": 5.0,
        },
    ],
}

ONLINE_FEATURE_DESCRIPTIONS = {
    "abs_speed_mps": "|odometry forward speed| at decision time",
    "abs_yaw_rate_radps": "|IMU yaw rate| at decision time",
    "abs_accel_mps2": "causal |odometry acceleration| from current/previous sample",
    "abs_wheel_imu_disagreement_radps": "|wheel yaw - IMU yaw| at decision time",
    "curvature_abs_radpm": "|IMU yaw rate| / max(|speed|, 0.1 m/s)",
    "elapsed_s": "elapsed time since twin/trajectory start (clock/context only)",
}


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if len(x) else float("nan")


def safe_quantile(x: Iterable[float], q: float) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if len(a) else float("nan")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in user.items():
            cfg[k] = v
    return cfg


def sequence_from_path(path: Path) -> str:
    s = str(path).lower()
    for seq in EXPECTED_SEQUENCES:
        if seq in s:
            return seq
    raise ValueError(f"Cannot identify held-out sequence from path: {path}")


def base_seed_from_path(path: Path) -> int:
    s = str(path)
    m = re.search(r"replicate_\d+_base(\d+)", s, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"base[_-]?(\d+)", s, flags=re.I)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot identify base seed from path: {path}")


def locate_raw_result_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        Path("results/i2nav_v2_full_loso/i2nav_v2_full_loso"),
        Path("results/i2nav_v2_full_loso"),
        Path("results"),
    ])
    seen: set[Path] = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            pass
        if c in seen or not c.exists():
            continue
        seen.add(c)
        hits = list(c.rglob("v2_evaluated_trajectory.csv"))
        identified = []
        for p in hits:
            try:
                sequence_from_path(p)
                base_seed_from_path(p)
                identified.append(p)
            except ValueError:
                pass
        pairs = {(sequence_from_path(p), base_seed_from_path(p)) for p in identified}
        if len(pairs) >= 30:
            # Return the narrowest common-ish root supplied/discovered. rglob below is robust.
            return c
    raise FileNotFoundError(
        "Could not locate the 30-run frozen V2 result tree. Expected a path such as "
        "results/i2nav_v2_full_loso/i2nav_v2_full_loso containing "
        "v2_evaluated_trajectory.csv files. Use --input-root if needed."
    )


def locate_frozen_summary(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    names = ["full_loso_per_sequence.csv"]
    for name in names:
        candidates = [
            Path("results/i2nav_v2_full_loso_summary") / name,
            Path("results/i2nav_v2_full_loso/i2nav_v2_full_loso_summary") / name,
        ]
        for p in candidates:
            if p.exists():
                return p
        found = sorted(Path("results").rglob(name)) if Path("results").exists() else []
        if len(found) == 1:
            return found[0]
    return None


def relative_pose(x: np.ndarray, y: np.ndarray, th: np.ndarray, i: int, j: int) -> tuple[float, float, float]:
    dx = float(x[j] - x[i])
    dy = float(y[j] - y[i])
    c = math.cos(float(th[i]))
    s = math.sin(float(th[i]))
    # World displacement expressed in the pose-i body frame.
    rx = c * dx + s * dy
    ry = -s * dx + c * dy
    rth = float(wrap_angle(float(th[j] - th[i])))
    return rx, ry, rth


def future_index_at_or_after(t: np.ndarray, start_i: int, horizon_s: float, max_gap_s: float) -> int | None:
    """Match the original frozen RPE convention: first sample at/after t+h."""
    target = float(t[start_i] + horizon_s)
    j = int(np.searchsorted(t, target, side="left"))
    if j <= start_i or j >= len(t):
        return None
    return j if float(t[j] - target) <= max_gap_s else None


def nonoverlap_start_indices(t: np.ndarray, horizon_s: float) -> list[int]:
    if len(t) == 0:
        return []
    out: list[int] = []
    next_allowed = float(t[0])
    for i, ti in enumerate(t):
        ti = float(ti)
        if ti + 1e-12 < next_allowed:
            continue
        out.append(i)
        next_allowed = ti + max(horizon_s, 1e-9)
    return out


def causal_acceleration(t: np.ndarray, speed: np.ndarray) -> np.ndarray:
    a = np.zeros(len(t), dtype=float)
    if len(t) < 2:
        return a
    dt = np.diff(t)
    dv = np.diff(speed)
    good = dt > 1e-9
    vals = np.zeros_like(dv, dtype=float)
    vals[good] = dv[good] / dt[good]
    a[1:] = vals
    a[0] = a[1] if len(a) > 1 else 0.0
    return a


def merge_online_features(traj: pd.DataFrame, trace_path: Path | None) -> pd.DataFrame:
    d = traj.copy().sort_values("time_s").drop_duplicates("time_s").reset_index(drop=True)
    req = [
        "time_s", "gt_east_m", "gt_north_m", "gt_heading_rad",
        "estimate_east_m", "estimate_north_m", "estimate_heading_rad",
    ]
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise ValueError(f"Trajectory missing required columns: {miss}")

    # Online/candidate observables. Prefer actual logged proprioceptive channels.
    speed_col = next((c for c in ["odo_speed_mps", "corrected_v_mps"] if c in d.columns), None)
    yaw_col = next((c for c in ["imu_yaw_rate_radps", "corrected_omega_radps"] if c in d.columns), None)
    if speed_col is None or yaw_col is None:
        raise ValueError("Need odo_speed_mps/corrected_v_mps and imu_yaw_rate_radps/corrected_omega_radps")

    if trace_path and trace_path.exists():
        q = pd.read_csv(trace_path).sort_values("time_s").drop_duplicates("time_s")
        keep = [c for c in [
            "time_s", "wheel_yaw_radps", "imu_yaw_radps",
            "wheel_imu_yaw_disagreement_radps",
        ] if c in q.columns]
        if len(keep) > 1:
            q = q[keep].copy()
            dt = np.diff(d["time_s"].to_numpy(float))
            dt = dt[np.isfinite(dt) & (dt > 0)]
            tol = max(0.05, 2.5 * float(np.median(dt))) if len(dt) else 0.2
            d = pd.merge_asof(
                d.sort_values("time_s"), q.sort_values("time_s"), on="time_s",
                direction="nearest", tolerance=tol, suffixes=("", "_trace")
            )

    t = d["time_s"].to_numpy(float)
    speed = d[speed_col].to_numpy(float)
    yaw = d[yaw_col].to_numpy(float)
    accel = causal_acceleration(t, speed)

    disagreement = np.full(len(d), np.nan, dtype=float)
    if "wheel_imu_yaw_disagreement_radps" in d.columns:
        disagreement = d["wheel_imu_yaw_disagreement_radps"].to_numpy(float)
    elif "wheel_yaw_radps" in d.columns and "imu_yaw_radps" in d.columns:
        disagreement = d["wheel_yaw_radps"].to_numpy(float) - d["imu_yaw_radps"].to_numpy(float)
    # Fallback is missing, never ground truth.

    elapsed = t - float(t[0])
    d["abs_speed_mps"] = np.abs(speed)
    d["abs_yaw_rate_radps"] = np.abs(yaw)
    d["abs_accel_mps2"] = np.abs(accel)
    d["abs_wheel_imu_disagreement_radps"] = np.abs(disagreement)
    d["curvature_abs_radpm"] = np.abs(yaw) / np.maximum(np.abs(speed), 0.1)
    d["elapsed_s"] = elapsed
    return d


def build_run_windows(path: Path, horizons_s: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sequence = sequence_from_path(path)
    base_seed = base_seed_from_path(path)
    traj = pd.read_csv(path)
    trace_path = path.with_name("v2_prediction_trace.csv")
    d = merge_online_features(traj, trace_path if trace_path.exists() else None)

    t = d["time_s"].to_numpy(float)
    gx = d["gt_east_m"].to_numpy(float)
    gy = d["gt_north_m"].to_numpy(float)
    gh = d["gt_heading_rad"].to_numpy(float)
    ex = d["estimate_east_m"].to_numpy(float)
    ey = d["estimate_north_m"].to_numpy(float)
    eh = d["estimate_heading_rad"].to_numpy(float)

    dt = np.diff(t)
    dt_good = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(dt_good)) if len(dt_good) else 0.1
    max_gap = max(0.20, 2.5 * median_dt)

    feature_cols = [c for c in DEFAULT_CONFIG["online_features"] if c in d.columns]
    local_rows: list[dict] = []
    for h in horizons_s:
        for ordinal, i in enumerate(nonoverlap_start_indices(t, float(h))):
            j = future_index_at_or_after(t, i, float(h), max_gap)
            if j is None:
                continue
            gp = relative_pose(gx, gy, gh, i, j)
            ep = relative_pose(ex, ey, eh, i, j)
            rpe_t = float(math.hypot(ep[0] - gp[0], ep[1] - gp[1]))
            rpe_h = abs(math.degrees(float(wrap_angle(ep[2] - gp[2]))))
            gp_end = float(math.hypot(ex[j] - gx[j], ey[j] - gy[j]))
            gh_end = abs(math.degrees(float(wrap_angle(eh[j] - gh[j]))))
            row = {
                "sequence": sequence,
                "base_seed": base_seed,
                "family": "local_relative_motion",
                "horizon_s": float(h),
                "window_ordinal": ordinal,
                "start_time_s": float(t[i]),
                "end_time_s": float(t[j]),
                "local_position_error_m": rpe_t,
                "local_heading_error_deg": rpe_h,
                "global_position_error_m": gp_end,
                "global_heading_error_deg": gh_end,
            }
            for c in feature_cols:
                row[c] = float(d[c].iloc[i]) if pd.notna(d[c].iloc[i]) else np.nan
            local_rows.append(row)

    # Global synchronized-state service is sampled at ~1 s intervals to avoid
    # giving high-rate timestamps pseudoreplication weight.
    global_rows: list[dict] = []
    sample_step = 1.0
    global_indices = [i for i in nonoverlap_start_indices(t, sample_step) if i != 0]
    for ordinal, i in enumerate(global_indices):
        row = {
            "sequence": sequence,
            "base_seed": base_seed,
            "family": "global_synchronization",
            "horizon_s": 0.0,
            "window_ordinal": ordinal,
            "start_time_s": float(t[i]),
            "end_time_s": float(t[i]),
            "local_position_error_m": np.nan,
            "local_heading_error_deg": np.nan,
            "global_position_error_m": float(math.hypot(ex[i] - gx[i], ey[i] - gy[i])),
            "global_heading_error_deg": abs(math.degrees(float(wrap_angle(eh[i] - gh[i])))),
        }
        for c in feature_cols:
            row[c] = float(d[c].iloc[i]) if pd.notna(d[c].iloc[i]) else np.nan
        global_rows.append(row)

    # Recompute headline signature from the full saved trajectory.
    dp = np.hypot(ex - gx, ey - gy)
    dh = np.abs(np.degrees(wrap_angle(eh - gh)))
    sig = {
        "sequence": sequence,
        "base_seed": base_seed,
        "ate_m": rmse(dp),
        "heading_mae_deg": float(np.nanmean(dh)),
        "dp_p95_m": safe_quantile(dp, 0.95),
        "dtheta_p95_deg": safe_quantile(dh, 0.95),
    }
    for h in horizons_s:
        vals = [r["local_position_error_m"] for r in local_rows if r["horizon_s"] == float(h)]
        # IMPORTANT: frozen headline RPE uses dense start points, not the
        # dependence-reduced windows above. Recompute dense for signature.
        dense = []
        for i in range(len(t)):
            j = future_index_at_or_after(t, i, float(h), max_gap)
            if j is None:
                continue
            gp = relative_pose(gx, gy, gh, i, j)
            ep = relative_pose(ex, ey, eh, i, j)
            dense.append(math.hypot(ep[0] - gp[0], ep[1] - gp[1]))
        sig[f"rpe{int(h)}_m"] = rmse(np.asarray(dense, dtype=float))
    return pd.DataFrame(local_rows), pd.DataFrame(global_rows), sig


def seed_average_physical_windows(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    # Round time only as a stable cross-seed key. All seeds evaluate the same
    # held-out physical trajectory.
    x = rows.copy()
    x["start_time_key"] = x["start_time_s"].round(3)
    keys = ["sequence", "family", "horizon_s", "start_time_key"]
    num = [
        "start_time_s", "end_time_s", "local_position_error_m",
        "local_heading_error_deg", "global_position_error_m",
        "global_heading_error_deg",
    ] + [f for f in features if f in x.columns]
    agg = {c: "mean" for c in num}
    agg["base_seed"] = "nunique"
    y = x.groupby(keys, as_index=False).agg(agg).rename(columns={"base_seed": "n_seeds"})
    return y


def recompute_signature_from_runs(sigs: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in sigs.columns if c not in ["sequence", "base_seed"]]
    return sigs.groupby("sequence", as_index=False)[metric_cols].mean()


def summary_signature_from_csv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    seq_col = "test_sequence" if "test_sequence" in d.columns else "sequence"
    mapping = {
        "v2_ate_m_mean": "ate_m",
        "v2_heading_mae_deg_mean": "heading_mae_deg",
        "v2_rpe_1s_m_mean": "rpe1_m",
        "v2_rpe_5s_m_mean": "rpe5_m",
        "v2_rpe_10s_m_mean": "rpe10_m",
        "fidelity_Dp_p95_m_mean": "dp_p95_m",
        "fidelity_Dtheta_p95_deg_mean": "dtheta_p95_deg",
    }
    missing = [c for c in mapping if c not in d.columns]
    if missing:
        raise ValueError(f"Frozen summary is missing expected columns: {missing}")
    out = d[[seq_col] + list(mapping)].rename(columns={seq_col: "sequence", **mapping})
    return out


def verify_frozen_signature(per_sequence: pd.DataFrame, rel_tol: float = 0.02) -> pd.DataFrame:
    rows = []
    for seq, expected in FROZEN_SIGNATURE.items():
        q = per_sequence[per_sequence["sequence"] == seq]
        if len(q) != 1:
            raise RuntimeError(f"Frozen signature check: expected one row for {seq}, found {len(q)}")
        r = q.iloc[0]
        for metric, exp in expected.items():
            obs = float(r[metric])
            rel = abs(obs - exp) / max(abs(exp), 1e-12)
            rows.append({
                "sequence": seq, "metric": metric,
                "expected": exp, "observed": obs,
                "relative_error": rel, "passes": bool(rel <= rel_tol),
            })
    out = pd.DataFrame(rows)
    if not bool(out["passes"].all()):
        bad = out[~out["passes"]]
        raise RuntimeError(
            "Frozen signature mismatch. Refusing to continue because this may be the wrong result tree.\n" +
            bad.to_string(index=False)
        )
    return out


def parking_verification(per_sequence: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ate_m", "heading_mae_deg", "rpe1_m", "rpe5_m", "rpe10_m",
        "dp_p95_m", "dtheta_p95_deg",
    ]
    a = per_sequence.set_index("sequence").loc["parking00"]
    b = per_sequence.set_index("sequence").loc["parking02"]
    rows = []
    for m in metrics:
        av, bv = float(a[m]), float(b[m])
        rows.append({
            "metric": m,
            "parking00": av,
            "parking02": bv,
            "parking02_over_parking00": bv / av if abs(av) > 1e-12 else np.nan,
            "parking02_lower_is_better": bool(bv < av),
        })
    return pd.DataFrame(rows)


def write_parking_markdown(v: pd.DataFrame, path: Path) -> None:
    x = v.set_index("metric")
    def f(m):
        r = x.loc[m]
        return float(r.parking00), float(r.parking02), float(r.parking02_over_parking00)
    p00_ate,p02_ate,ate_ratio=f("ate_m")
    p00_rpe,p02_rpe,rpe_ratio=f("rpe10_m")
    p00_dp,p02_dp,dp_ratio=f("dp_p95_m")
    p00_dh,p02_dh,dh_ratio=f("dtheta_p95_deg")
    text = f"""# Frozen parking00 vs parking02 verification

This file is generated before the service-relative analysis and is a guard
against changing the research question based on a misremembered result.

| Metric | parking00 | parking02 | parking02 / parking00 |
|---|---:|---:|---:|
| ATE (m) | {p00_ate:.6f} | {p02_ate:.6f} | {ate_ratio:.3f}x |
| RPE10 (m) | {p00_rpe:.6f} | {p02_rpe:.6f} | {rpe_ratio:.3f}x |
| Dp p95 (m) | {p00_dp:.6f} | {p02_dp:.6f} | {dp_ratio:.3f}x |
| Dtheta p95 (deg) | {p00_dh:.6f} | {p02_dh:.6f} | {dh_ratio:.3f}x |

**Verified interpretation:** parking02 has substantially *lower* 10-s relative
translation error than parking00, yet much larger accumulated/global position
and heading disagreement. This is an empirical inversion between two service
claims, not evidence that RPE or ATE is mathematically wrong. The underlying
fact that local errors can coexist with global drift is not itself claimed as
novel; the new experiment asks whether a service-relative fidelity protocol
can turn that distinction into a defensible validity decision.
"""
    path.write_text(text, encoding="utf-8")


def make_service_grid(cfg: dict) -> pd.DataFrame:
    rows = []
    for h in cfg["horizons_s"]:
        for pt in cfg["local_position_tolerances_m"]:
            for ht in cfg["local_heading_tolerances_deg"]:
                rows.append({
                    "family": "local_relative_motion", "horizon_s": float(h),
                    "position_tolerance_m": float(pt), "heading_tolerance_deg": float(ht),
                })
    for pt in cfg["global_position_tolerances_m"]:
        for ht in cfg["global_heading_tolerances_deg"]:
            rows.append({
                "family": "global_synchronization", "horizon_s": 0.0,
                "position_tolerance_m": float(pt), "heading_tolerance_deg": float(ht),
            })
    return pd.DataFrame(rows)


def error_columns(family: str) -> tuple[str, str]:
    if family == "local_relative_motion":
        return "local_position_error_m", "local_heading_error_deg"
    if family == "global_synchronization":
        return "global_position_error_m", "global_heading_error_deg"
    raise ValueError(family)


def service_pass_rates(windows: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, spec in grid.iterrows():
        family = str(spec.family)
        h = float(spec.horizon_s)
        pcol, hcol = error_columns(family)
        d = windows[(windows.family == family) & np.isclose(windows.horizon_s, h)].copy()
        for seq, g in d.groupby("sequence"):
            truth = (g[pcol] <= float(spec.position_tolerance_m)) & (g[hcol] <= float(spec.heading_tolerance_deg))
            rows.append({
                "sequence": seq, "family": family, "horizon_s": h,
                "position_tolerance_m": float(spec.position_tolerance_m),
                "heading_tolerance_deg": float(spec.heading_tolerance_deg),
                "n_physical_windows": int(len(g)),
                "service_valid_fraction": float(truth.mean()) if len(g) else np.nan,
                "position_pass_fraction": float((g[pcol] <= float(spec.position_tolerance_m)).mean()) if len(g) else np.nan,
                "heading_pass_fraction": float((g[hcol] <= float(spec.heading_tolerance_deg)).mean()) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


@dataclass
class BinDefinition:
    held_out_sequence: str
    family: str
    horizon_s: float
    feature: str
    q50: float
    q80: float
    n_train_windows: int


def feature_bin(values: pd.Series, q50: float, q80: float) -> np.ndarray:
    a = values.to_numpy(float)
    # Missing online signal -> unknown; caller will fall back to unconditional.
    out = np.full(len(a), -1, dtype=int)
    ok = np.isfinite(a)
    out[ok & (a <= q50)] = 0
    out[ok & (a > q50) & (a <= q80)] = 1
    out[ok & (a > q80)] = 2
    return out


def fit_condition_envelope(
    train: pd.DataFrame,
    family: str,
    horizon_s: float,
    features: list[str],
    q: float,
    qcuts: tuple[float, float],
    min_bin_windows: int,
) -> tuple[dict, list[dict]]:
    pcol, hcol = error_columns(family)
    d = train[(train.family == family) & np.isclose(train.horizon_s, horizon_s)].copy()
    if len(d) < 20:
        raise RuntimeError(f"Too few training windows for {family} h={horizon_s}: {len(d)}")
    uncond = {
        "p": safe_quantile(d[pcol], q),
        "h": safe_quantile(d[hcol], q),
    }
    model = {"unconditional": uncond, "features": {}}
    defs: list[dict] = []
    for feat in features:
        if feat not in d.columns or d[feat].notna().sum() < max(20, min_bin_windows):
            continue
        vals = d[feat].dropna().to_numpy(float)
        c50, c80 = [float(np.quantile(vals, qq)) for qq in qcuts]
        bins = feature_bin(d[feat], c50, c80)
        fb = {}
        for b in [0,1,2]:
            g = d[bins == b]
            if len(g) >= min_bin_windows:
                fb[str(b)] = {
                    "p": safe_quantile(g[pcol], q),
                    "h": safe_quantile(g[hcol], q),
                    "n": int(len(g)),
                }
            else:
                fb[str(b)] = {"p": uncond["p"], "h": uncond["h"], "n": int(len(g)), "fallback": True}
        model["features"][feat] = {"q50": c50, "q80": c80, "bins": fb}
        defs.append({
            "feature": feat, "q50": c50, "q80": c80,
            "n_train_windows": int(len(d)),
        })
    return model, defs


def predict_envelope(model: dict, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    up = float(model["unconditional"]["p"])
    uh = float(model["unconditional"]["h"])
    p = np.full(len(test), -np.inf, dtype=float)
    h = np.full(len(test), -np.inf, dtype=float)
    used = np.zeros(len(test), dtype=int)
    # Conservative composition across the *conditional* univariate envelopes.
    # Unlike max(unconditional, conditional), this can become sharper in nominal
    # conditions while widening when any observed condition is severe.
    for feat, meta in model["features"].items():
        bins = feature_bin(test[feat], float(meta["q50"]), float(meta["q80"])) if feat in test.columns else np.full(len(test), -1)
        for b in [0,1,2]:
            mask = bins == b
            if not np.any(mask):
                continue
            p[mask] = np.maximum(p[mask], float(meta["bins"][str(b)]["p"]))
            h[mask] = np.maximum(h[mask], float(meta["bins"][str(b)]["h"]))
            used[mask] += 1
    # Unknown/missing context falls back to the unconditional envelope.
    p[used == 0] = up
    h[used == 0] = uh
    return p, h


def decision_metrics(truth: np.ndarray, supported: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=bool)
    supported = np.asarray(supported, dtype=bool)
    n = len(truth)
    fs = supported & ~truth
    fr = ~supported & truth
    tp = supported & truth
    tn = ~supported & ~truth
    support_n = int(supported.sum())
    valid_n = int(truth.sum())
    return {
        "n_windows": n,
        "actual_valid_rate": float(truth.mean()) if n else np.nan,
        "support_rate": float(supported.mean()) if n else np.nan,
        "false_safe_fraction": float(fs.mean()) if n else np.nan,
        "unsafe_among_supported": float(fs.sum() / support_n) if support_n else np.nan,
        "false_reject_fraction": float(fr.mean()) if n else np.nan,
        "valid_captured_fraction": float(tp.sum() / valid_n) if valid_n else np.nan,
        "accuracy": float((tp | tn).mean()) if n else np.nan,
    }


def aggregate_scalar_support(test: pd.DataFrame, family: str, pt: float, ht: float) -> np.ndarray:
    """Retrospective scalar certification baseline; deliberately non-deployable.

    If the sequence-level aggregate error satisfies the service specification,
    the scalar baseline certifies every window; otherwise it rejects all.
    """
    pcol, hcol = error_columns(family)
    # Match common conventions: RMSE for translation, MAE for heading.
    pscore = rmse(test[pcol].to_numpy(float))
    hscore = float(np.nanmean(test[hcol].to_numpy(float)))
    passed = bool(pscore <= pt and hscore <= ht)
    return np.full(len(test), passed, dtype=bool)


def loso_monitor_analysis(windows: pd.DataFrame, grid: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rows = []
    bin_defs = []
    calibration_rows = []
    features = [f for f in cfg["online_features"] if f in windows.columns]
    q = float(cfg["envelope_quantile"])
    qcuts = tuple(float(x) for x in cfg["condition_quantiles"])
    min_bin = int(cfg["min_bin_windows"])

    for held in EXPECTED_SEQUENCES:
        train_all = windows[windows.sequence != held]
        test_all = windows[windows.sequence == held]
        if test_all.empty:
            continue
        unique_specs = grid[["family","horizon_s"]].drop_duplicates()
        for _, fh in unique_specs.iterrows():
            family = str(fh.family); horizon = float(fh.horizon_s)
            train = train_all[(train_all.family == family) & np.isclose(train_all.horizon_s,horizon)]
            test = test_all[(test_all.family == family) & np.isclose(test_all.horizon_s,horizon)].copy()
            if test.empty:
                continue
            model, defs = fit_condition_envelope(train_all, family, horizon, features, q, qcuts, min_bin)
            cp, ch = predict_envelope(model, test)
            up = np.full(len(test), float(model["unconditional"]["p"]))
            uh = np.full(len(test), float(model["unconditional"]["h"]))
            for dd in defs:
                bin_defs.append({
                    "held_out_sequence": held, "family": family, "horizon_s": horizon,
                    **dd,
                })

            pcol, hcol = error_columns(family)
            actual_p=test[pcol].to_numpy(float); actual_h=test[hcol].to_numpy(float)
            for method,bp,bh in [
                ("unconditional_loso_envelope",up,uh),
                ("condition_aware_loso_envelope",cp,ch),
            ]:
                calibration_rows.append({
                    "sequence":held,"family":family,"horizon_s":horizon,"method":method,
                    "n_windows":int(len(test)),
                    "position_coverage":float(np.mean(actual_p <= bp)),
                    "heading_coverage":float(np.mean(actual_h <= bh)),
                    "joint_coverage":float(np.mean((actual_p <= bp) & (actual_h <= bh))),
                    "mean_position_bound_m":float(np.mean(bp)),
                    "mean_heading_bound_deg":float(np.mean(bh)),
                })
            specs = grid[(grid.family == family) & np.isclose(grid.horizon_s,horizon)]
            for _, spec in specs.iterrows():
                pt=float(spec.position_tolerance_m); ht=float(spec.heading_tolerance_deg)
                truth = (test[pcol].to_numpy(float) <= pt) & (test[hcol].to_numpy(float) <= ht)
                supports = {
                    "retrospective_aggregate_scalar": aggregate_scalar_support(test, family, pt, ht),
                    "unconditional_loso_envelope": (up <= pt) & (uh <= ht),
                    "condition_aware_loso_envelope": (cp <= pt) & (ch <= ht),
                }
                for method, sup in supports.items():
                    m = decision_metrics(truth, sup)
                    rows.append({
                        "sequence": held, "family": family, "horizon_s": horizon,
                        "position_tolerance_m": pt, "heading_tolerance_deg": ht,
                        "method": method, **m,
                        "mean_position_bound_m": float(np.mean(cp if method=="condition_aware_loso_envelope" else up)) if method!="retrospective_aggregate_scalar" else np.nan,
                        "mean_heading_bound_deg": float(np.mean(ch if method=="condition_aware_loso_envelope" else uh)) if method!="retrospective_aggregate_scalar" else np.nan,
                    })
    return pd.DataFrame(rows), pd.DataFrame(bin_defs), pd.DataFrame(calibration_rows)


def macro_decisions(per_seq: pd.DataFrame) -> pd.DataFrame:
    if per_seq.empty:
        return per_seq.copy()
    group = ["family","horizon_s","position_tolerance_m","heading_tolerance_deg","method"]
    metrics = [
        "actual_valid_rate","support_rate","false_safe_fraction",
        "unsafe_among_supported","false_reject_fraction","valid_captured_fraction","accuracy",
    ]
    rows=[]
    for key,g in per_seq.groupby(group, dropna=False):
        r={k:v for k,v in zip(group,key)}
        r["n_sequences"] = int(g.sequence.nunique())
        for m in metrics:
            vals=g[m].to_numpy(float); vals=vals[np.isfinite(vals)]
            r[f"{m}_sequence_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            r[f"{m}_sequence_median"] = float(np.median(vals)) if len(vals) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def representative_service_rows(per_seq: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows=[]
    for spec in cfg["representative_services"]:
        q=per_seq[
            (per_seq.family==spec["family"]) &
            np.isclose(per_seq.horizon_s,float(spec["horizon_s"])) &
            np.isclose(per_seq.position_tolerance_m,float(spec["position_tolerance_m"])) &
            np.isclose(per_seq.heading_tolerance_deg,float(spec["heading_tolerance_deg"]))
        ].copy()
        if q.empty:
            # Representative tolerance may not be in the Cartesian sweep; skip.
            continue
        q["service_id"] = spec["service_id"]
        rows.append(q)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def build_report(
    out: Path,
    per_sequence_signature: pd.DataFrame,
    parking: pd.DataFrame,
    service_rates: pd.DataFrame,
    per_seq_decisions: pd.DataFrame,
    macro: pd.DataFrame,
    cfg: dict,
) -> None:
    p = parking.set_index("metric")
    ate_ratio=float(p.loc["ate_m","parking02_over_parking00"])
    rpe_ratio=float(p.loc["rpe10_m","parking02_over_parking00"])
    dp_ratio=float(p.loc["dp_p95_m","parking02_over_parking00"])
    dh_ratio=float(p.loc["dtheta_p95_deg","parking02_over_parking00"])

    lines=[]
    lines += [
        "# Service-relative operational-fidelity audit",
        "",
        "## 1. Pre-flight verification",
        "",
        "The frozen parking contrast was reproduced before the new analysis.",
        f"parking02 / parking00: ATE **{ate_ratio:.2f}x**, RPE10 **{rpe_ratio:.2f}x**, "
        f"Dp p95 **{dp_ratio:.2f}x**, Dtheta p95 **{dh_ratio:.2f}x**.",
        "",
        "Because lower error is better, this means parking02 is substantially better at 10-s relative motion while being dramatically worse as a globally synchronized physical-virtual state. This empirical inversion is real. It is motivation, not by itself a novelty claim.",
        "",
        "## 2. What is new in this analysis",
        "",
        "The analysis converts fidelity into explicit service claims. A local-relative-motion service and a global-synchronization service are evaluated separately. Ground truth defines whether each service actually met its tolerance. A leave-one-physical-sequence-out envelope then attempts to decide support using only prevalidated distributions and online-observable operating context.",
        "",
        "The retrospective aggregate scalar is a deliberately strong hindsight comparator and is **not deployable online**. The unconditional and condition-aware LOSO envelopes do not use held-out-sequence ground truth at decision time; held-out ground truth is used only for scoring.",
        "",
        "## 3. Primary falsification question",
        "",
        "> Does condition-aware, service-relative support reduce false-safe use of the twin compared with a condition-blind envelope and a retrospective aggregate scalar, across a broad tolerance sweep, without simply rejecting almost every window?",
        "",
    ]

    # Across-grid comparison of the two deployable-ish envelopes.
    if not macro.empty:
        wide = macro.pivot_table(
            index=["family","horizon_s","position_tolerance_m","heading_tolerance_deg"],
            columns="method", values=["false_safe_fraction_sequence_mean","support_rate_sequence_mean"],
        )
        try:
            fs_c = wide[("false_safe_fraction_sequence_mean","condition_aware_loso_envelope")]
            fs_u = wide[("false_safe_fraction_sequence_mean","unconditional_loso_envelope")]
            sp_c = wide[("support_rate_sequence_mean","condition_aware_loso_envelope")]
            sp_u = wide[("support_rate_sequence_mean","unconditional_loso_envelope")]
            ok=np.isfinite(fs_c)&np.isfinite(fs_u)&np.isfinite(sp_c)&np.isfinite(sp_u)
            improve = (fs_c[ok] < fs_u[ok]-1e-12)
            equal = np.isclose(fs_c[ok],fs_u[ok],atol=1e-12)
            coverage_delta=(sp_c[ok]-sp_u[ok])
            lines += [
                "## 4. Across-tolerance summary",
                "",
                f"Comparable grid points: **{int(ok.sum())}**.",
                f"Condition-aware envelope has lower sequence-mean false-safe fraction at **{int(improve.sum())}/{int(ok.sum())}** grid points; equal at **{int(equal.sum())}/{int(ok.sum())}**.",
                f"Median support-rate change (condition-aware minus unconditional): **{float(np.nanmedian(coverage_delta)):+.3f}**.",
                "",
            ]
            # Deliberately conservative interpretation.
            if ok.sum() and improve.mean() >= 0.5 and np.nanmedian(coverage_delta) >= -0.10:
                lines += [
                    "**Interpretation:** the new service-envelope idea receives meaningful support across the predeclared sweep. This is not proof of universal superiority; prospective UGV01 validation is still required.",
                    "",
                ]
            else:
                lines += [
                    "**Interpretation:** the current i2Nav evidence does not robustly establish that condition-aware service support is better than the simpler condition-blind envelope. Do not promote the operational-monitoring claim without revising or prospectively validating it.",
                    "",
                ]
        except KeyError:
            pass

    reps = representative_service_rows(per_seq_decisions, cfg)
    if not reps.empty:
        lines += ["## 5. Representative illustrative services", ""]
        for sid,g in reps.groupby("service_id"):
            lines.append(f"### {sid}")
            lines.append("")
            for method,mm in g.groupby("method"):
                lines.append(
                    f"- {method}: mean support={mm.support_rate.mean():.3f}, "
                    f"false-safe={mm.false_safe_fraction.mean():.3f}, "
                    f"false-reject={mm.false_reject_fraction.mean():.3f}."
                )
            lines.append("")

    lines += [
        "## 6. Claim boundary",
        "",
        "This study does **not** claim that ATE/RPE are wrong, that local/global divergence is a newly discovered mathematical phenomenon, that the illustrative tolerances are safety standards, or that ground-truth TFP discrepancies are directly available online. The contribution under test is narrower: a service-relative validity protocol for a synchronized sensor-lightweight twin, together with an auditable operating-context envelope and prospective physical validation path.",
        "",
        "## 7. Next physical step",
        "",
        "Freeze the service definitions and monitor before the final UGV01 outdoor experiment. That run should be treated prospectively: no retuning after inspecting its ground-truth outcomes.",
    ]
    (out/"service_relative_fidelity_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def make_figures(out: Path, parking: pd.DataFrame, service_rates: pd.DataFrame, macro: pd.DataFrame, cfg: dict) -> None:
    if plt is None:
        return
    fdir=out/"figures"; fdir.mkdir(parents=True,exist_ok=True)
    p=parking.set_index("metric")

    fig,ax=plt.subplots(figsize=(7.2,4.2))
    metrics=["rpe1_m","rpe5_m","rpe10_m"]
    x=np.arange(len(metrics)); w=.36
    ax.bar(x-w/2,[p.loc[m,"parking00"] for m in metrics],w,label="parking00")
    ax.bar(x+w/2,[p.loc[m,"parking02"] for m in metrics],w,label="parking02")
    ax.set_xticks(x,["RPE1","RPE5","RPE10"])
    ax.set_ylabel("Relative translation error [m]")
    ax.set_title("Local relative-motion fidelity: parking02 is better")
    ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(fdir/"parking_local_contrast.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.2))
    metrics=["ate_m","dp_p95_m"]
    x=np.arange(len(metrics)); w=.36
    ax.bar(x-w/2,[p.loc[m,"parking00"] for m in metrics],w,label="parking00")
    ax.bar(x+w/2,[p.loc[m,"parking02"] for m in metrics],w,label="parking02")
    ax.set_xticks(x,["ATE","Dp p95"])
    ax.set_ylabel("Global position disagreement [m]")
    ax.set_title("Global synchronization: parking02 is much worse")
    ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(fdir/"parking_global_contrast.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    # Representative false-safe/support plot if exact service exists in sweep.
    reps=representative_service_rows(macro.rename(columns={
        "support_rate_sequence_mean":"support_rate",
        "false_safe_fraction_sequence_mean":"false_safe_fraction",
        "false_reject_fraction_sequence_mean":"false_reject_fraction",
    }),cfg) if not macro.empty else pd.DataFrame()
    if not reps.empty:
        for sid,g in reps.groupby("service_id"):
            fig,ax=plt.subplots(figsize=(7.2,4.2))
            methods=list(g.method)
            fs=[float(v) for v in g.false_safe_fraction]
            sr=[float(v) for v in g.support_rate]
            x=np.arange(len(methods)); w=.36
            ax.bar(x-w/2,fs,w,label="false-safe fraction")
            ax.bar(x+w/2,sr,w,label="support rate")
            ax.set_xticks(x,[m.replace("_","\n") for m in methods])
            ax.set_ylim(0,1)
            ax.set_ylabel("Fraction")
            ax.set_title(f"Service decision summary: {sid}")
            ax.legend(); ax.grid(axis="y",alpha=.25)
            fig.tight_layout(); fig.savefig(fdir/f"{sid}_decision_summary.png",dpi=220,bbox_inches="tight"); plt.close(fig)


def run_verify_only(summary_path: Path, out: Path) -> None:
    perseq=summary_signature_from_csv(summary_path)
    ver=verify_frozen_signature(perseq)
    parking=parking_verification(perseq)
    out.mkdir(parents=True,exist_ok=True)
    ver.to_csv(out/"frozen_signature_verification.csv",index=False)
    parking.to_csv(out/"parking00_vs_parking02_verification.csv",index=False)
    write_parking_markdown(parking,out/"parking00_vs_parking02_verification.md")
    print("FROZEN SIGNATURE: PASS")
    print(parking.to_string(index=False))


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",default=None,help="Frozen 30-run V2 result root")
    ap.add_argument("--summary",default=None,help="Optional frozen full_loso_per_sequence.csv")
    ap.add_argument("--config",default=None,help="JSON config override")
    ap.add_argument("--output-root",default="results/service_relative_fidelity")
    ap.add_argument("--verify-only",action="store_true",help="Verify known frozen signature without running window analysis")
    ap.add_argument("--signature-relative-tolerance",type=float,default=0.02)
    args=ap.parse_args()

    cfg=load_config(args.config)
    out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True)

    summary_path=locate_frozen_summary(args.summary)
    if args.verify_only and summary_path is not None:
        run_verify_only(summary_path,out)
        return 0

    root=locate_raw_result_root(args.input_root)
    files=[]
    seen=set()
    for p in sorted(root.rglob("v2_evaluated_trajectory.csv")):
        try:
            key=(sequence_from_path(p),base_seed_from_path(p))
        except ValueError:
            continue
        # A broad results/ root can contain old smoke/pilot experiments. Keep
        # only one file per sequence/base-seed and prefer paths containing
        # 'full_loso'. Ambiguity is fatal rather than silently choosing.
        files.append((key,p))
    by={}
    for key,p in files:
        by.setdefault(key,[]).append(p)
    selected=[]
    for key,ps in sorted(by.items()):
        full=[p for p in ps if "full_loso" in str(p).lower()]
        cand=full if full else ps
        if len(cand)!=1:
            raise RuntimeError(f"Ambiguous frozen trajectory for {key}: {cand}")
        selected.append(cand[0])
    pairs={(sequence_from_path(p),base_seed_from_path(p)) for p in selected}
    expected={(s,b) for s in EXPECTED_SEQUENCES for b in EXPECTED_BASE_SEEDS}
    if pairs != expected:
        missing=sorted(expected-pairs); extra=sorted(pairs-expected)
        raise RuntimeError(f"Expected exact 10x3 frozen runs. missing={missing} extra={extra}")

    # Build all run-level windows and recompute the frozen signature.
    local=[]; global_=[]; sigs=[]
    for k,p in enumerate(selected,1):
        print(f"[{k:02d}/30] {sequence_from_path(p)} base{base_seed_from_path(p)}")
        L,G,S=build_run_windows(p,[float(x) for x in cfg["horizons_s"]])
        local.append(L); global_.append(G); sigs.append(S)
    run_windows=pd.concat(local+global_,ignore_index=True)
    sigs_df=pd.DataFrame(sigs)
    perseq_sig=recompute_signature_from_runs(sigs_df)

    # If authoritative frozen summary is available, use it for exact signature
    # verification; otherwise use independent raw recomputation.
    if summary_path is not None:
        auth=summary_signature_from_csv(summary_path)
    else:
        auth=perseq_sig
    verification=verify_frozen_signature(auth,rel_tol=float(args.signature_relative_tolerance))
    verification.to_csv(out/"frozen_signature_verification.csv",index=False)

    # Also record differences between recomputation and the authoritative table.
    if summary_path is not None:
        merged=auth.merge(perseq_sig,on="sequence",suffixes=("_frozen","_recomputed"))
        diffs=[]
        for _,r in merged.iterrows():
            for m in ["ate_m","heading_mae_deg","rpe1_m","rpe5_m","rpe10_m","dp_p95_m","dtheta_p95_deg"]:
                a=float(r[f"{m}_frozen"]); b=float(r[f"{m}_recomputed"])
                diffs.append({"sequence":r.sequence,"metric":m,"frozen":a,"recomputed":b,
                              "relative_difference":abs(b-a)/max(abs(a),1e-12)})
        pd.DataFrame(diffs).to_csv(out/"raw_recomputation_vs_frozen_summary.csv",index=False)

    parking=parking_verification(auth)
    parking.to_csv(out/"parking00_vs_parking02_verification.csv",index=False)
    write_parking_markdown(parking,out/"parking00_vs_parking02_verification.md")
    print("\nFROZEN SIGNATURE: PASS")
    print(parking.to_string(index=False))

    features=[f for f in cfg["online_features"] if f in run_windows.columns]
    physical=seed_average_physical_windows(run_windows,features)
    physical.to_csv(out/"physical_windows_seed_averaged.csv",index=False)

    grid=make_service_grid(cfg)
    grid.to_csv(out/"service_tolerance_grid.csv",index=False)
    rates=service_pass_rates(physical,grid)
    rates.to_csv(out/"service_pass_rates_per_sequence.csv",index=False)

    decisions,bindefs,calibration=loso_monitor_analysis(physical,grid,cfg)
    decisions.to_csv(out/"loso_monitor_decisions.csv",index=False)
    bindefs.to_csv(out/"condition_bin_definitions.csv",index=False)
    calibration.to_csv(out/"loso_envelope_calibration.csv",index=False)
    macro=macro_decisions(decisions)
    macro.to_csv(out/"loso_monitor_macro.csv",index=False)

    build_report(out,auth,parking,rates,decisions,macro,cfg)
    make_figures(out,parking,rates,macro,cfg)

    audit={
        "status":"PASS",
        "input_root":str(root),
        "n_runs":len(selected),
        "n_sequences":len({sequence_from_path(p) for p in selected}),
        "base_seeds":sorted({base_seed_from_path(p) for p in selected}),
        "source_full_loso_commit":SOURCE_FULL_LOSO_COMMIT,
        "v2_schema":V2_SCHEMA,
        "summary_path":str(summary_path) if summary_path else None,
        "ground_truth_role":"evaluation/calibration target only; never online feature",
        "online_features":features,
        "physical_unit_of_inference":"held-out physical sequence",
    }
    (out/"input_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")

    manifest={
        "analysis":"i2nav_service_relative_fidelity",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),
        "numpy":np.__version__,
        "pandas":pd.__version__,
        "source_full_loso_commit":SOURCE_FULL_LOSO_COMMIT,
        "v2_schema":V2_SCHEMA,
        "config":cfg,
        "method_notes":{
            "seed_aggregation":"seed-average within matched physical sequence/time window before cross-sequence analysis",
            "window_dependence":"non-overlapping-by-horizon local windows; ~1 s global samples",
            "loso":"held-out physical sequence never enters envelope calibration",
            "condition_envelope":"q95 per predeclared online feature bin; conservative max across active feature bins",
            "retrospective_scalar":"hindsight comparator only, not online/deployable",
            "tolerances":"stress-test sweep; not safety standards",
        },
        "input_files":[{"path":str(p),"sha256":sha256_file(p)} for p in selected],
    }
    (out/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    print(f"\nDONE: {out}")
    print(f"Report: {out/'service_relative_fidelity_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
