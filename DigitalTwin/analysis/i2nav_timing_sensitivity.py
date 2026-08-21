#!/usr/bin/env python3
"""
Architecture-independent physical--virtual timing sensitivity analysis.

This operates on SAVED frozen V2 trajectory CSVs. It does not retrain V2 and
does not change any checkpoint. It perturbs the timing relationship between
the physical trajectory and the already-produced virtual-state stream.

Three analyses are provided:
  1) Fixed communication/update delay: 0, 25, 50, 100, 200 ms.
     The virtual state is delivered by causal zero-order hold.
  2) Constant clock/timestamp offset: -100 ... +100 ms.
     The virtual trajectory is time-shifted and interpolated.
  3) Timestamp jitter: 0, 10, 25, 50 ms standard deviation with fixed seeds.

IMPORTANT CLAIM BOUNDARY:
This is a synchronization/timestamp-sensitivity experiment on frozen virtual
trajectories. It is NOT a test of how delayed raw IMU/odometry inputs alter
the neural-network correction itself. Phrase it accordingly in the paper.

Run from repository root:
    python -m DigitalTwin.analysis.i2nav_timing_sensitivity --input-root results

Outputs:
    results/i2nav_timing_sensitivity/
      timing_sensitivity_all_runs.csv
      timing_sensitivity_sequence_summary.csv
      timing_sensitivity_dataset_summary.csv
      timing_sensitivity_paired_statistics.csv
      timing_sensitivity_report.md
      delay_sensitivity.png
      clock_offset_sensitivity.png
      jitter_sensitivity.png
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(f"matplotlib is required: {exc}")


METRICS = [
    "ate_m",
    "heading_mae_deg",
    "rpe1_m",
    "rpe5_m",
    "rpe10_m",
    "dp_p95_m",
    "dtheta_p95_deg",
]

COLUMN_ALIASES = {
    "time": [
        "time_s", "timestamp_s", "time", "t", "timestamp",
        "bag_time_s", "elapsed_s",
    ],
    "gt_x": [
        "gt_east_m", "gt_x", "x_gt", "ground_truth_x", "truth_x", "physical_x", "xp",
        "gt_east", "east_gt", "true_x", "reference_x",
    ],
    "gt_y": [
        "gt_north_m", "gt_y", "y_gt", "ground_truth_y", "truth_y", "physical_y", "yp",
        "gt_north", "north_gt", "true_y", "reference_y",
    ],
    "gt_theta": [
        "gt_heading_rad", "gt_heading", "heading_gt", "gt_yaw", "yaw_gt", "theta_gt",
        "physical_heading", "physical_yaw", "thetap", "true_heading",
        "reference_heading",
    ],
    "est_x": [
        "estimate_east_m", "est_x", "x_est", "pred_x", "x_pred", "twin_x", "xt",
        "estimate_x", "estimated_x", "pred_east", "est_east",
    ],
    "est_y": [
        "estimate_north_m", "est_y", "y_est", "pred_y", "y_pred", "twin_y", "yt",
        "estimate_y", "estimated_y", "pred_north", "est_north",
    ],
    "est_theta": [
        "estimate_heading_rad", "est_heading", "heading_est", "pred_heading", "heading_pred",
        "est_yaw", "yaw_est", "pred_yaw", "twin_heading", "twin_yaw",
        "thetat", "estimate_heading", "estimated_heading",
    ],
}


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--input-root", default="results")
    p.add_argument(
        "--glob",
        default="**/v2_evaluated_trajectory.csv",
        help="CSV glob under --input-root.",
    )
    p.add_argument(
        "--output",
        default="results/i2nav_timing_sensitivity",
    )
    p.add_argument(
        "--delay-ms",
        default="0,25,50,100,200",
        help="Positive fixed update-delivery delays.",
    )
    p.add_argument(
        "--clock-offset-ms",
        default="-100,-50,-25,0,25,50,100",
        help="Signed timestamp offsets.",
    )
    p.add_argument(
        "--jitter-ms",
        default="0,10,25,50",
        help="Timestamp jitter standard deviations.",
    )
    p.add_argument("--jitter-seeds", default="0,1,2,3,4")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument(
        "--heading-unit",
        choices=["auto", "rad", "deg"],
        default="auto",
    )
    p.add_argument(
        "--include-path-regex",
        default=r".*",
        help="Path regex used during auto-discovery.",
    )
    p.add_argument(
        "--exclude-path-regex",
        default=r"(?i)official|aifarms|terrasentia|ugv01|conditioned|envelope|pilot|summary|aggregate",
    )
    p.add_argument(
        "--min-rows",
        type=int,
        default=50,
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=100,
    )
    p.add_argument(
        "--baseline-check",
        action="store_true",
        help=(
            "Compare the discovered 0-ms macro means against the paper's frozen V2 "
            "headline metrics and stop if they are far away. Use only when the input "
            "set is the complete 10-sequence x 3-seed operational V2 trajectory archive."
        ),
    )
    p.add_argument(
        "--baseline-rel-tol",
        type=float,
        default=0.08,
        help="Relative tolerance for --baseline-check.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def first_existing_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    by_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in by_lower:
            return by_lower[a.lower()]
    return None


def resolve_columns(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        col = first_existing_column(df, aliases)
        if col is None:
            return None
        mapping[canonical] = col
    return mapping


def normalize_time_seconds(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return x
    xf = x[finite]
    diffs = np.diff(xf)
    diffs = diffs[diffs > 0]
    if not len(diffs):
        return x
    md = float(np.median(diffs))

    # Infer scale from spacing, not absolute epoch magnitude.
    if md > 1e6:      # nanoseconds
        x = x / 1e9
    elif md > 1e3:    # microseconds
        x = x / 1e6
    elif md > 10:     # milliseconds for normal robotics sample rates
        x = x / 1e3

    # Work in elapsed seconds for numerical stability.
    finite = np.isfinite(x)
    if finite.any():
        x = x - x[finite][0]
    return x


def heading_to_rad(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if mode == "rad":
        return x
    if mode == "deg":
        return np.deg2rad(x)
    finite = np.abs(x[np.isfinite(x)])
    if len(finite) == 0:
        return x
    # Conservative heuristic: very large magnitudes are probably degrees.
    # Typical unwrapped indoor heading in radians remains well below 20.
    q99 = float(np.quantile(finite, 0.99))
    return np.deg2rad(x) if q99 > 20.0 else x


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def interp_angle(t_new: np.ndarray, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    s = np.interp(t_new, t, np.sin(theta), left=np.nan, right=np.nan)
    c = np.interp(t_new, t, np.cos(theta), left=np.nan, right=np.nan)
    return np.arctan2(s, c)


def stable_unique_time(t: np.ndarray, *values: np.ndarray) -> Tuple[np.ndarray, ...]:
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    vals = [v[order] for v in values]

    # Keep the last occurrence of duplicate timestamps.
    rev_unique_idx = np.unique(ts[::-1], return_index=True)[1]
    keep = (len(ts) - 1 - rev_unique_idx)
    keep = np.sort(keep)
    return (ts[keep],) + tuple(v[keep] for v in vals)


def causal_hold(
    query_t: np.ndarray,
    arrival_t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrival_t, x, y, theta = stable_unique_time(arrival_t, x, y, theta)
    idx = np.searchsorted(arrival_t, query_t, side="right") - 1
    valid = idx >= 0
    xo = np.full_like(query_t, np.nan, dtype=float)
    yo = np.full_like(query_t, np.nan, dtype=float)
    to = np.full_like(query_t, np.nan, dtype=float)
    xo[valid] = x[idx[valid]]
    yo[valid] = y[idx[valid]]
    to[valid] = theta[idx[valid]]
    return xo, yo, to


def interpolated_shift(
    query_t: np.ndarray,
    stamped_t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stamped_t, x, y, theta = stable_unique_time(stamped_t, x, y, theta)
    xo = np.interp(query_t, stamped_t, x, left=np.nan, right=np.nan)
    yo = np.interp(query_t, stamped_t, y, left=np.nan, right=np.nan)
    to = interp_angle(query_t, stamped_t, theta)
    outside = (query_t < stamped_t[0]) | (query_t > stamped_t[-1])
    xo[outside] = np.nan
    yo[outside] = np.nan
    to[outside] = np.nan
    return xo, yo, to


def relative_translation(
    x0: np.ndarray, y0: np.ndarray, th0: np.ndarray,
    x1: np.ndarray, y1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    dx = x1 - x0
    dy = y1 - y0
    c = np.cos(th0)
    s = np.sin(th0)
    return c * dx + s * dy, -s * dx + c * dy


def rpe_translation(
    t: np.ndarray,
    gx: np.ndarray, gy: np.ndarray, gth: np.ndarray,
    ex: np.ndarray, ey: np.ndarray, eth: np.ndarray,
    horizon_s: float,
) -> float:
    if len(t) < 3:
        return float("nan")

    # For each start time, find nearest future sample at t+h.
    targets = t + horizon_s
    j = np.searchsorted(t, targets, side="left")
    valid = j < len(t)
    i = np.arange(len(t))
    i = i[valid]
    j = j[valid]
    if len(i) == 0:
        return float("nan")

    # Require the selected endpoint to be reasonably close to the target horizon.
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    tol = max(0.15, 2.5 * float(np.median(dt))) if len(dt) else 0.25
    close = np.abs(t[j] - (t[i] + horizon_s)) <= tol
    i = i[close]
    j = j[close]
    if len(i) == 0:
        return float("nan")

    finite = (
        np.isfinite(gx[i]) & np.isfinite(gy[i]) & np.isfinite(gth[i]) &
        np.isfinite(ex[i]) & np.isfinite(ey[i]) & np.isfinite(eth[i]) &
        np.isfinite(gx[j]) & np.isfinite(gy[j]) & np.isfinite(gth[j]) &
        np.isfinite(ex[j]) & np.isfinite(ey[j]) & np.isfinite(eth[j])
    )
    i = i[finite]
    j = j[finite]
    if len(i) == 0:
        return float("nan")

    gdx, gdy = relative_translation(gx[i], gy[i], gth[i], gx[j], gy[j])
    edx, edy = relative_translation(ex[i], ey[i], eth[i], ex[j], ey[j])

    err2 = (edx - gdx) ** 2 + (edy - gdy) ** 2
    return float(np.sqrt(np.mean(err2)))


def metrics(
    t: np.ndarray,
    gx: np.ndarray, gy: np.ndarray, gth: np.ndarray,
    ex: np.ndarray, ey: np.ndarray, eth: np.ndarray,
) -> Dict[str, float]:
    finite = (
        np.isfinite(t) &
        np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gth) &
        np.isfinite(ex) & np.isfinite(ey) & np.isfinite(eth)
    )

    valid_fraction = float(finite.mean()) if len(finite) else 0.0
    if finite.sum() < 3:
        return {m: float("nan") for m in METRICS} | {
            "valid_fraction": valid_fraction,
            "n_valid": int(finite.sum()),
        }

    tt = t[finite]
    gxx, gyy, gtt = gx[finite], gy[finite], gth[finite]
    exx, eyy, ett = ex[finite], ey[finite], eth[finite]

    dp = np.hypot(gxx - exx, gyy - eyy)
    dth = np.abs(wrap(gtt - ett))

    out = {
        "ate_m": float(np.sqrt(np.mean(dp ** 2))),
        "heading_mae_deg": float(np.rad2deg(np.mean(dth))),
        "rpe1_m": rpe_translation(tt, gxx, gyy, gtt, exx, eyy, ett, 1.0),
        "rpe5_m": rpe_translation(tt, gxx, gyy, gtt, exx, eyy, ett, 5.0),
        "rpe10_m": rpe_translation(tt, gxx, gyy, gtt, exx, eyy, ett, 10.0),
        "dp_p95_m": float(np.quantile(dp, 0.95)),
        "dtheta_p95_deg": float(np.rad2deg(np.quantile(dth, 0.95))),
        "valid_fraction": valid_fraction,
        "n_valid": int(finite.sum()),
    }
    return out


def infer_sequence_id(path: Path) -> str:
    s = str(path).lower()
    patterns = [
        r"(building[_-]?\d+)",
        r"(parking[_-]?\d+)",
        r"(street[_-]?\d+)",
        r"(seq(?:uence)?[_-]?\d+)",
        r"(fold[_-]?\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1).replace("-", "")
    # Fallback to nearest useful parent.
    return path.parent.name or path.stem


def infer_seed(path: Path) -> str:
    s = str(path).lower()
    for pat in (r"seed[_-]?(\d+)", r"base[_-]?(\d+)", r"rep(?:licate)?[_-]?(\d+)"):
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return "unknown"


def candidate_files(args: argparse.Namespace, root: Path) -> List[Path]:
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = root / input_root
    include_re = re.compile(args.include_path_regex)
    exclude_re = re.compile(args.exclude_path_regex)

    candidates = []
    for p in input_root.glob(args.glob):
        if not p.is_file():
            continue
        ps = str(p)
        if exclude_re.search(ps):
            continue
        if not include_re.search(ps):
            continue
        try:
            header = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        if resolve_columns(header) is None:
            continue
        candidates.append(p)

    return sorted(candidates, key=lambda p: str(p).lower())[: args.max_files]


def load_trajectory(path: Path, heading_unit: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) < 3:
        return None
    cols = resolve_columns(df)
    if cols is None:
        return None

    arr = {}
    for k, c in cols.items():
        arr[k] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)

    arr["time"] = normalize_time_seconds(arr["time"])
    arr["gt_theta"] = heading_to_rad(arr["gt_theta"], heading_unit)
    arr["est_theta"] = heading_to_rad(arr["est_theta"], heading_unit)

    finite_t = np.isfinite(arr["time"])
    if finite_t.sum() < 3:
        return None

    # Sort once by nominal time and keep all pose arrays aligned.
    order = np.argsort(arr["time"], kind="mergesort")
    for k in arr:
        arr[k] = arr[k][order]
    return arr


def fixed_delay_variant(a: Dict[str, np.ndarray], delay_s: float):
    t = a["time"]
    arrival = t + delay_s
    ex, ey, eth = causal_hold(t, arrival, a["est_x"], a["est_y"], a["est_theta"])
    return ex, ey, eth


def clock_offset_variant(a: Dict[str, np.ndarray], offset_s: float):
    t = a["time"]
    stamped = t + offset_s
    return interpolated_shift(t, stamped, a["est_x"], a["est_y"], a["est_theta"])


def jitter_variant(a: Dict[str, np.ndarray], sigma_s: float, seed: int):
    t = a["time"]
    if sigma_s == 0:
        stamped = t.copy()
    else:
        rng = np.random.default_rng(seed)
        stamped = t + rng.normal(0.0, sigma_s, size=len(t))
    return interpolated_shift(t, stamped, a["est_x"], a["est_y"], a["est_theta"])


def add_result(
    rows: List[Dict],
    path: Path,
    root: Path,
    seq: str,
    seed: str,
    kind: str,
    value_ms: float,
    jitter_seed: Optional[int],
    a: Dict[str, np.ndarray],
    ex: np.ndarray,
    ey: np.ndarray,
    eth: np.ndarray,
):
    m = metrics(
        a["time"], a["gt_x"], a["gt_y"], a["gt_theta"],
        ex, ey, eth,
    )
    rows.append(
        {
            "file": str(path.relative_to(root) if path.is_relative_to(root) else path),
            "sequence": seq,
            "seed": seed,
            "perturbation": kind,
            "value_ms": float(value_ms),
            "jitter_seed": jitter_seed,
            **m,
        }
    )


def sequence_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    # First average stochastic jitter repeats within the original run/checkpoint,
    # then average checkpoint/seed runs within each physical sequence.
    run_group = ["file", "sequence", "seed", "perturbation", "value_ms"]
    run_mean = (
        all_runs.groupby(run_group, dropna=False)[METRICS + ["valid_fraction"]]
        .mean()
        .reset_index()
    )
    seq = (
        run_mean.groupby(["sequence", "perturbation", "value_ms"], dropna=False)[METRICS + ["valid_fraction"]]
        .mean()
        .reset_index()
    )
    return seq


def dataset_summary(seq: pd.DataFrame) -> pd.DataFrame:
    return (
        seq.groupby(["perturbation", "value_ms"], dropna=False)[METRICS + ["valid_fraction"]]
        .mean()
        .reset_index()
    )


def bootstrap_ci(values: np.ndarray, nboot: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(nboot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_stats(seq: pd.DataFrame, nboot: int, seed: int) -> pd.DataFrame:
    rows = []
    for perturbation in seq["perturbation"].unique():
        base = seq[(seq["perturbation"] == perturbation) & (seq["value_ms"] == 0)]
        if base.empty:
            continue
        for value in sorted(seq.loc[seq["perturbation"] == perturbation, "value_ms"].unique()):
            if value == 0:
                continue
            cur = seq[(seq["perturbation"] == perturbation) & (seq["value_ms"] == value)]
            merged = base.merge(cur, on="sequence", suffixes=("_base", "_cur"))
            for metric in METRICS:
                d = merged[f"{metric}_cur"].to_numpy(float) - merged[f"{metric}_base"].to_numpy(float)
                b = merged[f"{metric}_base"].to_numpy(float)
                valid = np.isfinite(d) & np.isfinite(b)
                d = d[valid]
                b = b[valid]
                lo, hi = bootstrap_ci(d, nboot, seed)
                mean_diff = float(np.mean(d)) if len(d) else float("nan")
                mean_base = float(np.mean(b)) if len(b) else float("nan")
                rel = (
                    100.0 * mean_diff / mean_base
                    if np.isfinite(mean_base) and abs(mean_base) > 1e-12
                    else float("nan")
                )
                rows.append(
                    {
                        "perturbation": perturbation,
                        "value_ms": value,
                        "metric": metric,
                        "n_sequences": int(len(d)),
                        "mean_paired_difference": mean_diff,
                        "paired_ci95_low": lo,
                        "paired_ci95_high": hi,
                        "relative_change_pct_vs_0ms": rel,
                    }
                )
    return pd.DataFrame(rows)


def plot_relative_change(
    paired: pd.DataFrame,
    perturbation: str,
    outpath: Path,
    title: str,
):
    sub = paired[paired["perturbation"] == perturbation].copy()
    if sub.empty:
        return
    wanted = ["ate_m", "heading_mae_deg", "rpe1_m", "rpe5_m", "rpe10_m"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for metric in wanted:
        s = sub[sub["metric"] == metric].sort_values("value_ms")
        if s.empty:
            continue
        # Add explicit zero reference.
        x = np.r_[0.0, s["value_ms"].to_numpy(float)]
        y = np.r_[0.0, s["relative_change_pct_vs_0ms"].to_numpy(float)]
        ax.plot(x, y, marker="o", label=metric)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Timing perturbation (ms)")
    ax.set_ylabel("Mean paired change vs 0 ms (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def paper_baseline_check(ds: pd.DataFrame, rel_tol: float) -> Tuple[bool, List[str]]:
    expected = {
        "ate_m": 2.398,
        "heading_mae_deg": 2.569,
        "rpe1_m": 0.0611,
        "rpe5_m": 0.1603,
        "rpe10_m": 0.2532,
    }
    # Use delay=0 baseline; all perturbation families should agree at zero.
    b = ds[(ds["perturbation"] == "fixed_delay") & (ds["value_ms"] == 0)]
    if len(b) != 1:
        return False, ["Could not identify exactly one fixed-delay 0-ms dataset baseline."]
    row = b.iloc[0]
    messages = []
    ok = True
    for m, e in expected.items():
        got = float(row[m])
        rel = abs(got - e) / max(abs(e), 1e-12)
        messages.append(f"{m}: got {got:.6g}, expected {e:.6g}, rel diff {100*rel:.2f}%")
        if rel > rel_tol:
            ok = False
    return ok, messages


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    outdir = Path(args.output)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    files = candidate_files(args, root)
    if not files:
        print("No compatible frozen i2Nav V2 trajectory CSVs were auto-discovered.")
        print("The script only accepts CSVs containing time + GT x/y/heading + twin x/y/heading.")
        print("If your result directory has a different name, rerun with, for example:")
        print(
            '  python -m DigitalTwin.analysis.i2nav_timing_sensitivity '
            '--input-root results/YOUR_FROZEN_V2_DIR --include-path-regex ".*"'
        )
        return 2

    print(f"Discovered {len(files)} candidate trajectory CSVs:")
    for p in files:
        print("  ", p)

    delays = parse_float_list(args.delay_ms)
    offsets = parse_float_list(args.clock_offset_ms)
    jitters = parse_float_list(args.jitter_ms)
    jitter_seeds = parse_int_list(args.jitter_seeds)

    rows: List[Dict] = []
    accepted_files = []

    for path in files:
        a = load_trajectory(path, args.heading_unit)
        if a is None or len(a["time"]) < args.min_rows:
            if args.verbose:
                print(f"Skipping {path}: unreadable or too few rows")
            continue

        seq = infer_sequence_id(path)
        seed = infer_seed(path)
        accepted_files.append(path)

        for d_ms in delays:
            ex, ey, eth = fixed_delay_variant(a, d_ms / 1000.0)
            add_result(rows, path, root, seq, seed, "fixed_delay", d_ms, None, a, ex, ey, eth)

        for o_ms in offsets:
            ex, ey, eth = clock_offset_variant(a, o_ms / 1000.0)
            add_result(rows, path, root, seq, seed, "clock_offset", o_ms, None, a, ex, ey, eth)

        for j_ms in jitters:
            seeds = [0] if j_ms == 0 else jitter_seeds
            for js in seeds:
                # Combine user seed with a stable per-file term so files do not share
                # identical random jitter patterns.
                file_term = sum(path.as_posix().encode("utf-8")) % 100000
                eff_seed = int(js + 100003 * file_term)
                ex, ey, eth = jitter_variant(a, j_ms / 1000.0, eff_seed)
                add_result(rows, path, root, seq, seed, "timestamp_jitter", j_ms, js, a, ex, ey, eth)

    if not rows:
        print("No trajectory files survived parsing.")
        return 3

    all_runs = pd.DataFrame(rows)
    seq = sequence_summary(all_runs)
    ds = dataset_summary(seq)
    paired = paired_stats(seq, args.bootstrap, args.bootstrap_seed)

    all_runs.to_csv(outdir / "timing_sensitivity_all_runs.csv", index=False)
    seq.to_csv(outdir / "timing_sensitivity_sequence_summary.csv", index=False)
    ds.to_csv(outdir / "timing_sensitivity_dataset_summary.csv", index=False)
    paired.to_csv(outdir / "timing_sensitivity_paired_statistics.csv", index=False)

    plot_relative_change(
        paired, "fixed_delay", outdir / "delay_sensitivity.png",
        "Physical–virtual fidelity sensitivity to fixed update delay",
    )
    plot_relative_change(
        paired, "clock_offset", outdir / "clock_offset_sensitivity.png",
        "Physical–virtual fidelity sensitivity to clock/timestamp offset",
    )
    plot_relative_change(
        paired, "timestamp_jitter", outdir / "jitter_sensitivity.png",
        "Physical–virtual fidelity sensitivity to timestamp jitter",
    )

    baseline_messages = []
    if args.baseline_check:
        ok, baseline_messages = paper_baseline_check(ds, args.baseline_rel_tol)
        print("\nFrozen V2 paper-baseline check:")
        for m in baseline_messages:
            print("  ", m)
        if not ok:
            print(
                "\nSTOP: baseline does not reproduce the frozen V2 headline within tolerance. "
                "Do not use the perturbation results yet. The most likely cause is that "
                "auto-discovery selected the wrong trajectory archive or the local metric "
                "convention differs from the frozen evaluator."
            )
            # Still keep outputs for diagnosis.
            return_code = 4
        else:
            print("Baseline check PASSED.")
            return_code = 0
    else:
        return_code = 0

    report = []
    report.append("# i2Nav Frozen-V2 Timing Sensitivity\n")
    report.append("## Scope and claim boundary\n")
    report.append(
        "This analysis perturbs the timing relationship between the saved physical trajectory "
        "and the already-generated frozen Twin V2 virtual-state stream. It characterizes "
        "**physical–virtual synchronization sensitivity** to update delay, clock/timestamp "
        "offset, and timestamp jitter. It does **not** claim to emulate how delayed raw IMU/"
        "odometry samples would change the V2 neural correction internally."
    )
    report.append(f"\n- Accepted trajectory files: **{len(accepted_files)}**")
    report.append(f"- Distinct inferred physical sequences: **{seq['sequence'].nunique()}**")
    report.append(f"- Fixed delays (ms): `{delays}`")
    report.append(f"- Clock offsets (ms): `{offsets}`")
    report.append(f"- Jitter standard deviations (ms): `{jitters}`")
    report.append(f"- Jitter seeds for nonzero jitter: `{jitter_seeds}`")
    report.append(
        "- Statistical hierarchy: jitter replicates are averaged within an original run; "
        "original runs/checkpoints are averaged within a physical sequence; the physical "
        "sequence is the primary unit for paired confidence intervals."
    )

    report.append("\n## Zero-perturbation dataset macro means\n")
    base = ds[(ds["perturbation"] == "fixed_delay") & (ds["value_ms"] == 0)]
    if len(base):
        r = base.iloc[0]
        for m in METRICS:
            report.append(f"- {m}: **{float(r[m]):.6g}**")

    report.append("\n## Paired sensitivity summary\n")
    for perturbation in ["fixed_delay", "clock_offset", "timestamp_jitter"]:
        sub = paired[paired["perturbation"] == perturbation]
        if sub.empty:
            continue
        report.append(f"\n### {perturbation}\n")
        report.append(
            "| value (ms) | metric | mean paired difference | 95% CI | relative change vs 0 ms | n seq |"
        )
        report.append("|---:|---|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            report.append(
                f"| {r['value_ms']:.0f} | {r['metric']} | "
                f"{r['mean_paired_difference']:.6g} | "
                f"[{r['paired_ci95_low']:.6g}, {r['paired_ci95_high']:.6g}] | "
                f"{r['relative_change_pct_vs_0ms']:.2f}% | {int(r['n_sequences'])} |"
            )

    if baseline_messages:
        report.append("\n## Frozen headline baseline check\n")
        report.extend([f"- {m}" for m in baseline_messages])

    report.append("\n## Interpretation rule\n")
    report.append(
        "A timing level should be described as materially degrading fidelity only when the "
        "sequence-level paired change is meaningful in magnitude and its uncertainty supports "
        "that interpretation. Do not treat timestamp samples or jitter draws as independent "
        "physical experiments."
    )

    (outdir / "timing_sensitivity_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(f"\nWrote timing sensitivity outputs to: {outdir}")
    print(f"Accepted {len(accepted_files)} files across {seq['sequence'].nunique()} inferred sequences.")
    print("Inspect timing_sensitivity_report.md before using any number in the paper.")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
