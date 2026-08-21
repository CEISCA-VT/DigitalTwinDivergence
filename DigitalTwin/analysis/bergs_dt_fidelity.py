#!/usr/bin/env python3
"""Bergs-style mobile-robot DT trajectory validation baseline.

Computes trajectory/path measures that are straightforward to reproduce on the
same physical/virtual traces: symmetric Hausdorff distance, bidirectional
nearest-path distance, terminal position/heading error, path-length mismatch,
and ATE for context.

Hausdorff is geometry-based and does not preserve timestamp correspondence;
this is precisely why it is useful as a complementary evaluator to synchronized
TFP metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import load_pose_trajectory, wrap_angle
from .fidelity_common import load_manifest, percentile_bootstrap_mean

METRICS = ["hausdorff_m", "mean_bidirectional_nearest_m", "terminal_position_error_m", "terminal_heading_error_deg", "path_length_abs_error_m", "path_length_rel_error_pct", "ate_m"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="results/i2nav_fidelity_baselines/trajectory_manifest.csv")
    p.add_argument("--output", default="results/i2nav_fidelity_baselines/bergs")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--max-points", type=int, default=5000, help="Uniform arc-length samples per path for Hausdorff; 0 uses all")
    p.add_argument("--bootstrap", type=int, default=5000)
    return p.parse_args()


def path_length(P: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0


def arc_resample(P: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(P) <= max_points:
        return P
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    if s[-1] <= 1e-12:
        idx = np.linspace(0, len(P)-1, max_points).round().astype(int)
        return P[idx]
    q = np.linspace(0.0, s[-1], max_points)
    return np.column_stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])])


def nearest_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
        return cKDTree(B).query(A, k=1, workers=-1)[0]
    except Exception:
        # Memory-bounded NumPy fallback.
        out = np.empty(len(A), float)
        batch = max(32, min(512, int(2_000_000 / max(len(B), 1))))
        for s in range(0, len(A), batch):
            X = A[s:s+batch]
            d2 = np.sum((X[:, None, :] - B[None, :, :]) ** 2, axis=2)
            out[s:s+len(X)] = np.sqrt(np.min(d2, axis=1))
        return out


def evaluate(path: str | Path, max_points: int) -> dict:
    d = load_pose_trajectory(path)
    G = d[["gt_east_m", "gt_north_m"]].to_numpy(float)
    E = d[["estimate_east_m", "estimate_north_m"]].to_numpy(float)
    Gs = arc_resample(G, max_points); Es = arc_resample(E, max_points)
    gde = nearest_distances(Gs, Es); edg = nearest_distances(Es, Gs)
    haus = max(float(np.max(gde)), float(np.max(edg)))
    mean_bi = 0.5 * (float(np.mean(gde)) + float(np.mean(edg)))
    terminal_pos = float(np.linalg.norm(G[-1] - E[-1]))
    terminal_heading = float(np.degrees(abs(float(wrap_angle(d["gt_heading_rad"].iloc[-1] - d["estimate_heading_rad"].iloc[-1])))))
    lg = path_length(G); le = path_length(E)
    dp = np.linalg.norm(G - E, axis=1)
    return {
        "n_original_points": len(G), "n_hausdorff_gt": len(Gs), "n_hausdorff_est": len(Es),
        "hausdorff_m": haus,
        "mean_bidirectional_nearest_m": mean_bi,
        "terminal_position_error_m": terminal_pos,
        "terminal_heading_error_deg": terminal_heading,
        "gt_path_length_m": lg,
        "estimate_path_length_m": le,
        "path_length_abs_error_m": abs(le-lg),
        "path_length_rel_error_pct": 100.0 * abs(le-lg) / lg if lg > 1e-9 else np.nan,
        "ate_m": float(np.sqrt(np.mean(dp ** 2))),
    }


def main():
    a = parse_args(); repo = Path(a.repo_root).resolve(); out = Path(a.output); out = out if out.is_absolute() else repo/out; out.mkdir(parents=True, exist_ok=True)
    m = load_manifest(a.manifest if Path(a.manifest).is_absolute() else repo/a.manifest, repo)
    rows=[]
    for k,r in m.iterrows():
        print(f"[{k+1:03d}/{len(m):03d}] Bergs-style {r.method} {r.sequence} {r.seed}")
        rows.append({"method":r.method,"sequence":r.sequence,"seed":r.seed,"trajectory":r.trajectory,**evaluate(r.trajectory_abs,a.max_points)})
    run=pd.DataFrame(rows); run.to_csv(out/"bergs_per_run.csv",index=False)
    seq=run.groupby(["method","sequence"])[METRICS].mean().reset_index(); seq.to_csv(out/"bergs_per_sequence.csv",index=False)
    sr=[]
    for method,g in seq.groupby("method"):
        row={"method":method,"n_sequences":g.sequence.nunique()}
        for metric in METRICS:
            c,lo,hi=percentile_bootstrap_mean(g[metric].to_numpy(float),a.bootstrap,2026)
            row[metric]=c; row[f"{metric}_ci_low"]=lo; row[f"{metric}_ci_high"]=hi
        sr.append(row)
    summary=pd.DataFrame(sr); summary.to_csv(out/"bergs_dataset_summary.csv",index=False)
    (out/"bergs_report.md").write_text(
        "# Bergs-style DT trajectory evaluation\n\n"
        "Hausdorff and nearest-path distances compare path geometry without preserving timestamp correspondence. "
        "The reported max-points setting is an arc-length resampling cap used to bound computation.\n\n"
        f"- max_points: **{a.max_points}**\n\n```\n{summary.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")

if __name__=="__main__": main()
