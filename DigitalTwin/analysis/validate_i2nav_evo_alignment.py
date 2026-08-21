#!/usr/bin/env python3
"""Run i2Nav-style evo SE(3) alignment/APE on planar saved trajectories.

The public i2Nav `evaluate_odometry` evaluator:
  * associates trajectories with max timestamp difference 0.005 s;
  * aligns the estimated trajectory to reference by SE(3), without scale;
  * reports APE translation RMSE.

This adapter applies that exact evo workflow to the available planar CSVs by
embedding (east,north) in z=0 and converting yaw to a yaw-only quaternion.

Caution
-------
If this still does not reproduce a previously frozen *headline aggregation*,
do not alter trajectories to force a match. Compare per-sequence values and
audit the original aggregation/source files.
"""
from __future__ import annotations
import argparse, copy, math
from pathlib import Path
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import load_pose_trajectory, sequence_id, seed_id

try:
    from evo.core import metrics, sync, trajectory
except Exception as exc:
    raise SystemExit(
        "The official i2Nav evaluator depends on evo. Install it in this environment with:\n"
        "  python -m pip install evo\n"
        f"Import error: {exc}"
    )

MAX_TIME_SYNC_DIFF=0.005


def _traj(df: pd.DataFrame, gt: bool):
    if gt:
        x=df["gt_east_m"].to_numpy(float)
        y=df["gt_north_m"].to_numpy(float)
        yaw=df["gt_heading_rad"].to_numpy(float)
    else:
        x=df["estimate_east_m"].to_numpy(float)
        y=df["estimate_north_m"].to_numpy(float)
        yaw=df["estimate_heading_rad"].to_numpy(float)
    xyz=np.column_stack([x,y,np.zeros(len(df))])
    # evo PoseTrajectory3D expects quaternions in wxyz.
    q=np.column_stack([
        np.cos(yaw/2.0),
        np.zeros(len(df)),
        np.zeros(len(df)),
        np.sin(yaw/2.0),
    ])
    return trajectory.PoseTrajectory3D(
        positions_xyz=xyz,
        orientations_quat_wxyz=q,
        timestamps=df["time_s"].to_numpy(float),
    )


def eval_file(path: Path):
    d=load_pose_trajectory(path)
    ref_raw=_traj(d,True)
    est_raw=_traj(d,False)
    ref,est=sync.associate_trajectories(ref_raw,est_raw,max_diff=MAX_TIME_SYNC_DIFF)
    aligned=copy.deepcopy(est)
    aligned.align(ref,correct_scale=False,correct_only_scale=False)
    metric=metrics.APE(metrics.PoseRelation.translation_part)
    metric.process_data((ref,aligned))
    stats=metric.get_all_statistics()
    return float(stats["rmse"]), len(metric.error)


def gather(method, root, pattern):
    files=sorted(Path(root).glob(pattern),key=lambda p:str(p).lower())
    rows=[]
    for f in files:
        rmse,n=eval_file(f)
        rows.append({
            "method":method,
            "sequence":sequence_id(f),
            "seed":seed_id(f),
            "ape_translation_rmse_m":rmse,
            "n_associated_poses":n,
            "path":str(f),
        })
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v2-root",default="results/i2nav_v2_full_loso/i2nav_v2_full_loso")
    p.add_argument("--v2-glob",default="**/v2_evaluated_trajectory.csv")
    p.add_argument("--fixed-root",default="results/i2nav_external_baselines/Fixed_Physics_Recomputed")
    p.add_argument("--fixed-glob",default="**/evaluated_trajectory.csv")
    p.add_argument("--v2-target-m",type=float,default=1.635)
    p.add_argument("--fixed-target-m",type=float,default=3.299)
    p.add_argument("--output-root",default="results/i2nav_fidelity_baselines/validation")
    a=p.parse_args()

    frames=[]
    if Path(a.v2_root).exists():
        frames.append(gather("Twin V2",a.v2_root,a.v2_glob))
    if Path(a.fixed_root).exists():
        frames.append(gather("Fixed Physics (recomputed)",a.fixed_root,a.fixed_glob))
    if not frames:
        raise SystemExit("No trajectory roots found.")

    run=pd.concat(frames,ignore_index=True)
    seq=(run.groupby(["method","sequence"],as_index=False)
         .agg(ape_translation_rmse_m=("ape_translation_rmse_m","mean"),
              n_runs=("seed","size"),
              n_associated_poses=("n_associated_poses","mean")))

    summary=[]
    targets={"Twin V2":a.v2_target_m,"Fixed Physics (recomputed)":a.fixed_target_m}
    for method,g in seq.groupby("method"):
        x=g["ape_translation_rmse_m"].to_numpy(float)
        target=targets.get(method,np.nan)
        macro=float(np.mean(x))
        rms_across=float(np.sqrt(np.mean(x*x)))
        summary.append({
            "method":method,
            "macro_mean_of_sequence_rmse_m":macro,
            "rms_across_sequence_rmse_m":rms_across,
            "target_headline_m":target,
            "macro_relative_difference_pct":(
                100*abs(macro-target)/abs(target) if np.isfinite(target) and target!=0 else np.nan
            ),
        })
    S=pd.DataFrame(summary)

    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    run.to_csv(out/"i2nav_evo_ape_per_run.csv",index=False)
    seq.to_csv(out/"i2nav_evo_ape_per_sequence.csv",index=False)
    S.to_csv(out/"i2nav_evo_ape_summary.csv",index=False)

    print("\nOfficial-style evo SE(3) APE per-sequence")
    print(seq.to_string(index=False))
    print("\nAggregation diagnostics")
    print(S.to_string(index=False))
    print("\nDo not force the macro number to match a frozen headline if per-sequence values agree.")
    print("A mismatch can be caused by a different aggregation table, trajectory source, or preprocessing protocol.")
    print(f"\nWrote files under {out}")


if __name__=="__main__":
    main()
