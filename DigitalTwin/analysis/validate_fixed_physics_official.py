#!/usr/bin/env python3
"""Validate recomputed Fixed Physics against the separate official aligned metric.

The TFP suite reports synchronized/un-aligned operational ATE, while the official
i2Nav benchmark uses standardized post-hoc alignment. A large raw ATE therefore
cannot be compared directly with the frozen official APE.

This script first sanity-checks our 2-D alignment procedure on frozen Twin V2
(default official target 1.635 m). Only if that magnitude is reproduced should
we use the same alignment as evidence about the recomputed Fixed Physics target
(default 3.299 m).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import load_pose_trajectory, sequence_id, seed_id


def umeyama_2d(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    src=np.asarray(src,float); dst=np.asarray(dst,float)
    mu_s=src.mean(axis=0); mu_d=dst.mean(axis=0)
    X=src-mu_s; Y=dst-mu_d
    cov=(Y.T@X)/len(src)
    U,D,Vt=np.linalg.svd(cov)
    S=np.eye(2)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[-1,-1]=-1
    R=U@S@Vt
    if with_scale:
        var=np.mean(np.sum(X*X,axis=1))
        scale=float(np.trace(np.diag(D)@S)/max(var,1e-12))
    else: scale=1.0
    trans=mu_d-scale*(R@mu_s)
    return scale,R,trans


def rmse_xy(est,gt):
    e=np.linalg.norm(est-gt,axis=1)
    return float(np.sqrt(np.mean(e*e)))


def eval_file(path:Path):
    d=load_pose_trajectory(path)
    gt=d[['gt_east_m','gt_north_m']].to_numpy(float)
    est=d[['estimate_east_m','estimate_north_m']].to_numpy(float)
    raw=rmse_xy(est,gt)
    s1,R1,t1=umeyama_2d(est,gt,False); se2=rmse_xy((s1*(R1@est.T)).T+t1,gt)
    s2,R2,t2=umeyama_2d(est,gt,True); sim2=rmse_xy((s2*(R2@est.T)).T+t2,gt)
    return raw,se2,sim2,s2


def eval_method(name,root,pattern):
    files=sorted(Path(root).glob(pattern),key=lambda q:str(q).lower())
    if not files: raise FileNotFoundError(f'No trajectories for {name}: {root}/{pattern}')
    rows=[]
    for f in files:
        raw,se2,sim2,scale=eval_file(f)
        rows.append(dict(method=name,sequence=sequence_id(f),seed=seed_id(f),path=str(f),
                         raw_ate_rmse_m=raw,se2_aligned_ape_rmse_m=se2,
                         sim2_aligned_ape_rmse_m=sim2,sim2_scale=scale))
    run=pd.DataFrame(rows)
    seq=run.groupby(['method','sequence'],as_index=False).agg(
        raw_ate_rmse_m=('raw_ate_rmse_m','mean'),
        se2_aligned_ape_rmse_m=('se2_aligned_ape_rmse_m','mean'),
        sim2_aligned_ape_rmse_m=('sim2_aligned_ape_rmse_m','mean'),
        sim2_scale=('sim2_scale','mean'),n_runs=('seed','size'))
    macro=seq[['raw_ate_rmse_m','se2_aligned_ape_rmse_m','sim2_aligned_ape_rmse_m']].mean().to_dict()
    return run,seq,macro


def rel(x,target): return abs(float(x)-float(target))/max(abs(float(target)),1e-12)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixed-root',default='results/i2nav_external_baselines/Fixed_Physics_Recomputed')
    p.add_argument('--fixed-glob',default='**/evaluated_trajectory.csv')
    p.add_argument('--v2-root',default='results/i2nav_v2_full_loso/i2nav_v2_full_loso')
    p.add_argument('--v2-glob',default='**/v2_evaluated_trajectory.csv')
    p.add_argument('--fixed-target-m',type=float,default=3.299)
    p.add_argument('--v2-target-m',type=float,default=1.635)
    p.add_argument('--compat-relative-tol',type=float,default=0.15)
    p.add_argument('--output-root',default='results/i2nav_fidelity_baselines/validation')
    a=p.parse_args()

    frun,fseq,fmacro=eval_method('Fixed Physics (recomputed)',a.fixed_root,a.fixed_glob)
    vrun,vseq,vmacro=eval_method('Twin V2',a.v2_root,a.v2_glob)
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    pd.concat([frun,vrun],ignore_index=True).to_csv(out/'official_alignment_validation_per_run.csv',index=False)
    seq=pd.concat([fseq,vseq],ignore_index=True); seq.to_csv(out/'official_alignment_validation_per_sequence.csv',index=False)

    summary=[]
    for method,macro,target in [('Twin V2',vmacro,a.v2_target_m),('Fixed Physics (recomputed)',fmacro,a.fixed_target_m)]:
        for kind in ['raw_ate_rmse_m','se2_aligned_ape_rmse_m','sim2_aligned_ape_rmse_m']:
            summary.append(dict(method=method,metric=kind,macro_mean_m=macro[kind],official_target_m=target,
                                relative_difference_pct=100*rel(macro[kind],target)))
    S=pd.DataFrame(summary); S.to_csv(out/'official_alignment_validation_summary.csv',index=False)
    print(S.to_string(index=False))

    v_se2=rel(vmacro['se2_aligned_ape_rmse_m'],a.v2_target_m)
    v_sim2=rel(vmacro['sim2_aligned_ape_rmse_m'],a.v2_target_m)
    alignment='SE2' if v_se2<=v_sim2 else 'Sim2'
    v_best=min(v_se2,v_sim2)
    f_value=fmacro['se2_aligned_ape_rmse_m'] if alignment=='SE2' else fmacro['sim2_aligned_ape_rmse_m']
    f_rel=rel(f_value,a.fixed_target_m)
    print('\nProtocol interpretation')
    print(f'V2 chooses {alignment} as closer to official target; V2 relative difference={100*v_best:.2f}%')
    if v_best>a.compat_relative_tol:
        print('ALIGNMENT_PROTOCOL_NOT_VALIDATED: our simple 2-D alignment does not reproduce frozen V2 closely enough.')
        print('Do not use this script to judge Fixed Physics equivalence yet; the official scorer may use a different SE(3)/alignment protocol.')
    else:
        print('ALIGNMENT_MAGNITUDE_VALIDATED_ON_V2')
        print(f'Using {alignment}, recomputed Fixed Physics relative difference from official target={100*f_rel:.2f}%')
        if f_rel<=a.compat_relative_tol:
            print('FIXED_RECOMPUTATION_COMPATIBLE_WITH_OFFICIAL_MAGNITUDE')
            print('The large raw/synchronized ATE is therefore not evidence of a bug; it measures operational drift that post-hoc alignment removes.')
        else:
            print('FIXED_RECOMPUTATION_NEEDS_PROPAGATION_AUDIT')
            print('Alignment is plausible on V2 but does not recover the Fixed-Physics target, so inspect the original physics propagation/configuration.')
    print(f'Wrote validation files under {out}')

if __name__=='__main__': main()
