#!/usr/bin/env python3
"""Diagnose the planar EKF-IW inputs and corrected outputs fold-by-fold.

This script is meant to prevent publishing a degenerate classical baseline.
It checks whether wheel/IMU yaw channels have the expected sign/scale on each
training fold and summarizes held-out EKF heading/position behavior.
"""
from __future__ import annotations
import argparse, math
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import discover_i2nav_corpus, wrap_angle
from DigitalTwin.baselines.ekf_iw import fit_config


def _rmse(a):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a*a))) if len(a) else float('nan')


def _corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.std(a[ok])<1e-12 or np.std(b[ok])<1e-12: return float('nan')
    return float(np.corrcoef(a[ok],b[ok])[0,1])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-root',default='results/i2nav_v2_full_loso/i2nav_v2_full_loso')
    p.add_argument('--glob',default='**/v2_evaluated_trajectory.csv')
    p.add_argument('--ekf-root',default='results/i2nav_external_baselines/EKF_IW')
    p.add_argument('--output-root',default='results/i2nav_fidelity_baselines/validation')
    a=p.parse_args()

    corpus=discover_i2nav_corpus(a.input_root,a.glob)
    rows=[]
    for test in sorted(corpus):
        train=[corpus[s].data for s in sorted(corpus) if s!=test]
        cfg=fit_config(train)
        d=corpus[test].data
        gt=d['gt_yaw_rate_radps'].to_numpy(float)
        imu=d['imu_yaw_rate_radps'].to_numpy(float)
        imu_c=cfg.imu_yaw_scale*imu+cfg.imu_yaw_bias_radps
        wheel=d['wheel_yaw_radps'].to_numpy(float) if 'wheel_yaw_radps' in d else np.full(len(d),np.nan)
        wheel_c=cfg.wheel_yaw_scale*wheel+cfg.wheel_yaw_bias_radps
        row={
            'sequence':test,
            'imu_raw_corr_gt':_corr(imu,gt),
            'imu_cal_corr_gt':_corr(imu_c,gt),
            'imu_raw_rmse_radps':_rmse(imu-gt),
            'imu_cal_rmse_radps':_rmse(imu_c-gt),
            'wheel_raw_corr_gt':_corr(wheel,gt),
            'wheel_cal_corr_gt':_corr(wheel_c,gt),
            'wheel_raw_rmse_radps':_rmse(wheel-gt),
            'wheel_cal_rmse_radps':_rmse(wheel_c-gt),
            **asdict(cfg),
        }
        f=Path(a.ekf_root)/test/'evaluated_trajectory.csv'
        if f.exists():
            e=pd.read_csv(f)
            dx=e['estimate_east_m'].to_numpy(float)-e['gt_east_m'].to_numpy(float)
            dy=e['estimate_north_m'].to_numpy(float)-e['gt_north_m'].to_numpy(float)
            dh=np.rad2deg(wrap_angle(e['estimate_heading_rad'].to_numpy(float)-e['gt_heading_rad'].to_numpy(float)))
            row['ekf_ate_rmse_m']=_rmse(np.sqrt(dx*dx+dy*dy))
            row['ekf_heading_mae_deg']=float(np.mean(np.abs(dh)))
            row['ekf_heading_p95_deg']=float(np.quantile(np.abs(dh),.95))
            if 'wheel_yaw_update_acceptance_fraction' in e:
                row['heldout_update_acceptance_fraction']=float(e['wheel_yaw_update_acceptance_fraction'].iloc[0])
        rows.append(row)

    df=pd.DataFrame(rows)
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    path=out/'ekf_iw_diagnostics.csv'; df.to_csv(path,index=False)
    show=[c for c in ['sequence','imu_raw_corr_gt','imu_cal_corr_gt','imu_raw_rmse_radps','imu_cal_rmse_radps',
                      'wheel_raw_corr_gt','wheel_cal_corr_gt','wheel_raw_rmse_radps','wheel_cal_rmse_radps',
                      'use_wheel_yaw_updates','wheel_yaw_training_corr','ekf_ate_rmse_m','ekf_heading_mae_deg',
                      'heldout_update_acceptance_fraction'] if c in df]
    print(df[show].to_string(index=False))
    print('\nKey check: if corrected EKF heading remains near 90 deg MAE, do NOT publish it as a fair baseline.')
    print('The removed v≈0=>omega=0 pseudo-update was invalid for in-place/low-speed turning.')
    print(f'Wrote {path}')

if __name__=='__main__': main()
