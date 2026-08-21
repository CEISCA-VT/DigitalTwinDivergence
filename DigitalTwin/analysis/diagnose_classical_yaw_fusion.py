#!/usr/bin/env python3
"""Diagnose classical wheel/IMU yaw-fusion choices under strict LOSO.

Purpose
-------
The earlier planar EKF-IW remained degenerate even after removing an invalid
zero-speed=>zero-yaw pseudo-measurement. This script determines *why* before
we publish or tune another baseline.

For each held-out i2Nav sequence it:
  1) fits speed/yaw affine calibration on the other 9 sequences only;
  2) evaluates several classical yaw strategies on the held-out sequence;
  3) separately chooses a strategy using TRAINING-SEQUENCE macro ATE only;
  4) reports the held-out result of that preselected strategy.

Ground truth is used on the held-out sequence ONLY for final evaluation.
No held-out GT is used to select/calibrate the baseline.

Candidates
----------
- raw_imu                  : raw odometer speed + raw IMU yaw
- calibrated_imu           : train-calibrated speed + train-calibrated IMU yaw
- calibrated_wheel         : train-calibrated speed + train-calibrated wheel yaw
- inverse_variance_fusion  : static train-noise-weighted IMU/wheel yaw fusion
- robust_gated_fusion      : same fusion but rejects large inter-sensor disagreement
- current_dynamic_bias_ekf : current repaired planar EKF-IW implementation

The script is a DIAGNOSTIC. Do not cherry-pick a held-out winner for publication.
Use `selected_by_train_ate` if you need a predeclared classical-fusion baseline.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import (
    discover_i2nav_corpus, integrate_planar, wrap_angle
)
from DigitalTwin.baselines.ekf_iw import fit_config, run_ekf_iw

STATIC_CANDIDATES = [
    "raw_imu",
    "calibrated_imu",
    "calibrated_wheel",
    "inverse_variance_fusion",
    "robust_gated_fusion",
]


def _metrics(d: pd.DataFrame, x, y, h):
    gx=d["gt_east_m"].to_numpy(float)
    gy=d["gt_north_m"].to_numpy(float)
    gh=d["gt_heading_rad"].to_numpy(float)
    pe=np.hypot(np.asarray(x)-gx, np.asarray(y)-gy)
    he=np.abs(np.rad2deg(wrap_angle(np.asarray(h)-gh)))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(pe*pe))),
        "position_p95_m": float(np.quantile(pe, .95)),
        "heading_mae_deg": float(np.mean(he)),
        "heading_p95_deg": float(np.quantile(he, .95)),
        "terminal_position_error_m": float(pe[-1]),
        "terminal_heading_error_deg": float(he[-1]),
    }


def _candidate_rates(d: pd.DataFrame, cfg, name: str):
    raw_v=d["odo_speed_mps"].to_numpy(float)
    imu_raw=d["imu_yaw_rate_radps"].to_numpy(float)
    vcal=cfg.speed_scale*raw_v + cfg.speed_bias_mps
    imu=cfg.imu_yaw_scale*imu_raw + cfg.imu_yaw_bias_radps

    if "wheel_yaw_radps" in d.columns:
        wheel_raw=d["wheel_yaw_radps"].to_numpy(float)
        wheel=cfg.wheel_yaw_scale*wheel_raw + cfg.wheel_yaw_bias_radps
    else:
        wheel=np.full(len(d), np.nan)

    if name=="raw_imu":
        return raw_v, imu_raw

    if name=="calibrated_imu":
        return vcal, imu

    if name=="calibrated_wheel":
        w=np.where(np.isfinite(wheel), wheel, imu)
        return vcal, w

    si=max(float(cfg.imu_yaw_sigma_radps), 1e-6)
    sw=max(float(cfg.wheel_yaw_sigma_radps), 1e-6)
    wi=1.0/(si*si)
    ww=1.0/(sw*sw)
    both=np.isfinite(wheel)
    fused=imu.copy()
    fused[both]=(wi*imu[both] + ww*wheel[both])/(wi+ww)

    if name=="inverse_variance_fusion":
        return vcal, fused

    if name=="robust_gated_fusion":
        # Gate is fixed from TRAINING residual scales, not tuned on held-out GT.
        gate=3.0*math.sqrt(si*si+sw*sw)
        ok=both & (np.abs(imu-wheel) <= gate)
        out=imu.copy()
        out[ok]=(wi*imu[ok] + ww*wheel[ok])/(wi+ww)
        # When sensors strongly disagree, trust the sensor with lower TRAINING sigma.
        bad=both & ~ok
        if sw < si:
            out[bad]=wheel[bad]
        return vcal, out

    raise KeyError(name)


def _integrate_candidate(d, cfg, name):
    v,w=_candidate_rates(d,cfg,name)
    init=(float(d["gt_east_m"].iloc[0]),
          float(d["gt_north_m"].iloc[0]),
          float(d["gt_heading_rad"].iloc[0]))
    x,y,h=integrate_planar(d["time_s"].to_numpy(float),v,w,init)
    return x,y,h,v,w


def _training_macro_ate(train_frames, cfg, name):
    vals=[]
    for d in train_frames:
        x,y,h,_,_=_integrate_candidate(d,cfg,name)
        vals.append(_metrics(d,x,y,h)["ate_rmse_m"])
    return float(np.mean(vals))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root", default="results/i2nav_v2_full_loso/i2nav_v2_full_loso")
    p.add_argument("--glob", default="**/v2_evaluated_trajectory.csv")
    p.add_argument("--output-root", default="results/i2nav_fidelity_baselines/validation")
    p.add_argument("--smoke", action="store_true")
    a=p.parse_args()

    corpus=discover_i2nav_corpus(a.input_root,a.glob)
    tests=sorted(corpus)
    if a.smoke:
        tests=[s for s in ("parking00","parking02") if s in corpus]

    rows=[]
    selected_rows=[]
    selections=[]

    for i,test in enumerate(tests,1):
        train_names=[s for s in sorted(corpus) if s!=test]
        train=[corpus[s].data for s in train_names]
        d=corpus[test].data
        cfg=fit_config(train)

        print("="*84)
        print(f"{i}/{len(tests)} held-out={test}")

        train_scores={}
        for name in STATIC_CANDIDATES:
            train_scores[name]=_training_macro_ate(train,cfg,name)

        selected=min(train_scores, key=train_scores.get)
        selections.append(selected)
        print("  train-selected:", selected,
              f"(macro train ATE={train_scores[selected]:.3f} m)")

        for name in STATIC_CANDIDATES:
            x,y,h,v,w=_integrate_candidate(d,cfg,name)
            m=_metrics(d,x,y,h)
            gt_rate=d["gt_yaw_rate_radps"].to_numpy(float)
            rate_rmse=float(np.sqrt(np.mean((w-gt_rate)**2)))
            rows.append({
                "sequence":test,
                "candidate":name,
                "selected_by_train_ate": bool(name==selected),
                "training_macro_ate_m_for_selection":train_scores[name],
                "heldout_yaw_rate_rmse_radps":rate_rmse,
                **m,
            })
            if name==selected:
                selected_rows.append({
                    "sequence":test,
                    "candidate":"selected_by_train_ate",
                    "selected_strategy":selected,
                    **m,
                })

        # Current dynamic-bias EKF is included as a diagnostic, never as a selector candidate.
        e=run_ekf_iw(d,cfg)
        m=_metrics(
            e,
            e["estimate_east_m"].to_numpy(float),
            e["estimate_north_m"].to_numpy(float),
            e["estimate_heading_rad"].to_numpy(float)
        )
        rows.append({
            "sequence":test,
            "candidate":"current_dynamic_bias_ekf",
            "selected_by_train_ate":False,
            "training_macro_ate_m_for_selection":np.nan,
            "heldout_yaw_rate_rmse_radps":float(np.sqrt(np.mean(
                (e["corrected_omega_radps"].to_numpy(float)-d["gt_yaw_rate_radps"].to_numpy(float))**2
            ))),
            **m,
        })

    per=pd.DataFrame(rows)
    sel=pd.DataFrame(selected_rows)
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    per_path=out/"classical_yaw_fusion_per_sequence.csv"
    sel_path=out/"classical_yaw_fusion_selected_by_training.csv"
    sum_path=out/"classical_yaw_fusion_summary.csv"
    per.to_csv(per_path,index=False)
    sel.to_csv(sel_path,index=False)

    summary=(per.groupby("candidate",as_index=False)
             .agg(
                 n_sequences=("sequence","nunique"),
                 ate_rmse_m=("ate_rmse_m","mean"),
                 heading_mae_deg=("heading_mae_deg","mean"),
                 position_p95_m=("position_p95_m","mean"),
                 heading_p95_deg=("heading_p95_deg","mean"),
                 yaw_rate_rmse_radps=("heldout_yaw_rate_rmse_radps","mean"),
             )
             .sort_values("ate_rmse_m"))
    if len(sel):
        extra=pd.DataFrame([{
            "candidate":"selected_by_train_ate",
            "n_sequences":sel["sequence"].nunique(),
            "ate_rmse_m":sel["ate_rmse_m"].mean(),
            "heading_mae_deg":sel["heading_mae_deg"].mean(),
            "position_p95_m":sel["position_p95_m"].mean(),
            "heading_p95_deg":sel["heading_p95_deg"].mean(),
            "yaw_rate_rmse_radps":np.nan,
        }])
        summary=pd.concat([summary,extra],ignore_index=True).sort_values("ate_rmse_m")
    summary.to_csv(sum_path,index=False)

    print("\nHELD-OUT MACRO SUMMARY")
    print(summary.to_string(index=False))
    print("\nTraining-selected strategy counts:", dict(Counter(selections)))
    print("\nInterpretation rule:")
    print("- If calibrated_imu beats current_dynamic_bias_ekf substantially, dynamic bias updates are the problem.")
    print("- If calibrated_wheel/fusion wins, retain classical wheel+IMU fusion but do not call it an exact WING EKF-IW.")
    print("- If every classical candidate still has large global drift despite low yaw-rate RMSE, that is an observability/drift result, not automatically a code bug.")
    print(f"\nWrote {per_path}")
    print(f"Wrote {sel_path}")
    print(f"Wrote {sum_path}")


if __name__=="__main__":
    main()
