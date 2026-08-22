#!/usr/bin/env python3
"""Controlled replay-based yaw-bias sensitivity on frozen V2 trajectories.

This is a mechanism/sensitivity experiment, NOT attack detection.

A constant signed bias is added to the saved corrected yaw-rate channel,
the twin is re-integrated from its original initial state, and fidelity
metrics are recomputed. Positive and negative biases are both evaluated;
headline summaries aggregate by absolute bias magnitude to avoid choosing
a favorable sign post hoc.

A zero-bias reconstruction audit is mandatory. If re-integrating the saved
corrected velocity/yaw-rate does not reproduce the saved V2 trajectory
closely, the script labels the experiment NOT_READY rather than silently
claiming a valid replay.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

SEQ=["building00","building01","building02","parking00","parking01","parking02",
     "playground00","street00","street01","street02"]

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def seqid(p):
    s=str(p).lower()
    for x in SEQ:
        if x in s:return x
    return "unknown"

def seedid(p):
    s=str(p)
    for pat in [r"replicate_(\d+)_base(\d+)",r"seed[_-]?(\d+)"]:
        m=re.search(pat,s,re.I)
        if m:return "_".join(m.groups())
    return p.parent.name

def integrate(t,v,w,x0,y0,h0):
    n=len(t); x=np.empty(n); y=np.empty(n); h=np.empty(n)
    x[0]=x0;y[0]=y0;h[0]=h0
    for k in range(n-1):
        dt=max(float(t[k+1]-t[k]),0.0)
        wm=0.5*(w[k]+w[k+1]); vm=0.5*(v[k]+v[k+1])
        hm=h[k]+0.5*wm*dt
        x[k+1]=x[k]+vm*dt*np.cos(hm)
        y[k+1]=y[k]+vm*dt*np.sin(hm)
        h[k+1]=wrap(h[k]+wm*dt)
    return x,y,h

def relative_pose(x,y,h,i,j):
    dx=x[j]-x[i]; dy=y[j]-y[i]
    c=np.cos(h[i]); s=np.sin(h[i])
    return np.array([c*dx+s*dy,-s*dx+c*dy,wrap(h[j]-h[i])])

def rpe(t,gx,gy,gh,x,y,h,horizon):
    errs=[]
    j=0
    for i in range(len(t)):
        target=t[i]+horizon
        j=max(j,i+1)
        while j<len(t) and t[j]<target: j+=1
        if j>=len(t): break
        gp=relative_pose(gx,gy,gh,i,j); ep=relative_pose(x,y,h,i,j)
        # relative transform error translation in the physical start frame
        errs.append(np.linalg.norm(ep[:2]-gp[:2]))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan

def metrics(d,x,y,h,bias):
    gx=d.gt_east_m.to_numpy(float); gy=d.gt_north_m.to_numpy(float)
    gh=d.gt_heading_rad.to_numpy(float); t=d.time_s.to_numpy(float)
    dp=np.hypot(x-gx,y-gy); dh=np.abs(np.rad2deg(wrap(h-gh)))
    # pose-derived reference yaw rate
    dt=np.diff(t); dgh=wrap(np.diff(gh))
    gwr=np.zeros(len(t))
    good=dt>0
    gwr[1:][good]=dgh[good]/dt[good]
    if len(t)>1:gwr[0]=gwr[1]
    return {
      "bias_radps":bias,"bias_degps":float(np.rad2deg(bias)),
      "ate_m":float(np.sqrt(np.mean(dp**2))),
      "heading_mae_deg":float(np.mean(dh)),
      "dp_p95_m":float(np.quantile(dp,.95)),
      "dtheta_p95_deg":float(np.quantile(dh,.95)),
      "rpe1_m":rpe(t,gx,gy,gh,x,y,h,1.0),
      "rpe5_m":rpe(t,gx,gy,gh,x,y,h,5.0),
      "rpe10_m":rpe(t,gx,gy,gh,x,y,h,10.0),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",default="results/i2nav_v2_full_loso/i2nav_v2_full_loso")
    ap.add_argument("--glob",default="**/v2_evaluated_trajectory.csv")
    ap.add_argument("--bias-radps",default="0,0.0005,-0.0005,0.001,-0.001,0.0025,-0.0025,0.005,-0.005")
    ap.add_argument("--max-zero-replay-rmse-m",type=float,default=0.05)
    ap.add_argument("--output-root",default="results/publication_hardening")
    a=ap.parse_args()

    files=sorted(Path(a.input_root).glob(a.glob))
    if not files: raise SystemExit("No frozen V2 trajectories found")
    biases=[float(x) for x in a.bias_radps.split(",")]
    rows=[]; audits=[]
    for p in files:
        d=pd.read_csv(p)
        req=["time_s","gt_east_m","gt_north_m","gt_heading_rad",
             "estimate_east_m","estimate_north_m","estimate_heading_rad",
             "corrected_v_mps","corrected_omega_radps"]
        miss=[c for c in req if c not in d.columns]
        if miss: raise SystemExit(f"{p}: missing {miss}")
        t=d.time_s.to_numpy(float); v=d.corrected_v_mps.to_numpy(float)
        w=d.corrected_omega_radps.to_numpy(float)
        x0=float(d.estimate_east_m.iloc[0]); y0=float(d.estimate_north_m.iloc[0])
        h0=float(d.estimate_heading_rad.iloc[0])

        xz,yz,hz=integrate(t,v,w,x0,y0,h0)
        dx=xz-d.estimate_east_m.to_numpy(float)
        dy=yz-d.estimate_north_m.to_numpy(float)
        replay_rmse=float(np.sqrt(np.mean(dx*dx+dy*dy)))
        audits.append({"sequence":seqid(p),"seed":seedid(p),"path":str(p),
                       "zero_bias_replay_position_rmse_m":replay_rmse,
                       "passes_replay_gate":replay_rmse<=a.max_zero_replay_rmse_m})

        for b in biases:
            x,y,h=integrate(t,v,w+b,x0,y0,h0)
            rr={"sequence":seqid(p),"seed":seedid(p),"path":str(p),
                "zero_bias_replay_position_rmse_m":replay_rmse}
            rr.update(metrics(d,x,y,h,b))
            rows.append(rr)

    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    A=pd.DataFrame(audits); R=pd.DataFrame(rows)
    A.to_csv(out/"yaw_bias_zero_replay_audit.csv",index=False)
    R.to_csv(out/"yaw_bias_per_run.csv",index=False)
    # seed -> sequence -> dataset; preserve sequence as physical unit
    S=(R.groupby(["sequence","bias_radps","bias_degps"],as_index=False)
       .agg({m:"mean" for m in ["ate_m","heading_mae_deg","dp_p95_m","dtheta_p95_deg","rpe1_m","rpe5_m","rpe10_m"]}))
    S.to_csv(out/"yaw_bias_per_sequence.csv",index=False)

    S["abs_bias_radps"]=S.bias_radps.abs()
    M=(S.groupby("abs_bias_radps",as_index=False)
       .agg(n_sequences=("sequence","nunique"),
            ate_m=("ate_m","mean"),heading_mae_deg=("heading_mae_deg","mean"),
            dp_p95_m=("dp_p95_m","mean"),dtheta_p95_deg=("dtheta_p95_deg","mean"),
            rpe1_m=("rpe1_m","mean"),rpe5_m=("rpe5_m","mean"),rpe10_m=("rpe10_m","mean")))
    M["abs_bias_degps"]=np.rad2deg(M.abs_bias_radps)
    M.to_csv(out/"yaw_bias_macro_by_magnitude.csv",index=False)

    ready=bool(A.passes_replay_gate.all())
    (out/"yaw_bias_replay_status.txt").write_text(
        ("PUBLICATION_REPLAY_GATE: PASS\n" if ready else "PUBLICATION_REPLAY_GATE: FAIL\n")+
        f"max zero-bias reconstruction RMSE = {A.zero_bias_replay_position_rmse_m.max():.6f} m\n"+
        "Use this experiment only as controlled replay-based yaw-bias sensitivity, not attack detection.\n",
        encoding="utf-8")
    print(A[["sequence","seed","zero_bias_replay_position_rmse_m","passes_replay_gate"]].to_string(index=False))
    print("\nMacro by absolute injected bias:")
    print(M.to_string(index=False))
    print("\nSTATUS:", "PASS" if ready else "FAIL -- do not publish yaw-bias results until replay is reconciled")

if __name__=="__main__":
    main()
