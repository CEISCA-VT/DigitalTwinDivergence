#!/usr/bin/env python3
"""Muñoz-style trace-alignment fidelity baseline for multiple methods.

Paper-faithful adaptation of Muñoz et al., IEEE TSE 2024; NOT the authors'
official implementation. Position and heading are evaluated separately to
avoid mixing meters and radians/degrees. MAD values are a declared sensitivity
grid. The affine-gap dynamic program is vectorized by anti-diagonals for speed.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import load_pose_trajectory, wrap_angle
from .fidelity_common import load_manifest


def flist(s): return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest",default="results/i2nav_fidelity_baselines/trajectory_manifest.csv")
    p.add_argument("--output",default="results/i2nav_fidelity_baselines/munoz")
    p.add_argument("--repo-root",default=".")
    p.add_argument("--alignment-hz",type=float,default=1.0)
    p.add_argument("--max-snapshots",type=int,default=2500)
    p.add_argument("--position-mads-m",default="0.25,0.5,1.0")
    p.add_argument("--heading-mads-deg",default="2,5,10")
    p.add_argument("--gap-open",type=float,default=-1.0)
    p.add_argument("--gap-extend",type=float,default=-0.1)
    p.add_argument("--lcaw",type=float,default=1.0,help="1 disables low-complexity attenuation")
    p.add_argument("--stationary-speed-mps",type=float,default=0.05)
    p.add_argument("--methods",default="",help="Optional comma-separated method names")
    p.add_argument("--resume",action="store_true")
    return p.parse_args()


def interp_angle(q,t,a):
    return np.arctan2(np.interp(q,t,np.sin(a)),np.interp(q,t,np.cos(a)))


def load_resampled(path,hz,maxn):
    d=load_pose_trajectory(path)
    a=d[["time_s","gt_east_m","gt_north_m","gt_heading_rad","estimate_east_m","estimate_north_m","estimate_heading_rad"]].to_numpy(float)
    t=a[:,0]; dur=float(t[-1]-t[0]); n=int(math.floor(dur*hz))+1; ehz=float(hz)
    if n>maxn:
        n=maxn; ehz=(maxn-1)/dur
    n=max(n,3)
    q=np.linspace(t[0],t[-1],n)
    out={
        "t":q-q[0],
        "gx":np.interp(q,t,a[:,1]), "gy":np.interp(q,t,a[:,2]), "gh":interp_angle(q,t,a[:,3]),
        "ex":np.interp(q,t,a[:,4]), "ey":np.interp(q,t,a[:,5]), "eh":interp_angle(q,t,a[:,6]),
        "duration_s":dur,"effective_hz":ehz,
    }
    dt=np.diff(out["t"],prepend=out["t"][0]); dt[0]=np.median(dt[1:]); dt=np.maximum(dt,1e-9)
    out["gspeed"]=np.hypot(np.diff(out["gx"],prepend=out["gx"][0]),np.diff(out["gy"],prepend=out["gy"][0]))/dt
    out["espeed"]=np.hypot(np.diff(out["ex"],prepend=out["ex"][0]),np.diff(out["ey"],prepend=out["ey"][0]))/dt
    return out


def _affine_align_from_similarity(sim: np.ndarray, gap_open: float, gap_extend: float):
    """Affine-gap global-style alignment with free leading H boundaries as in prior adaptation.

    Returns list of (physical_index, twin_index), -1 denoting a gap.
    """
    n,m=sim.shape
    neg=np.float32(-1e30)
    H=np.zeros((n+1,m+1),np.float32)
    X=np.full((n+1,m+1),neg,np.float32)
    Y=np.full((n+1,m+1),neg,np.float32)
    state=np.full((n+1,m+1),-1,np.int8)
    x_extend=np.zeros((n+1,m+1),np.bool_)
    y_extend=np.zeros((n+1,m+1),np.bool_)
    X[1:,0]=gap_open+gap_extend
    Y[0,1:]=gap_open+gap_extend
    # Anti-diagonal vectorization: all cells with i+j=s are mutually independent.
    for s in range(2,n+m+1):
        i0=max(1,s-m); i1=min(n,s-1)
        if i0>i1: continue
        ii=np.arange(i0,i1+1); jj=s-ii
        mm=H[ii-1,jj-1]+sim[ii-1,jj-1]
        ox=H[ii-1,jj]+gap_open+gap_extend; ex=X[ii-1,jj]+gap_extend
        xv=np.maximum(ox,ex); xe=ex>=ox
        oy=H[ii,jj-1]+gap_open+gap_extend; ey=Y[ii,jj-1]+gap_extend
        yv=np.maximum(oy,ey); ye=ey>=oy
        vals=np.stack([mm,xv,yv],axis=0)
        st=np.argmax(vals,axis=0).astype(np.int8)
        H[ii,jj]=np.take_along_axis(vals,st[None,:],axis=0)[0]
        X[ii,jj]=xv; Y[ii,jj]=yv; state[ii,jj]=st; x_extend[ii,jj]=xe; y_extend[ii,jj]=ye

    i,j=n,m; cur=int(state[i,j]) if i and j else -1; out=[]
    while i>0 or j>0:
        if i==0: out.append((-1,j-1)); j-=1; cur=2; continue
        if j==0: out.append((i-1,-1)); i-=1; cur=1; continue
        if cur==0:
            out.append((i-1,j-1)); i-=1; j-=1; cur=int(state[i,j]) if i and j else -1
        elif cur==1:
            out.append((i-1,-1)); keep=bool(x_extend[i,j]); i-=1; cur=1 if keep else (int(state[i,j]) if i and j else -1)
        elif cur==2:
            out.append((-1,j-1)); keep=bool(y_extend[i,j]); j-=1; cur=2 if keep else (int(state[i,j]) if i and j else -1)
        else:
            if i: out.append((i-1,-1)); i-=1
            else: out.append((-1,j-1)); j-=1
    return out[::-1]


def gapstats(al):
    groups=[]; run=0; ng=0
    for i,j in al:
        if i<0 or j<0: run+=1; ng+=1
        elif run: groups.append(run); run=0
    if run: groups.append(run)
    return ng,len(groups),(float(np.mean(groups)) if groups else 0.0)


def _lc_weights(low_g,low_e,lcaw):
    if lcaw>=1.0: return np.ones((len(low_g),len(low_e)),np.float32)
    A=low_g[:,None]; B=low_e[None,:]
    W=np.ones((len(low_g),len(low_e)),np.float32)
    W[A^B]=lcaw; W[A&B]=lcaw/2.0
    return W


def score_position(a,mad,go,ge,lcaw,stationary):
    gx=a["gx"][:,None]; ex=a["ex"][None,:]; gy=a["gy"][:,None]; ey=a["ey"][None,:]
    dx=np.abs(gx-ex); dy=np.abs(gy-ey)
    valid=(dx<mad)&(dy<mad)
    sim=np.where(valid,0.5*((1-dx/mad)+(1-dy/mad)),0.0).astype(np.float32)
    sim*= _lc_weights(a["gspeed"]<stationary,a["espeed"]<stationary,lcaw)
    al=_affine_align_from_similarity(sim,go,ge)
    d=[]; nm=nx=0
    lg=a["gspeed"]<stationary; le=a["espeed"]<stationary
    for i,j in al:
        if i<0 or j<0: continue
        if sim[i,j]>0:
            nm+=1
            if not (lcaw<1 and (lg[i] or le[j])):
                d.append(math.hypot(a["gx"][i]-a["ex"][j],a["gy"][i]-a["ey"][j]))
        else: nx+=1
    den=max(len(a["gx"]),len(a["ex"])); ng,ngr,av=gapstats(al)
    return dict(pct_matched=100*nm/den,pct_mismatched=100*nx/den,pct_gaps=100-100*(nm+nx)/den,
                ed_mean_m=float(np.mean(d)) if d else np.nan,fd_max_m=float(np.max(d)) if d else np.nan,
                n_gap_groups=ngr,avg_gap_group_len=av,n_gaps_alignment=ng)


def score_heading(a,mad_deg,go,ge,lcaw,stationary):
    mr=math.radians(mad_deg)
    diff=np.abs(wrap_angle(a["gh"][:,None]-a["eh"][None,:]))
    valid=diff<mr
    sim=np.where(valid,1-diff/mr,0.0).astype(np.float32)
    sim*=_lc_weights(a["gspeed"]<stationary,a["espeed"]<stationary,lcaw)
    al=_affine_align_from_similarity(sim,go,ge)
    d=[]; nm=nx=0; lg=a["gspeed"]<stationary; le=a["espeed"]<stationary
    for i,j in al:
        if i<0 or j<0: continue
        if sim[i,j]>0:
            nm+=1
            if not (lcaw<1 and (lg[i] or le[j])): d.append(abs(math.degrees(float(wrap_angle(a["gh"][i]-a["eh"][j])))))
        else: nx+=1
    den=max(len(a["gh"]),len(a["eh"])); ng,ngr,av=gapstats(al)
    return dict(pct_matched=100*nm/den,pct_mismatched=100*nx/den,pct_gaps=100-100*(nm+nx)/den,
                ed_mean_deg=float(np.mean(d)) if d else np.nan,fd_max_deg=float(np.max(d)) if d else np.nan,
                n_gap_groups=ngr,avg_gap_group_len=av,n_gaps_alignment=ng)


def main():
    z=parse_args(); repo=Path(z.repo_root).resolve(); out=Path(z.output); out=out if out.is_absolute() else repo/out; out.mkdir(parents=True,exist_ok=True)
    m=load_manifest(z.manifest if Path(z.manifest).is_absolute() else repo/z.manifest,repo)
    if z.methods:
        wanted={x.strip() for x in z.methods.split(",") if x.strip()}; m=m[m.method.isin(wanted)].reset_index(drop=True)
    pm=flist(z.position_mads_m); hm=flist(z.heading_mads_deg)
    prior=pd.DataFrame()
    prior_path=out/"munoz_per_run.csv"
    if z.resume and prior_path.exists():
        prior=pd.read_csv(prior_path)
    rows=[] if prior.empty else prior.to_dict("records")
    done=set()
    if not prior.empty:
        for _,r in prior.iterrows(): done.add((str(r.method),str(r.sequence),str(r.seed),str(r.domain),float(r.mad)))
    for k,r in m.iterrows():
        todo_pos=[mad for mad in pm if (str(r.method),str(r.sequence),str(r.seed),"position_xy",float(mad)) not in done]
        todo_head=[mad for mad in hm if (str(r.method),str(r.sequence),str(r.seed),"heading",float(mad)) not in done]
        if not todo_pos and not todo_head: continue
        a=load_resampled(r.trajectory_abs,z.alignment_hz,z.max_snapshots)
        print(f"[{k+1:03d}/{len(m):03d}] {r.method} {r.sequence} {r.seed}: {len(a['t'])} snapshots @ {a['effective_hz']:.3f} Hz")
        common=dict(method=r.method,sequence=r.sequence,seed=r.seed,trajectory=r.trajectory,
                    alignment_hz=z.alignment_hz,effective_hz=a["effective_hz"],duration_s=a["duration_s"],gap_open=z.gap_open,gap_extend=z.gap_extend,lcaw=z.lcaw)
        for mad in todo_pos:
            rows.append({**common,"domain":"position_xy","mad":mad,"mad_unit":"m_per_coordinate",**score_position(a,mad,z.gap_open,z.gap_extend,z.lcaw,z.stationary_speed_mps)})
        for mad in todo_head:
            rows.append({**common,"domain":"heading","mad":mad,"mad_unit":"deg",**score_heading(a,mad,z.gap_open,z.gap_extend,z.lcaw,z.stationary_speed_mps)})
        pd.DataFrame(rows).to_csv(prior_path,index=False)
    run=pd.DataFrame(rows); run.to_csv(prior_path,index=False)
    metrics=[c for c in ["pct_matched","pct_mismatched","pct_gaps","ed_mean_m","fd_max_m","ed_mean_deg","fd_max_deg","n_gap_groups","avg_gap_group_len"] if c in run]
    seq=run.groupby(["method","sequence","domain","mad","mad_unit"],dropna=False)[metrics].mean().reset_index(); seq.to_csv(out/"munoz_per_sequence.csv",index=False)
    ds=seq.groupby(["method","domain","mad","mad_unit"],dropna=False)[metrics].mean().reset_index(); ds.to_csv(out/"munoz_dataset_summary.csv",index=False)
    report=["# Multi-method Muñoz-style trace-alignment baseline","",f"- trajectories: **{m.shape[0]}**",f"- alignment sampling: **{z.alignment_hz:g} Hz**",f"- max snapshots: **{z.max_snapshots}**",f"- gap open/extend: **{z.gap_open:g}/{z.gap_extend:g}**",f"- LCAW: **{z.lcaw:g}**","",
            "This is a paper-faithful adaptation, not the authors' official artifact. Position and heading are evaluated separately. MAD values are a predeclared sensitivity grid and must not be cherry-picked after seeing results.","","```",ds.to_string(index=False),"```"]
    (out/"munoz_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Wrote {out}")

if __name__=="__main__": main()
