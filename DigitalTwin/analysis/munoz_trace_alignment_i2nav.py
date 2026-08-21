#!/usr/bin/env python3
"""Muñoz-style trace-alignment fidelity baseline for frozen i2Nav V2 trajectories.

Paper-faithful Python reimplementation/adaptation of Muñoz et al. (TSE 2024).
Not the authors' official implementation.

Mobile-robot adaptation:
  * position (x,y) and heading are aligned/evaluated separately to avoid mixing units;
  * MAD thresholds are swept rather than post-hoc cherry-picked;
  * affine-gap Needleman-Wunsch uses the paper's intended negative gap-score convention;
  * default 1 Hz resampling keeps the O(N*M) alignment tractable and is recorded.
"""
from __future__ import annotations
import argparse, math, re
from pathlib import Path
import numpy as np
import pandas as pd


def flist(s): return [float(x.strip()) for x in s.split(',') if x.strip()]
def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',default='.')
    p.add_argument('--input-root',required=True)
    p.add_argument('--glob',default='**/v2_evaluated_trajectory.csv')
    p.add_argument('--method-name',default='Twin V2')
    p.add_argument('--output',default='results/munoz_trace_alignment')
    p.add_argument('--alignment-hz',type=float,default=1.0)
    p.add_argument('--max-snapshots',type=int,default=2500)
    p.add_argument('--position-mads-m',default='0.25,0.5,1.0')
    p.add_argument('--heading-mads-deg',default='2,5,10')
    p.add_argument('--gap-open',type=float,default=-1.0)
    p.add_argument('--gap-extend',type=float,default=-0.1)
    p.add_argument('--lcaw',type=float,default=1.0,help='1.0 disables low-complexity masking')
    p.add_argument('--stationary-speed-mps',type=float,default=0.05)
    return p.parse_args()

def seqid(p):
    m=re.search(r'(building\d+|parking\d+|playground\d+|street\d+)',str(p).lower())
    return m.group(1) if m else p.parent.name

def seedid(p):
    m=re.search(r'replicate_(\d+)_base(\d+)',str(p).lower())
    return f'rep{m.group(1)}_base{m.group(2)}' if m else 'unknown'

def interp_ang(q,t,a):
    return np.arctan2(np.interp(q,t,np.sin(a)),np.interp(q,t,np.cos(a)))

def load(path,hz,maxn):
    df=pd.read_csv(path)
    c=['time_s','gt_east_m','gt_north_m','gt_heading_rad','estimate_east_m','estimate_north_m','estimate_heading_rad']
    miss=[x for x in c if x not in df]
    if miss: raise ValueError(f'{path}: missing {miss}')
    a=df[c].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    a=a[np.isfinite(a).all(axis=1)]
    a=a[np.argsort(a[:,0],kind='mergesort')]
    a=a[np.r_[True,np.diff(a[:,0])>0]]
    if len(a)<3: raise ValueError(f'{path}: too few rows')
    t=a[:,0]; dur=t[-1]-t[0]
    n=int(math.floor(dur*hz))+1
    ehz=hz
    if n>maxn:
        n=maxn; ehz=(maxn-1)/dur
    q=np.linspace(t[0],t[-1],n)
    out={'t':q-q[0], 'gx':np.interp(q,t,a[:,1]), 'gy':np.interp(q,t,a[:,2]),
         'gh':interp_ang(q,t,a[:,3]), 'ex':np.interp(q,t,a[:,4]), 'ey':np.interp(q,t,a[:,5]),
         'eh':interp_ang(q,t,a[:,6]), 'duration_s':float(dur), 'effective_hz':float(ehz)}
    dt=np.diff(out['t'],prepend=out['t'][0]); dt[0]=np.median(dt[1:]); dt=np.maximum(dt,1e-9)
    out['gspeed']=np.hypot(np.diff(out['gx'],prepend=out['gx'][0]),np.diff(out['gy'],prepend=out['gy'][0]))/dt
    out['espeed']=np.hypot(np.diff(out['ex'],prepend=out['ex'][0]),np.diff(out['ey'],prepend=out['ey'][0]))/dt
    return out

def apply_lc(score,la,lb,lcaw):
    if lcaw>=1: return score
    if la and lb: return score*lcaw/2
    if la or lb: return score*lcaw
    return score

def align(n,m,sim,go,ge):
    neg=np.float32(-1e30)
    H=np.zeros((n+1,m+1),np.float32)
    M=np.full_like(H,neg); X=np.full_like(H,neg); Y=np.full_like(H,neg)
    st=np.full((n+1,m+1),-1,np.int8); xx=np.zeros_like(st,bool); yy=np.zeros_like(st,bool)
    X[1:,0]=go+ge; Y[0,1:]=go+ge
    for i in range(1,n+1):
        for j in range(1,m+1):
            M[i,j]=H[i-1,j-1]+sim(i-1,j-1)
            ox=H[i-1,j]+go+ge; ex=X[i-1,j]+ge
            X[i,j]=ex if ex>=ox else ox; xx[i,j]=ex>=ox
            oy=H[i,j-1]+go+ge; ey=Y[i,j-1]+ge
            Y[i,j]=ey if ey>=oy else oy; yy[i,j]=ey>=oy
            vals=(M[i,j],X[i,j],Y[i,j]); s=int(np.argmax(vals)); H[i,j]=vals[s]; st[i,j]=s
    i,j=n,m; cur=int(st[i,j]) if i and j else -1; out=[]
    while i>0 or j>0:
        if i==0: out.append((-1,j-1)); j-=1; cur=2; continue
        if j==0: out.append((i-1,-1)); i-=1; cur=1; continue
        if cur==0:
            out.append((i-1,j-1)); i-=1; j-=1; cur=int(st[i,j]) if i and j else -1
        elif cur==1:
            out.append((i-1,-1)); keep=xx[i,j]; i-=1; cur=1 if keep else (int(st[i,j]) if i and j else -1)
        elif cur==2:
            out.append((-1,j-1)); keep=yy[i,j]; j-=1; cur=2 if keep else (int(st[i,j]) if i and j else -1)
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

def score_position(a,mad,go,ge,lcaw,stationary):
    lg=a['gspeed']<stationary; le=a['espeed']<stationary
    def sim(i,j):
        dx=abs(a['gx'][i]-a['ex'][j]); dy=abs(a['gy'][i]-a['ey'][j])
        if dx>=mad or dy>=mad: return 0.0
        return apply_lc(0.5*((1-dx/mad)+(1-dy/mad)),bool(lg[i]),bool(le[j]),lcaw)
    al=align(len(a['gx']),len(a['ex']),sim,go,ge)
    d=[]; nm=nx=0
    for i,j in al:
        if i<0 or j<0: continue
        if sim(i,j)>0:
            nm+=1
            if not (lcaw<1 and (lg[i] or le[j])): d.append(math.hypot(a['gx'][i]-a['ex'][j],a['gy'][i]-a['ey'][j]))
        else: nx+=1
    den=max(len(a['gx']),len(a['ex'])); ng,ngr,av=gapstats(al)
    return dict(pct_matched=100*nm/den,pct_mismatched=100*nx/den,pct_gaps=100-100*(nm+nx)/den,
                ed_mean_m=float(np.mean(d)) if d else np.nan,fd_max_m=float(np.max(d)) if d else np.nan,
                n_gap_groups=ngr,avg_gap_group_len=av,n_gaps_alignment=ng)

def score_heading(a,mad_deg,go,ge,lcaw,stationary):
    mr=math.radians(mad_deg); lg=a['gspeed']<stationary; le=a['espeed']<stationary
    def sim(i,j):
        d=abs(float(wrap(a['gh'][i]-a['eh'][j])))
        if d>=mr: return 0.0
        return apply_lc(1-d/mr,bool(lg[i]),bool(le[j]),lcaw)
    al=align(len(a['gh']),len(a['eh']),sim,go,ge)
    d=[]; nm=nx=0
    for i,j in al:
        if i<0 or j<0: continue
        if sim(i,j)>0:
            nm+=1
            if not (lcaw<1 and (lg[i] or le[j])): d.append(abs(math.degrees(float(wrap(a['gh'][i]-a['eh'][j])))))
        else: nx+=1
    den=max(len(a['gh']),len(a['eh'])); ng,ngr,av=gapstats(al)
    return dict(pct_matched=100*nm/den,pct_mismatched=100*nx/den,pct_gaps=100-100*(nm+nx)/den,
                ed_mean_deg=float(np.mean(d)) if d else np.nan,fd_max_deg=float(np.max(d)) if d else np.nan,
                n_gap_groups=ngr,avg_gap_group_len=av,n_gaps_alignment=ng)

def main():
    z=args(); repo=Path(z.repo_root).resolve(); root=Path(z.input_root); root=root if root.is_absolute() else repo/root
    out=Path(z.output); out=out if out.is_absolute() else repo/out; out.mkdir(parents=True,exist_ok=True)
    files=sorted(root.glob(z.glob),key=lambda p:str(p).lower())
    if not files: raise SystemExit(f'No files matched {root/z.glob}')
    rows=[]; pm=flist(z.position_mads_m); hm=flist(z.heading_mads_deg)
    for k,p in enumerate(files,1):
        a=load(p,z.alignment_hz,z.max_snapshots); seq=seqid(p); seed=seedid(p)
        print(f'[{k:02d}/{len(files)}] {seq} {seed}: {len(a["t"])} snapshots @ {a["effective_hz"]:.3f} Hz')
        common=dict(method=z.method_name,file=str(p.relative_to(repo) if p.is_relative_to(repo) else p),sequence=seq,seed=seed,
                    alignment_hz=z.alignment_hz,effective_hz=a['effective_hz'],duration_s=a['duration_s'],gap_open=z.gap_open,gap_extend=z.gap_extend,lcaw=z.lcaw)
        for mad in pm: rows.append({**common,'domain':'position_xy','mad':mad,'mad_unit':'m_per_coordinate',**score_position(a,mad,z.gap_open,z.gap_extend,z.lcaw,z.stationary_speed_mps)})
        for mad in hm: rows.append({**common,'domain':'heading','mad':mad,'mad_unit':'deg',**score_heading(a,mad,z.gap_open,z.gap_extend,z.lcaw,z.stationary_speed_mps)})
    run=pd.DataFrame(rows); run.to_csv(out/'munoz_per_run.csv',index=False)
    metrics=[c for c in ['pct_matched','pct_mismatched','pct_gaps','ed_mean_m','fd_max_m','ed_mean_deg','fd_max_deg','n_gap_groups','avg_gap_group_len'] if c in run]
    seq=run.groupby(['method','sequence','domain','mad','mad_unit'],dropna=False)[metrics].mean().reset_index(); seq.to_csv(out/'munoz_per_sequence.csv',index=False)
    ds=seq.groupby(['method','domain','mad','mad_unit'],dropna=False)[metrics].mean().reset_index(); ds.to_csv(out/'munoz_dataset_summary.csv',index=False)
    report=['# Muñoz-style trace-alignment baseline','',f'- Files: **{len(files)}**',f'- Alignment sampling: **{z.alignment_hz:g} Hz**',f'- Gap open/extend: **{z.gap_open:g}/{z.gap_extend:g}**',f'- LCAW: **{z.lcaw:g}**','',
            'This is a paper-faithful Python adaptation, not the authors\' official artifact. Position and heading are evaluated separately. MAD values are a sensitivity grid and must not be post-hoc cherry-picked.','','```\n'+ds.to_string(index=False)+'\n```']
    (out/'munoz_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Wrote {out}')
if __name__=='__main__': main()
