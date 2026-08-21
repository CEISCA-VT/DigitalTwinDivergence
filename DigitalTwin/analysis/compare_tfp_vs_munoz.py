#!/usr/bin/env python3
"""Compare Twin Fidelity Profile metrics with Muñoz-style trace-alignment metrics
on the exact same frozen i2Nav V2 trajectories.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',default='.')
    p.add_argument('--input-root',required=True)
    p.add_argument('--glob',default='**/v2_evaluated_trajectory.csv')
    p.add_argument('--munoz-csv',default='results/munoz_trace_alignment/munoz_per_sequence.csv')
    p.add_argument('--output',default='results/tfp_vs_munoz')
    return p.parse_args()

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def seqid(p):
    m=re.search(r'(building\d+|parking\d+|playground\d+|street\d+)',str(p).lower())
    return m.group(1) if m else p.parent.name

def seedid(p):
    m=re.search(r'replicate_(\d+)_base(\d+)',str(p).lower())
    return f'rep{m.group(1)}_base{m.group(2)}' if m else 'unknown'

def rel(x0,y0,h0,x1,y1):
    dx=x1-x0; dy=y1-y0; c=np.cos(h0); s=np.sin(h0)
    return c*dx+s*dy,-s*dx+c*dy

def rpe(t,gx,gy,gh,ex,ey,eh,H):
    j=np.searchsorted(t,t+H,'left'); i=np.arange(len(t)); ok=j<len(t); i=i[ok]; j=j[ok]
    if not len(i): return np.nan
    dt=np.diff(t); dt=dt[(dt>0)&np.isfinite(dt)]; tol=max(0.15,2.5*float(np.median(dt))) if len(dt) else 0.25
    ok=np.abs(t[j]-(t[i]+H))<=tol; i=i[ok]; j=j[ok]
    if not len(i): return np.nan
    gdx,gdy=rel(gx[i],gy[i],gh[i],gx[j],gy[j]); edx,edy=rel(ex[i],ey[i],eh[i],ex[j],ey[j])
    return float(np.sqrt(np.mean((edx-gdx)**2+(edy-gdy)**2)))

def tfp(path):
    c=['time_s','gt_east_m','gt_north_m','gt_heading_rad','estimate_east_m','estimate_north_m','estimate_heading_rad']
    d=pd.read_csv(path); miss=[x for x in c if x not in d]
    if miss: raise ValueError(f'{path}: missing {miss}')
    a=d[c].apply(pd.to_numeric,errors='coerce').to_numpy(float); a=a[np.isfinite(a).all(axis=1)]; a=a[np.argsort(a[:,0],kind='mergesort')]
    t=a[:,0]-a[0,0]; gx,gy,gh,ex,ey,eh=a[:,1],a[:,2],a[:,3],a[:,4],a[:,5],a[:,6]
    dp=np.hypot(gx-ex,gy-ey); dh=np.abs(wrap(gh-eh))
    return dict(ate_m=float(np.sqrt(np.mean(dp**2))),heading_mae_deg=float(np.degrees(np.mean(dh))),
                rpe1_m=rpe(t,gx,gy,gh,ex,ey,eh,1),rpe5_m=rpe(t,gx,gy,gh,ex,ey,eh,5),rpe10_m=rpe(t,gx,gy,gh,ex,ey,eh,10),
                dp_p95_m=float(np.quantile(dp,.95)),dtheta_p95_deg=float(np.degrees(np.quantile(dh,.95))),
                dp_max_m=float(np.max(dp)),dtheta_max_deg=float(np.degrees(np.max(dh))))

def spearman(x,y):
    z=pd.DataFrame({'x':x,'y':y}).dropna(); return float(z.x.rank().corr(z.y.rank())) if len(z)>=3 else np.nan

def main():
    z=args(); repo=Path(z.repo_root).resolve(); root=Path(z.input_root); root=root if root.is_absolute() else repo/root
    mc=Path(z.munoz_csv); mc=mc if mc.is_absolute() else repo/mc
    out=Path(z.output); out=out if out.is_absolute() else repo/out; out.mkdir(parents=True,exist_ok=True)
    files=sorted(root.glob(z.glob),key=lambda p:str(p).lower())
    if not files: raise SystemExit('No trajectories found')
    rows=[]
    for p in files: rows.append({'file':str(p.relative_to(repo) if p.is_relative_to(repo) else p),'sequence':seqid(p),'seed':seedid(p),**tfp(p)})
    run=pd.DataFrame(rows); run.to_csv(out/'tfp_per_run.csv',index=False)
    tm=['ate_m','heading_mae_deg','rpe1_m','rpe5_m','rpe10_m','dp_p95_m','dtheta_p95_deg','dp_max_m','dtheta_max_deg']
    seq=run.groupby('sequence')[tm].mean().reset_index(); seq.to_csv(out/'tfp_per_sequence.csv',index=False)
    m=pd.read_csv(mc); joined=seq.merge(m,on='sequence',how='inner'); joined.to_csv(out/'tfp_vs_munoz_joined.csv',index=False)
    cr=[]
    for (dom,mad),g in joined.groupby(['domain','mad']):
        pairs=[('ate_m','pct_matched'),('rpe10_m','pct_matched'),('dp_p95_m','pct_matched')]
        if dom=='position_xy': pairs += [('ate_m','ed_mean_m'),('ate_m','fd_max_m'),('rpe10_m','ed_mean_m'),('dp_p95_m','fd_max_m')]
        else: pairs=[('heading_mae_deg','pct_matched'),('dtheta_p95_deg','pct_matched'),('heading_mae_deg','ed_mean_deg'),('dtheta_p95_deg','fd_max_deg')]
        for a,b in pairs:
            if a in g and b in g: cr.append({'domain':dom,'mad':mad,'tfp_metric':a,'munoz_metric':b,'spearman':spearman(g[a],g[b]),'n_sequences':len(g[[a,b]].dropna())})
    corr=pd.DataFrame(cr); corr.to_csv(out/'evaluator_rank_correlations.csv',index=False)
    d=seq.copy(); d['rpe10_rank_pct']=d.rpe10_m.rank(pct=True); d['ate_rank_pct']=d.ate_m.rank(pct=True); d['local_good_global_bad']=(d.rpe10_rank_pct<=.5)&(d.ate_rank_pct>.5)
    d.sort_values(['local_good_global_bad','ate_m'],ascending=[False,False]).to_csv(out/'local_global_discordance.csv',index=False)
    pos=joined[joined.domain=='position_xy']
    if len(pos):
        mads=sorted(pos.mad.unique()); chosen=mads[len(mads)//2]; g=pos[pos.mad==chosen]
        fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.scatter(g.rpe10_m,g.pct_matched)
        for _,r in g.iterrows(): ax.annotate(r.sequence,(r.rpe10_m,r.pct_matched),fontsize=8)
        ax.set_xlabel('TFP RPE10 (m)'); ax.set_ylabel('Muñoz-style matched snapshots (%)'); ax.set_title(f'Local fidelity vs trace alignment (MAD={chosen:g} m)'); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(out/'rpe10_vs_munoz_match.png',dpi=220); plt.close(fig)
        if 'ed_mean_m' in g:
            fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.scatter(g.ate_m,g.ed_mean_m)
            for _,r in g.iterrows(): ax.annotate(r.sequence,(r.ate_m,r.ed_mean_m),fontsize=8)
            ax.set_xlabel('TFP/global ATE (m)'); ax.set_ylabel('Muñoz aligned mean distance ED (m)'); ax.set_title(f'Synchronized global error vs aligned distance (MAD={chosen:g} m)'); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(out/'ate_vs_munoz_ed.png',dpi=220); plt.close(fig)
    p2=joined[joined.sequence=='parking02']
    keep=[c for c in ['sequence','domain','mad','ate_m','heading_mae_deg','rpe1_m','rpe5_m','rpe10_m','dp_p95_m','dtheta_p95_deg','pct_matched','ed_mean_m','fd_max_m','ed_mean_deg','fd_max_deg','pct_gaps'] if c in p2]
    report=['# TFP vs Muñoz-style trace alignment','',
            'Both evaluators use the same frozen physical/virtual trajectories. TFP preserves synchronization and decomposes local/global/component/tail fidelity; Muñoz-style evaluation aligns behavior traces before computing %MS and aligned distances.','',
            '## TFP per sequence','','```\n'+seq.to_string(index=False)+'\n```','','## Local/global discordance','','```\n'+d.to_string(index=False)+'\n```','','## Rank correlations','','```\n'+corr.to_string(index=False)+'\n```']
    if len(p2): report += ['','## parking02 diagnostic','','```\n'+p2[keep].to_string(index=False)+'\n```']
    report += ['','## Claim rule','',
               'Do not claim universal superiority. A defensible claim is that TFP is more diagnostically informative for synchronized mobile-robot twins if it exposes local/global, component, tail, or condition-dependent failures that %MS/ED/FD compress or align away. Muñoz remains stronger when the intended behavior may be temporally shifted but behaviorally equivalent.']
    (out/'tfp_vs_munoz_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Wrote {out}')
if __name__=='__main__': main()
