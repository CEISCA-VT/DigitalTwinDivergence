#!/usr/bin/env python3
"""Sequence-bootstrap uncertainty for TFP vs Muñoz correlations.

Correlations are computed across physical sequences, never timestamps.
All Muñoz MAD values are retained as a sensitivity grid.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def locate(explicit):
    if explicit:
        p=Path(explicit)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    for p in [
        Path("results/i2nav_fidelity_baselines/comparison/combined_sequence_metrics_long.csv"),
        Path("results/i2nav_fidelity_baselines/combined_sequence_metrics_long.csv"),
    ]:
        if p.exists(): return p
    hits=sorted(Path("results").rglob("combined_sequence_metrics_long.csv"))
    if len(hits)==1:return hits[0]
    raise FileNotFoundError("Use --input to specify combined_sequence_metrics_long.csv")

def rho(a,b):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<4:return np.nan
    if len(np.unique(a[ok]))<2 or len(np.unique(b[ok]))<2:return np.nan
    return float(spearmanr(a[ok],b[ok]).statistic)

def boot(a,b,rng,nboot):
    ok=np.isfinite(a)&np.isfinite(b)
    a=a[ok]; b=b[ok]; n=len(a)
    vals=[]
    for _ in range(nboot):
        ix=rng.integers(0,n,n)
        r=rho(a[ix],b[ix])
        if np.isfinite(r): vals.append(r)
    if len(vals)<max(100,nboot//10): return (np.nan,np.nan,len(vals))
    lo,hi=np.quantile(vals,[.025,.975])
    return float(lo),float(hi),len(vals)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--bootstrap",type=int,default=20000)
    ap.add_argument("--bootstrap-seed",type=int,default=20260821)
    ap.add_argument("--output-root",default="results/publication_hardening")
    a=ap.parse_args()
    d=pd.read_csv(locate(a.input))
    need={"method","sequence","domain","mad","pct_matched"}
    if not need.issubset(d.columns): raise SystemExit(f"Missing columns: {sorted(need-set(d.columns))}")

    specs={
      "position_xy":[
        ("ate_m","pct_matched"),("rpe10_m","pct_matched"),("dp_p95_m","pct_matched"),
        ("ate_m","ed_mean_m"),("ate_m","fd_max_m"),
        ("rpe10_m","ed_mean_m"),("dp_p95_m","fd_max_m")],
      "heading":[
        ("heading_mae_deg","pct_matched"),("dtheta_p95_deg","pct_matched"),
        ("heading_mae_deg","ed_mean_deg"),("dtheta_p95_deg","fd_max_deg")],
    }
    rng=np.random.default_rng(a.bootstrap_seed); rows=[]
    for (method,domain,mad),g in d.groupby(["method","domain","mad"],dropna=False):
        if domain not in specs: continue
        # exactly one row per physical sequence/domain/MAD expected
        g=g.sort_values("sequence").drop_duplicates("sequence")
        for ma,mb in specs[domain]:
            if ma not in g.columns or mb not in g.columns: continue
            aa=g[ma].to_numpy(float); bb=g[mb].to_numpy(float)
            ok=np.isfinite(aa)&np.isfinite(bb)
            rr=rho(aa,bb)
            lo,hi,nb=boot(aa,bb,rng,a.bootstrap)
            rows.append({
                "method":method,"domain":domain,"mad":mad,
                "metric_a":ma,"metric_b":mb,"n_sequences":int(ok.sum()),
                "spearman":rr,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,
                "valid_bootstrap_replicates":nb,
                "interpretation":"exploratory sequence-level correlation",
            })
    R=pd.DataFrame(rows)
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    R.to_csv(out/"munoz_tfp_bootstrap_correlations.csv",index=False)

    # Compact representative middle-MAD table for manuscript use.
    middle=[]
    for domain,target in [("position_xy",0.5),("heading",5.0)]:
        q=R[(R.domain==domain)&np.isclose(R.mad.astype(float),target)]
        middle.append(q)
    M=pd.concat(middle,ignore_index=True) if middle else pd.DataFrame()
    M.to_csv(out/"munoz_tfp_bootstrap_correlations_midMAD.csv",index=False)
    print(M.to_string(index=False))
    print(f"\nWrote {out}")

if __name__=="__main__":
    main()
