#!/usr/bin/env python3
"""Generate compact publication-oriented figures from existing evaluator outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def locate(explicit):
    if explicit:
        p=Path(explicit)
        if not p.exists():raise FileNotFoundError(p)
        return p
    for p in [Path("results/i2nav_fidelity_baselines/comparison/combined_sequence_metrics_long.csv"),
              Path("results/i2nav_fidelity_baselines/combined_sequence_metrics_long.csv")]:
        if p.exists():return p
    hits=sorted(Path("results").rglob("combined_sequence_metrics_long.csv"))
    if len(hits)==1:return hits[0]
    raise FileNotFoundError("Use --input.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output-root",default="results/publication_hardening/figures")
    a=ap.parse_args()
    d=pd.read_csv(locate(a.input)); out=Path(a.output_root);out.mkdir(parents=True,exist_ok=True)

    # 1) V1/V2 local-vs-global scatter, only label parking02.
    base=d.drop_duplicates(["method","sequence"])
    q=base[base.method.isin(["Twin V1","Twin V2"])].copy()
    fig,ax=plt.subplots(figsize=(5.2,3.5))
    for method,g in q.groupby("method"):
        ax.scatter(g["rpe10_m"],g["ate_m"],label=method)
        p=g[g.sequence=="parking02"]
        if len(p):
            ax.annotate(f"{method} parking02",(float(p.rpe10_m.iloc[0]),float(p.ate_m.iloc[0])),
                        xytext=(5,5),textcoords="offset points",fontsize=7)
    ax.set_xlabel("RPE10 (m)")
    ax.set_ylabel("Operational ATE RMSE (m)")
    ax.set_title("Local motion agreement vs global synchronization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"v1_v2_local_vs_global_clean.png",dpi=300)
    plt.close(fig)

    # 2) parking02 TFP + Muñoz representative mid-MAD, compact two-panel.
    p=d[(d.sequence=="parking02") & (d.method.isin(["Twin V1","Twin V2"]))].copy()
    pos=p[(p.domain=="position_xy") & np.isclose(p.mad.astype(float),0.5)]
    head=p[(p.domain=="heading") & np.isclose(p.mad.astype(float),5.0)]
    methods=["Twin V1","Twin V2"]
    vals=[]
    for m in methods:
        b=base[(base.method==m)&(base.sequence=="parking02")].iloc[0]
        pp=pos[pos.method==m]; hh=head[head.method==m]
        vals.append({"method":m,"RPE10":b.rpe10_m,"ATE":b.ate_m,
                     "Dp95":b.dp_p95_m,"Dtheta95":b.dtheta_p95_deg,
                     "Muñoz position %MS":float(pp.pct_matched.iloc[0]) if len(pp) else np.nan,
                     "Muñoz heading %MS":float(hh.pct_matched.iloc[0]) if len(hh) else np.nan})
    V=pd.DataFrame(vals)
    fig,ax=plt.subplots(figsize=(5.2,3.5))
    x=np.arange(len(methods)); w=.24
    ax.bar(x-w,V["ATE"],width=w,label="ATE (m)")
    ax.bar(x,V["Dp95"],width=w,label="Dp p95 (m)")
    ax.bar(x+w,V["Dtheta95"],width=w,label="Dtheta p95 (deg)")
    ax.set_xticks(x,methods)
    ax.set_ylabel("Global/tail magnitude")
    ax.set_title("parking02: global divergence despite low RPE10")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"parking02_tfp_global_tail.png",dpi=300)
    plt.close(fig)

    V.to_csv(out/"parking02_figure_values.csv",index=False)
    print(f"Wrote figures to {out}")

if __name__=="__main__":
    main()
