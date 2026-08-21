#!/usr/bin/env python3
"""Compare conventional, Bergs-style, Muñoz-style, and TFP evaluations.

No scalar 'winner' is computed. The purpose is convergent validity plus
identification of diagnostic information that aggregate/geometry/aligned
metrics compress or intentionally align away.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .fidelity_common import spearman


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--tfp",default="results/i2nav_fidelity_baselines/tfp/tfp_per_sequence.csv")
    p.add_argument("--munoz",default="results/i2nav_fidelity_baselines/munoz/munoz_per_sequence.csv")
    p.add_argument("--bergs",default="results/i2nav_fidelity_baselines/bergs/bergs_per_sequence.csv")
    p.add_argument("--output",default="results/i2nav_fidelity_baselines/comparison")
    p.add_argument("--repo-root",default=".")
    return p.parse_args()


def rp(repo:Path,p:str)->Path:
    q=Path(p); return q if q.is_absolute() else repo/q


def rank_table(tfp:pd.DataFrame,bergs:pd.DataFrame,munoz:pd.DataFrame)->pd.DataFrame:
    # Dataset means after physical-sequence aggregation.
    t=tfp.groupby("method")[["ate_m","heading_mae_deg","rpe1_m","rpe5_m","rpe10_m","dp_p95_m","dtheta_p95_deg","accum_yaw_residual_absmax_deg"]].mean()
    b=bergs.groupby("method")[["hausdorff_m","mean_bidirectional_nearest_m","terminal_position_error_m"]].mean()
    # Middle MAD from each domain for one descriptive ranking column only; full sensitivity remains in long CSVs.
    mm=[]
    for dom,g in munoz.groupby("domain"):
        mads=sorted(g.mad.unique()); chosen=mads[len(mads)//2]
        q=g[g.mad==chosen].groupby("method")[["pct_matched"]].mean().rename(columns={"pct_matched":f"munoz_{dom}_pct_matched_midMAD"})
        mm.append(q)
    z=t.join(b,how="outer")
    for q in mm: z=z.join(q,how="outer")
    z=z.reset_index()
    for c in z.columns:
        if c=="method": continue
        ascending=not c.endswith("pct_matched_midMAD")
        z[c+"_rank"]=z[c].rank(method="min",ascending=ascending)
    return z


def correlations(tfp,munoz,bergs):
    rows=[]
    bmerge=bergs.drop(columns=["ate_m"],errors="ignore")
    tb=tfp.merge(bmerge,on=["method","sequence"],how="inner")
    position_pairs=[
        ("ate_m","hausdorff_m"),("ate_m","mean_bidirectional_nearest_m"),("rpe10_m","hausdorff_m"),("dp_p95_m","hausdorff_m"),
    ]
    for scope,g in [("ALL_METHOD_SEQUENCE_POINTS",tb)]+[(m,q) for m,q in tb.groupby("method")]:
        for a,b in position_pairs:
            rows.append({"scope":scope,"evaluator_a":"TFP","metric_a":a,"evaluator_b":"Bergs-style","metric_b":b,"spearman":spearman(g[a],g[b]),"n":len(g[[a,b]].dropna())})
    tm=tfp.merge(munoz,on=["method","sequence"],how="inner")
    for (dom,mad),g0 in tm.groupby(["domain","mad"]):
        scopes=[("ALL_METHOD_SEQUENCE_POINTS",g0)]+[(m,q) for m,q in g0.groupby("method")]
        if dom=="position_xy":
            pairs=[("ate_m","pct_matched"),("rpe10_m","pct_matched"),("dp_p95_m","pct_matched"),("ate_m","ed_mean_m"),("dp_p95_m","fd_max_m")]
        else:
            pairs=[("heading_mae_deg","pct_matched"),("dtheta_p95_deg","pct_matched"),("heading_mae_deg","ed_mean_deg"),("dtheta_p95_deg","fd_max_deg")]
        for scope,g in scopes:
            for a,b in pairs:
                if a in g and b in g:
                    rows.append({"scope":scope,"domain":dom,"mad":mad,"evaluator_a":"TFP","metric_a":a,"evaluator_b":"Munoz-style","metric_b":b,"spearman":spearman(g[a],g[b]),"n":len(g[[a,b]].dropna())})
    return pd.DataFrame(rows)


def main():
    a=parse_args(); repo=Path(a.repo_root).resolve(); out=rp(repo,a.output); out.mkdir(parents=True,exist_ok=True)
    tfp=pd.read_csv(rp(repo,a.tfp)); munoz=pd.read_csv(rp(repo,a.munoz)); bergs=pd.read_csv(rp(repo,a.bergs))
    ranks=rank_table(tfp,bergs,munoz); ranks.to_csv(out/"method_rankings.csv",index=False)
    corr=correlations(tfp,munoz,bergs); corr.to_csv(out/"evaluator_rank_correlations.csv",index=False)

    # Long combined table retains all MAD sensitivity points.
    combined=tfp.merge(bergs.drop(columns=["ate_m"],errors="ignore"),on=["method","sequence"],how="outer").merge(munoz,on=["method","sequence"],how="outer")
    combined.to_csv(out/"combined_sequence_metrics_long.csv",index=False)
    combined[combined.sequence=="parking02"].to_csv(out/"parking02_framework_comparison.csv",index=False)

    # Find local-good/global-bad sequences within every method and attach the middle-MAD Muñoz / Bergs diagnostics.
    flags=[]
    for method,g in tfp.groupby("method"):
        q=g.copy(); q["rpe10_rank_pct"]=q.rpe10_m.rank(pct=True); q["ate_rank_pct"]=q.ate_m.rank(pct=True)
        q["local_good_global_bad"]=(q.rpe10_rank_pct<=.5)&(q.ate_rank_pct>.5); flags.append(q)
    flags=pd.concat(flags,ignore_index=True)
    pos=munoz[munoz.domain=="position_xy"]
    if len(pos):
        mads=sorted(pos.mad.unique()); mid=mads[len(mads)//2]
        pm=pos[pos.mad==mid][["method","sequence","mad","pct_matched","ed_mean_m","fd_max_m","pct_gaps"]].rename(columns={"mad":"munoz_position_mad"})
        flags=flags.merge(pm,on=["method","sequence"],how="left")
    flags=flags.merge(bergs[["method","sequence","hausdorff_m","mean_bidirectional_nearest_m","terminal_position_error_m"]],on=["method","sequence"],how="left")
    flags.to_csv(out/"framework_disagreements.csv",index=False)

    # Diagnostic plots use matplotlib defaults; no metric is collapsed into a scalar score.
    try:
        fig,ax=plt.subplots(figsize=(7,5))
        for method,g in tfp.groupby("method"):
            ax.scatter(g.rpe10_m,g.ate_m,label=method)
            for _,r in g.iterrows():
                if r.sequence.startswith("parking"):
                    ax.annotate(r.sequence,(r.rpe10_m,r.ate_m),fontsize=7)
        ax.set_xlabel("RPE10 (m) — local error"); ax.set_ylabel("ATE RMSE (m) — global error"); ax.set_title("Local vs global fidelity across methods"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out/"local_vs_global_all_methods.png",dpi=220); plt.close(fig)

        if len(pos):
            g=tfp.merge(pm,on=["method","sequence"],how="inner")
            fig,ax=plt.subplots(figsize=(7,5))
            for method,q in g.groupby("method"):
                ax.scatter(q.ate_m,q.pct_matched,label=method)
            ax.set_xlabel("ATE RMSE (m)"); ax.set_ylabel(f"Muñoz-style matched snapshots (%) @ MAD={mid:g} m"); ax.set_title("Synchronized global error vs aligned trace match"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(out/"ate_vs_munoz_match_all_methods.png",dpi=220); plt.close(fig)
    except Exception as exc:
        print(f"Plot warning: {exc}")

    bad=flags[flags.local_good_global_bad]
    report=[
        "# Fidelity-framework comparison", "",
        "This report intentionally does not define a scalar winner. Conventional pose/RPE, Bergs-style path geometry, Muñoz-style aligned behavior, and TFP answer overlapping but non-identical questions.", "",
        "## Method-level descriptive ranking table", "", "```", ranks.to_string(index=False), "```", "",
        "## Local-good / global-bad cases exposed by synchronized TFP", "", "```", bad.to_string(index=False), "```", "",
        "## Publication-safe interpretation", "",
        "Use agreement between evaluators as convergent validity. Claim extra diagnostic resolution for TFP only where component, timescale, tail, persistence, or condition information changes the interpretation. Do not claim universal superiority over Muñoz; trace alignment is intentionally stronger when behavior may be temporally shifted yet equivalent. Bergs/Hausdorff is intentionally geometry-focused and time-agnostic.",
    ]
    (out/"fidelity_framework_comparison_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Wrote {out}")

if __name__=="__main__": main()
