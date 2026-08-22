#!/usr/bin/env python3
"""Publication-grade sequence-level paired statistics for Twin V1 vs Twin V2.

Primary statistical unit: held-out physical sequence.
Never treats timestamps or repeated seeds as independent physical samples.

Outputs:
  publication_sequence_statistics.csv
  publication_sequence_paired_values.csv
  publication_sequence_statistics.tex
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

DEFAULT_METRICS = [
    "ate_m","heading_mae_deg","rpe1_m","rpe5_m","rpe10_m",
    "dp_p95_m","dtheta_p95_deg","accum_yaw_residual_absmax_deg",
    "hausdorff_m","mean_bidirectional_nearest_m","terminal_position_error_m",
]
LABELS = {
    "ate_m":"ATE RMSE (m)",
    "heading_mae_deg":r"Heading MAE ($^\circ$)",
    "rpe1_m":"RPE1 (m)",
    "rpe5_m":"RPE5 (m)",
    "rpe10_m":"RPE10 (m)",
    "dp_p95_m":r"$D_{p,95}$ (m)",
    "dtheta_p95_deg":r"$D_{\theta,95}$ ($^\circ$)",
    "accum_yaw_residual_absmax_deg":r"$|I_\omega|_{\max}$ ($^\circ$)",
    "hausdorff_m":"Hausdorff (m)",
    "mean_bidirectional_nearest_m":"Mean bidirectional nearest (m)",
    "terminal_position_error_m":"Terminal position error (m)",
}

def locate(explicit):
    if explicit:
        p=Path(explicit)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    hits = [
        Path("results/i2nav_fidelity_baselines/comparison/combined_sequence_metrics_long.csv"),
        Path("results/i2nav_fidelity_baselines/combined_sequence_metrics_long.csv"),
    ]
    for p in hits:
        if p.exists(): return p
    found=sorted(Path("results").rglob("combined_sequence_metrics_long.csv"))
    if len(found)==1: return found[0]
    raise FileNotFoundError("Use --input to specify combined_sequence_metrics_long.csv")

def rb_effect(diff):
    nz=np.asarray(diff,float)
    nz=nz[np.isfinite(nz) & (nz!=0)]
    if not len(nz): return 0.0
    r=rankdata(np.abs(nz))
    pos=float(r[nz>0].sum()); neg=float(r[nz<0].sum())
    return (pos-neg)/float(r.sum())

def bci(x, rng, nboot, func):
    x=np.asarray(x,float)
    vals=np.empty(nboot,float)
    for i in range(nboot):
        vals[i]=func(rng.choice(x,size=len(x),replace=True))
    return np.quantile(vals,[.025,.975])

def q25(x): return float(np.quantile(x,.25))
def q75(x): return float(np.quantile(x,.75))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--method-a",default="Twin V1")
    ap.add_argument("--method-b",default="Twin V2")
    ap.add_argument("--metrics",default=",".join(DEFAULT_METRICS))
    ap.add_argument("--bootstrap",type=int,default=50000)
    ap.add_argument("--bootstrap-seed",type=int,default=20260821)
    ap.add_argument("--output-root",default="results/publication_hardening")
    a=ap.parse_args()

    src=locate(a.input)
    d=pd.read_csv(src)
    metrics=[m.strip() for m in a.metrics.split(",") if m.strip()]
    missing=[m for m in metrics if m not in d.columns]
    if missing: raise SystemExit(f"Missing metrics: {missing}")

    # Muñoz sensitivity rows repeat the same TFP values. Collapse them.
    base=d[["method","sequence"]+metrics].drop_duplicates(["method","sequence"])
    A=base[base.method==a.method_a].set_index("sequence").sort_index()
    B=base[base.method==a.method_b].set_index("sequence").sort_index()
    seq=A.index.intersection(B.index)
    if len(seq)<3: raise SystemExit(f"Only {len(seq)} paired sequences found")
    A=A.loc[seq]; B=B.loc[seq]
    rng=np.random.default_rng(a.bootstrap_seed)

    rows=[]; long=[]
    for m in metrics:
        x=A[m].to_numpy(float); y=B[m].to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)
        x=x[ok]; y=y[ok]; ss=np.asarray(seq)[ok]
        diff=y-x
        pct=np.where(np.abs(x)>1e-15,100*diff/np.abs(x),np.nan)
        ci_mean=bci(diff,rng,a.bootstrap,np.mean)
        ci_med=bci(diff,rng,a.bootstrap,np.median)
        try:
            w=wilcoxon(y,x,alternative="two-sided",zero_method="wilcox")
            wp=float(w.pvalue)
        except ValueError:
            wp=np.nan
        improved=int(np.sum(y<x))  # all listed metrics are lower-better
        rows.append({
            "metric":m,"label":LABELS.get(m,m),"n_sequences":len(x),
            "v1_mean":float(np.mean(x)),"v2_mean":float(np.mean(y)),
            "v1_median":float(np.median(x)),"v2_median":float(np.median(y)),
            "v1_sd":float(np.std(x,ddof=1)),"v2_sd":float(np.std(y,ddof=1)),
            "v1_iqr":q75(x)-q25(x),"v2_iqr":q75(y)-q25(y),
            "mean_paired_difference_v2_minus_v1":float(np.mean(diff)),
            "median_paired_difference_v2_minus_v1":float(np.median(diff)),
            "mean_relative_change_pct":float(np.nanmean(pct)),
            "median_relative_change_pct":float(np.nanmedian(pct)),
            "mean_difference_ci_low":float(ci_mean[0]),
            "mean_difference_ci_high":float(ci_mean[1]),
            "median_difference_ci_low":float(ci_med[0]),
            "median_difference_ci_high":float(ci_med[1]),
            "wilcoxon_two_sided_p":wp,
            "matched_pairs_rank_biserial_v2_minus_v1":rb_effect(diff),
            "v2_improved_sequences":improved,
            "v2_worse_sequences":int(np.sum(y>x)),
            "ties":int(np.sum(y==x)),
        })
        for s,aa,bb,dd,pp in zip(ss,x,y,diff,pct):
            long.append({"metric":m,"sequence":s,"v1":aa,"v2":bb,
                         "difference_v2_minus_v1":dd,"relative_change_pct":pp})

    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    R=pd.DataFrame(rows); L=pd.DataFrame(long)
    R.to_csv(out/"publication_sequence_statistics.csv",index=False)
    L.to_csv(out/"publication_sequence_paired_values.csv",index=False)

    tex=[]
    tex += [r"\begin{table*}[t]",
            r"\caption{Sequence-level paired statistics for frozen Twin V1 versus Twin V2. The physical sequence is the statistical unit ($n=10$). Negative paired differences favor V2 because all reported errors are lower-better.}",
            r"\label{tab:paired_stats}",
            r"\centering\scriptsize",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Metric & V1 med. & V2 med. & Med. $\Delta$ & 95\% CI of mean $\Delta$ & Wilcoxon $p$ & Improved \\",
            r"\midrule"]
    for _,r in R.iterrows():
        tex.append(
            f"{r['label']} & {r.v1_median:.4g} & {r.v2_median:.4g} & "
            f"{r.median_paired_difference_v2_minus_v1:.4g} & "
            f"[{r.mean_difference_ci_low:.4g},{r.mean_difference_ci_high:.4g}] & "
            f"{r.wilcoxon_two_sided_p:.4g} & {int(r.v2_improved_sequences)}/{int(r.n_sequences)} \\\\"
        )
    tex += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (out/"publication_sequence_statistics.tex").write_text("\n".join(tex),encoding="utf-8")

    print(f"Input: {src}")
    print(R[["metric","n_sequences","v1_median","v2_median",
             "mean_difference_ci_low","mean_difference_ci_high",
             "wilcoxon_two_sided_p","matched_pairs_rank_biserial_v2_minus_v1",
             "v2_improved_sequences"]].to_string(index=False))
    print(f"\nWrote {out}")

if __name__=="__main__":
    main()
