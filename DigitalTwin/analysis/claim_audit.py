#!/usr/bin/env python3
"""Audit manuscript numerical/terminology claims against frozen result tables.

This deliberately flags unresolved official-benchmark discrepancies rather
than rewriting the manuscript to match whichever number is convenient.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

def first_existing(paths):
    for p in paths:
        p=Path(p)
        if p.exists(): return p
    return None

def find_tex(explicit):
    if explicit:
        p=Path(explicit)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    candidates=sorted(Path(".").glob("*.tex"))
    preferred=[p for p in candidates if "iotj" in p.name.lower() and "fidelity" in p.name.lower()]
    if preferred:return preferred[-1]
    if len(candidates)==1:return candidates[0]
    raise FileNotFoundError("Use --tex to specify the manuscript .tex file")

def approx_present(text,val,tol_digits=3):
    variants={f"{val:.3f}",f"{val:.4f}",f"{val:.2f}"}
    return any(v in text for v in variants)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tex")
    ap.add_argument("--output-root",default="results/publication_hardening")
    a=ap.parse_args()
    tex=find_tex(a.tex); text=tex.read_text(encoding="utf-8",errors="ignore")
    rows=[]

    tfp=first_existing([
      "results/i2nav_fidelity_baselines/tfp/tfp_dataset_summary.csv",
      "results/i2nav_fidelity_baselines/tfp_dataset_summary.csv",
    ])
    if not tfp:
        hits=sorted(Path("results").rglob("tfp_dataset_summary.csv"))
        tfp=hits[0] if hits else None
    if tfp:
        d=pd.read_csv(tfp)
        for method in ["Twin V1","Twin V2"]:
            q=d[d.method==method]
            if len(q):
                r=q.iloc[0]
                for col,label in [("ate_m","ATE"),("heading_mae_deg","heading MAE"),
                                  ("rpe1_m","RPE1"),("rpe5_m","RPE5"),("rpe10_m","RPE10")]:
                    val=float(r[col])
                    rows.append({"claim":f"{method} {label}","source":str(tfp),
                                 "expected_value":val,
                                 "status":"PASS" if approx_present(text,val) else "CHECK",
                                 "note":"Value should appear consistently in manuscript if it is a headline claim."})

    evo=first_existing([
      "results/i2nav_fidelity_baselines/validation/i2nav_evo_ape_summary.csv",
      "results/publication_hardening/i2nav_evo_ape_summary.csv",
    ])
    if evo:
        e=pd.read_csv(evo)
        q=e[e.method=="Twin V2"]
        if len(q):
            macro=float(q.iloc[0]["macro_mean_of_sequence_rmse_m"])
            frozen=float(q.iloc[0]["target_headline_m"])
            delta=abs(macro-frozen)
            rows.append({
              "claim":"Official V2 aligned APE headline provenance",
              "source":str(evo),"expected_value":f"recomputed macro={macro:.6f}; frozen headline={frozen:.6f}",
              "status":"UNRESOLVED" if delta>0.01 else "PASS",
              "note":"Do not silently relabel one aggregation as the other. Audit original official benchmark source/preprocessing/aggregation."
            })

    outdated = "No equivalent frozen V1 export is available" in text
    rows.append({
      "claim":"Frozen V1 availability wording","source":str(tex),"expected_value":"30 matched V1/V2 pairs exist",
      "status":"FAIL" if outdated else "PASS",
      "note":"30/30 sequence-replicate pairs were protocol-validated; remove obsolete no-V1-export sentence if still present."
    })

    for phrase,why in [
      ("universal superiority","Avoid universal superiority over trace alignment."),
      ("state-of-the-art","Only retain if explicitly negated/qualified."),
      ("SOTA","Only retain if explicitly negated/qualified."),
    ]:
        count=text.lower().count(phrase.lower())
        rows.append({"claim":f"Risk phrase: {phrase}","source":str(tex),"expected_value":count,
                     "status":"REVIEW" if count else "PASS","note":why})

    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    R=pd.DataFrame(rows)
    R.to_csv(out/"claim_audit.csv",index=False)
    lines=["# Manuscript claim audit","",f"Manuscript: `{tex}`",""]
    for _,r in R.iterrows():
        lines.append(f"- **{r.status} — {r['claim']}**: {r['note']} Source: `{r['source']}`.")
    (out/"claim_audit.md").write_text("\n".join(lines),encoding="utf-8")
    print(R.to_string(index=False))
    print(f"\nWrote {out/'claim_audit.csv'}")

if __name__=="__main__":
    main()
