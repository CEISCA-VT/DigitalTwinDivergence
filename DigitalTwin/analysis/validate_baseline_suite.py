#!/usr/bin/env python3
"""Protocol and output audit for the external-baseline / fidelity suite."""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import load_pose_trajectory

V2_FROZEN = {
    "ate_m": 2.398,
    "heading_mae_deg": 2.569,
    "rpe1_m": 0.0611,
    "rpe5_m": 0.1603,
    "rpe10_m": 0.2532,
}


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",default=".")
    p.add_argument("--manifest",default="results/i2nav_fidelity_baselines/trajectory_manifest.csv")
    p.add_argument("--generated-manifest",default="results/i2nav_external_baselines/baseline_manifest.csv")
    p.add_argument("--tfp-summary",default="results/i2nav_fidelity_baselines/tfp/tfp_dataset_summary.csv")
    p.add_argument("--output",default="results/i2nav_fidelity_baselines/validation")
    p.add_argument("--v2-tolerance-pct",type=float,default=1.0)
    return p.parse_args()


def rp(repo,p):
    q=Path(p); return q if q.is_absolute() else repo/q


def main():
    a=parse_args(); repo=Path(a.repo_root).resolve(); out=rp(repo,a.output); out.mkdir(parents=True,exist_ok=True)
    checks=[]
    def check(name,passed,detail,critical=True):
        checks.append({"check":name,"passed":bool(passed),"critical":critical,"detail":detail})
        print(("PASS" if passed else "FAIL")+f" | {name} | {detail}")

    mf=rp(repo,a.manifest)
    if not mf.exists():
        check("trajectory_manifest_exists",False,str(mf)); pd.DataFrame(checks).to_csv(out/"validation_checks.csv",index=False); raise SystemExit(2)
    m=pd.read_csv(mf)
    check("manifest_nonempty",len(m)>0,f"rows={len(m)}")
    for method,g in m.groupby("method"):
        check(f"{method}:sequence_coverage",g.sequence.nunique()>=1,f"sequences={g.sequence.nunique()}, files={len(g)}",critical=False)
        missing=[]; bad_time=[]; bad_pose=[]
        for p in g.trajectory.astype(str):
            q=rp(repo,p)
            if not q.exists(): missing.append(str(q)); continue
            try:
                d=load_pose_trajectory(q)
                t=d.time_s.to_numpy(float)
                if not np.all(np.diff(t)>0): bad_time.append(str(q))
                pose=d[["gt_east_m","gt_north_m","gt_heading_rad","estimate_east_m","estimate_north_m","estimate_heading_rad"]].to_numpy(float)
                if not np.isfinite(pose).all(): bad_pose.append(str(q))
            except Exception as exc: bad_pose.append(f"{q}: {exc}")
        check(f"{method}:files_exist",not missing,f"missing={len(missing)}")
        check(f"{method}:monotonic_time",not bad_time,f"bad={len(bad_time)}")
        check(f"{method}:finite_pose",not bad_pose,f"bad={len(bad_pose)}")

    gm=rp(repo,a.generated_manifest)
    if gm.exists():
        g=pd.read_csv(gm)
        ok=g[g.status=="ok"] if "status" in g.columns else g
        leaks=[]
        for _,r in ok.iterrows():
            train={x for x in str(r.get("train_sequences","")).split(";") if x}
            test=str(r.get("test_sequence",r.get("sequence","")))
            if test in train: leaks.append((r.get("method"),test,r.get("seed")))
        check("generated_LOSO_no_test_in_train",not leaks,f"leaks={leaks[:10]}")
        errors=g[g.status!="ok"] if "status" in g.columns else pd.DataFrame()
        check("generated_jobs_completed",len(errors)==0,f"failed_jobs={len(errors)}")
    else:
        check("generated_manifest_present",False,f"not found: {gm}",critical=False)

    ts=rp(repo,a.tfp_summary)
    if ts.exists():
        s=pd.read_csv(ts)
        v=s[s.method=="Twin V2"]
        if len(v)==1:
            r=v.iloc[0]
            for metric,expected in V2_FROZEN.items():
                got=float(r[metric]); rel=100*abs(got-expected)/max(abs(expected),1e-12)
                check(f"V2_frozen_{metric}",rel<=a.v2_tolerance_pct,f"got={got:.6g}, expected={expected:.6g}, rel={rel:.3f}%")
        else:
            check("V2_frozen_summary_present",False,f"Twin V2 rows={len(v)}",critical=False)
    else:
        check("tfp_summary_present",False,f"not found: {ts}",critical=False)

    c=pd.DataFrame(checks); c.to_csv(out/"validation_checks.csv",index=False)
    critical_fail=c[(c.critical==True)&(c.passed==False)]
    report=["# Baseline-suite validation","",f"Critical failures: **{len(critical_fail)}**","","```",c.to_string(index=False),"```",""]
    (out/"validation_report.md").write_text("\n".join(report),encoding="utf-8")
    if len(critical_fail):
        raise SystemExit(2)
    print(f"Validation passed. Wrote {out}")

if __name__=="__main__": main()
