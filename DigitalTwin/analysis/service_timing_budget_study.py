"""Held-out service timing-budget and delivery-remediability study.

Uses only frozen i2Nav V2 trajectories. No training or modification of the twin.
The compact predictor is trained inside each leave-one-physical-sequence-out fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/i2nav_v2_full_loso/i2nav_v2_full_loso"
OUT = ROOT / "results/service_timing_budget"
RATES = (10.0, 5.0, 2.0, 1.0)
DELAYS_MS = (0, 25, 50, 100, 200)
TARGET = 0.80
MIN_COVERAGE = 0.95
PREFIX_S = 30.0
THRESHOLDS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
MATCHED_ACCEPTANCE_TARGETS = (0.25, 0.30, 0.35, 0.40)
PRIMARY_MATCHED_ACCEPTANCE = 0.30
SERVICES = (
    {"service":"local_1s", "family":"local", "horizon_s":1.0, "pos_tol_m":0.1, "heading_tol_deg":2.0, "aoi_limit_s":0.6},
    {"service":"local_5s", "family":"local", "horizon_s":5.0, "pos_tol_m":0.2, "heading_tol_deg":5.0, "aoi_limit_s":1.0},
    {"service":"local_10s", "family":"local", "horizon_s":10.0, "pos_tol_m":0.5, "heading_tol_deg":10.0, "aoi_limit_s":1.5},
    {"service":"global", "family":"global", "horizon_s":0.0, "pos_tol_m":1.0, "heading_tol_deg":5.0, "aoi_limit_s":1.0},
)
NUMERIC_FEATURES = ["rate_hz", "delay_ms", "horizon_s", "pos_tol_m", "heading_tol_deg", "aoi_limit_s",
                    "speed_mean", "speed_p95", "yaw_abs_mean", "yaw_abs_p95", "accel_abs_p95", "turn_fraction"]


def wrap(x): return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi


def discover():
    p = sorted(SOURCE.glob("replicate_*/fold_*/v2_evaluated_trajectory.csv"))
    if len(p) != 30: raise RuntimeError(f"Expected 30 frozen trajectories, found {len(p)}")
    return p


def load(path):
    d = pd.read_csv(path)
    needed = ["time_s","gt_east_m","gt_north_m","gt_heading_rad","estimate_east_m","estimate_north_m",
              "estimate_heading_rad","odo_speed_mps","imu_yaw_rate_radps"]
    if not set(needed).issubset(d): raise ValueError(f"Missing columns: {path}")
    a = {k:d[k].to_numpy(dtype=float, copy=True) for k in needed}
    a["time_s"] -= a["time_s"][0]
    if not np.all(np.diff(a["time_s"]) > 0): raise ValueError(f"Nonmonotone time: {path}")
    if abs(np.median(np.diff(a["time_s"])) - .1) > .005: raise ValueError(f"Not 10 Hz: {path}")
    return a


def relative(x, y, th, i, j):
    dx, dy = x[j]-x[i], y[j]-y[i]; c, s = np.cos(th[i]), np.sin(th[i])
    return c*dx+s*dy, -s*dx+c*dy, wrap(th[j]-th[i])


def delivered_indices(t, rate, delay_s):
    stride = int(round(10/rate))
    src = np.arange(0, len(t), stride)
    clock_ns = np.rint(np.asarray(t) * 1_000_000_000).astype(np.int64)
    arrivals = clock_ns[src] + int(round(delay_s * 1_000_000_000))
    p = np.searchsorted(arrivals, clock_ns, side="right") - 1
    idx = np.full(len(t), -1, int); valid = p >= 0; idx[valid] = src[p[valid]]
    return idx


def episodes(mask, dt=.1):
    if not len(mask): return (0, 0.0, 0.0)
    z = np.r_[False, mask, False].astype(int); edges = np.diff(z)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = (ends-starts)*dt
    return len(lengths), float(lengths.sum()), float(lengths.max()) if len(lengths) else 0.0


def evaluate(a, service, rate, delay_ms, start_s=PREFIX_S, end_s=None):
    t = a["time_s"]; idx = delivered_indices(t, rate, delay_ms/1000)
    horizon_n = int(round(service["horizon_s"]*10))
    eligible = np.arange(len(t)) >= int(round(start_s*10)) + horizon_n
    if end_s is not None: eligible &= np.arange(len(t)) < int(round(end_s*10))
    observable = eligible & (idx >= 0)
    if horizon_n: observable &= np.r_[np.zeros(horizon_n,bool), idx[:-horizon_n] >= 0]
    pos = np.full(len(t), np.nan); head = np.full(len(t), np.nan)
    use = np.flatnonzero(observable)
    if service["family"] == "global":
        k = idx[use]
        pos[use] = np.hypot(a["estimate_east_m"][k]-a["gt_east_m"][use], a["estimate_north_m"][k]-a["gt_north_m"][use])
        head[use] = np.abs(np.rad2deg(wrap(a["estimate_heading_rad"][k]-a["gt_heading_rad"][use])))
    else:
        i = use-horizon_n; ki, kj = idx[i], idx[use]
        gp = relative(a["gt_east_m"],a["gt_north_m"],a["gt_heading_rad"],i,use)
        ep = relative(a["estimate_east_m"],a["estimate_north_m"],a["estimate_heading_rad"],ki,kj)
        pos[use] = np.hypot(ep[0]-gp[0], ep[1]-gp[1])
        head[use] = np.abs(np.rad2deg(wrap(ep[2]-gp[2])))
    age = np.full(len(t), np.inf)
    clock_ns = np.rint(t * 1_000_000_000).astype(np.int64)
    age[use] = (clock_ns[use] - clock_ns[idx[use]]) / 1_000_000_000
    physical = observable & (pos <= service["pos_tol_m"]) & (head <= service["heading_tol_deg"])
    fresh = observable & (age <= service["aoi_limit_s"])
    joint = physical & fresh
    denom = max(1, int(observable.sum()))
    fail_count, fail_s, fail_max = episodes(observable & ~joint)
    return {"coverage":float(observable.sum()/max(1,eligible.sum())),
            "physical_satisfaction":float(physical.sum()/denom), "freshness_satisfaction":float(fresh.sum()/denom),
            "joint_satisfaction":float(joint.sum()/denom), "failure_episodes":fail_count,
            "failure_duration_s":fail_s, "max_failure_episode_s":fail_max,
            "mean_position_discrepancy_m":float(np.mean(pos[observable])) if observable.any() else np.nan,
            "p95_position_discrepancy_m":float(np.quantile(pos[observable],.95)) if observable.any() else np.nan,
            "mean_heading_discrepancy_deg":float(np.mean(head[observable])) if observable.any() else np.nan,
            "p95_heading_discrepancy_deg":float(np.quantile(head[observable],.95)) if observable.any() else np.nan,
            "mean_aoi_s":float(np.mean(age[observable])) if observable.any() else np.nan,
            "max_aoi_s":float(np.max(age[observable])) if observable.any() else np.nan,
            "eligible_samples":int(eligible.sum()), "observable_samples":int(observable.sum())}


def motion_features(a):
    n = min(len(a["time_s"]), int(PREFIX_S*10)+1); speed=a["odo_speed_mps"][:n]; yaw=np.abs(a["imu_yaw_rate_radps"][:n])
    accel=np.abs(np.diff(speed)/.1)
    return {"speed_mean":float(np.mean(np.abs(speed))), "speed_p95":float(np.quantile(np.abs(speed),.95)),
            "yaw_abs_mean":float(np.mean(yaw)), "yaw_abs_p95":float(np.quantile(yaw,.95)),
            "accel_abs_p95":float(np.quantile(accel,.95)) if len(accel) else 0.0,
            "turn_fraction":float(np.mean(yaw>.1))}


def response_rows(paths):
    rows=[]
    for path in paths:
        a=load(path); seq=path.parent.name.split("_",2)[-1]; rep=path.parent.parent.name; feat=motion_features(a)
        for s in SERVICES:
            for rate in RATES:
                for delay in DELAYS_MS:
                    m=evaluate(a,s,rate,delay)
                    cal=evaluate(a,s,rate,delay,start_s=0.0,end_s=PREFIX_S)
                    qualified=(m["coverage"]>=MIN_COVERAGE and m["physical_satisfaction"]>=TARGET and
                               m["freshness_satisfaction"]>=TARGET and m["joint_satisfaction"]>=TARGET)
                    rows.append({"sequence":seq,"replicate":rep,"source":str(path.relative_to(ROOT)),**s,
                                 "rate_hz":rate,"interval_s":1/rate,"delay_ms":delay,**feat,**m,"qualified":int(qualified)})
                    rows[-1].update({f"calibration_{k}":cal[k] for k in ("coverage","physical_satisfaction","freshness_satisfaction","joint_satisfaction")})
    return pd.DataFrame(rows)


def sequence_rows(runs):
    keys=["sequence","service","family","horizon_s","pos_tol_m","heading_tol_deg","aoi_limit_s","rate_hz","interval_s","delay_ms"]
    metrics=["coverage","physical_satisfaction","freshness_satisfaction","joint_satisfaction","failure_duration_s","max_failure_episode_s",
             "mean_aoi_s","max_aoi_s","calibration_coverage","calibration_physical_satisfaction",
             "calibration_freshness_satisfaction","calibration_joint_satisfaction"]+NUMERIC_FEATURES[6:]
    metrics += [name for name in ("mean_position_discrepancy_m","p95_position_discrepancy_m",
                                  "mean_heading_discrepancy_deg","p95_heading_discrepancy_deg")
                if name in runs.columns]
    d=runs.groupby(keys,as_index=False)[metrics].mean()
    d["qualified"]=((d.coverage>=MIN_COVERAGE)&(d.physical_satisfaction>=TARGET)&
                    (d.freshness_satisfaction>=TARGET)&(d.joint_satisfaction>=TARGET)).astype(int)
    return d


def probability_models(seq):
    rows=[]; sequences=sorted(seq.sequence.unique())
    categorical=["service"]
    pre=ColumnTransformer([("num",StandardScaler(),NUMERIC_FEATURES),("cat",OneHotEncoder(handle_unknown="ignore"),categorical)])
    for held in sequences:
        train=seq[seq.sequence!=held].copy(); test=seq[seq.sequence==held].copy()
        common=train.groupby(["rate_hz","delay_ms"]).qualified.mean()
        empirical=train.groupby(["service","rate_hz","delay_ms"]).qualified.mean()
        X=train[NUMERIC_FEATURES+categorical]; y=train.qualified
        if y.nunique()>1:
            model=make_pipeline(pre,LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000,random_state=42))
            model.fit(X,y); proposed=model.predict_proba(test[NUMERIC_FEATURES+categorical])[:,1]
        else: proposed=np.full(len(test),float(y.iloc[0]))
        for j,(_,r) in enumerate(test.iterrows()):
            scores={"common":float(common.loc[(r.rate_hz,r.delay_ms)]),
                    # Positive headroom means the observed worst-case age remains below the service limit.
                    "aoi_only":float(np.round(r.aoi_limit_s-r.max_aoi_s,6)),
                    "service_empirical":float(empirical.loc[(r.service,r.rate_hz,r.delay_ms)]),
                    "motion_logistic":float(proposed[j])}
            for method,score in scores.items(): rows.append({"sequence":held,"service":r.service,"rate_hz":r.rate_hz,
                "delay_ms":r.delay_ms,"actual_qualified":int(r.qualified),"method":method,"score":score,
                "training_sequences":";".join(sorted(train.sequence.unique()))})
    return pd.DataFrame(rows)


def prediction_summary(pred):
    rows=[]
    for (seq,method),g in pred.groupby(["sequence","method"]):
        for th in THRESHOLDS:
            accept=g.score>=th; actual=g.actual_qualified.astype(bool)
            rows.append({"sequence":seq,"method":method,"threshold":th,"accepted":int(accept.sum()),"conditions":len(g),
                         "acceptance":float(accept.mean()),"false_qualification":float((accept&~actual).sum()/accept.sum()) if accept.any() else np.nan,
                         "unnecessary_restriction":float((~accept&actual).sum()/actual.sum()) if actual.any() else np.nan})
    per=pd.DataFrame(rows)
    macro=per.groupby(["method","threshold"],as_index=False).agg(
        acceptance=("acceptance","mean"),false_qualification=("false_qualification","mean"),
        unnecessary_restriction=("unnecessary_restriction","mean"),sequences=("sequence","nunique"))
    return per,macro


def _acceptance_threshold(scores, target):
    """Choose an operating point by acceptance only; labels are never consulted."""
    values = np.asarray(scores, dtype=float)
    candidates = np.r_[np.inf, np.unique(values)[::-1], -np.inf]
    ranked = []
    for threshold in candidates:
        acceptance = float(np.mean(values >= threshold))
        ranked.append((abs(acceptance-target), -float(threshold), float(threshold), acceptance))
    _, _, threshold, acceptance = min(ranked)
    return threshold, acceptance


def matched_acceptance_analysis(seq, pred, targets=MATCHED_ACCEPTANCE_TARGETS):
    """Nested LOSO comparison at acceptance selected inside each outer training fold."""
    methods = ("aoi_only", "service_empirical", "motion_logistic")
    rows=[]
    for held in sorted(seq.sequence.unique()):
        outer_train=seq[seq.sequence != held].copy()
        inner=probability_models(outer_train)
        outer=pred[pred.sequence == held]
        for target in targets:
            for method in methods:
                train_scores=inner[inner.method == method].score.to_numpy(float)
                threshold, training_acceptance=_acceptance_threshold(train_scores,target)
                test=outer[outer.method == method]
                accepted=test.score.to_numpy(float) >= threshold
                actual=test.actual_qualified.to_numpy(int).astype(bool)
                accepted_n=int(accepted.sum()); actual_n=int(actual.sum())
                false_n=int((accepted & ~actual).sum())
                restriction_n=int((~accepted & actual).sum())
                rows.append({"heldout_sequence":held,"method":method,"target_acceptance":target,
                    "selected_threshold":threshold,"training_acceptance":training_acceptance,
                    "accepted_count":accepted_n,"condition_count":len(test),
                    "false_qualified_count":false_n,"actual_qualified_count":actual_n,
                    "unnecessary_restriction_count":restriction_n,
                    "heldout_acceptance":accepted_n/len(test),
                    "false_qualification_rate":false_n/accepted_n if accepted_n else np.nan,
                    "unnecessary_restriction_rate":restriction_n/actual_n if actual_n else np.nan,
                    "operating_point_training_sequences":";".join(sorted(outer_train.sequence.unique()))})
    per=pd.DataFrame(rows)
    rng=np.random.default_rng(42); summaries=[]
    for (target,method),g in per.groupby(["target_acceptance","method"]):
        seqs=sorted(g.heldout_sequence.unique()); draws=[]
        for _ in range(5000):
            sampled=rng.choice(seqs,len(seqs),replace=True)
            h=pd.concat([g[g.heldout_sequence==s] for s in sampled],ignore_index=True)
            accepted=int(h.accepted_count.sum()); false=int(h.false_qualified_count.sum())
            draws.append((h.heldout_acceptance.mean(), false/accepted if accepted else np.nan))
        draws=np.asarray(draws,float)
        accepted=int(g.accepted_count.sum()); false=int(g.false_qualified_count.sum())
        summaries.append({"method":method,"target_acceptance":target,"n_sequences":len(seqs),
            "accepted_count":accepted,"condition_count":int(g.condition_count.sum()),
            "false_qualified_count":false,
            "achieved_acceptance":accepted/g.condition_count.sum(),
            "acceptance_sequence_macro":g.heldout_acceptance.mean(),
            "acceptance_ci_low":np.nanquantile(draws[:,0],.025),"acceptance_ci_high":np.nanquantile(draws[:,0],.975),
            "false_qualification":false/accepted if accepted else np.nan,
            "false_qualification_sequence_macro":g.false_qualification_rate.mean(),
            "false_qualification_ci_low":np.nanquantile(draws[:,1],.025),
            "false_qualification_ci_high":np.nanquantile(draws[:,1],.975)})
    summary=pd.DataFrame(summaries)
    pairs=[]; primary=per[np.isclose(per.target_acceptance,PRIMARY_MATCHED_ACCEPTANCE)]
    for first,second in (("service_empirical","aoi_only"),("service_empirical","motion_logistic"),("motion_logistic","aoi_only")):
        a=primary[primary.method==first].set_index("heldout_sequence")
        b=primary[primary.method==second].set_index("heldout_sequence")
        common=sorted(set(a.index)&set(b.index)); diffs=[]
        for _ in range(5000):
            sampled=rng.choice(common,len(common),replace=True)
            def pooled(frame):
                h=frame.loc[list(sampled)]; return h.false_qualified_count.sum()/h.accepted_count.sum()
            diffs.append(pooled(a)-pooled(b))
        observed=(a.false_qualified_count.sum()/a.accepted_count.sum())-(b.false_qualified_count.sum()/b.accepted_count.sum())
        pairs.append({"target_acceptance":PRIMARY_MATCHED_ACCEPTANCE,"first_method":first,"second_method":second,
            "false_qualification_difference_first_minus_second":observed,
            "ci_low":np.quantile(diffs,.025),"ci_high":np.quantile(diffs,.975),"n_sequences":len(common)})
    return per,summary,pd.DataFrame(pairs)


def _failure_component(row):
    if row.coverage < MIN_COVERAGE: return "unobservable"
    physical=row.physical_satisfaction >= TARGET
    fresh=row.freshness_satisfaction >= TARGET
    if physical and fresh: return "none"
    if not physical and fresh: return "physical_only"
    if physical and not fresh: return "freshness_only"
    return "physical_and_freshness"


def _contract_status(row):
    if row.coverage < MIN_COVERAGE: return "unobservable"
    return "qualified" if int(row.qualified) else "unqualified"


def remediability(seq,pred):
    key=["sequence","service"]
    def condition(rate,delay): return seq[(seq.rate_hz==rate)&(seq.delay_ms==delay)].set_index(key)
    ideal, degraded, feasible=condition(10,0),condition(2,200),condition(5,50)
    rows=[]
    for k in ideal.index:
        iq,dq,fq=int(ideal.loc[k].qualified),int(degraded.loc[k].qualified),int(feasible.loc[k].qualified)
        if min(ideal.loc[k].coverage,degraded.loc[k].coverage,feasible.loc[k].coverage)<MIN_COVERAGE: cls="unresolved"
        elif not iq: cls="persistent_under_ideal"
        elif not dq: cls="delivery_remediable"
        else: cls="already_qualified_degraded"
        ir,dr,fr=ideal.loc[k],degraded.loc[k],feasible.loc[k]
        row={"sequence":k[0],"service":k[1],"ideal_qualified":iq,"degraded_qualified":dq,
             "practical_qualified":fq,"classification":cls,"practical_restores":int(not dq and fq),
             "observability":int(min(ir.coverage,dr.coverage,fr.coverage)>=MIN_COVERAGE),
             "degraded_delivery_status":_contract_status(dr),"ideal_delivery_status":_contract_status(ir),
             "practical_intervention_status":_contract_status(fr),
             "degraded_failure_component":_failure_component(dr),
             "ideal_failure_component":_failure_component(ir),
             "practical_failure_component":_failure_component(fr)}
        for prefix,r in (("degraded",dr),("ideal",ir),("practical",fr)):
            for metric in ("coverage","physical_satisfaction","freshness_satisfaction","joint_satisfaction",
                           "mean_position_discrepancy_m","p95_position_discrepancy_m",
                           "mean_heading_discrepancy_deg","p95_heading_discrepancy_deg"):
                if metric not in r.index:
                    continue
                row[f"{prefix}_{metric}"]=float(r[metric])
        if "degraded_mean_position_discrepancy_m" in row:
            row["held_minus_ideal_mean_position_m"] = (
                row["degraded_mean_position_discrepancy_m"] - row["ideal_mean_position_discrepancy_m"]
            )
            row["held_minus_ideal_mean_heading_deg"] = (
                row["degraded_mean_heading_discrepancy_deg"] - row["ideal_mean_heading_discrepancy_deg"]
            )
        rows.append(row)
    out=pd.DataFrame(rows)
    # Any non-ideal qualification when ideal fails is recorded explicitly.
    nonideal=seq.merge(ideal.reset_index()[key+["qualified"]].rename(columns={"qualified":"ideal_q"}),on=key)
    odd=nonideal[(nonideal.ideal_q==0)&(nonideal.qualified==1)&~((nonideal.rate_hz==10)&(nonideal.delay_ms==0))]
    out.attrs["nonideal_qualifies_ideal_fails"]=len(odd)
    return out


def intervention_results(seq, pred, rem):
    keys=["sequence","service"]
    truth=rem.set_index(keys)
    selected=pred.copy()
    scores=selected.pivot_table(index=keys+["rate_hz","delay_ms"],columns="method",values="score").reset_index()
    degraded=scores[(scores.rate_hz==2)&(scores.delay_ms==200)].set_index(keys)
    feasible=scores[(scores.rate_hz==5)&(scores.delay_ms==50)].set_index(keys)
    dseq=seq[(seq.rate_hz==2)&(seq.delay_ms==200)].set_index(keys)
    fseq=seq[(seq.rate_hz==5)&(seq.delay_ms==50)].set_index(keys)
    rows=[]; methods=["always","aoi_only","service_empirical","motion_logistic","calibration_diagnostic"]
    opportunity=(truth.practical_restores==1)
    for method in methods:
        if method=="always": recommend=pd.Series(True,index=truth.index)
        elif method=="calibration_diagnostic":
            dpass=(dseq.calibration_coverage>=MIN_COVERAGE)&(dseq.calibration_joint_satisfaction>=TARGET)
            fpass=(fseq.calibration_coverage>=MIN_COVERAGE)&(fseq.calibration_joint_satisfaction>=TARGET)
            recommend=(~dpass)&fpass
        else: recommend=(degraded[method]<TARGET)&(feasible[method]>=TARGET)
        tp=int((recommend&opportunity).sum()); n=int(recommend.sum()); opp=int(opportunity.sum())
        rows.append({"method":method,"recommendations":n,"restoration_opportunities":opp,"true_recommendations":tp,
                     "precision":tp/n if n else np.nan,"recall":tp/opp if opp else np.nan,
                     "unnecessary_interventions":n-tp,"missed_opportunities":opp-tp})
    return pd.DataFrame(rows)


def recovery_component_summary(rem):
    rows=[]
    groups=(("delivery_remediable",rem.classification=="delivery_remediable"),
            ("practical_recoveries",rem.practical_restores==1))
    for population,mask in groups:
        h=rem[mask]
        for component,count in h.degraded_failure_component.value_counts().items():
            rows.append({"population":population,"failure_component":component,"cases":int(count),
                         "population_denominator":len(h),"fraction":count/len(h) if len(h) else np.nan})
    return pd.DataFrame(rows)


def plots(seq,macro,rem,out,intervention=None,matched=None):
    out.mkdir(parents=True,exist_ok=True)
    # Median response surfaces across sequences.
    fig,axes=plt.subplots(1,4,figsize=(15,3.5),sharey=True)
    for ax,s in zip(axes,[x["service"] for x in SERVICES]):
        g=seq[seq.service==s].groupby(["rate_hz","delay_ms"]).joint_satisfaction.median().unstack()
        im=ax.imshow(g.loc[list(RATES),list(DELAYS_MS)],vmin=0,vmax=1,aspect="auto",cmap="viridis")
        ax.set(title=s,xticks=range(len(DELAYS_MS)),xticklabels=DELAYS_MS,yticks=range(len(RATES)),yticklabels=RATES,
               xlabel="delay (ms)"); axes[0].set_ylabel("delivery rate (Hz)")
    fig.colorbar(im,ax=axes,label="median joint satisfaction",shrink=.8); fig.savefig(out/"timing_response_surfaces.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    for method,g in macro.groupby("method"):
        ax.plot(g.acceptance,g.false_qualification,marker="o",label=method)
        selected=g[np.isclose(g.threshold,TARGET)]
        if len(selected): ax.scatter(selected.acceptance,selected.false_qualification,marker="x",s=90,linewidths=2)
    ax.set(xlabel="accepted condition fraction",ylabel="false qualification",title="Held-out timing-budget risk versus coverage")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"heldout_risk_coverage.png",dpi=180); plt.close(fig)
    counts=rem.classification.value_counts().rename(index={"delivery_remediable":"Delivery-remediable",
        "persistent_under_ideal":"Persistent under ideal","already_qualified_degraded":"Already qualified"})
    fig,ax=plt.subplots(figsize=(6.5,3.8)); counts.plot.bar(ax=ax,color="#2878B5")
    ax.set(ylabel="sequence-service cases",title="Delivery remediability: 2 Hz/200 ms versus ideal delivery")
    ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(out/"delivery_remediability.png",dpi=180); plt.close(fig)
    if intervention is not None:
        fig,ax=plt.subplots(figsize=(7.2,4.0)); x=np.arange(len(intervention)); w=.36
        ax.bar(x-w/2,intervention.precision,w,label="precision",color="#2B7A78")
        ax.bar(x+w/2,intervention.recall,w,label="recall",color="#D95F02")
        ax.set(ylim=(0,1.05),ylabel="fraction",title="Prediction of restoration by 5 Hz / 50 ms delivery",
               xticks=x,xticklabels=[m.replace("_","\n") for m in intervention.method])
        ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(out/"intervention_performance.png",dpi=180); plt.close(fig)
    if matched is not None:
        fig,ax=plt.subplots(figsize=(7.2,4.2))
        for method,g in matched.groupby("method"):
            ax.errorbar(g.achieved_acceptance,g.false_qualification,
                yerr=[g.false_qualification-g.false_qualification_ci_low,g.false_qualification_ci_high-g.false_qualification],
                marker="o",capsize=3,label=method.replace("_"," "))
        ax.set(xlabel="held-out acceptance (explicit denominator: 800 cells per method)",
               ylabel="false qualifications / accepted cells",title="Matched-acceptance held-out comparison")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"matched_acceptance_comparison.png",dpi=180); plt.close(fig)


def bootstrap_selected(per):
    rng=np.random.default_rng(42); rows=[]
    g=per[per.threshold==TARGET]
    for method,h in g.groupby("method"):
        row={"method":method,"threshold":TARGET,"n_sequences":h.sequence.nunique()}
        for metric in ("acceptance","false_qualification","unnecessary_restriction"):
            x=h[metric].dropna().to_numpy(float); draw=rng.choice(x,(5000,len(x)),replace=True).mean(1)
            row[metric]=float(np.mean(x)); row[metric+"_ci_low"],row[metric+"_ci_high"]=np.quantile(draw,[.025,.975])
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_and_monotonicity(seq, pred):
    rate_order={1.0:0,2.0:1,5.0:2,10.0:3}; none_code=4
    rows=[]
    for (sequence,service,delay),g in pred.groupby(["sequence","service","delay_ms"]):
        actual=g.drop_duplicates(["rate_hz"])[["rate_hz","actual_qualified"]]
        q=actual[actual.actual_qualified==1].rate_hz
        true=min(q) if len(q) else np.nan; true_code=rate_order.get(true,none_code)
        for method,h in g.groupby("method"):
            accepted=h[h.score>=TARGET].rate_hz; predicted=min(accepted) if len(accepted) else np.nan
            pred_code=rate_order.get(predicted,none_code)
            rows.append({"sequence":sequence,"service":service,"delay_ms":delay,"method":method,
                         "true_min_rate_hz":true,"predicted_min_rate_hz":predicted,
                         "boundary_step_error":abs(pred_code-true_code)})
    boundaries=pd.DataFrame(rows)
    mono=[]
    for (sequence,service,rate),g in seq.groupby(["sequence","service","rate_hz"]):
        y=g.sort_values("delay_ms").joint_satisfaction.to_numpy(); mono.append({"sequence":sequence,"service":service,
            "axis":"delay","fixed_value":rate,"comparisons":len(y)-1,"violations":int(np.sum(np.diff(y)>1e-9))})
    for (sequence,service,delay),g in seq.groupby(["sequence","service","delay_ms"]):
        y=g.sort_values("rate_hz").joint_satisfaction.to_numpy(); mono.append({"sequence":sequence,"service":service,
            "axis":"rate","fixed_value":delay,"comparisons":len(y)-1,"violations":int(np.sum(np.diff(y)<-1e-9))})
    return boundaries,pd.DataFrame(mono)


def write_four_checks_report(out, matched, pairwise, rem, recovery, literature_path):
    primary=matched[np.isclose(matched.target_acceptance,PRIMARY_MATCHED_ACCEPTANCE)].sort_values("method")
    empirical_aoi=pairwise[(pairwise.first_method=="service_empirical")&(pairwise.second_method=="aoi_only")].iloc[0]
    empirical_logistic=pairwise[(pairwise.first_method=="service_empirical")&(pairwise.second_method=="motion_logistic")].iloc[0]
    lines=["# Four-check timing-budget audit","",
      "This audit uses frozen V2 outputs only. Physical sequences, rather than timestamps or seeds, are the statistical units.","",
      "## 1. Matched acceptance","",
      f"Operating thresholds were selected independently inside each outer LOSO training fold to target {PRIMARY_MATCHED_ACCEPTANCE:.0%} acceptance. Labels from the held-out sequence were not used for operating-point selection.","",
      "| Method | Accepted / evaluated | Achieved acceptance | False qualifications / accepted | False-qualification rate (95% sequence bootstrap CI) |","|---|---:|---:|---:|---:|"]
    for _,r in primary.iterrows():
        lines.append(f"| {r.method} | {int(r.accepted_count)} / {int(r.condition_count)} | {r.achieved_acceptance:.1%} | {int(r.false_qualified_count)} / {int(r.accepted_count)} | {r.false_qualification:.1%} [{r.false_qualification_ci_low:.1%}, {r.false_qualification_ci_high:.1%}] |")
    lines += ["", "The confidence intervals resample the ten held-out physical sequences. They do not treat the 800 condition cells as independent experiments.",
      f"At comparable acceptance, service-specific empirical budgets reduce pooled false qualification by {-100*empirical_aoi.false_qualification_difference_first_minus_second:.1f} percentage points versus AoI-only (95% paired sequence-bootstrap CI for the reduction: {-100*empirical_aoi.ci_high:.1f} to {-100*empirical_aoi.ci_low:.1f} points). Their {-100*empirical_logistic.false_qualification_difference_first_minus_second:.1f}-point reduction versus the logistic model is not statistically resolved by ten sequences (CI: {-100*empirical_logistic.ci_high:.1f} to {-100*empirical_logistic.ci_low:.1f} points).","",
      "## 2. What delivery improvement repaired","",
      "| Population | Degraded failure component | Cases | Denominator | Fraction |","|---|---|---:|---:|---:|"]
    for _,r in recovery.iterrows():
        lines.append(f"| {r.population} | {r.failure_component} | {int(r.cases)} | {int(r.population_denominator)} | {r.fraction:.1%} |")
    counts=rem.classification.value_counts()
    lines += ["",f"At 2 Hz / 200 ms, {int(counts.get('delivery_remediable',0))} cases fail but pass at ideal delivery, {int(counts.get('persistent_under_ideal',0))} still fail at ideal delivery, and {int(counts.get('already_qualified_degraded',0))} already pass under degraded delivery.",
      f"Together these account for all {len(rem)} sequence-service cases; {int(counts.get('already_qualified_degraded',0))} were not degraded-delivery failures.","",
      "## 3. Complete case accounting","",
      "`sequence_service_case_ledger.csv` contains one row for each of 10 sequences x 4 services, with degraded, practical, and ideal coverage, physical satisfaction, freshness satisfaction, joint satisfaction, qualification state, observability, and failure-component labels.","",
      "## 4. Literature distinction","",
      f"The capability-level comparison is in `{literature_path.name}`. The defensible contribution is an empirical, service-relative physical-virtual fidelity and delivery-remediability audit. It is not a new AoI metric, a new stochastic synchronization optimizer, or evidence of communication savings.","",
      "## Interpretation","",
      "The matched-acceptance result should replace comparisons made at one shared numerical score threshold. Recovery-component counts determine whether the practical intervention repairs physical discrepancy, freshness compliance, or both; freshness-only recovery must not be presented as improved physical fidelity."]
    (out/"four_checks_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def write_literature_comparison(path):
    text="""# Capability-level literature distinction

This comparison distinguishes actual problem formulations and demonstrated capabilities. It does not claim novelty from terminology alone.

| Work | Actual capability | Decision/evaluation object | What the present study adds, and does not add |
|---|---|---|---|
| Maatouk *et al.*, [Age of Incorrect Information](https://doi.org/10.1109/TWC.2022.3213227) | Defines a semantics-aware age process that grows while a receiver's estimate is incorrect and studies update policies under source/channel models. | Communication update policy and long-run AoII. | The present study measures empirical physical-virtual errors for distinct robot services and distinguishes physical discrepancy from delivery freshness. It does **not** replace AoII theory or prove an optimal update policy. |
| Chiariotti *et al.*, [Query Age of Information](https://doi.org/10.1109/TCOMM.2022.3141786) | Optimizes freshness at query instants for periodic and stochastic pull-based use, rather than treating all times equally. | Query-timed AoI and communication scheduling. | Service horizons and tolerances make the consequence of delivery service-specific, but the present study does **not** formulate a query process or optimize QAoI. |
| Tan and Matta, [Digital twin synchronization problem](https://doi.org/10.1080/24725854.2023.2253869) | Formulates synchronization as stochastic control balancing synchronization cost against prediction-bias cost; derives state-independent, state-dependent, approximate, and full-information policies for an unreliable production system. | When to synchronize a stochastic production-system twin. | The present study empirically tests whether a frozen mobile-robot twin meets componentwise service contracts across rate-delay conditions and unseen physical sequences. It does **not** claim optimal synchronization or cost minimization. |
| Li *et al.*, [Budget-Constrained DT Synchronization and Fidelity-Aware Queries](https://doi.org/10.1109/TMC.2024.3455357) | Selects physical objects for synchronization under an update budget and evaluates fidelity-aware edge queries, using DT-state staleness and query cost in a UAV-assisted sensor network. | Budget allocation, state staleness, and query evaluation cost. | The present study supplies measured physical-virtual trajectory disagreement, multi-horizon service outcomes, and a delivery-remediability decomposition. It does **not** offer their budget-allocation algorithm or UAV collection model. |
| Wanigarathna *et al.*, [Adaptive DT synchronization under bandwidth constraints](https://doi.org/10.1016/j.comnet.2026.112718) | Uses GRU quantile forecasts, online coverage checks, and KL-divergence drift monitoring to switch between predictive and direct synchronization; reports smartphone traffic and MAE reductions on LTE/5G data. | Event-driven mode selection, predictive reliability, traffic, and prediction error. | The present study evaluates whether service-specific physical fidelity can be inferred from sensor context and delivery conditions. Its compact predictor is diagnostic and its current replay does **not** demonstrate live traffic reduction. |

## Exact distinction

The closest synchronization papers decide **when or what to update** under modeled costs, budgets, queries, or predictive uncertainty. The present analysis instead asks whether a fixed physical-virtual model remains adequate for a specified service after a particular delivery process, and whether changing delivery can repair the observed failure. Its distinctive evidence is the joint use of componentwise physical fidelity, service horizons/tolerances, explicit observability, held-out physical-sequence evaluation, and a physical-versus-freshness recovery decomposition.

This is a narrower empirical contribution than a general synchronization optimizer. The study should not claim that AoI-only methods ignore task context universally, that existing DT synchronization lacks fidelity awareness, or that the current method saves bandwidth. The supportable claim is that freshness alone is insufficient to identify the measured service-validity boundary in these frozen robot traces, and that some apparent recoveries may be freshness compliance rather than repaired physical fidelity.
"""
    path.write_text(text,encoding="utf-8")


def main():
    global OUT
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=OUT); p.add_argument("--freeze-only",action="store_true")
    p.add_argument("--summarize-existing",action="store_true"); args=p.parse_args(); OUT=args.output
    paths=discover(); OUT.mkdir(parents=True,exist_ok=True)
    manifest={"schema":"service_timing_budget_v1","status":"retrospective_protocol_frozen_before_outcome_evaluation",
      "created_from_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),
      "frozen_v2_commit":"6540c01f90f3c1074de0d8dae9964a5276fbbc91","sources":[str(x.relative_to(ROOT)) for x in paths],
      "source_sha256":{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
      "services":SERVICES,"rates_hz":RATES,"delays_ms":DELAYS_MS,"target_satisfaction":TARGET,"minimum_coverage":MIN_COVERAGE,
      "prefix_s":PREFIX_S,"prediction_thresholds":THRESHOLDS,"selected_prediction_threshold":TARGET,
      "matched_acceptance_targets":MATCHED_ACCEPTANCE_TARGETS,"primary_matched_acceptance":PRIMARY_MATCHED_ACCEPTANCE,
      "matched_acceptance_selection":"nested within each outer LOSO training fold; acceptance-only criterion; no held-out labels",
      "aoi_only_score":"service AoI limit minus measured maximum AoI, rounded to 1e-6 s; larger is safer",
      "ideal_delivery":"10 Hz recorded source, zero added delay","degraded":"2 Hz, 200 ms","feasible_improvement":"5 Hz, 50 ms",
      "reconstruction":"causal zero-order hold, no queue, newest arrived source sample","evaluation_clock_hz":10,
      "clock_comparison":"source and delay timestamps rounded to integer nanoseconds before arrival and freshness comparisons",
      "frame":"saved common local ENU; no alignment","quality_flags":"none in evaluated trajectory files",
      "statistical_unit":"physical sequence; three seeds aggregated within sequence"}
    (OUT/"protocol_manifest.json").write_text(json.dumps(manifest,indent=2,default=list)+"\n",encoding="utf-8")
    if args.freeze_only: print(OUT/"protocol_manifest.json"); return
    if args.summarize_existing: runs=pd.read_csv(OUT/"per_run_response_surface.csv")
    else: runs=response_rows(paths); runs.to_csv(OUT/"per_run_response_surface.csv",index=False)
    seq=sequence_rows(runs); seq.to_csv(OUT/"per_sequence_response_surface.csv",index=False)
    pred=probability_models(seq); pred.to_csv(OUT/"heldout_predictions.csv",index=False)
    per,macro=prediction_summary(pred); per.to_csv(OUT/"prediction_per_sequence.csv",index=False); macro.to_csv(OUT/"prediction_risk_coverage.csv",index=False)
    table=bootstrap_selected(per); table.to_csv(OUT/"baseline_comparison.csv",index=False)
    rem=remediability(seq,pred); rem.to_csv(OUT/"delivery_remediability.csv",index=False)
    rem.to_csv(OUT/"sequence_service_case_ledger.csv",index=False)
    recovery=recovery_component_summary(rem); recovery.to_csv(OUT/"recovery_component_summary.csv",index=False)
    matched_per,matched_summary,matched_pairs=matched_acceptance_analysis(seq,pred)
    matched_per.to_csv(OUT/"matched_acceptance_per_sequence.csv",index=False)
    matched_summary.to_csv(OUT/"matched_acceptance_summary.csv",index=False)
    matched_pairs.to_csv(OUT/"matched_acceptance_pairwise_bootstrap.csv",index=False)
    intervention=intervention_results(seq,pred,rem); intervention.to_csv(OUT/"intervention_comparison.csv",index=False)
    boundaries,mono=boundary_and_monotonicity(seq,pred); boundaries.to_csv(OUT/"boundary_prediction.csv",index=False)
    boundaries.groupby("method",as_index=False).boundary_step_error.mean().to_csv(OUT/"boundary_prediction_summary.csv",index=False)
    mono.to_csv(OUT/"monotonicity_audit.csv",index=False)
    literature=OUT/"literature_capability_comparison.md"; write_literature_comparison(literature)
    plots(seq,macro,rem,OUT,intervention,matched_summary)
    write_four_checks_report(OUT,matched_summary,matched_pairs,rem,recovery,literature)
    print(OUT)

if __name__=="__main__": main()
