"""Secondary robustness checks for frozen i2Nav service timing replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DigitalTwin.analysis import service_timing_budget_study as base
from DigitalTwin.analysis import timing_reviewer_checks as reviewer

OUT = base.ROOT / "results/service_timing_robustness"
REQUIREMENTS = (0.70, 0.80, 0.90)
TARGETS = reviewer.TARGETS
FIXED_ROOT = base.ROOT / "results/i2nav_final_model_study/phase1_official_fixed_v5"


def identity_scores(train, test):
    """Training-nine service frequency, constant over all rate-delay cells."""
    frequencies = train.groupby("service").qualified.mean()
    if not set(test.service).issubset(frequencies.index):
        raise ValueError("Missing service in training sequences")
    return test.service.map(frequencies).to_numpy(float)


def nested_identity(seq):
    outer = []
    for held in sorted(seq.sequence.unique()):
        train, test = seq[seq.sequence != held], seq[seq.sequence == held]
        inner = []
        for dev in sorted(train.sequence.unique()):
            inner_train, inner_test = train[train.sequence != dev], train[train.sequence == dev]
            inner.extend(identity_scores(inner_train, inner_test))
        score = identity_scores(train, test)
        for target in TARGETS:
            threshold, inner_acceptance = base._acceptance_threshold(np.asarray(inner), target)
            accept = score >= threshold
            for service in sorted(test.service.unique()):
                mask = test.service.to_numpy() == service
                qualified = test.qualified.to_numpy(bool)
                outer.append({"sequence": held, "service": service, "method": "service_identity_only",
                              "target_acceptance": target, "threshold": threshold,
                              "inner_acceptance": inner_acceptance,
                              "accepted": int((accept & mask).sum()), "conditions": int(mask.sum()),
                              "false_qualified": int((accept & mask & ~qualified).sum())})
    return pd.DataFrame(outer)


def evaluate_service_scores(seq, scores, identity):
    """Apply nested training thresholds from the existing study per service."""
    original = pd.read_csv(base.ROOT / "results/service_timing_reviewer_checks/multi_coverage_per_sequence.csv")
    # The existing per-sequence file is overall, so reconstruct per-service decisions
    # using exactly its already frozen outer-fold thresholds.
    rows = []
    predictions = pd.read_csv(base.OUT / "heldout_predictions.csv")
    for r in original.itertuples():
        sequence = r.sequence
        if r.method == "kinematic_staleness":
            test = scores[(scores.sequence == sequence)].copy()
            test = test.merge(seq[seq.sequence == sequence][["sequence", "service", "rate_hz", "delay_ms", "qualified"]],
                              on=["sequence", "service", "rate_hz", "delay_ms"], validate="one_to_one")
            values = test.score.to_numpy(float)
        else:
            test = predictions[(predictions.sequence == sequence) & (predictions.method == r.method)]
            values = test.score.to_numpy(float)
        selected = values >= r.threshold
        for service in sorted(test.service.unique()):
            use = test.service.to_numpy() == service
            truth = test.qualified.to_numpy(bool) if r.method == "kinematic_staleness" else test.actual_qualified.to_numpy(bool)
            rows.append({"sequence": sequence, "service": service, "method": r.method,
                         "target_acceptance": r.target_acceptance, "accepted": int((use & selected).sum()),
                         "conditions": int(use.sum()), "false_qualified": int((use & selected & ~truth).sum())})
    return pd.concat([pd.DataFrame(rows), identity], ignore_index=True)


def sequence_bootstrap(per, draws=3000):
    rng = np.random.default_rng(947)
    rows = []
    for (service, target, method), g in per.groupby(["service", "target_acceptance", "method"]):
        g = g.sort_values("sequence")
        accepted = int(g.accepted.sum()); false = int(g.false_qualified.sum())
        indices = rng.integers(0, len(g), size=(draws, len(g)))
        na = g.accepted.to_numpy()[indices].sum(1)
        nf = g.false_qualified.to_numpy()[indices].sum(1)
        risk = np.divide(nf, na, out=np.full(draws, np.nan), where=na>0)
        finite = risk[np.isfinite(risk)]
        lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        rows.append({"service": service, "target_acceptance": target, "method": method,
                     "accepted": accepted, "conditions": int(g.conditions.sum()), "false_qualified": false,
                     "achieved_acceptance": accepted / g.conditions.sum(),
                     "false_qualification": false / accepted if accepted else np.nan,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def identity_overall_pairwise(overall, draws=5000):
    rng=np.random.default_rng(397)
    rows=[]
    for target in TARGETS:
        d=overall[np.isclose(overall.target_acceptance,target)]
        first=d[d.method=="service_empirical"].set_index("sequence").sort_index()
        second=d[d.method=="service_identity_only"].set_index("sequence").reindex(first.index)
        indices=rng.integers(0,len(first),size=(draws,len(first)))
        def risk(frame):
            n=frame.accepted.to_numpy()[indices].sum(1)
            f=frame.false_qualified.to_numpy()[indices].sum(1)
            return np.divide(f,n,out=np.full(draws,np.nan),where=n>0)
        sample=risk(first)-risk(second);sample=sample[np.isfinite(sample)]
        a,b=first.accepted.sum(),second.accepted.sum()
        lo,hi=np.quantile(sample,[.025,.975]) if len(sample) else (np.nan,np.nan)
        rows.append({"target_acceptance":target,"surface_accepted":a,"identity_accepted":b,
                     "surface_false":first.false_qualified.sum(),"identity_false":second.false_qualified.sum(),
                     "surface_acceptance":a/first.conditions.sum(),"identity_acceptance":b/second.conditions.sum(),
                     "difference_surface_minus_identity":first.false_qualified.sum()/a-second.false_qualified.sum()/b if a and b else np.nan,
                     "ci_low":lo,"ci_high":hi,"n_sequences":len(first)})
    return pd.DataFrame(rows)


def paired_service_differences(per, target=.30, draws=3000):
    rng = np.random.default_rng(112)
    rows=[]
    for service in sorted(per.service.unique()):
        g=per[(per.service==service)&np.isclose(per.target_acceptance,target)]
        ref=g[g.method=="service_empirical"].set_index("sequence").sort_index()
        for method in sorted(set(g.method)-{"service_empirical"}):
            alt=g[g.method==method].set_index("sequence").reindex(ref.index)
            a0,a1=ref.accepted.sum(),alt.accepted.sum()
            # Within-service matched acceptance is required; otherwise no comparison.
            matched=a0>0 and a1>0 and abs(a0-a1)/ref.conditions.sum()<=.05
            if not matched:
                rows.append({"service":service,"method":method,"target_acceptance":target,"matched":False,
                             "surface_accepted":a0,"comparator_accepted":a1,"difference":np.nan,"ci_low":np.nan,"ci_high":np.nan})
                continue
            ii=rng.integers(0,len(ref),size=(draws,len(ref)))
            def risks(x):
                n=x.accepted.to_numpy()[ii].sum(1); f=x.false_qualified.to_numpy()[ii].sum(1)
                return np.divide(f,n,out=np.full(draws,np.nan),where=n>0)
            diff=risks(ref)-risks(alt); diff=diff[np.isfinite(diff)]
            lo,hi=np.quantile(diff,[.025,.975]) if len(diff) else (np.nan,np.nan)
            rows.append({"service":service,"method":method,"target_acceptance":target,"matched":True,
                         "surface_accepted":a0,"comparator_accepted":a1,
                         "difference":ref.false_qualified.sum()/a0-alt.false_qualified.sum()/a1,
                         "ci_low":lo,"ci_high":hi})
    return pd.DataFrame(rows)


def qualification_sensitivity(seq):
    rows=[]
    for target in REQUIREMENTS:
        d=seq.copy()
        d["qualified_sensitivity"]=((d.coverage>=base.MIN_COVERAGE)&
            (d.physical_satisfaction>=target)&(d.freshness_satisfaction>=target)&
            (d.joint_satisfaction>=target)).astype(int)
        d["qualification_requirement"]=target
        rows.append(d)
    return pd.concat(rows,ignore_index=True)


def classify_cases(grid, label="qualified"):
    rows=[]
    for (sequence,service),g in grid.groupby(["sequence","service"]):
        def value(rate,delay):
            h=g[(g.rate_hz==rate)&(g.delay_ms==delay)]
            if len(h)!=1: raise ValueError("Incomplete case grid")
            return bool(h[label].iloc[0])
        ideal,degraded,practical=value(10,0),value(2,200),value(5,50)
        cls="persistent_under_ideal" if not ideal else ("already_qualified_degraded" if degraded else "delivery_remediable")
        rows.append({"sequence":sequence,"service":service,"classification":cls,
                     "ideal_qualified":int(ideal),"degraded_qualified":int(degraded),
                     "practical_qualified":int(practical)})
    return pd.DataFrame(rows)


def causal_indices(t, rate, delay_ms, phase=0):
    stride=round(10/rate)
    if not 0<=phase<stride: raise ValueError("Phase must be a native 10-Hz source index within delivery stride")
    src=np.arange(phase,len(t),stride)
    clock_ns=np.rint(np.asarray(t)*1_000_000_000).astype(np.int64)
    arrivals=clock_ns[src]+int(round(delay_ms*1_000_000))
    p=np.searchsorted(arrivals,clock_ns,side="right")-1
    idx=np.full(len(t),-1,dtype=int)
    valid=p>=0; idx[valid]=src[p[valid]]
    return idx


def history_signature(idx):
    return hashlib.sha256(np.asarray(idx,dtype="<i4").tobytes()).hexdigest()


def extrapolate_delivered(a, idx):
    """Constant-turn arc from last delivered pose/rates, no hidden samples."""
    t=a["time_s"]
    x=np.full(len(t),np.nan); y=x.copy(); th=x.copy()
    valid=idx>=0; k=idx[valid]; age=t[valid]-t[k]
    v=a["corrected_v_mps"][k]; w=a["corrected_omega_radps"][k]
    heading=a["estimate_heading_rad"][k]
    turn=np.abs(w)>1e-8
    dx=v*age*np.cos(heading); dy=v*age*np.sin(heading)
    dx[turn]=v[turn]/w[turn]*(np.sin(heading[turn]+w[turn]*age[turn])-np.sin(heading[turn]))
    dy[turn]=-v[turn]/w[turn]*(np.cos(heading[turn]+w[turn]*age[turn])-np.cos(heading[turn]))
    x[valid]=a["estimate_east_m"][k]+dx
    y[valid]=a["estimate_north_m"][k]+dy
    th[valid]=base.wrap(heading+w*age)
    return x,y,th


def evaluate_indices(a, service, idx, target=.8, reconstruct=False):
    t=a["time_s"]
    h=round(service["horizon_s"]*10)
    eligible=np.arange(len(t))>=round(base.PREFIX_S*10)+h
    observable=eligible&(idx>=0)
    if h: observable &= np.r_[np.zeros(h,bool),idx[:-h]>=0]
    use=np.flatnonzero(observable)
    if reconstruct: x,y,th=extrapolate_delivered(a,idx)
    else: x,y,th=(a["estimate_east_m"][idx.clip(min=0)],a["estimate_north_m"][idx.clip(min=0)],a["estimate_heading_rad"][idx.clip(min=0)])
    pos=np.empty(len(use)); heading=np.empty(len(use))
    if service["family"]=="global":
        pos=np.hypot(x[use]-a["gt_east_m"][use],y[use]-a["gt_north_m"][use])
        heading=np.abs(np.rad2deg(base.wrap(th[use]-a["gt_heading_rad"][use])))
    else:
        i=use-h
        rp=base.relative(x,y,th,i,use)
        gp=base.relative(a["gt_east_m"],a["gt_north_m"],a["gt_heading_rad"],i,use)
        pos=np.hypot(rp[0]-gp[0],rp[1]-gp[1])
        heading=np.abs(np.rad2deg(base.wrap(rp[2]-gp[2])))
    clock_ns=np.rint(t*1_000_000_000).astype(np.int64)
    age=(clock_ns[use]-clock_ns[idx[use]])/1_000_000_000
    physical=(pos<=service["pos_tol_m"])&(heading<=service["heading_tol_deg"])
    fresh=age<=service["aoi_limit_s"]
    denom=len(use)
    if not denom:
        return {"eligible":int(eligible.sum()),"observable":0,"physical_satisfaction":np.nan,
                "freshness_satisfaction":np.nan,"joint_satisfaction":np.nan,"qualified":False,"mean_position_discrepancy_m":np.nan}
    coverage=denom/eligible.sum()
    ps=physical.mean(); fs=fresh.mean(); js=(physical&fresh).mean()
    return {"eligible":int(eligible.sum()),"observable":denom,"coverage":coverage,
            "physical_satisfaction":ps,"freshness_satisfaction":fs,"joint_satisfaction":js,
            "qualified":bool(coverage>=base.MIN_COVERAGE and min(ps,fs,js)>=target),
            "mean_position_discrepancy_m":float(np.mean(pos)),"mean_heading_discrepancy_deg":float(np.mean(heading))}


def receiver_phase_study(paths):
    rows=[]; signatures=[]
    for path in paths:
        a=base.load(path); d=pd.read_csv(path,usecols=["corrected_v_mps","corrected_omega_radps"])
        for c in d: a[c]=d[c].to_numpy(float)
        seq=path.parent.name.split("_",2)[-1]; replicate=path.parent.parent.name
        for rate in base.RATES:
            for delay in base.DELAYS_MS:
                for phase in range(round(10/rate)):
                    idx=causal_indices(a["time_s"],rate,delay,phase)
                    signatures.append({"sequence":seq,"replicate":replicate,"rate_hz":rate,"delay_ms":delay,
                                       "phase_native_samples":phase,"history_sha256":history_signature(idx)})
                    for service in base.SERVICES:
                        for receiver in ("zoh","constant_turn_extrapolation"):
                            metric=evaluate_indices(a,service,idx,reconstruct=receiver!="zoh")
                            rows.append({"sequence":seq,"replicate":replicate,"service":service["service"],
                                         "rate_hz":rate,"delay_ms":delay,"phase_native_samples":phase,
                                         "receiver":receiver,**metric})
    return pd.DataFrame(rows),pd.DataFrame(signatures)


def fixed_physics_paths(v2paths):
    result={}
    for sequence in sorted({p.parent.name.split("_",2)[-1] for p in v2paths}):
        path=FIXED_ROOT/sequence/"fixed_v5_estimate_traj.txt"
        if not path.exists(): raise FileNotFoundError(path)
        result[sequence]=path
    if len(result)!=10: raise ValueError("Expected ten deterministic fixed-physics sequences")
    return result


def load_fixed_on_v2(v2path, fixedpath):
    a=base.load(v2path)
    fixed=np.loadtxt(fixedpath)
    if fixed.shape!=(len(a["time_s"]),8): raise ValueError(f"Fixed trajectory length mismatch: {fixedpath}")
    rawtime=pd.read_csv(v2path,usecols=["time_s"]).time_s.to_numpy(float)
    if not np.allclose(fixed[:,0],rawtime,rtol=0,atol=1e-6): raise ValueError("Fixed/V2 clock mismatch")
    if not np.allclose(fixed[0,1:3],[a["gt_east_m"][0],a["gt_north_m"][0]],atol=1e-5):
        raise ValueError("Fixed/V2 initial frame mismatch")
    a["estimate_east_m"]=fixed[:,1]; a["estimate_north_m"]=fixed[:,2]
    a["estimate_heading_rad"]=base.wrap(2*np.arctan2(fixed[:,6],fixed[:,7]))
    return a


def configuration_study(paths):
    fixed=fixed_physics_paths(paths)
    first={}
    for p in paths:
        seq=p.parent.name.split("_",2)[-1]
        first.setdefault(seq,p)
    rows=[]; paired=[]
    for sequence,p in first.items():
        a=load_fixed_on_v2(p,fixed[sequence])
        for service in base.SERVICES:
            for rate in base.RATES:
                for delay in base.DELAYS_MS:
                    metric=base.evaluate(a,service,rate,delay)
                    rows.append({"sequence":sequence,"configuration":"fixed_v5_deterministic","service":service["service"],
                                 "rate_hz":rate,"delay_ms":delay,**metric,
                                 "qualified":int(metric["coverage"]>=base.MIN_COVERAGE and min(metric["physical_satisfaction"],
                                                metric["freshness_satisfaction"],metric["joint_satisfaction"])>=base.TARGET)})
            for rate,delay in reviewer.CONDITIONS:
                tr=reviewer.discrepancy_trace(a,service,rate,delay)
                paired.append({"sequence":sequence,"configuration":"fixed_v5_deterministic","service":service["service"],
                               "rate_hz":rate,"delay_ms":delay,**reviewer.trace_summary(tr,service)})
    return pd.DataFrame(rows),pd.DataFrame(paired),fixed


def figures(service_summary, receiver_seq, config_seq, out):
    g=service_summary[np.isclose(service_summary.target_acceptance,.3)]
    fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for ax,(service,part) in zip(axs.flat,g.groupby("service")):
        part=part.sort_values("method"); y=np.arange(len(part))
        ax.barh(y,part.false_qualification,color="#2878B5",height=.65)
        ax.set(title=service,xlim=(0,1),yticks=y,yticklabels=[m.replace("_"," ") for m in part.method],xlabel="False / accepted")
        ax.invert_yaxis()
        ax.set_ylim(len(part)-.85,-.85)
        for i,r in enumerate(part.itertuples()):
            label=f"{r.false_qualified}/{r.accepted}" if r.accepted else "NA (0 accepted)"
            xpos=min(float(r.false_qualification)+.02,.83) if r.accepted else .02
            ax.text(xpos,i,label,va="center",fontsize=8)
        ax.grid(axis="x",alpha=.2)
    fig.tight_layout(); fig.savefig(out/"per_service_qualification.png",dpi=180); fig.savefig(out/"per_service_qualification.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    e=service_summary[(service_summary.method.isin(["service_empirical","service_identity_only"]))&
                      (np.isclose(service_summary.target_acceptance,.3))]
    for m,h in e.groupby("method"):
        ax.scatter(h.achieved_acceptance,h.false_qualification,label=m.replace("_"," "),s=60)
    ax.set(xlabel="Achieved within-service acceptance",ylabel="False / accepted",ylim=(0,1))
    ax.legend(frameon=False); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out/"surface_vs_identity.png",dpi=180); fig.savefig(out/"surface_vs_identity.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    h=receiver_seq[(receiver_seq.rate_hz==2)&(receiver_seq.delay_ms==200)&(receiver_seq.phase_native_samples==0)]
    for receiver,part in h.groupby("receiver"):
        v=part.groupby("service").joint_satisfaction.mean(); ax.plot(v.index,v.values,marker="o",label=receiver)
    ax.set(ylabel="Mean sequence-level joint satisfaction",ylim=(0,1)); ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/"receiver_robustness.png",dpi=180); fig.savefig(out/"receiver_robustness.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    c=config_seq[(config_seq.rate_hz==2)&(config_seq.delay_ms==200)]
    for config,part in c.groupby("configuration"):
        v=part.groupby("service").joint_satisfaction.mean(); ax.plot(v.index,v.values,marker="o",label=config)
    ax.set(ylabel="Mean sequence-level joint satisfaction",ylim=(0,1));ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"configuration_robustness.png",dpi=180);fig.savefig(out/"configuration_robustness.pdf");plt.close(fig)


def run(out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    paths=base.discover()
    seq=pd.read_csv(base.OUT/"per_sequence_response_surface.csv")
    sensitivity=qualification_sensitivity(seq)
    sensitivity.to_csv(out/"contract_sensitivity_per_sequence.csv",index=False)
    contract=sensitivity.groupby(["qualification_requirement","service","rate_hz","delay_ms"],as_index=False).qualified_sensitivity.agg(["sum","count"])
    contract.to_csv(out/"contract_sensitivity_summary.csv",index=False)
    identity=nested_identity(seq)
    identity.to_csv(out/"identity_nested_per_sequence.csv",index=False)
    identity_overall=identity.groupby(["sequence","method","target_acceptance"],as_index=False)[["accepted","conditions","false_qualified"]].sum()
    identity_overall.to_csv(out/"identity_nested_overall.csv",index=False)
    original_overall=pd.read_csv(base.ROOT/"results/service_timing_reviewer_checks/multi_coverage_per_sequence.csv")
    overall_comparison=pd.concat([original_overall[["sequence","method","target_acceptance","accepted","conditions","false_qualified"]],
                                  identity_overall],ignore_index=True)
    overall_comparison.to_csv(out/"all_methods_nested_overall.csv",index=False)
    identity_overall_pairwise(overall_comparison).to_csv(out/"identity_vs_surface_paired_bootstrap.csv",index=False)
    kin=pd.read_csv(base.ROOT/"results/service_timing_reviewer_checks/kinematic_staleness_scores.csv")
    by_service=evaluate_service_scores(seq,kin,identity)
    by_service.to_csv(out/"per_service_nested_predictions.csv",index=False)
    summary=sequence_bootstrap(by_service)
    summary.to_csv(out/"per_service_nested_summary.csv",index=False)
    paired=paired_service_differences(by_service)
    paired.to_csv(out/"per_service_paired_bootstrap.csv",index=False)
    cases=[]
    for requirement,g in sensitivity.groupby("qualification_requirement"):
        c=classify_cases(g,"qualified_sensitivity");c["qualification_requirement"]=requirement;cases.append(c)
    pd.concat(cases,ignore_index=True).to_csv(out/"contract_remediability_by_requirement.csv",index=False)
    receiver,signatures=receiver_phase_study(paths)
    receiver.to_csv(out/"receiver_phase_per_run.csv",index=False)
    signatures.to_csv(out/"represented_history_signatures.csv",index=False)
    aliases=signatures.groupby(["sequence","replicate","rate_hz","phase_native_samples"],as_index=False).agg(
        unique_histories=("history_sha256","nunique"),tested_delays=("delay_ms","nunique"))
    aliases.to_csv(out/"represented_history_alias_summary.csv",index=False)
    receiver_seq=receiver.groupby(["sequence","service","rate_hz","delay_ms","phase_native_samples","receiver"],as_index=False).mean(numeric_only=True)
    receiver_seq["qualified_at_sequence_aggregate"]=((receiver_seq.coverage>=base.MIN_COVERAGE)&
        (receiver_seq.physical_satisfaction>=base.TARGET)&(receiver_seq.freshness_satisfaction>=base.TARGET)&
        (receiver_seq.joint_satisfaction>=base.TARGET)).astype(int)
    receiver_seq.to_csv(out/"receiver_phase_per_sequence.csv",index=False)
    receiver_cases=[]
    for (receiver,phase),g in receiver_seq.groupby(["receiver","phase_native_samples"]):
        # A complete 10/0, 5/50, 2/200 case grid exists only for phases 0 and 1;
        # ideal 10-Hz delivery has only phase 0, so phase comparisons use that
        # common ideal reference rather than inventing sub-sample ideal histories.
        if phase>1: continue
        current=g[g.phase_native_samples==phase]
        ideal=receiver_seq[(receiver_seq.receiver==receiver)&(receiver_seq.rate_hz==10)&(receiver_seq.delay_ms==0)]
        if phase:
            current=pd.concat([current,ideal],ignore_index=True)
        relevant=current[((current.rate_hz==10)&(current.delay_ms==0))|
                         ((current.rate_hz==5)&(current.delay_ms==50))|
                         ((current.rate_hz==2)&(current.delay_ms==200))]
        if relevant.groupby(["sequence","service","rate_hz","delay_ms"]).size().max()>1:
            raise ValueError("Duplicate receiver case rows")
        if len(relevant)==120:
            case=classify_cases(relevant,"qualified_at_sequence_aggregate")
            case["receiver"]=receiver;case["phase_native_samples"]=phase;receiver_cases.append(case)
    pd.concat(receiver_cases,ignore_index=True).to_csv(out/"receiver_remediability.csv",index=False)
    alt,alt_paired,fixed=fixed_physics_study(paths)
    alt.to_csv(out/"fixed_physics_per_sequence.csv",index=False)
    alt_paired.to_csv(out/"fixed_physics_paired_discrepancy.csv",index=False)
    primary=seq[["sequence","service","rate_hz","delay_ms","qualified","joint_satisfaction"]].copy()
    primary["configuration"]="frozen_v2"
    comparable=pd.concat([primary,alt[primary.columns]],ignore_index=True)
    comparable.to_csv(out/"configuration_comparison.csv",index=False)
    config_cases=[]
    for config,g in comparable.groupby("configuration"):
        z=classify_cases(g);z["configuration"]=config;config_cases.append(z)
    pd.concat(config_cases,ignore_index=True).to_csv(out/"configuration_remediability.csv",index=False)
    primary_paired=pd.read_csv(base.ROOT/"results/service_timing_reviewer_checks/paired_discrepancy_per_sequence.csv")
    primary_paired["configuration"]="frozen_v2"
    pd.concat([primary_paired,alt_paired],ignore_index=True).to_csv(out/"configuration_paired_discrepancy.csv",index=False)
    figures(summary,receiver_seq,comparable,out)
    manifest={"schema":"service_timing_robustness_v1","primary_unchanged":True,
              "physical_sequences":10,"v2_seeds_per_sequence":3,"requirements":REQUIREMENTS,"acceptance_targets":TARGETS,
              "rates_hz":base.RATES,"delays_ms":base.DELAYS_MS,
              "phase_policy":"native 10-Hz sample indices within each delivery stride; no sub-sample interpolation",
              "receiver":"ZOH versus last-delivered constant-turn arc using corrected_v_mps and corrected_omega_radps",
              "alternate_configuration":"deterministic fixed_v5, one trajectory per sequence; paired sensitivity only",
              "fixed_sources":{k:str(v.relative_to(base.ROOT)) for k,v in fixed.items()},
              "inference_unit":"physical sequence; seeds and phase variants nested"}
    (out/"protocol_manifest.json").write_text(json.dumps(manifest,indent=2,default=list)+"\n",encoding="utf-8")
    return summary,receiver_seq,comparable


# Kept separate from the runner so compatibility checks run before any results are written.
fixed_physics_study=configuration_study

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args();run(args.output);print(args.output)
