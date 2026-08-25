#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, re, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPECTED_SEQUENCES=["building00","building01","building02","parking00","parking01","parking02","playground00","street00","street01","street02"]
EXPECTED_SEEDS=[42,1042,2042]
SOURCE_COMMIT="6540c01f90f3c1074de0d8dae9964a5276fbbc91"
V2_SCHEMA="i2nav_twin_v2_slow_additive_sensor_consistency_v1"

FORBIDDEN_TOKENS=("gt_","ground_truth","reference_","local_position_error","local_heading_error","global_position_error","global_heading_error","true_delta","remaining_yaw_error")


def load_json(p:Path): return json.loads(p.read_text(encoding='utf-8'))


def verify_prior_freeze(cfg):
    vpath=Path('results/service_relative_fidelity/frozen_signature_verification.csv')
    if vpath.exists():
        v=pd.read_csv(vpath)
        if 'passes' not in v.columns or not v['passes'].astype(str).str.lower().eq('true').all():
            raise RuntimeError('Prior frozen-signature verification is not a clean PASS.')
    ppath=Path('results/service_relative_fidelity/parking00_vs_parking02_verification.csv')
    if ppath.exists():
        d=pd.read_csv(ppath).set_index('metric')
        if not (float(d.loc['rpe10_m','parking02']) < float(d.loc['rpe10_m','parking00']) and float(d.loc['ate_m','parking02']) > float(d.loc['ate_m','parking00'])):
            raise RuntimeError('Expected parking00/parking02 local/global inversion was not reproduced.')
    mpath=Path('results/service_relative_fidelity/analysis_manifest.json')
    if mpath.exists():
        old=load_json(mpath).get('config',{}).get('representative_services',[])
        oldmap={x.get('service_id'):x for x in old}
        for x in cfg['services']:
            if x['service_id'] not in oldmap:
                raise RuntimeError(f"Service {x['service_id']} was not frozen in the prior manifest.")
            y=oldmap[x['service_id']]
            for k in ['family','horizon_s','position_tolerance_m','heading_tolerance_deg']:
                if str(y.get(k)) != str(x.get(k)):
                    # numeric formatting may differ; compare numerically when possible
                    try:
                        if abs(float(y.get(k))-float(x.get(k))) < 1e-12: continue
                    except Exception: pass
                    raise RuntimeError(f"Frozen service definition drift for {x['service_id']} field {k}: prior={y.get(k)} new={x.get(k)}")

def locate_window_table(explicit:str|None)->Path:
    cand=[]
    if explicit: cand.append(Path(explicit))
    cand += [Path('results/service_relative_fidelity/physical_windows_seed_averaged.csv'), Path('/mnt/data/physical_windows_seed_averaged.csv')]
    for p in cand:
        if p.exists(): return p
    raise FileNotFoundError('Cannot locate physical_windows_seed_averaged.csv; run the prior service-relative audit first or pass --window-table.')

def locate_raw_root(explicit:str|None)->Path|None:
    cand=[]
    if explicit: cand.append(Path(explicit))
    cand += [Path('results/i2nav_v2_full_loso/i2nav_v2_full_loso'),Path('results/i2nav_v2_full_loso')]
    for p in cand:
        if p.exists() and list(p.rglob('v2_evaluated_trajectory.csv')):
            return p
    return None

def seq_from_path(p:Path)->str:
    s=str(p).lower()
    for q in EXPECTED_SEQUENCES:
        if q in s: return q
    raise ValueError(f'Cannot identify sequence from {p}')

def seed_from_path(p:Path)->int:
    m=re.search(r'base(\d+)',str(p),re.I)
    if not m: raise ValueError(f'Cannot identify seed from {p}')
    return int(m.group(1))

def causal_accel(t,v):
    t=np.asarray(t,float); v=np.asarray(v,float); out=np.full(len(v),np.nan)
    if len(v)>1:
        dt=np.diff(t); dv=np.diff(v); good=np.isfinite(dt)&(dt>0)&np.isfinite(dv)
        x=np.full(len(dt),np.nan); x[good]=dv[good]/dt[good]; out[1:]=x
        if len(out)>1: out[0]=out[1]
    return out

def sample_1hz_indices(t):
    t=np.asarray(t,float); out=[]; target=t[0]
    for i,ti in enumerate(t):
        if ti+1e-9>=target:
            out.append(i); target=ti+1.0
    return out

def build_signed_context(raw_root:Path)->pd.DataFrame:
    files=sorted(raw_root.rglob('v2_evaluated_trajectory.csv'))
    if len(files)!=30:
        raise RuntimeError(f'Expected exactly 30 frozen v2_evaluated_trajectory.csv files, found {len(files)} under {raw_root}')
    rows=[]
    seen=set()
    for p in files:
        seq=seq_from_path(p); seed=seed_from_path(p); seen.add((seq,seed))
        tr=pd.read_csv(p)
        req=['time_s','odo_speed_mps','imu_yaw_rate_radps']
        miss=[c for c in req if c not in tr.columns]
        if miss: raise RuntimeError(f'{p} missing {miss}')
        d=tr[req].copy().sort_values('time_s')
        trace=p.with_name('v2_prediction_trace.csv')
        if not trace.exists(): raise RuntimeError(f'Missing adjacent prediction trace: {trace}')
        q=pd.read_csv(trace)
        keep=[c for c in ['time_s','wheel_yaw_radps','imu_yaw_radps','wheel_imu_yaw_disagreement_radps'] if c in q.columns]
        if 'wheel_imu_yaw_disagreement_radps' not in keep and not {'wheel_yaw_radps','imu_yaw_radps'}.issubset(keep):
            raise RuntimeError(f'{trace} lacks signed wheel/IMU context')
        q=q[keep].sort_values('time_s')
        dt=np.diff(d.time_s.to_numpy(float)); dt=dt[np.isfinite(dt)&(dt>0)]
        tol=max(0.05,2.5*float(np.median(dt))) if len(dt) else 0.2
        d=pd.merge_asof(d,q,on='time_s',direction='nearest',tolerance=tol,suffixes=('','_trace'))
        if 'wheel_imu_yaw_disagreement_radps' in d.columns:
            dis=d['wheel_imu_yaw_disagreement_radps'].to_numpy(float)
        else:
            wy=d['wheel_yaw_radps'].to_numpy(float); iy=d['imu_yaw_radps'].to_numpy(float); dis=wy-iy
        t=d.time_s.to_numpy(float); speed=d.odo_speed_mps.to_numpy(float); yaw=d.imu_yaw_rate_radps.to_numpy(float)
        acc=causal_accel(t,speed); curv=yaw/np.maximum(np.abs(speed),0.1); elapsed=t-t[0]
        idx=sample_1hz_indices(t)
        for i in idx:
            rows.append(dict(sequence=seq,base_seed=seed,time_s=float(t[i]),time_key=round(float(t[i]),3),
                             speed_signed_mps=float(speed[i]),abs_speed_mps=abs(float(speed[i])),
                             yaw_rate_signed_radps=float(yaw[i]),abs_yaw_rate_radps=abs(float(yaw[i])),
                             accel_signed_mps2=float(acc[i]) if np.isfinite(acc[i]) else np.nan,abs_accel_mps2=abs(float(acc[i])) if np.isfinite(acc[i]) else np.nan,
                             wheel_imu_disagreement_signed_radps=float(dis[i]) if np.isfinite(dis[i]) else np.nan,
                             abs_wheel_imu_disagreement_radps=abs(float(dis[i])) if np.isfinite(dis[i]) else np.nan,
                             curvature_signed_radpm=float(curv[i]) if np.isfinite(curv[i]) else np.nan,
                             curvature_abs_radpm=abs(float(curv[i])) if np.isfinite(curv[i]) else np.nan,
                             elapsed_s=float(elapsed[i])))
    expected={(s,z) for s in EXPECTED_SEQUENCES for z in EXPECTED_SEEDS}
    if seen!=expected: raise RuntimeError(f'Frozen run identity mismatch. missing={sorted(expected-seen)} extra={sorted(seen-expected)}')
    r=pd.DataFrame(rows)
    num=[c for c in r.columns if c not in ['sequence','base_seed','time_key']]
    a=r.groupby(['sequence','time_key'],as_index=False)[num].mean(numeric_only=True)
    return a.sort_values(['sequence','time_s']).reset_index(drop=True)

def build_abs_context_from_window_table(w:pd.DataFrame)->pd.DataFrame:
    g=w[w.family.eq('global_synchronization')].copy()
    keep=['sequence','start_time_s','abs_speed_mps','abs_yaw_rate_radps','abs_accel_mps2','abs_wheel_imu_disagreement_radps','curvature_abs_radpm','elapsed_s']
    g=g[keep].rename(columns={'start_time_s':'time_s'}).copy()
    g['speed_signed_mps']=g['abs_speed_mps']; g['yaw_rate_signed_radps']=g['abs_yaw_rate_radps']; g['accel_signed_mps2']=g['abs_accel_mps2']
    g['wheel_imu_disagreement_signed_radps']=g['abs_wheel_imu_disagreement_radps']; g['curvature_signed_radpm']=g['curvature_abs_radpm']
    return g.sort_values(['sequence','time_s']).reset_index(drop=True)

def add_history(ctx:pd.DataFrame, windows):
    out=[]
    for seq,s in ctx.groupby('sequence',sort=False):
        s=s.sort_values('time_s').copy(); t=s.time_s.to_numpy(float)
        # rolling functions are causal: rows are indexed only up through current t.
        specs={
          'wheel_imu_disagreement_signed_radps':['mean','absmean','std','integral','signcons'],
          'yaw_rate_signed_radps':['mean','absmean'],
          'accel_signed_mps2':['absmean'],
          'curvature_signed_radpm':['absmean'],
          'speed_signed_mps':['mean'],
        }
        for W in windows:
            for feat,stats in specs.items():
                v=s[feat].to_numpy(float)
                vals={x:[] for x in stats}
                for i,ti in enumerate(t):
                    j=np.searchsorted(t,ti-float(W),side='left'); z=v[j:i+1]; tt=t[j:i+1]
                    z=z[np.isfinite(z)]
                    if len(z)==0:
                        for x in stats: vals[x].append(np.nan)
                        continue
                    mean=float(np.mean(z)); absm=float(np.mean(np.abs(z))); sd=float(np.std(z))
                    for x in stats:
                        if x=='mean': val=mean
                        elif x=='absmean': val=absm
                        elif x=='std': val=sd
                        elif x=='integral': val=mean*float(W)
                        elif x=='signcons': val=abs(mean)/(absm+1e-12)
                        vals[x].append(val)
                for x in stats: s[f'{feat}_{x}_{int(W)}s']=vals[x]
        out.append(s)
    return pd.concat(out,ignore_index=True)

def merge_context(labels:pd.DataFrame,ctx:pd.DataFrame)->pd.DataFrame:
    frames=[]
    context_cols=[c for c in ctx.columns if c not in ['sequence','time_s','time_key']]
    for seq,q in labels.groupby('sequence',sort=False):
        a=q.sort_values('start_time_s').copy(); c=ctx[ctx.sequence.eq(seq)].sort_values('time_s')
        # Replace any prior diagnostic context columns with the newly reconstructed
        # causal context; never keep ambiguous _x/_y duplicates.
        a=a.drop(columns=[x for x in context_cols if x in a.columns],errors='ignore')
        m=pd.merge_asof(a,c[['time_s']+context_cols],left_on='start_time_s',right_on='time_s',direction='backward',tolerance=1.25)
        m=m.drop(columns=['time_s'],errors='ignore'); frames.append(m)
    return pd.concat(frames,ignore_index=True)

def service_frame(w,cfg):
    q=w[(w.family==cfg['family']) & np.isclose(w.horizon_s,float(cfg['horizon_s']))].copy()
    if cfg['family']=='local_relative_motion':
        valid=(q.local_position_error_m<=cfg['position_tolerance_m'])&(q.local_heading_error_deg<=cfg['heading_tolerance_deg'])
    else:
        valid=(q.global_position_error_m<=cfg['position_tolerance_m'])&(q.global_heading_error_deg<=cfg['heading_tolerance_deg'])
    q['service_valid']=valid.astype(int); q['service_fail']=(~valid).astype(int); q['service_id']=cfg['service_id']; return q

def assert_feature_safety(features):
    bad=[f for f in features if any(tok in f.lower() for tok in FORBIDDEN_TOKENS)]
    if bad: raise RuntimeError(f'Forbidden/leaky feature(s): {bad}')

def feature_sets(df):
    m0=['elapsed_s']
    m1=[x for x in ['speed_signed_mps','abs_speed_mps','yaw_rate_signed_radps','abs_yaw_rate_radps','accel_signed_mps2','abs_accel_mps2','wheel_imu_disagreement_signed_radps','abs_wheel_imu_disagreement_radps','curvature_signed_radpm','curvature_abs_radpm','elapsed_s'] if x in df.columns]
    hist=[c for c in df.columns if re.search(r'_(mean|absmean|std|integral|signcons)_\d+s$',c)]
    m2=m1+sorted(hist)
    for x in [m0,m1,m2]: assert_feature_safety(x)
    return {'M0_elapsed_only':m0,'M1_instant_context':m1,'M2_context_plus_history':m2}

def make_pipe(C,max_iter):
    return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('lr',LogisticRegression(C=float(C),max_iter=int(max_iter),solver='newton-cholesky'))])

def fit_predict(train,test,features,C,max_iter):
    y=train.service_fail.to_numpy(int)
    if len(np.unique(y))<2: return np.full(len(test),float(np.mean(y))),None
    model=make_pipe(C,max_iter); model.fit(train[features],y); return model.predict_proba(test[features])[:,1],model

def inner_splits(seqs,nfold):
    seqs=sorted(seqs); folds=[[] for _ in range(nfold)]
    for i,s in enumerate(seqs): folds[i%nfold].append(s)
    return folds

def seq_macro_brier(y,p,seq):
    vals=[]
    for s in sorted(pd.unique(seq)):
        k=np.asarray(seq)==s; vals.append(brier_score_loss(np.asarray(y)[k],np.asarray(p)[k]))
    return float(np.mean(vals))

def inner_oof(dev,features,C,nfold,max_iter):
    parts=[]; folds=inner_splits(dev.sequence.unique(),nfold)
    for val_seqs in folds:
        va=dev[dev.sequence.isin(val_seqs)]; tr=dev[~dev.sequence.isin(val_seqs)]
        p,_=fit_predict(tr,va,features,C,max_iter)
        parts.append(pd.DataFrame({'idx':va.index,'sequence':va.sequence.to_numpy(),'y':va.service_fail.to_numpy(int),'p':p}))
    return pd.concat(parts,ignore_index=True)

def choose_C(dev,features,grid,nfold,max_iter):
    rows=[]; oofs={}
    for C in grid:
        o=inner_oof(dev,features,C,nfold,max_iter); oofs[float(C)]=o
        rows.append((float(C),seq_macro_brier(o.y,o.p,o.sequence)))
    rows.sort(key=lambda z:(z[1],abs(math.log10(z[0]))))
    C=rows[0][0]; return C,oofs[C],rows

def decision_metrics(y,support):
    y=np.asarray(y,int); support=np.asarray(support,bool); fail=y.astype(bool); valid=~fail; n=len(y)
    supp=support.sum(); unsafe=(support&fail).sum(); frej=((~support)&valid).sum(); fsafe=unsafe
    return dict(support_rate=float(supp/n),unsafe_among_supported=float(unsafe/supp) if supp else np.nan,false_safe_fraction=float(fsafe/n),false_reject_fraction=float(frej/n),valid_captured_fraction=float((support&valid).sum()/max(valid.sum(),1)))

def sequence_macro_decision(frame,thr):
    vals=[]
    for s,g in frame.groupby('sequence'):
        vals.append(decision_metrics(g.y.to_numpy(),g.p.to_numpy()<=thr))
    return {k:float(np.nanmean([v[k] for v in vals])) for k in vals[0]}

def select_threshold(oof,target):
    p=oof.p.to_numpy(float); cand=np.unique(np.quantile(p,np.linspace(0,1,101)))
    feasible=[]
    for th in cand:
        m=sequence_macro_decision(oof,float(th))
        if np.isfinite(m['unsafe_among_supported']) and m['unsafe_among_supported']<=target:
            feasible.append((m['support_rate'],float(th),m))
    if not feasible: return None,None
    feasible.sort(key=lambda z:(z[0],z[1]),reverse=True); return feasible[0][1],feasible[0][2]

def risk_at_coverage(y,p,cov):
    y=np.asarray(y,int); p=np.asarray(p,float); n=len(y); k=max(1,min(n,int(math.ceil(float(cov)*n))))
    idx=np.argsort(p,kind='mergesort')[:k]; return float(np.mean(y[idx]))

def aurc(y,p,grid): return float(np.mean([risk_at_coverage(y,p,c) for c in grid]))

def safe_auc(y,p):
    y=np.asarray(y,int)
    return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan

def safe_ap(y,p):
    y=np.asarray(y,int)
    return float(average_precision_score(y,p)) if y.sum()>0 else np.nan

def model_coefficients(model,features,held,service_id,model_name):
    if model is None: return []
    imp=model.named_steps['imp']; lr=model.named_steps['lr']
    try: names=imp.get_feature_names_out(features)
    except Exception: names=np.array(features,dtype=object)
    coef=lr.coef_.ravel(); return [dict(service_id=service_id,held_sequence=held,model=model_name,feature=str(n),standardized_coef=float(c)) for n,c in zip(names,coef)]

def load_q95_baseline(path:Path|None,services):
    if path is None or not path.exists(): return pd.DataFrame()
    d=pd.read_csv(path); out=[]
    for s in services:
        q=d[(d.family==s['family']) & np.isclose(d.horizon_s,float(s['horizon_s'])) & np.isclose(d.position_tolerance_m,float(s['position_tolerance_m'])) & np.isclose(d.heading_tolerance_deg,float(s['heading_tolerance_deg']))]
        for _,r in q.iterrows():
            if r['method'] in ('unconditional_loso_envelope','condition_aware_loso_envelope'):
                out.append(dict(service_id=s['service_id'],held_sequence=r['sequence'],method=r['method'],support_rate=r['support_rate'],unsafe_among_supported=r['unsafe_among_supported'],false_safe_fraction=r['false_safe_fraction'],false_reject_fraction=r['false_reject_fraction'],valid_captured_fraction=r['valid_captured_fraction']))
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='service_risk_estimator/service_risk_config.json'); ap.add_argument('--window-table'); ap.add_argument('--raw-root'); ap.add_argument('--allow-abs-fallback',action='store_true'); ap.add_argument('--baseline-decisions'); ap.add_argument('--out',default='results/service_risk_estimator_nested_loso'); args=ap.parse_args()
    cfg=load_json(Path(args.config)); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    verify_prior_freeze(cfg)
    wpath=locate_window_table(args.window_table); w=pd.read_csv(wpath)
    if sorted(w.sequence.unique())!=sorted(EXPECTED_SEQUENCES): raise RuntimeError('Window table does not contain exactly the ten expected physical sequences.')
    # verify strongest frozen service contrast before any modeling
    sig=pd.read_csv('results/service_relative_fidelity/parking00_vs_parking02_verification.csv') if Path('results/service_relative_fidelity/parking00_vs_parking02_verification.csv').exists() else None
    raw=locate_raw_root(args.raw_root)
    if raw is not None:
        ctx=build_signed_context(raw); context_mode='signed_raw_frozen_context'
    elif args.allow_abs_fallback:
        ctx=build_abs_context_from_window_table(w); context_mode='ABS_FALLBACK_DIAGNOSTIC_ONLY'
    else:
        raise RuntimeError('Frozen raw V2 trajectories/traces were not found. Default analysis requires signed causal context. Pass --raw-root, or use --allow-abs-fallback only for code diagnostics (not publication claims).')
    ctx=add_history(ctx,cfg['history_windows_s'])
    full=merge_context(w,ctx)
    fsets=feature_sets(full)
    pd.DataFrame({'model':sum([[k]*len(v) for k,v in fsets.items()],[]),'feature':sum([v for v in fsets.values()],[])}).to_csv(out/'feature_manifest.csv',index=False)
    # hard guard: labels/errors can exist in dataframe but none are in features
    for v in fsets.values(): assert_feature_safety(v)
    allpred=[]; outer_rows=[]; tuning=[]; coeff=[]; threshold_rows=[]
    for svc in cfg['services']:
        q=service_frame(full,svc).reset_index(drop=True)
        for model_name,features in fsets.items():
            print(f"[service {svc['service_id']}] model={model_name}", flush=True)
            for held in EXPECTED_SEQUENCES:
                dev=q[q.sequence!=held].copy(); test=q[q.sequence==held].copy()
                C,oof,cvrows=choose_C(dev,features,cfg['regularization_C_grid'],int(cfg['inner_group_folds']),int(cfg['max_iter']))
                for Cx,b in cvrows: tuning.append(dict(service_id=svc['service_id'],model=model_name,held_sequence=held,C=Cx,inner_sequence_macro_brier=b,selected=(Cx==C)))
                p,model=fit_predict(dev,test,features,C,int(cfg['max_iter']))
                base_rate=float(np.mean([g.service_fail.mean() for _,g in dev.groupby('sequence')]))
                y=test.service_fail.to_numpy(int)
                rec=dict(service_id=svc['service_id'],family=svc['family'],horizon_s=svc['horizon_s'],held_sequence=held,model=model_name,n_windows=len(test),fail_rate=float(y.mean()),C=C,
                         brier=float(brier_score_loss(y,p)),roc_auc=safe_auc(y,p),pr_auc=safe_ap(y,p),aurc=aurc(y,p,cfg['risk_coverage_grid']),constant_aurc=base_rate)
                for cov in cfg['report_coverages']: rec[f'risk_at_{int(100*cov)}pct_coverage']=risk_at_coverage(y,p,cov)
                outer_rows.append(rec); coeff += model_coefficients(model,features,held,svc['service_id'],model_name)
                tmp=test[['sequence','family','horizon_s','start_time_s','end_time_s','service_valid','service_fail']].copy(); tmp['service_id']=svc['service_id']; tmp['model']=model_name; tmp['predicted_failure_risk']=p; tmp['held_sequence']=held; allpred.append(tmp)
                for target in cfg['target_unsafe_rates']:
                    th,inner_m=select_threshold(oof,float(target))
                    support=np.zeros(len(test),bool) if th is None else p<=th
                    m=decision_metrics(y,support)
                    threshold_rows.append(dict(service_id=svc['service_id'],model=model_name,held_sequence=held,target_unsafe_rate=target,selected_risk_cutoff=th if th is not None else np.nan,inner_feasible=th is not None,inner_support_rate=(inner_m or {}).get('support_rate',np.nan),inner_unsafe_among_supported=(inner_m or {}).get('unsafe_among_supported',np.nan),**m))
    outer=pd.DataFrame(outer_rows); outer.to_csv(out/'outer_sequence_metrics.csv',index=False)
    pd.concat(allpred,ignore_index=True).to_csv(out/'outer_window_predictions.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'inner_regularization_selection.csv',index=False)
    pd.DataFrame(threshold_rows).to_csv(out/'outer_operational_decisions.csv',index=False)
    pd.DataFrame(coeff).to_csv(out/'standardized_logistic_coefficients.csv',index=False)
    # macro summaries; each sequence gets equal weight
    metric_cols=['fail_rate','brier','roc_auc','pr_auc','aurc','constant_aurc']+[f'risk_at_{int(100*c)}pct_coverage' for c in cfg['report_coverages']]
    macro=outer.groupby(['service_id','model'],as_index=False)[metric_cols].mean(numeric_only=True); macro.to_csv(out/'service_model_macro.csv',index=False)
    td=pd.DataFrame(threshold_rows); op=td.groupby(['service_id','model','target_unsafe_rate'],as_index=False)[['support_rate','unsafe_among_supported','false_safe_fraction','false_reject_fraction','valid_captured_fraction','inner_feasible']].mean(numeric_only=True); op.to_csv(out/'operational_macro.csv',index=False)
    # q95 baseline points from previous experiment
    bpath=Path(args.baseline_decisions) if args.baseline_decisions else Path('results/service_relative_fidelity/loso_monitor_decisions.csv')
    q95=load_q95_baseline(bpath if bpath.exists() else None,cfg['services']); q95.to_csv(out/'previous_q95_baseline_points.csv',index=False)
    # GO/NO-GO, frozen criteria
    go_cfg=cfg['go_no_go']; verdict_rows=[]
    for svc in cfg['services']:
        s=svc['service_id']; z=outer[outer.service_id.eq(s)]
        means=z.groupby('model').aurc.mean(); m0=means.get('M0_elapsed_only',np.nan); m1=means.get('M1_instant_context',np.nan); m2=means.get('M2_context_plus_history',np.nan)
        const=float(z[z.model.eq('M2_context_plus_history')].constant_aurc.mean())
        rel=(const-m2)/const if const>1e-12 else np.nan
        pair=z.pivot(index='held_sequence',columns='model',values='aurc'); wins=int((pair['M2_context_plus_history']<pair['M1_instant_context']).sum()) if {'M2_context_plus_history','M1_instant_context'}.issubset(pair.columns) else 0
        verdict_rows.append(dict(service_id=s,M0_aurc=m0,M1_aurc=m1,M2_aurc=m2,constant_aurc=const,M2_relative_improvement_vs_constant=rel,M2_beats_M1=(m2<m1),M2_beats_constant=(m2<const),outer_sequence_wins_vs_M1=wins,sequence_majority=(wins>=int(go_cfg['min_outer_sequences_history_beats_current_per_service']))))
    verdict=pd.DataFrame(verdict_rows); verdict.to_csv(out/'go_no_go_by_service.csv',index=False)
    n_m1=int(verdict.M2_beats_M1.sum()); n_const=int((verdict.M2_beats_constant & (verdict.M2_relative_improvement_vs_constant>=float(go_cfg['minimum_relative_aurc_improvement_vs_constant']))).sum()); n_major=int(verdict.sequence_majority.sum())
    GO=(n_m1>=int(go_cfg['min_services_history_beats_current_aurc']) and n_const>=int(go_cfg['min_services_history_beats_constant_aurc']) and n_major>=int(go_cfg['required_services_with_sequence_majority']) and context_mode=='signed_raw_frozen_context')
    verdict_text='GO' if GO else ('INCONCLUSIVE' if context_mode!='signed_raw_frozen_context' else 'NO-GO')
    # top coefficients for interpretability
    cdf=pd.DataFrame(coeff); top=[]
    if len(cdf):
        for (svc,feat),g in cdf[cdf.model.eq('M2_context_plus_history')].groupby(['service_id','feature']):
            top.append(dict(service_id=svc,feature=feat,median_coef=float(g.standardized_coef.median()),median_abs_coef=float(g.standardized_coef.abs().median()),sign_consistency=float(max((g.standardized_coef>0).mean(),(g.standardized_coef<0).mean()))))
    top=pd.DataFrame(top).sort_values(['service_id','median_abs_coef'],ascending=[True,False]) if top else pd.DataFrame(); top.to_csv(out/'history_model_feature_stability.csv',index=False)
    report=[]
    report += ['# Nested causal service-risk estimator','',f'**Verdict: {verdict_text}**','',f'Context mode: `{context_mode}`.','']
    report += ['## Safeguards',f'- Outer test unit: one complete physical sequence; 10 outer folds.',f'- Inner calibration: {cfg["inner_group_folds"]} group-disjoint sequence folds over the other nine sequences; logistic C is frozen at 1.0.',f'- Frozen source commit: `{SOURCE_COMMIT}`; schema `{V2_SCHEMA}`.',f'- GT/local/global error columns are labels/evaluation targets only and are forbidden from model features.',('- Signed causal history is reconstructed from odometry/IMU/wheel context available at or before decision time.' if context_mode=='signed_raw_frozen_context' else '- Diagnostic fallback uses only absolute causal context; signed persistence is unavailable in this run.'),'']
    report += ['## GO/NO-GO summary','',verdict.to_markdown(index=False,floatfmt='.4f'),'']
    primary=float(cfg['primary_target_unsafe_rate']); report += [f'## Operational decisions at target unsafe-among-supported <= {primary:.0%}','']
    report += [op[np.isclose(op.target_unsafe_rate,primary)].to_markdown(index=False,floatfmt='.4f'),'']
    report += ['## Claim rule','A GO promotes the online service-risk monitor only if the history model improves matched-coverage risk over simpler baselines across the predeclared majority of services/sequences. A NO-GO means the paper should retain service-relative fidelity as the contribution but should not claim a superior online risk monitor.','']
    if context_mode!='signed_raw_frozen_context': report += ['## Diagnostic-only warning','This run used the absolute-context fallback because frozen raw traces were unavailable. It validates code flow only and cannot establish the online-monitor claim. Re-run in the repository with the raw 30-run V2 result tree.','']
    (out/'service_risk_report.md').write_text('\n'.join(report),encoding='utf-8')
    manifest=dict(analysis='nested_causal_service_risk',generated_utc=datetime.now(timezone.utc).isoformat(),verdict=verdict_text,context_mode=context_mode,source_full_loso_commit=SOURCE_COMMIT,v2_schema=V2_SCHEMA,window_table=str(wpath),raw_root=str(raw) if raw else None,config=cfg,feature_sets=fsets,forbidden_feature_tokens=FORBIDDEN_TOKENS)
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Nested service-risk analysis complete.')
    print('Context mode:',context_mode)
    print('Verdict:',verdict_text)
    print('Report:',out/'service_risk_report.md')

if __name__=='__main__': main()
