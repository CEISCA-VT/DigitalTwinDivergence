#!/usr/bin/env python3
"""Repair/recompute only Fixed-Physics and EKF-IW outputs without retraining learned baselines.

Existing LWOI/YNet entries in baseline_manifest.csv are preserved. This is useful
after the EKF-IW correction because the full learned baseline training need not be
repeated merely to replace the deterministic Fixed/EKF trajectories.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import pandas as pd

from .common import discover_i2nav_corpus, save_json
from .fixed_physics import run_fixed_physics, METHOD_NAME as FIXED_NAME
from .ekf_iw import fit_config, run_ekf_iw, METHOD_NAME as EKF_NAME


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-root',default='results/i2nav_v2_full_loso/i2nav_v2_full_loso')
    p.add_argument('--glob',default='**/v2_evaluated_trajectory.csv')
    p.add_argument('--output-root',default='results/i2nav_external_baselines')
    p.add_argument('--test-sequences',default='')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()

    corpus=discover_i2nav_corpus(a.input_root,a.glob)
    seqs=sorted(corpus)
    if a.test_sequences.strip():
        tests=[s.strip() for s in a.test_sequences.split(',') if s.strip()]
    elif a.smoke:
        tests=[s for s in ['parking00','parking02'] if s in corpus]
    else:
        tests=seqs

    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'baseline_manifest.csv'
    old=pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    # Preserve all unrelated methods, and during a smoke repair preserve untouched
    # Fixed/EKF sequences as well. Only rows that are about to be rewritten are removed.
    if len(old):
        drop = old['method_key'].isin(['fixed_recomputed','ekf_iw']) & old['sequence'].astype(str).isin(tests)
        old=old[~drop].copy()

    new=[]
    for i,test in enumerate(tests,1):
        train_seq=[s for s in seqs if s!=test]
        train=[corpus[s].data for s in train_seq]
        d=corpus[test].data
        print('='*84); print(f'REPAIR {i}/{len(tests)} test={test}')

        fp=out/'Fixed_Physics_Recomputed'/test/'evaluated_trajectory.csv'
        fp.parent.mkdir(parents=True,exist_ok=True)
        run_fixed_physics(d).to_csv(fp,index=False)
        new.append(dict(method_key='fixed_recomputed',method=FIXED_NAME,sequence=test,seed='deterministic',
                        trajectory=str(fp),test_sequence=test,train_sequences=';'.join(train_seq),
                        n_train_sequences=len(train_seq),adaptation_level='recomputed raw-input sanity baseline; operational/un-aligned TFP must be kept separate from official aligned i2Nav APE',status='ok',error=''))
        print('  WROTE',fp)

        cfg=fit_config(train)
        ep=out/'EKF_IW'/test/'evaluated_trajectory.csv'; ep.parent.mkdir(parents=True,exist_ok=True)
        save_json(ep.with_name('ekf_config.json'),asdict(cfg))
        run_ekf_iw(d,cfg).to_csv(ep,index=False)
        new.append(dict(method_key='ekf_iw',method=EKF_NAME,sequence=test,seed='deterministic',
                        trajectory=str(ep),test_sequence=test,train_sequences=';'.join(train_seq),
                        n_train_sequences=len(train_seq),adaptation_level='corrected planar classical EKF-IW: LOSO sensor calibration, wheel-yaw innovation gating, no invalid stationary-zero-yaw assumption; not WING full-state reproduction',status='ok',error=''))
        print('  WROTE',ep)

    merged=pd.concat([old,pd.DataFrame(new)],ignore_index=True) if len(old) else pd.DataFrame(new)
    merged=merged.sort_values(['method_key','sequence','seed']).reset_index(drop=True)
    merged.to_csv(manifest_path,index=False)
    print('='*84)
    print('Preserved non-repaired learned-baseline manifest entries.')
    print('Wrote',manifest_path)

if __name__=='__main__': main()
