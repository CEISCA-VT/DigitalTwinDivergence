#!/usr/bin/env python3
from __future__ import annotations
import numpy as np, pandas as pd
from DigitalTwin.analysis.i2nav_nested_service_risk import add_history, assert_feature_safety, inner_splits, select_threshold

def main():
    # Causality: a future spike at t=20 must not alter a history statistic at t=10.
    t=np.arange(0,21,dtype=float); z=np.zeros_like(t); z[-1]=1000
    d=pd.DataFrame({'sequence':'s','time_s':t,'speed_signed_mps':0.,'abs_speed_mps':0.,'yaw_rate_signed_radps':0.,'abs_yaw_rate_radps':0.,'accel_signed_mps2':0.,'abs_accel_mps2':0.,'wheel_imu_disagreement_signed_radps':z,'abs_wheel_imu_disagreement_radps':np.abs(z),'curvature_signed_radpm':0.,'curvature_abs_radpm':0.,'elapsed_s':t})
    h=add_history(d,[5.,10.]); v=float(h.loc[h.time_s.eq(10),'wheel_imu_disagreement_signed_radps_mean_10s'].iloc[0]); assert abs(v)<1e-12,v
    # Leakage guard.
    try:
        assert_feature_safety(['abs_speed_mps','global_position_error_m']); raise AssertionError('leakage guard failed')
    except RuntimeError: pass
    # Group split contains whole sequences only.
    folds=inner_splits(list('abcdefghi'),3); assert sorted(sum(folds,[]))==list('abcdefghi'); assert all(len(x)==3 for x in folds)
    # No feasible support if every supported point is unsafe at target zero.
    o=pd.DataFrame({'sequence':['a','a','b','b'],'y':[1,1,1,1],'p':[.1,.2,.3,.4]}); th,_=select_threshold(o,0.0); assert th is None
    print('service_risk_estimator_selftest: PASS')
if __name__=='__main__': main()
