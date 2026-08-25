#!/usr/bin/env python3
"""Small deterministic self-test for service-relative fidelity helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from DigitalTwin.analysis.i2nav_service_relative_fidelity import (
    relative_pose, wrap_angle, decision_metrics, feature_bin
)

def main():
    # Translation offset should disappear from relative motion, but not global state.
    t=np.arange(0,11,dtype=float)
    gx=t.copy(); gy=np.zeros_like(t); gh=np.zeros_like(t)
    ex=t+5.0; ey=np.full_like(t,2.0); eh=np.zeros_like(t)
    gp=relative_pose(gx,gy,gh,0,10)
    ep=relative_pose(ex,ey,eh,0,10)
    assert np.allclose(gp,ep,atol=1e-12), (gp,ep)
    assert abs(np.hypot(ex[-1]-gx[-1],ey[-1]-gy[-1])-np.sqrt(29))<1e-12

    # Angle wrapping.
    assert abs(float(wrap_angle(np.deg2rad(181))) - np.deg2rad(-179)) < 1e-12

    # Decision accounting.
    truth=np.array([1,1,0,0],bool); supported=np.array([1,0,1,0],bool)
    m=decision_metrics(truth,supported)
    assert abs(m["false_safe_fraction"]-.25)<1e-12
    assert abs(m["false_reject_fraction"]-.25)<1e-12

    # Stable bins.
    b=feature_bin(pd.Series([0,1,2,3,np.nan]),1.0,2.0)
    assert b.tolist()==[0,0,1,2,-1],b
    print("service_relative_fidelity_selftest: PASS")

if __name__=="__main__": main()
