from __future__ import annotations
import numpy as np, pandas as pd
from DigitalTwin.analysis.service_contract_e1_e2 import relative_pose, compute_trace_windows, service_grid_from_windows, DEFAULT_CONFIG

def main():
    t=np.arange(0,31,.1); x=t; y=np.zeros_like(t); th=np.zeros_like(t)
    aligned=pd.DataFrame({'time_s':t,'rtk_east_m':x,'rtk_north_m':y,'reference_heading_rad':th})
    # Perfect twin must pass every tolerance.
    trace=pd.DataFrame({'time_s':t,'x_T_m':x,'y_T_m':y,'theta_T_rad':th})
    w=compute_trace_windows(aligned,trace,[1.,5.,10.]); g=service_grid_from_windows(w,DEFAULT_CONFIG)
    assert len(w)>0 and np.allclose(g.position_valid_fraction,1) and np.allclose(g.joint_valid_fraction,1)
    # Constant global offset should not affect local relative motion but should affect global service.
    trace2=trace.copy(); trace2['x_T_m'] += 10.0
    w2=compute_trace_windows(aligned,trace2,[1.,5.,10.])
    assert float(w2[w2.family=='local_relative_motion'].position_error_m.max()) < 1e-9
    assert float(w2[w2.family=='global_synchronization'].position_error_m.min()) > 9.9
    print('service_contract_e1_e2_selftest: PASS')
if __name__=='__main__': main()
