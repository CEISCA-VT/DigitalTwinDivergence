import numpy as np
import pandas as pd

from DigitalTwin.analysis.service_timing_budget_study import (
    SERVICES,
    _acceptance_threshold,
    delivered_indices,
    evaluate,
    prediction_summary,
    recovery_component_summary,
)


def trace(n=501):
    t=np.arange(n)*.1
    return {"time_s":t,"gt_east_m":t,"gt_north_m":np.zeros(n),"gt_heading_rad":np.zeros(n),
            "estimate_east_m":t.copy(),"estimate_north_m":np.zeros(n),"estimate_heading_rad":np.zeros(n),
            "odo_speed_mps":np.ones(n),"imu_yaw_rate_radps":np.zeros(n)}


def test_delivery_is_causal_and_source_rate_limited():
    t=trace()["time_s"]
    ideal=delivered_indices(t,10,0)
    assert np.array_equal(ideal,np.arange(len(t)))
    held=delivered_indices(t,2,.2)
    valid=held>=0
    assert np.all(t[held[valid]]+.2 <= t[valid]+1e-12)
    assert set(np.diff(np.unique(held[valid]))) == {5}


def test_startup_horizon_and_separate_contract_components():
    a=trace()
    r=evaluate(a,SERVICES[0],2,200)
    assert r["coverage"] > .99
    assert r["physical_satisfaction"] == 1.0
    assert r["freshness_satisfaction"] == 1.0
    assert r["joint_satisfaction"] == 1.0
    assert r["eligible_samples"] < len(a["time_s"])


def test_zero_acceptance_is_undefined_error():
    pred=pd.DataFrame([{"sequence":"a","method":"m","score":0.0,"actual_qualified":1},
                       {"sequence":"b","method":"m","score":0.0,"actual_qualified":0}])
    per,_=prediction_summary(pred)
    row=per[per.threshold==.8].iloc[0]
    assert row.acceptance == 0
    assert np.isnan(row.false_qualification)


def test_matched_acceptance_threshold_does_not_need_labels():
    threshold, achieved=_acceptance_threshold([.9,.8,.7,.6,.5,.4,.3,.2,.1,0],.3)
    assert threshold == .7
    assert achieved == .3


def test_recovery_components_keep_explicit_denominators():
    rem=pd.DataFrame([
        {"classification":"delivery_remediable","practical_restores":1,"degraded_failure_component":"physical_only"},
        {"classification":"delivery_remediable","practical_restores":1,"degraded_failure_component":"freshness_only"},
        {"classification":"delivery_remediable","practical_restores":0,"degraded_failure_component":"physical_and_freshness"},
        {"classification":"persistent_under_ideal","practical_restores":0,"degraded_failure_component":"physical_only"},
    ])
    out=recovery_component_summary(rem)
    delivery=out[out.population=="delivery_remediable"]
    practical=out[out.population=="practical_recoveries"]
    assert delivery.population_denominator.unique().tolist() == [3]
    assert delivery.cases.sum() == 3
    assert practical.population_denominator.unique().tolist() == [2]
    assert practical.cases.sum() == 2
