import numpy as np
import pandas as pd
import pytest

from DigitalTwin.analysis import service_timing_budget_study as base
from DigitalTwin.analysis.service_timing_robustness import (
    causal_indices, evaluate_indices, extrapolate_delivered, fixed_physics_paths,
    history_signature, identity_scores, qualification_sensitivity,
    load_fixed_on_v2,
)


def trace(n=401):
    t=np.arange(n)*.1
    return {"time_s":t,"gt_east_m":t.copy(),"gt_north_m":np.zeros(n),"gt_heading_rad":np.zeros(n),
            "estimate_east_m":t.copy(),"estimate_north_m":np.zeros(n),"estimate_heading_rad":np.zeros(n),
            "corrected_v_mps":np.ones(n),"corrected_omega_radps":np.zeros(n)}


def test_identity_uses_training_services_not_target_labels_or_delivery():
    train=pd.DataFrame([{"service":"a","qualified":1},{"service":"a","qualified":0},
                        {"service":"b","qualified":0}])
    test=pd.DataFrame([{"service":"a","rate_hz":1,"qualified":0},
                       {"service":"a","rate_hz":10,"qualified":1}])
    assert identity_scores(train,test).tolist()==[.5,.5]
    test["qualified"]=[1,0]
    assert identity_scores(train,test).tolist()==[.5,.5]


def test_identity_tied_scores_select_entire_service_or_none():
    train=pd.DataFrame([{"service":"a","qualified":1}, {"service":"a","qualified":0}])
    cells=pd.DataFrame({"service":["a"]*20,"rate_hz":[1,2,5,10]*5,"delay_ms":list(range(20))})
    scores=identity_scores(train,cells)
    assert np.unique(scores).tolist()==[.5]
    threshold,_=base._acceptance_threshold(scores,.3)
    assert int((scores>=threshold).sum()) in (0,len(cells))


def test_phase_variants_are_native_samples_only():
    t=np.arange(50)*.1
    assert len({history_signature(causal_indices(t,2,0,p)) for p in range(5)})==5
    with pytest.raises(ValueError,match="Phase"):
        causal_indices(t,2,0,5)


def test_phase_outputs_keep_physical_sequence_as_unit():
    path=base.ROOT / "results/service_timing_robustness/receiver_phase_per_sequence.csv"
    if not path.exists():
        pytest.skip("Run the robustness analysis before checking its output hierarchy")
    rows=pd.read_csv(path)
    assert rows.sequence.nunique()==10
    assert not rows.duplicated(["sequence","service","rate_hz","delay_ms","phase_native_samples","receiver"]).any()


def test_25_and_50_ms_same_represented_history_at_10hz_clock():
    t=np.arange(50)*.1
    assert history_signature(causal_indices(t,5,25))==history_signature(causal_indices(t,5,50))


def test_causal_receiver_uses_only_last_delivered_rate():
    a=trace(); idx=causal_indices(a["time_s"],2,200)
    x0=extrapolate_delivered(a,idx)[0]
    a["corrected_v_mps"][idx[35]+1:]=99
    # Compare a point before the changed source was deliverable.
    x1=extrapolate_delivered(a,idx)[0]
    assert x0[35]==x1[35]


def test_startup_and_denominator():
    a=trace(); idx=np.full(len(a["time_s"]),-1)
    r=evaluate_indices(a,base.SERVICES[0],idx)
    assert r["observable"]==0 and np.isnan(r["joint_satisfaction"])
    assert r["eligible"]<len(idx)


def test_zoh_matches_original_contract_denominator():
    a=trace(); idx=causal_indices(a["time_s"],2,200)
    other=evaluate_indices(a,base.SERVICES[0],idx)
    original=base.evaluate(a,base.SERVICES[0],2,200)
    assert other["observable"]==original["observable_samples"]
    assert other["eligible"]==original["eligible_samples"]
    assert other["joint_satisfaction"]==original["joint_satisfaction"]


def test_contract_sensitivity_keeps_all_settings():
    row={"sequence":"s","service":"global","rate_hz":10,"delay_ms":0,"coverage":1.,
         "physical_satisfaction":.85,"freshness_satisfaction":1.,"joint_satisfaction":.85}
    out=qualification_sensitivity(pd.DataFrame([row]))
    assert out.qualification_requirement.tolist()==[.7,.8,.9]
    assert out.qualified_sensitivity.tolist()==[1,1,0]


def test_fixed_configuration_requires_ten_sources():
    with pytest.raises(ValueError,match="ten"):
        fixed_physics_paths([])


def test_fixed_configuration_matches_saved_clock_and_frame():
    paths=base.discover()
    sequence=paths[0].parent.name.split("_",2)[-1]
    fixed=fixed_physics_paths(paths)[sequence]
    a=load_fixed_on_v2(paths[0],fixed)
    assert len(a["estimate_east_m"])==len(a["gt_east_m"])
    assert abs(a["estimate_east_m"][0]-a["gt_east_m"][0])<1e-5
