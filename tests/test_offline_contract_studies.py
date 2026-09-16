from DigitalTwin.analysis.e1_rank_inference import mahonian_counts
from DigitalTwin.analysis.live_position_contract_study import evaluate_position


QUALITY = {
    "minimum_satellites": 4,
    "maximum_hdop": 2.5,
    "maximum_gps_age_s": 1.5,
    "maximum_horizon_match_error_s": 0.35,
}


def point(t: float, twin_x: float = 0.0) -> dict:
    return {
        "t": t,
        "gps_valid": True,
        "gps_x": t,
        "gps_y": 0.0,
        "twin_x": twin_x,
        "twin_y": 0.0,
        "satellites": 10,
        "hdop": 0.9,
        "gps_age_s": 0.1,
        "aoi_s": 0.1,
    }


def test_position_only_contract_has_three_states() -> None:
    spec = {"service_id": "global", "horizon_s": 0.0, "position_tolerance_m": 1.0, "maximum_aoi_s": 1.0}
    assert evaluate_position([point(0.0)], spec, QUALITY)["state"] == "pass"
    assert evaluate_position([point(0.0, twin_x=2.0)], spec, QUALITY)["state"] == "fail"
    invalid = point(0.0)
    invalid["gps_valid"] = False
    assert evaluate_position([invalid], spec, QUALITY)["state"] == "unobservable"


def test_exact_inversion_distribution_for_ten_sequences() -> None:
    counts = mahonian_counts(10)
    assert len(counts) == 46
    assert sum(counts) == 3_628_800
    assert counts == list(reversed(counts))
