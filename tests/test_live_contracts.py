import math

from DigitalTwin.dashboard.contracts import ContractEngine, ResourcePolicy, load_contract_config


def _point(t: float, *, twin_x: float | None = None, aoi_s: float = 0.05) -> dict[str, object]:
    x = t if twin_x is None else twin_x
    return {
        "t": t,
        "twin_x": x,
        "twin_y": 0.0,
        "twin_global_theta": 0.0,
        "gps_x": t,
        "gps_y": 0.0,
        "gps_heading_rad": 0.0,
        "gps_valid": True,
        "contract_reference_valid": True,
        "satellites": 10,
        "hdop": 0.9,
        "gps_age_s": 0.05,
        "aoi_s": aoi_s,
    }


def test_contract_engine_observability_and_violation() -> None:
    config = load_contract_config()
    engine = ContractEngine(config)
    history = [_point(index * 0.2) for index in range(56)]

    contracts, _ = engine.evaluate(history)
    by_id = {item["service_id"]: item for item in contracts}
    assert all(item["status"] == "qualified" for item in contracts)
    assert math.isclose(by_id["local_1s_tight"]["position_error_m"], 0.0, abs_tol=1e-12)

    history.append(_point(11.2, twin_x=12.5))
    contracts, events = engine.evaluate(history)
    by_id = {item["service_id"]: item for item in contracts}
    assert by_id["global_state_tracking"]["status"] == "withdrawn"
    assert any(event["to"] == "withdrawn" for event in events)


def test_contract_engine_marks_missing_heading_unobservable() -> None:
    engine = ContractEngine(load_contract_config())
    point = _point(0.0)
    point["gps_heading_rad"] = None
    point["contract_reference_valid"] = False
    contracts, _ = engine.evaluate([point])
    assert all(item["status"] == "unobservable" for item in contracts)


def test_global_contract_withdraws_on_known_position_violation_without_course() -> None:
    engine = ContractEngine(load_contract_config())
    point = _point(0.0, twin_x=2.0)
    point["gps_heading_rad"] = None
    point["twin_global_theta"] = None
    point["contract_reference_valid"] = False
    contracts, _ = engine.evaluate([point])
    by_id = {item["service_id"]: item for item in contracts}
    assert by_id["global_state_tracking"]["status"] == "withdrawn"
    assert by_id["local_1s_tight"]["status"] == "unobservable"


def test_contract_aware_policy_escalates_and_holds() -> None:
    config = load_contract_config()
    policy = ResourcePolicy("contract-aware", config)
    contracts = [{"status": "withdrawn"}]
    event = policy.update(3.0, 0.05, contracts)
    assert event is not None
    assert policy.mode == "high"
    assert policy.update_rate_hz == 10.0
