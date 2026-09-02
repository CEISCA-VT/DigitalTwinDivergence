from DigitalTwin.dashboard.automated_motion import PROFILES, scaled_plan


def test_motion_profiles_have_safety_stops() -> None:
    for name, steps in PROFILES.items():
        assert steps
        assert steps[0].command == "stop"
        assert steps[-1].command == "stop"
        assert all(step.duration_s > 0.0 for step in steps)
        assert name == "stop_only" or any(step.command != "stop" for step in steps)


def test_scaled_plan_preserves_order_and_duration() -> None:
    plan = scaled_plan("turning_intensive", 60.0)

    assert plan[0].command == "stop"
    assert plan[-1].command == "stop"
    assert [step.command for step in plan] == [step.command for step in PROFILES["turning_intensive"]]
    assert abs(sum(step.duration_s for step in plan) - 60.0) < 1e-9
