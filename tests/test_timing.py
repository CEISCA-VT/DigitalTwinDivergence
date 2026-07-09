from DigitalTwin.timing import SessionClockCalibrator


def test_session_clock_calibrator_uses_median_offset():
    clock = SessionClockCalibrator(window_size=5, min_samples=3)

    first = clock.observe(remote_time_s=10.0, edge_time_s=110.0)
    second = clock.observe(remote_time_s=11.0, edge_time_s=111.1)
    third = clock.observe(remote_time_s=12.0, edge_time_s=112.0)

    assert first.calibrated is False
    assert second.calibrated is False
    assert third.calibrated is True
    assert round(third.offset_s, 3) == 100.0
    assert round(clock.edge_time_from_remote(20.0), 3) == 120.0
