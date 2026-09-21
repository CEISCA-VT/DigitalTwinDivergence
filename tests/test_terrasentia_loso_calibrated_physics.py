import numpy as np
import pandas as pd

from DigitalTwin.analysis.terrasentia_loso_calibrated_physics import delivered_indices, propagate


def test_propagate_straight_line() -> None:
    frame = pd.DataFrame({
        "time_s": [0.0, 0.1, 0.2], "rtk_east_m": [0.0, 0.1, 0.2],
        "rtk_north_m": [0.0, 0.0, 0.0], "reference_heading_rad": [0.0, 0.0, 0.0],
        "forward_motor_speed_mps": [1.0, 1.0, 1.0], "imu_yaw_rate_radps": [0.0, 0.0, 0.0],
    })
    trace = propagate(frame, np.array([1.0, 1.0, 0.0]), "test")
    np.testing.assert_allclose(trace.x_T_m, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(trace.y_T_m, 0.0, atol=1e-12)


def test_delivery_indices_include_delay() -> None:
    t = np.arange(0.0, 1.01, 0.1)
    indices = delivered_indices(t, rate_hz=2.0, delay_s=0.2)
    assert indices[1] == -1
    assert indices[2] == 0
    assert indices[7] == 5
