import numpy as np

from DigitalTwin.ekf import RoverEKF


def test_ekf_predict_update_reduces_position_error():
    ekf = RoverEKF(initial_state=np.array([0.0, 0.0, 0.0]), initial_covariance=np.diag([1.0, 1.0, 0.1]))
    ekf.predict(1.0, 0.0, 1.0, np.diag([0.01, 0.01, 0.001]))
    before = np.linalg.norm(ekf.state.x[:2] - np.array([1.2, 0.1]))
    ekf.update_gps(np.array([1.2, 0.1]), np.diag([0.04, 0.04]))
    after = np.linalg.norm(ekf.state.x[:2] - np.array([1.2, 0.1]))
    assert after < before
