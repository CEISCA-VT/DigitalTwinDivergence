"""Extended Kalman filter for GPS-aided differential-drive localization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .kinematics import integrate_unicycle, wrap_angle


@dataclass(slots=True)
class EKFState:
    x: np.ndarray
    P: np.ndarray


class RoverEKF:
    """State x = [east_m, north_m, heading_rad]."""

    def __init__(
        self,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        self.state = EKFState(
            x=np.array(initial_state if initial_state is not None else [0.0, 0.0, 0.0], dtype=float),
            P=np.array(initial_covariance if initial_covariance is not None else np.diag([2.0, 2.0, 0.5]), dtype=float),
        )
        self.last_innovation = np.zeros(2)
        self.last_S = np.eye(2)
        self.last_K = np.zeros((3, 2))
        self.last_mahalanobis = 0.0

    def gps_innovation(self, z_xy: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        z_xy = np.array(z_xy, dtype=float)
        innovation = z_xy - H @ self.state.x
        S = H @ self.state.P @ H.T + R
        return innovation, S

    def predict(self, v_mps: float, omega_radps: float, dt_s: float, Q: np.ndarray) -> EKFState:
        x = self.state.x
        theta = float(x[2])
        theta_mid = theta + 0.5 * omega_radps * dt_s

        F = np.eye(3)
        F[0, 2] = -v_mps * math.sin(theta_mid) * dt_s
        F[1, 2] = v_mps * math.cos(theta_mid) * dt_s

        self.state.x = integrate_unicycle(x, v_mps, omega_radps, dt_s)
        self.state.P = F @ self.state.P @ F.T + Q
        self.state.P = 0.5 * (self.state.P + self.state.P.T)
        return self.state

    def update_gps(self, z_xy: np.ndarray, R: np.ndarray, *, measurement_weight: float = 1.0) -> EKFState:
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        weight = max(float(measurement_weight), 1e-6)
        R = np.asarray(R, dtype=float) / (weight * weight)
        innovation, S = self.gps_innovation(z_xy, R)
        S = H @ self.state.P @ H.T + R
        K = self.state.P @ H.T @ np.linalg.inv(S)

        self.state.x = self.state.x + K @ innovation
        self.state.x[2] = wrap_angle(float(self.state.x[2]))
        I = np.eye(3)
        self.state.P = (I - K @ H) @ self.state.P @ (I - K @ H).T + K @ R @ K.T
        self.state.P = 0.5 * (self.state.P + self.state.P.T)

        self.last_innovation = innovation
        self.last_S = S
        self.last_K = K
        self.last_mahalanobis = float(innovation.T @ np.linalg.inv(S) @ innovation)
        return self.state
