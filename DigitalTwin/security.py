"""GPS-independent security prediction and trusted innovation gating."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .ekf import RoverEKF


@dataclass(frozen=True, slots=True)
class CovarianceAdaptationPolicy:
    """Bounds and smoothing for the process-covariance recursion."""

    smoothing_factor: float = 0.90
    minimum_diagonal: tuple[float, float, float] = (1e-10, 1e-10, 1e-12)
    maximum_diagonal: tuple[float, float, float] = (0.25, 0.25, 0.04)


@dataclass(frozen=True, slots=True)
class TrustedGatePolicy:
    """Frozen parameters for the PDF's trusted-whitened evidence gate."""

    soft_nis_threshold: float = 10.480551254279684
    persistent_bias_threshold: float = 155.30477241316595
    bias_memory: float = 0.90
    timing_mismatch_s: float = 0.20
    reject_stale_packets: bool = True
    reject_sequence_gaps: bool = True


@dataclass(frozen=True, slots=True)
class GateDecision:
    allowed: bool
    edge_evidence_ok: bool
    trusted_nis: float
    persistent_bias_score: float
    whitened_innovation: np.ndarray


DEFAULT_COVARIANCE_ADAPTATION_POLICY = CovarianceAdaptationPolicy()
DEFAULT_TRUSTED_GATE_POLICY = TrustedGatePolicy()


class BoundedCovarianceAdapter:
    """Implement Q_k = phi Q_(k-1) + (1-phi) clip(Q_tilde_k)."""

    def __init__(
        self,
        policy: CovarianceAdaptationPolicy = DEFAULT_COVARIANCE_ADAPTATION_POLICY,
    ) -> None:
        self.policy = policy
        self.current: np.ndarray | None = None
        self.accepted_updates = 0

    def _clip(self, proposal: np.ndarray) -> np.ndarray:
        matrix = np.asarray(proposal, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("process covariance proposal must be 3x3")
        diagonal = np.diag(matrix)
        clipped = np.clip(
            diagonal,
            np.asarray(self.policy.minimum_diagonal, dtype=float),
            np.asarray(self.policy.maximum_diagonal, dtype=float),
        )
        return np.diag(clipped)

    def update(self, proposal: np.ndarray, *, allowed: bool = True) -> np.ndarray:
        clipped = self._clip(proposal)
        if self.current is None:
            self.current = clipped
            self.accepted_updates = int(allowed)
            return self.current.copy()
        if allowed:
            phi = self.policy.smoothing_factor
            self.current = phi * self.current + (1.0 - phi) * clipped
            self.current = self._clip(self.current)
            self.accepted_updates += 1
        return self.current.copy()


class TrustedInnovationGate:
    """Trusted-whitened instantaneous and persistent-bias gate."""

    def __init__(
        self,
        measurement_dim: int = 2,
        policy: TrustedGatePolicy = DEFAULT_TRUSTED_GATE_POLICY,
    ) -> None:
        self.measurement_dim = int(measurement_dim)
        self.policy = policy
        self.bias_state = np.zeros(self.measurement_dim, dtype=float)
        self.updates = 0

    def evaluate(
        self,
        innovation: np.ndarray,
        trusted_covariance: np.ndarray,
        *,
        edge_evidence_ok: bool,
    ) -> GateDecision:
        innovation = np.asarray(innovation, dtype=float)
        covariance = np.asarray(trusted_covariance, dtype=float)
        if innovation.shape != (self.measurement_dim,):
            raise ValueError("innovation has the wrong measurement dimension")
        if covariance.shape != (self.measurement_dim, self.measurement_dim):
            raise ValueError("trusted covariance has the wrong shape")

        chol = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(chol, innovation)
        trusted_nis = float(whitened @ whitened)

        memory = self.policy.bias_memory
        self.bias_state = memory * self.bias_state + (1.0 - memory) * whitened
        self.updates += 1
        variance = (1.0 - memory) / (1.0 + memory)
        variance *= 1.0 - memory ** (2 * self.updates)
        persistent_score = float(
            self.bias_state @ self.bias_state / max(variance, np.finfo(float).eps)
        )
        allowed = bool(
            edge_evidence_ok
            and trusted_nis <= self.policy.soft_nis_threshold
            and persistent_score <= self.policy.persistent_bias_threshold
        )
        return GateDecision(
            allowed=allowed,
            edge_evidence_ok=bool(edge_evidence_ok),
            trusted_nis=trusted_nis,
            persistent_bias_score=persistent_score,
            whitened_innovation=whitened.copy(),
        )


class SecurityPredictor:
    """EKF-style prediction branch that never accepts GPS corrections."""

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray) -> None:
        self.filter = RoverEKF(initial_state, initial_covariance)

    @property
    def state(self):
        return self.filter.state

    def predict(self, v_mps: float, omega_radps: float, dt_s: float, Q: np.ndarray) -> None:
        self.filter.predict(v_mps, omega_radps, dt_s, Q)

    def innovation(
        self,
        gps_xy: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.filter.gps_innovation(gps_xy, measurement_covariance)


def initial_security_heading(
    gps_xy: np.ndarray,
    start_index: int,
    *,
    lookahead_updates: int = 5,
) -> float:
    """Estimate the initial local-frame heading from the clean settling prefix."""

    points = np.asarray(gps_xy, dtype=float)
    if len(points) < 2:
        return 0.0
    first = min(max(int(start_index), 0), len(points) - 1)
    last = min(first + max(int(lookahead_updates), 1), len(points) - 1)
    displacement = points[last] - points[first]
    if float(np.linalg.norm(displacement)) < 1e-6:
        return 0.0
    return math.atan2(float(displacement[1]), float(displacement[0]))
