"""Innovation-based detectability and confidence-envelope utilities.

This module implements the proposal's structural bound:

    epsilon_min(v, tau, l) = sqrt(lambda_star * lambda_max(S_k(v, tau, l)))

The same eigenvalue expression is also the instantaneous maximum stealth bound
when lambda_star is replaced by the detector threshold gamma_star.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class DetectionResult:
    detected: bool
    mahalanobis: float
    threshold: float
    lambda_star: float
    lambda_max_s: float
    epsilon_min_m: float
    epsilon_stealth_max_m: float
    confidence: float
    envelope_region: str


def chi_square_threshold(df: int, false_alarm_probability: float) -> float:
    """Return gamma* for P(delta > gamma* | H0) = P_FA.

    For the GPS position detector m=2, the chi-square survival function has the
    closed form exp(-x/2), so no SciPy dependency is needed.  Other dimensions
    use SciPy when available and otherwise raise a clear error.
    """
    if df == 2:
        return float(-2.0 * math.log(false_alarm_probability))
    try:
        from scipy.stats import chi2
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("scipy is required for chi-square thresholds when df != 2") from exc
    return float(chi2.isf(false_alarm_probability, df))


def noncentrality_for_detection_probability(
    df: int,
    threshold: float,
    detection_probability: float,
) -> float:
    """Solve lambda* where P(delta > threshold | H1, lambda*) = P_D."""
    try:
        from scipy.optimize import brentq
        from scipy.stats import ncx2

        def objective(nc: float) -> float:
            return float(ncx2.sf(threshold, df, nc) - detection_probability)

        high = max(1.0, threshold)
        while objective(high) < 0.0:
            high *= 2.0
        return float(brentq(objective, 0.0, high, xtol=1e-9))
    except Exception:
        # Conservative analytic fallback from a normal approximation to the
        # noncentral chi-square.  SciPy is preferred for paper figures.
        z_beta = _normal_quantile(detection_probability)
        return float(max(0.0, (math.sqrt(threshold) + z_beta) ** 2 - df))


def _normal_quantile(probability: float) -> float:
    # Acklam's rational approximation, sufficient for fallback use.
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    plow = 0.02425
    phigh = 1.0 - plow
    if probability < plow:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if probability <= phigh:
        q = probability - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def lambda_max(matrix: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(np.asarray(matrix, dtype=float)).max())


def structural_detectability_bound(S: np.ndarray, lambda_star: float) -> float:
    """Paper Eq. (8): sqrt(lambda* * lambda_max(S_k))."""
    return float(math.sqrt(max(lambda_star, 0.0) * lambda_max(S)))


def instantaneous_stealth_bound(S: np.ndarray, threshold: float) -> float:
    """Paper Section 6.4: sqrt(gamma* * lambda_max(S_k))."""
    return structural_detectability_bound(S, threshold)


def confidence_score(
    mahalanobis: float,
    threshold: float,
    epsilon_min_m: float,
    blind_epsilon_m: float,
) -> float:
    divergence_margin = max(0.0, 1.0 - mahalanobis / max(threshold, 1e-12))
    sensitivity_margin = max(0.0, 1.0 - epsilon_min_m / max(blind_epsilon_m, 1e-12))
    return float(max(0.0, min(1.0, 0.55 * divergence_margin + 0.45 * sensitivity_margin)))


def envelope_region(confidence: float) -> str:
    if confidence >= 0.90:
        return "safe"
    if confidence >= 0.50:
        return "warning"
    return "blind"


class InnovationDetector:
    def __init__(
        self,
        measurement_dim: int = 2,
        false_alarm_probability: float = 0.05,
        target_detection_probability: float = 0.95,
        blind_epsilon_m: float = 5.0,
    ) -> None:
        self.measurement_dim = measurement_dim
        self.false_alarm_probability = false_alarm_probability
        self.target_detection_probability = target_detection_probability
        self.blind_epsilon_m = blind_epsilon_m
        self.threshold = chi_square_threshold(measurement_dim, false_alarm_probability)
        self.lambda_star = noncentrality_for_detection_probability(
            measurement_dim,
            self.threshold,
            target_detection_probability,
        )

    def evaluate(self, innovation: np.ndarray, S: np.ndarray) -> DetectionResult:
        innovation = np.asarray(innovation, dtype=float)
        inv_s = np.linalg.inv(S)
        mahalanobis = float(innovation.T @ inv_s @ innovation)
        eps_min = structural_detectability_bound(S, self.lambda_star)
        eps_stealth = instantaneous_stealth_bound(S, self.threshold)
        confidence = confidence_score(mahalanobis, self.threshold, eps_min, self.blind_epsilon_m)
        return DetectionResult(
            detected=mahalanobis > self.threshold,
            mahalanobis=mahalanobis,
            threshold=self.threshold,
            lambda_star=self.lambda_star,
            lambda_max_s=lambda_max(S),
            epsilon_min_m=eps_min,
            epsilon_stealth_max_m=eps_stealth,
            confidence=confidence,
            envelope_region=envelope_region(confidence),
        )
