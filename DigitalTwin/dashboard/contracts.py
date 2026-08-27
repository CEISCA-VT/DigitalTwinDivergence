"""Causal operational service contracts for the UGV01 live twin.

The engine implements the manuscript's quantity x horizon x tolerance x age
contract. It consumes synchronized, common-frame GPS and twin samples and does
not train, align, or tune the twin from evaluation outcomes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ugv01_live_service_contracts.json"
STATUS_ORDER = {"qualified": 0, "at_risk": 1, "unobservable": 2, "withdrawn": 3}
MODE_ORDER = {"economy": 0, "normal": 1, "high": 2}


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def relative_pose(x: float, y: float, heading: float, x1: float, y1: float, heading1: float) -> tuple[float, float, float]:
    dx = float(x1) - float(x)
    dy = float(y1) - float(y)
    c = math.cos(float(heading))
    s = math.sin(float(heading))
    return c * dx + s * dy, -s * dx + c * dy, wrap_angle(float(heading1) - float(heading))


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    label: str
    family: str
    horizon_s: float
    position_tolerance_m: float
    heading_tolerance_deg: float
    maximum_aoi_s: float


def load_contract_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema") != "ugv01_live_service_contracts_v1":
        raise ValueError("Unsupported UGV01 live contract configuration")
    return config


class ContractEngine:
    """Evaluate and retain the lifecycle state of each live service contract."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.specs = [ServiceSpec(**item) for item in config["services"]]
        self._states = {spec.service_id: "unobservable" for spec in self.specs}
        self._recovery_since: dict[str, float | None] = {spec.service_id: None for spec in self.specs}
        self._observed = {spec.service_id: 0 for spec in self.specs}
        self._qualified = {spec.service_id: 0 for spec in self.specs}

    def evaluate(self, history: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if not history:
            return [], []
        now = history[-1]
        results: list[dict[str, object]] = []
        events: list[dict[str, object]] = []
        for spec in self.specs:
            result = self._evaluate_service(spec, history, now)
            previous = self._states[spec.service_id]
            status = self._lifecycle_status(spec, result, previous, float(now["t"]))
            result["status"] = status
            if status != "unobservable":
                self._observed[spec.service_id] += 1
                if status in {"qualified", "at_risk"}:
                    self._qualified[spec.service_id] += 1
            result["observed_evaluations"] = self._observed[spec.service_id]
            result["satisfaction_fraction"] = (
                self._qualified[spec.service_id] / self._observed[spec.service_id]
                if self._observed[spec.service_id]
                else None
            )
            if status != previous:
                events.append(
                    {
                        "t": float(now["t"]),
                        "type": "contract_state",
                        "service_id": spec.service_id,
                        "from": previous,
                        "to": status,
                        "reason": result["reason"],
                    }
                )
            self._states[spec.service_id] = status
            results.append(result)
        return results, events

    def _evaluate_service(self, spec: ServiceSpec, history: list[dict[str, object]], now: dict[str, object]) -> dict[str, object]:
        base = {
            "service_id": spec.service_id,
            "label": spec.label,
            "family": spec.family,
            "horizon_s": spec.horizon_s,
            "position_tolerance_m": spec.position_tolerance_m,
            "heading_tolerance_deg": spec.heading_tolerance_deg,
            "maximum_aoi_s": spec.maximum_aoi_s,
            "position_error_m": None,
            "heading_error_deg": None,
            "aoi_s": now.get("aoi_s"),
            "position_margin_m": None,
            "heading_margin_deg": None,
            "freshness_margin_s": None,
            "normalized_margin": None,
            "observable": False,
            "raw_pass": False,
            "reason": "reference unavailable",
        }
        require_heading = spec.family != "global_synchronization"
        quality_ok, quality_reason = self._reference_quality(now, require_heading=require_heading)
        if not quality_ok:
            base["reason"] = quality_reason
            return base
        aoi = now.get("aoi_s")
        if aoi is None or not math.isfinite(float(aoi)):
            base["reason"] = "source age unavailable"
            return base

        if spec.family == "global_synchronization":
            position_error = math.hypot(
                float(now["twin_x"]) - float(now["gps_x"]),
                float(now["twin_y"]) - float(now["gps_y"]),
            )
            heading_available = now.get("gps_heading_rad") is not None and now.get("twin_global_theta") is not None
            if not heading_available:
                position_margin = spec.position_tolerance_m - position_error
                freshness_margin = spec.maximum_aoi_s - float(aoi)
                base.update(
                    {
                        "position_error_m": position_error,
                        "position_margin_m": position_margin,
                        "freshness_margin_s": freshness_margin,
                    }
                )
                if position_margin >= 0.0 and freshness_margin >= 0.0:
                    base["reason"] = "GPS course unobservable at low speed"
                    return base
                normalized_margin = min(
                    position_margin / spec.position_tolerance_m,
                    freshness_margin / spec.maximum_aoi_s,
                )
                base.update(
                    {
                        "normalized_margin": normalized_margin,
                        "observable": True,
                        "raw_pass": False,
                        "reason": self._failure_reason(position_margin, math.inf, freshness_margin),
                    }
                )
                return base
            heading_error = abs(math.degrees(wrap_angle(float(now["twin_global_theta"]) - float(now["gps_heading_rad"]))))
        else:
            target_t = float(now["t"]) - spec.horizon_s
            candidates = [item for item in history[:-1] if item.get("contract_reference_valid")]
            if not candidates:
                base["reason"] = f"waiting for {spec.horizon_s:g} s history"
                return base
            start = min(candidates, key=lambda item: abs(float(item["t"]) - target_t))
            maximum_gap = float(self.config["reference_quality"]["maximum_horizon_match_error_s"])
            if abs(float(start["t"]) - target_t) > maximum_gap:
                base["reason"] = f"no synchronized {spec.horizon_s:g} s window"
                return base
            physical = relative_pose(
                float(start["gps_x"]), float(start["gps_y"]), float(start["gps_heading_rad"]),
                float(now["gps_x"]), float(now["gps_y"]), float(now["gps_heading_rad"]),
            )
            virtual = relative_pose(
                float(start["twin_x"]), float(start["twin_y"]), float(start["twin_global_theta"]),
                float(now["twin_x"]), float(now["twin_y"]), float(now["twin_global_theta"]),
            )
            position_error = math.hypot(virtual[0] - physical[0], virtual[1] - physical[1])
            heading_error = abs(math.degrees(wrap_angle(virtual[2] - physical[2])))

        position_margin = spec.position_tolerance_m - position_error
        heading_margin = spec.heading_tolerance_deg - heading_error
        freshness_margin = spec.maximum_aoi_s - float(aoi)
        normalized_margin = min(
            position_margin / spec.position_tolerance_m,
            heading_margin / spec.heading_tolerance_deg,
            freshness_margin / spec.maximum_aoi_s,
        )
        raw_pass = normalized_margin >= 0.0
        base.update(
            {
                "position_error_m": position_error,
                "heading_error_deg": heading_error,
                "position_margin_m": position_margin,
                "heading_margin_deg": heading_margin,
                "freshness_margin_s": freshness_margin,
                "normalized_margin": normalized_margin,
                "observable": True,
                "raw_pass": raw_pass,
                "reason": "within contract" if raw_pass else self._failure_reason(position_margin, heading_margin, freshness_margin),
            }
        )
        return base

    def _reference_quality(self, point: dict[str, object], *, require_heading: bool = True) -> tuple[bool, str]:
        if not point.get("gps_valid"):
            return False, "GPS fix unavailable"
        if int(point.get("satellites", 0)) < int(self.config["reference_quality"]["minimum_satellites"]):
            return False, "insufficient satellites"
        if float(point.get("hdop", math.inf)) > float(self.config["reference_quality"]["maximum_hdop"]):
            return False, "HDOP quality gate"
        gps_age = point.get("gps_age_s")
        if gps_age is None or float(gps_age) > float(self.config["reference_quality"]["maximum_gps_age_s"]):
            return False, "stale GPS fix"
        if require_heading and (point.get("gps_heading_rad") is None or point.get("twin_global_theta") is None):
            return False, "GPS course unobservable at low speed"
        return True, "reference valid"

    def _lifecycle_status(self, spec: ServiceSpec, result: dict[str, object], previous: str, now_s: float) -> str:
        if not result["observable"]:
            self._recovery_since[spec.service_id] = None
            return "unobservable"
        if not result["raw_pass"]:
            self._recovery_since[spec.service_id] = None
            return "withdrawn"

        warning = float(self.config["state_machine"]["warning_margin_fraction"])
        candidate = "at_risk" if float(result["normalized_margin"]) <= warning else "qualified"
        if previous == "withdrawn":
            if self._recovery_since[spec.service_id] is None:
                self._recovery_since[spec.service_id] = now_s
            recovery = float(self.config["state_machine"]["recovery_interval_s"])
            if now_s - float(self._recovery_since[spec.service_id]) < recovery:
                result["reason"] = "recovery interval"
                return "withdrawn"
        self._recovery_since[spec.service_id] = None
        return candidate

    @staticmethod
    def _failure_reason(position_margin: float, heading_margin: float, freshness_margin: float) -> str:
        failures = []
        if position_margin < 0.0:
            failures.append("position tolerance")
        if heading_margin < 0.0:
            failures.append("heading tolerance")
        if freshness_margin < 0.0:
            failures.append("AoI limit")
        return ", ".join(failures) or "contract violation"


class ResourcePolicy:
    """Select a frozen telemetry/update mode without changing the twin model."""

    def __init__(self, policy: str, config: dict[str, Any]) -> None:
        allowed = {"static-low", "static-high", "aoi-only", "contract-aware"}
        if policy not in allowed:
            raise ValueError(f"Unsupported resource policy: {policy}")
        self.policy = policy
        self.config = config
        self.mode = "economy" if policy == "static-low" else ("high" if policy == "static-high" else "normal")
        self._last_switch_s = 0.0
        self._downshift_candidate_since: float | None = None

    @property
    def update_rate_hz(self) -> float:
        return float(self.config["resource_modes"][self.mode]["update_rate_hz"])

    def snapshot(self, aoi_s: float | None, contracts: list[dict[str, object]]) -> dict[str, object]:
        desired, reason = self._desired_mode(aoi_s, contracts)
        mode_cfg = self.config["resource_modes"][self.mode]
        return {
            "policy": self.policy,
            "current_mode": self.mode,
            "desired_mode": desired,
            "reason": reason,
            "update_rate_hz": self.update_rate_hz,
            "relative_cost": float(mode_cfg["relative_cost"]),
            "aoi_s": aoi_s,
            "aoi_normal_trigger_s": float(self.config["resource_policy"]["aoi_normal_trigger_s"]),
            "aoi_high_trigger_s": float(self.config["resource_policy"]["aoi_high_trigger_s"]),
            "contract_statuses": {str(item["service_id"]): str(item["status"]) for item in contracts},
        }

    def update(self, now_s: float, aoi_s: float | None, contracts: list[dict[str, object]]) -> dict[str, object] | None:
        desired, reason = self._desired_mode(aoi_s, contracts)
        if desired == self.mode:
            self._downshift_candidate_since = None
            return None
        escalating = MODE_ORDER[desired] > MODE_ORDER[self.mode]
        policy_cfg = self.config["resource_policy"]
        if escalating:
            if now_s - self._last_switch_s < float(policy_cfg["minimum_dwell_s"]):
                return None
        else:
            if self._downshift_candidate_since is None:
                self._downshift_candidate_since = now_s
                return None
            if now_s - self._downshift_candidate_since < float(policy_cfg["downshift_dwell_s"]):
                return None
        previous = self.mode
        self.mode = desired
        self._last_switch_s = now_s
        self._downshift_candidate_since = None
        return {"t": now_s, "type": "resource_mode", "from": previous, "to": desired, "reason": reason}

    def _desired_mode(self, aoi_s: float | None, contracts: list[dict[str, object]]) -> tuple[str, str]:
        if self.policy == "static-low":
            return "economy", "static-low policy"
        if self.policy == "static-high":
            return "high", "static-high policy"
        cfg = self.config["resource_policy"]
        if self.policy == "aoi-only":
            if aoi_s is None or aoi_s >= float(cfg["aoi_high_trigger_s"]):
                return "high", "AoI high"
            if aoi_s >= float(cfg["aoi_normal_trigger_s"]):
                return "normal", "AoI elevated"
            return "economy", "AoI low"

        statuses = [str(item["status"]) for item in contracts]
        if "withdrawn" in statuses:
            return "high", "contract withdrawn"
        if "at_risk" in statuses:
            return "normal", "contract at risk"
        if statuses and all(status == "qualified" for status in statuses):
            return "economy", "all contracts qualified"
        if aoi_s is not None and aoi_s >= float(cfg["aoi_high_trigger_s"]):
            return "high", "reference unavailable and AoI high"
        return "normal", "contract evidence incomplete"
