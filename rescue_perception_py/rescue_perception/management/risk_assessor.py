"""场景级风险融合与安全策略映射。"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import (
    DegradationMode,
    EnvironmentQuality,
    EnvironmentRisk,
    GasRiskResult,
    ObjectClass,
    RiskAssessment,
    SafetyDecision,
    SafetyLevel,
    SmokeEstimateResult,
    TrackedObject,
)


class RiskAssessor:
    """把目标、环境和气体证据转换为风险等级及车辆动作约束。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        self.weights = self.config.risk["weights"]
        esc = self.config.risk["escalation"]
        self.critical_th = float(esc["critical"])
        self.high_th = float(esc["high"])
        self.medium_th = float(esc["medium"])

    def _factors(self,
                 tracks: List[TrackedObject],
                 env: EnvironmentQuality,
                 gas: GasRiskResult,
                 thermal_risk: int,
                 mode: DegradationMode,
                 radar_nearest: float) -> Dict[str, float]:
        """把不同量纲的输入转换为 0～1 风险因子。"""
        thermal = thermal_risk / 4.0
        gas_factor = gas.risk_level / 3.0
        if radar_nearest < 2.0:
            obstacle = 1.0
        elif radar_nearest < 5.0:
            obstacle = 0.6
        elif radar_nearest < 10.0:
            obstacle = 0.3
        else:
            obstacle = 0.0
        blockage = 0.8 if any(t.blockage_ratio > 0.7 for t in tracks) else 0.0
        valid_depths = [t.depth_valid for t in tracks] or [True]
        depth = 1.0 - float(np.mean(valid_depths))
        rgb = 1.0 - env.rgb_credibility()
        motion = 0.5 if any(np.linalg.norm(t.velocity) > 0.3 for t in tracks) else 0.0
        return {
            "thermal": thermal, "gas": gas_factor, "obstacle": obstacle,
            "blockage": blockage, "depth": depth, "rgb": rgb, "motion": motion,
        }

    def assess(self,
               tracks: List[TrackedObject],
               env: EnvironmentQuality,
               gas: GasRiskResult,
               smoke: SmokeEstimateResult,
               thermal_risk_level: int,
               mode: DegradationMode,
               radar_nearest_range: float,
               is_estop: bool = False) -> RiskAssessment:
        """计算综合环境风险、热区和离散事件。"""
        factors = self._factors(tracks, env, gas, thermal_risk_level, mode,
                                radar_nearest_range)
        if mode >= DegradationMode.HEAVY_SMOKE:
            w = self.weights["heavy"]
        elif mode == DegradationMode.LOW_VISIBILITY:
            w = self.weights["low_vis"]
        else:
            w = self.weights["clear"]

        weighted = sum(factors[k] * w[k] for k in factors)
        score = weighted / sum(w.values())
        # 加权平均不能稀释单项致命风险，因此高危热、气体和近障碍采用
        # 最低风险分数兜底。
        if factors["thermal"] >= 1.0:
            score = max(score, 0.95)
        if factors["gas"] >= 1.0:
            score = max(score, 0.90)
        if factors["obstacle"] >= 1.0:
            score = max(score, 0.85)
        if factors["blockage"] > 0:
            score = max(score, 0.70)

        if is_estop:
            risk = EnvironmentRisk.CRITICAL
            action = "ESTOP_IMMEDIATE"
        elif score >= self.critical_th:
            risk = EnvironmentRisk.CRITICAL
            action = "STOP_AND_WAIT_REMOTE_CONFIRM"
        elif score >= self.high_th:
            risk = EnvironmentRisk.HIGH
            action = "LOW_SPEED_WAIT_CONFIRMATION"
        elif score >= self.medium_th:
            risk = EnvironmentRisk.MEDIUM
            action = "LIMIT_SPEED_AND_LOG"
        else:
            risk = EnvironmentRisk.LOW
            action = "NORMAL_OPERATION"

        hot_zones = []
        events = []
        for tr in tracks:
            if tr.class_id in (ObjectClass.FLAME, ObjectClass.FIRE_CANDIDATE):
                pos = tr.position
                radius = tr.hot_zone_radius_m
                hot_zones.append({
                    "center": pos.tolist(),
                    "radius_m": radius,
                    "max_temperature": tr.temperature_max,
                    "severity": int(tr.risk),
                    "track_id": tr.track_id,
                })
                if tr.confirmed:
                    events.append(f"fire_confirmed:{tr.track_id}")
            if ((ObjectClass.PERSON_STANDING <= tr.class_id
                 <= ObjectClass.PERSON_OCCLUDED
                 or tr.class_id == ObjectClass.PERSON_CANDIDATE)
                    and tr.confirmed and tr.confidence >= 0.7):
                events.append(f"person_found:{tr.track_id}")
            if tr.blockage_ratio > 0.7:
                events.append(f"accident_vehicle_blocking:{tr.track_id}")
        if gas.risk_level >= 2:
            events.append("gas_danger_level")
        if mode == DegradationMode.PERCEPTION_DEGRADED:
            events.append("perception_lost")
        if smoke.smoke_score >= 0.65:
            events.append("visibility_blind")

        return RiskAssessment(
            risk_level=risk,
            combined_score=score,
            factors=factors,
            recommended_action=action,
            hot_zones=hot_zones,
            events=list(dict.fromkeys(events)),
        )

    def map_to_safety(self,
                      env_risk: EnvironmentRisk,
                      thermal_risk_level: int,
                      mode: DegradationMode,
                      radar_nearest_range: float,
                      is_estop: bool = False) -> SafetyDecision:
        """把环境风险映射为速度上限、停车要求和远程确认要求。"""
        if is_estop:
            return SafetyDecision(SafetyLevel.SAFETY_ESTOP, 0.0,
                                  "IMMEDIATE_ESTOP", False)
        if (thermal_risk_level >= 4
                or mode == DegradationMode.PERCEPTION_DEGRADED
                or radar_nearest_range < 1.0):
            return SafetyDecision(SafetyLevel.SAFETY_STOP_REQUIRED, 0.0,
                                  "STOP_AND_WAIT", True)
        if (thermal_risk_level >= 3
                or env_risk >= EnvironmentRisk.HIGH
                or radar_nearest_range < 3.0):
            return SafetyDecision(SafetyLevel.SAFETY_DEGRADED, 0.20,
                                  "LOW_SPEED_CAUTIOUS", True)
        if (thermal_risk_level >= 2
                or mode >= DegradationMode.LOW_VISIBILITY
                or radar_nearest_range < 5.0):
            return SafetyDecision(SafetyLevel.SAFETY_WARNING, 0.50,
                                  "REDUCED_SPEED", False)
        return SafetyDecision(SafetyLevel.SAFETY_OK, 1.0,
                              "NORMAL_OPERATION", False)
