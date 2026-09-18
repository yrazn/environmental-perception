"""带恢复滞后和振荡保护的四级感知退化状态机。"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

from ..config import PerceptionConfig, DegradationConfig
from ..types import (
    BehaviorRecommendation,
    DegradationMode,
    EnvironmentQuality,
    SensorWeights,
    SmokeEstimateResult,
)


class DegradationManager:
    """根据烟雾、亮度和传感器可信度选择感知模式与融合权重。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        deg: DegradationConfig = self.config.degradation
        self.base_weights = {
            DegradationMode.CLEAR: deg.base_weights["clear"],
            DegradationMode.LOW_VISIBILITY: deg.base_weights["low_vis"],
            DegradationMode.HEAVY_SMOKE: deg.base_weights["heavy_smoke"],
            DegradationMode.PERCEPTION_DEGRADED: deg.base_weights["degraded"],
        }
        self.behaviors = {
            DegradationMode.CLEAR: deg.behaviors["clear"],
            DegradationMode.LOW_VISIBILITY: deg.behaviors["low_vis"],
            DegradationMode.HEAVY_SMOKE: deg.behaviors["heavy_smoke"],
            DegradationMode.PERCEPTION_DEGRADED: deg.behaviors["degraded"],
        }
        self.depth_priority = deg.depth_priority
        self.current_mode = DegradationMode.CLEAR
        self.change_history: Deque[Tuple[DegradationMode, float]] = deque()
        self.lock_until = 0.0
        self.oscillation_window_s = 3.0
        self.oscillation_max_switches = 2

    def _transition_to(self, target: DegradationMode, now: float) -> bool:
        """执行状态转换；短时间频繁切换时暂时锁定当前状态。"""
        if now < self.lock_until:
            return False
        self.change_history.append((target, now))
        while self.change_history and now - self.change_history[0][1] > self.oscillation_window_s:
            self.change_history.popleft()
        if len(self.change_history) > self.oscillation_max_switches:
            self.lock_until = now + self.oscillation_window_s
            return False
        self.current_mode = target
        return True

    def update(self,
               env: EnvironmentQuality,
               smoke: SmokeEstimateResult,
               gas_failed: bool = False,
               now: float = 0.0) -> DegradationMode:
        """更新退化模式；进入危险状态快，恢复到正常状态更保守。"""
        if now < self.lock_until:
            return self.current_mode
        s = smoke.smoke_score
        rgb_enabled = (self.config.sensor_enabled("rgb_front")
                       or self.config.sensor_enabled("rgb_ptz"))
        rc = env.rgb_credibility() if rgb_enabled else 1.0
        lc = (env.lidar_credibility()
              if self.config.sensor_enabled("lidar_3d") else 1.0)
        tc = (env.thermal_credibility()
              if self.config.sensor_enabled("thermal") else 1.0)
        radar_c = (env.radar_credibility()
                   if self.config.sensor_enabled("radar_4d") else 1.0)
        brightness = env.illuminance_lux
        target = self.current_mode

        # 每个分支只允许相邻或合理恢复，避免一次噪声跨越多个等级。
        if self.current_mode == DegradationMode.CLEAR:
            if s > 0.15 or rc < 0.50 or brightness < 30.0:
                target = DegradationMode.LOW_VISIBILITY
        elif self.current_mode == DegradationMode.LOW_VISIBILITY:
            if s > 0.35 or (rc < 0.20 and lc < 0.30):
                target = DegradationMode.HEAVY_SMOKE
            elif s < 0.10 and rc > 0.70 and brightness > 50.0:
                target = DegradationMode.CLEAR
        elif self.current_mode == DegradationMode.HEAVY_SMOKE:
            if (rc < 0.10 and lc < 0.20 and tc < 0.30) or gas_failed:
                target = DegradationMode.PERCEPTION_DEGRADED
            elif s < 0.25 and rc > 0.30 and lc > 0.30:
                target = DegradationMode.LOW_VISIBILITY
        elif self.current_mode == DegradationMode.PERCEPTION_DEGRADED:
            reliable = sum(1 for c in (tc, radar_c, lc) if c > 0.50)
            if reliable >= 3 and s < 0.30:
                target = DegradationMode.LOW_VISIBILITY
            elif reliable >= 2 and s < 0.50:
                target = DegradationMode.HEAVY_SMOKE

        if target != self.current_mode:
            self._transition_to(target, now)
        return self.current_mode

    def get_weights(self, mode: DegradationMode,
                    env: EnvironmentQuality) -> SensorWeights:
        """按模式基础权重和实时可信度生成当前帧归一化权重。"""
        base = self.base_weights[mode]
        w = SensorWeights(*base)

        def apply_factor(weight: float, credibility: float) -> float:
            if credibility < 0.10:
                return 0.0
            if credibility < 0.30:
                return weight * credibility / 0.30
            return weight

        w.rgb = apply_factor(w.rgb, env.rgb_credibility())
        w.thermal = apply_factor(w.thermal, env.thermal_credibility())
        w.lidar = apply_factor(w.lidar, env.lidar_credibility())
        w.radar = apply_factor(w.radar, env.radar_credibility())
        w.gas = apply_factor(w.gas, env.gas_credibility())

        # 重烟下视觉和激光易受散射影响，毫米波与气体提供安全冗余。
        if mode >= DegradationMode.HEAVY_SMOKE:
            w.rgb *= 0.30
            w.radar = max(w.radar, 0.60)
            w.gas = max(w.gas, 0.50)
        if mode == DegradationMode.PERCEPTION_DEGRADED:
            w.lidar = 0.0
            w.rgb = 0.0
            w.radar = max(w.radar, 0.70)
        return w.normalize()

    def get_behavior(self, mode: DegradationMode) -> BehaviorRecommendation:
        b = self.behaviors[mode]
        return BehaviorRecommendation(
            max_speed_pct=b["max_speed_pct"],
            require_confirmation=b["require_confirmation"],
            allow_autonomous=b["allow_autonomous"],
            suggest_teleop=b.get("suggest_teleop", False),
        )
