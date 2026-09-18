"""多气体独立分级、趋势判断和综合风险评估。"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import GasReadings, GasRiskResult


class GasRiskAssessor:
    """评估中毒、缺氧、爆炸和火灾气体组合风险。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        thresholds = self.config.detectors["gas_risk"]["thresholds"]
        self.thresholds = {
            "co": tuple(thresholds["co"]),
            "co2": tuple(thresholds["co2"]),
            "o2": tuple(thresholds["o2"]),
            "ch4": tuple(thresholds["ch4_lel"]),
            "h2s": tuple(thresholds["h2s"]),
        }
        self.combo = self.config.detectors["gas_risk"]["fire_combo"]
        self.history_size = int(self.config.detectors["gas_risk"].get("history_size", 60))
        self.history: Deque[Dict[str, float]] = deque(maxlen=self.history_size)

    @staticmethod
    def _level(value: float, bounds, decreasing: bool = False) -> int:
        if decreasing:
            if value > bounds[0]:
                return 0
            if value > bounds[1]:
                return 1
            if value > bounds[2]:
                return 2
            return 3
        if value < bounds[0]:
            return 0
        if value < bounds[1]:
            return 1
        if value < bounds[2]:
            return 2
        return 3

    def _trend(self, key: str) -> float:
        values = [h[key] for h in self.history]
        if len(values) < 3:
            return 0.0
        return float(np.polyfit(np.arange(len(values)), values, 1)[0])

    def assess(self,
               readings: GasReadings,
               ambient_temp: float = 25.0,
               ambient_humidity: float = 50.0,
               now: float = 0.0) -> GasRiskResult:
        """根据当前读数和历史趋势生成 0～3 级气体风险。"""
        # 高湿和高温会影响部分电化学传感器，因此先做简化补偿。
        humidity_correction = 1.0
        if ambient_humidity > 90.0:
            humidity_correction = 0.70
        elif ambient_humidity > 70.0:
            humidity_correction = 0.85

        temp_correction = 1.0 + max(0.0, (ambient_temp - 40.0) * 0.01)
        corrected_co = readings.co_ppm * humidity_correction * temp_correction
        corrected_h2s = readings.h2s_ppm * humidity_correction

        individual = {
            "co": self._level(corrected_co, self.thresholds["co"]),
            "co2": self._level(readings.co2_ppm, self.thresholds["co2"]),
            "o2": self._level(readings.o2_pct, self.thresholds["o2"], decreasing=True),
            "ch4": self._level(readings.ch4_lel, self.thresholds["ch4"]),
            "h2s": self._level(corrected_h2s, self.thresholds["h2s"]),
        }
        # 安全系统采用最大风险原则，不允许平均值掩盖单项致命气体。
        risk_level = max(individual.values())

        self.history.append({
            "co": readings.co_ppm,
            "co2": readings.co2_ppm,
            "o2": readings.o2_pct,
            "temp": ambient_temp,
        })

        fire_signature = (
            self._trend("co") > self.combo["co_rise_rate_threshold"]
            and self._trend("co2") > self.combo["co2_rise_rate_threshold"]
            and self._trend("o2") < -self.combo["o2_drop_rate_threshold"]
            and self._trend("temp") > self.combo["temp_rise_threshold"]
        )
        explosion_risk = readings.ch4_lel >= 10.0
        asphyxiation_risk = readings.o2_pct < 19.5 or corrected_co > 200.0

        if risk_level >= 2 or fire_signature:
            alert = {
                "type": "GAS_DANGER_ALERT",
                "remote_notify": True,
                "mission_confirm": True,
                "auto_retreat": False,
            }
        elif risk_level >= 1:
            alert = {
                "type": "GAS_CAUTION_ALERT",
                "remote_notify": True,
                "mission_confirm": False,
                "auto_retreat": False,
            }
        else:
            alert = {
                "type": "GAS_NORMAL",
                "remote_notify": False,
                "mission_confirm": False,
                "auto_retreat": False,
            }

        return GasRiskResult(
            risk_level=risk_level,
            individual_levels=individual,
            fire_signature=fire_signature,
            explosion_risk=explosion_risk,
            asphyxiation_risk=asphyxiation_risk,
            alert=alert,
            timestamp=now,
        )
