"""可见光、热成像和气体证据的火源确认与融合决策矩阵。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..types import (
    Detection3D,
    EnvironmentRisk,
    GasRiskResult,
    ObjectClass,
    SensorType,
    SensorWeights,
    SourceMask,
    TrackedObject,
    clamp,
)


@dataclass
class FireConfirmationResult:
    fire_level: int = 0
    confidence: float = 0.0
    should_stop: bool = False
    hot_zone_radius_m: float = 1.5


class FireFusion:
    """输出已确认火源或疑似火源，并计算建议高温隔离半径。"""
    def __init__(self):
        self.next_id = 1

    @staticmethod
    def _gas_conf(gas: GasRiskResult) -> float:
        if gas.fire_signature:
            return 0.7
        if gas.risk_level >= 2:
            return 0.5
        return 0.0

    def confirm_fire(self,
                     rgb_conf: float,
                     thermal_conf: float,
                     max_temp: float,
                     hotspot_count: int,
                     gas: GasRiskResult,
                     weights: SensorWeights) -> FireConfirmationResult:
        """根据多源证据组合给出 0～4 级火源确认结果。"""
        gas_conf = self._gas_conf(gas)
        result = FireConfirmationResult()

        if ((thermal_conf > 0.8 and hotspot_count >= 3 and max_temp > 500.0)
                or (thermal_conf > 0.7 and gas_conf > 0.7
                    and gas.asphyxiation_risk)):
            result.fire_level = 4
            result.confidence = 0.95
        elif ((thermal_conf > 0.7 and max_temp > 300.0 and hotspot_count >= 2)
              or (thermal_conf > 0.6 and gas_conf > 0.6)):
            result.fire_level = 3
            result.confidence = 0.85
        elif ((rgb_conf > 0.7 and thermal_conf > 0.6)
              or (thermal_conf > 0.6 and gas_conf > 0.5)
              or (rgb_conf > 0.5 and gas_conf > 0.5 and thermal_conf > 0.4)):
            result.fire_level = 2
            result.confidence = 0.75
        elif ((rgb_conf > 0.5 and gas_conf > 0.4)
              or thermal_conf > 0.5
              or (rgb_conf > 0.6 and thermal_conf > 0.3)):
            result.fire_level = 1
            result.confidence = 0.50
        else:
            result.fire_level = 0
            result.confidence = 0.0

        if max_temp > 150.0:
            result.confidence = result.confidence * 0.7 + thermal_conf * weights.thermal * 0.3
        if gas.fire_signature:
            result.confidence = min(result.confidence + 0.10, 1.0)
        if gas.explosion_risk:
            result.confidence = min(result.confidence + 0.15, 1.0)
            result.fire_level = max(result.fire_level, 3)

        result.should_stop = (result.fire_level >= 3
                              or (result.fire_level == 2 and max_temp > 300.0))
        if max_temp > 500.0:
            result.hot_zone_radius_m = 8.0
        elif max_temp > 300.0:
            result.hot_zone_radius_m = 5.0
        elif max_temp > 150.0:
            result.hot_zone_radius_m = 3.0
        else:
            result.hot_zone_radius_m = 1.5
        return result

    def fuse(self,
             rgb_detections: List[Detection3D],
             thermal_detections: List[Detection3D],
             gas_risk: GasRiskResult,
             weights: SensorWeights,
             thermal_hotspots=None) -> List[TrackedObject]:
        """在空间邻域内配对 RGB 火焰和热热点，生成火源融合目标。"""
        results = []
        thermal_flames = [d for d in thermal_detections
                          if d.class_id in (ObjectClass.FLAME,
                                            ObjectClass.FIRE_CANDIDATE,
                                            ObjectClass.HOTSPOT,
                                            ObjectClass.HOT_ZONE)]
        rgb_flames = [d for d in rgb_detections
                      if d.class_id in (ObjectClass.FLAME, ObjectClass.SMOKE)]
        if not thermal_flames and not rgb_flames:
            if gas_risk.fire_signature:
                results.append(TrackedObject(
                    track_id=self.next_id,
                    class_id=ObjectClass.FIRE_CANDIDATE,
                    confidence=0.40,
                    source_mask=int(SourceMask.GAS),
                    confirmed=False,
                    risk=EnvironmentRisk.MEDIUM,
                ))
                self.next_id += 1
            return results

        hotspot_count = len(thermal_flames)
        max_temp = max([d.temperature_max for d in thermal_flames] + [0.0])
        for thermal in thermal_flames:
            matched_rgb = None
            best_dist = 2.0
            for rgb in rgb_flames:
                dist = float(np.linalg.norm(rgb.position - thermal.position))
                if dist < best_dist:
                    best_dist = dist
                    matched_rgb = rgb
            rgb_conf = matched_rgb.confidence if matched_rgb else 0.0
            thermal_conf = thermal.confidence
            conf = self.confirm_fire(rgb_conf, thermal_conf, thermal.temperature_max,
                                     hotspot_count, gas_risk, weights)
            if conf.fire_level == 0:
                continue
            fire = TrackedObject(
                track_id=self.next_id,
                class_id=(ObjectClass.FLAME if conf.fire_level >= 2
                          else ObjectClass.FIRE_CANDIDATE),
                confidence=conf.confidence,
                source_mask=((int(SourceMask.RGB) | int(SourceMask.THERMAL))
                             if matched_rgb else int(SourceMask.THERMAL)),
                confirmed=conf.fire_level >= 2,
                state=np.concatenate([thermal.position, np.zeros(5)]),
                risk={4: EnvironmentRisk.CRITICAL,
                      3: EnvironmentRisk.HIGH,
                      2: EnvironmentRisk.HIGH,
                      1: EnvironmentRisk.MEDIUM}[conf.fire_level],
                temperature_max=thermal.temperature_max,
                depth_valid=(thermal.depth_valid
                             or bool(matched_rgb and matched_rgb.depth_valid)),
                hot_zone_radius_m=conf.hot_zone_radius_m,
            )
            fire.extra["fire_level"] = conf.fire_level
            fire.extra["should_stop"] = conf.should_stop
            self.next_id += 1
            results.append(fire)
        return results
