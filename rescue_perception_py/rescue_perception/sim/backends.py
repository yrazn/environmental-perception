"""把场景真值转换成检测结果的合成推理后端。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..detectors.crack_segmenter import SegmentationBackend
from ..detectors.facility_inspector import FacilityBackend
from ..detectors.rgb_detector import DetectionBackend
from ..detectors.water_detector import WaterSegmentBackend
from ..types import (
    Detection3D,
    FacilityAnomaly,
    ObjectClass,
    SensorType,
    WaterRegion,
)
from .scenario import TunnelScenario


class SyntheticRgbBackend(DetectionBackend):
    """模拟真实 RGB 模型在烟雾增大时置信度下降。"""
    def __init__(self, scenario: TunnelScenario):
        self.scenario = scenario
        self.smoke_coverage = 0.0
        self.smoke_prob = 0.0
        self.smoke_centroid: Optional[np.ndarray] = None

    def detect(self, image: Optional[np.ndarray]) -> List[Detection3D]:
        t = self.scenario.current_t
        smoke = self.scenario.smoke_level(t)
        detections = []
        self.smoke_coverage = float(np.clip(smoke * 0.9, 0.0, 1.0))
        self.smoke_prob = float(np.clip(smoke * 0.8, 0.0, 1.0))
        self.smoke_centroid = np.array([160.0, 120.0]) if smoke > 0.05 else None

        for obj in self.scenario.objects:
            kind = obj["kind"]
            if kind == "fire":
                class_id = ObjectClass.FLAME
                conf = 0.90 if smoke < 0.35 else 0.60
            elif kind == "person":
                class_id = ObjectClass.PERSON_WALKING if obj.get("moving") else ObjectClass.PERSON_STANDING
                conf = 0.85 if smoke < 0.35 else max(0.30, 0.55 - smoke * 0.25)
            elif kind == "vehicle":
                class_id = ObjectClass.VEHICLE_CAR
                conf = 0.85 if smoke < 0.35 else 0.55
            else:
                continue
            detections.append(Detection3D(
                class_id=class_id,
                confidence=float(conf),
                source=SensorType.RGB,
                position=np.array([obj["x"], obj["y"], obj.get("z", 0.0)]),
                dimensions=np.array(obj.get("dims", [0.8, 0.5, 1.0])),
                velocity=np.array([obj.get("speed", 0.0), 0.0, 0.0]),
                temperature_max=obj.get("temperature", 0.0),
                extra={"accident_state": obj.get("accident_state", "unknown")},
            ))
        return detections


class SyntheticCrackBackend(SegmentationBackend):
    def segment(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2 - 20:h // 2 + 20, w // 2:w // 2 + 1] = 1
        return mask


class SyntheticWaterBackend(WaterSegmentBackend):
    def segment_water(self, rgb_image: np.ndarray,
                      thermal_image: Optional[np.ndarray] = None) -> List[WaterRegion]:
        return [WaterRegion(kind="puddle", depth_level="moderate", confidence=0.7)]


class SyntheticFacilityBackend(FacilityBackend):
    def inspect(self, rgb_image: np.ndarray,
                thermal_image: Optional[np.ndarray] = None) -> List[FacilityAnomaly]:
        return [FacilityAnomaly(facility_class="panel", anomaly_type="overheated",
                                confidence=0.8, thermal_temperature=66.0)]
