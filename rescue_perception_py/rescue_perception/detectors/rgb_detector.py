"""可见光目标检测与烟雾区域统计。

The heavy YOLO/ONNX inference is intentionally behind a backend interface so
the rest of the system can be developed and tested without model files. A
synthetic backend is provided in rescue_perception.sim.backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..types import (
    DegradationMode,
    Detection3D,
    EnvironmentQuality,
    ObjectClass,
    SensorType,
)


class DetectionBackend:
    """ONNX、TensorRT 或合成 RGB 推理后端的统一接口。"""

    def detect(self, image: Optional[np.ndarray]) -> List[Detection3D]:
        raise NotImplementedError


@dataclass
class RgbDetectResult:
    detections: List[Detection3D] = field(default_factory=list)
    smoke_coverage: float = 0.0
    smoke_centroid: Optional[np.ndarray] = None
    smoke_prob: float = 0.0


class RgbDetector:
    """对后端输出执行类别阈值、退化过滤和火焰时序验证。"""
    def __init__(self, config: Optional[PerceptionConfig] = None,
                 backend: Optional[DetectionBackend] = None):
        self.config = config or PerceptionConfig()
        params = self.config.detectors["rgb_detector"]
        self.flame_conf = float(params.get("flame_conf_threshold", 0.35))
        self.vehicle_conf = float(params.get("vehicle_conf_threshold", 0.45))
        self.person_conf = float(params.get("person_conf_threshold", 0.40))
        self.person_min_output = float(params.get("person_min_confidence_output", 0.50))
        self.temporal_window = int(params.get("temporal_window", 5))
        self.flicker_threshold = float(params.get("flicker_threshold", 0.15))
        self.backend = backend
        self._flame_history: List[Tuple[np.ndarray, float]] = []

    def set_backend(self, backend: DetectionBackend) -> None:
        self.backend = backend

    def _filter_detections(self, detections: List[Detection3D],
                           env: EnvironmentQuality,
                           mode: DegradationMode) -> List[Detection3D]:
        out = []
        for det in detections:
            if det.class_id == ObjectClass.FLAME and det.confidence >= self.flame_conf:
                if env.rgb_credibility() < 0.1 and mode >= DegradationMode.HEAVY_SMOKE:
                    continue
                out.append(self._verify_flame(det))
            elif (ObjectClass.PERSON_STANDING <= det.class_id <= ObjectClass.PERSON_OCCLUDED
                  and det.confidence >= self.person_conf):
                if det.confidence >= self.person_min_output:
                    out.append(det)
            elif (ObjectClass.VEHICLE_CAR <= det.class_id <= ObjectClass.VEHICLE_UNKNOWN
                  and det.confidence >= self.vehicle_conf):
                out.append(det)
        return out

    def _verify_flame(self, det: Detection3D) -> Detection3D:
        # 保存近期火焰位置；连续出现在同一区域的候选获得小幅置信度增益。
        recent = [p for p, _ in self._flame_history]
        stable = sum(float(np.linalg.norm(p - det.position[:2])) < 1.0
                     for p in recent)
        det.confidence = min(1.0, det.confidence + 0.05 * stable)
        self._flame_history.append((det.position[:2], det.confidence))
        if len(self._flame_history) > self.temporal_window:
            self._flame_history.pop(0)
        return det

    def analyze_smoke_mask(self, mask: Optional[np.ndarray]) -> RgbDetectResult:
        """从烟雾分割掩膜提取覆盖率、平均概率和像素质心。"""
        result = RgbDetectResult()
        if mask is None or mask.size == 0:
            return result
        mask = np.asarray(mask, dtype=float)
        coverage = float(np.mean(mask > 0.5)) if mask.max() > 0 else 0.0
        if coverage > 0:
            ys, xs = np.nonzero(mask > 0.5)
            result.smoke_centroid = np.array([xs.mean(), ys.mean()])
        result.smoke_coverage = coverage
        result.smoke_prob = float(np.mean(mask))
        return result

    def detect(self,
               image: Optional[np.ndarray],
               env_quality: EnvironmentQuality,
               mode: DegradationMode,
               smoke_mask: Optional[np.ndarray] = None) -> RgbDetectResult:
        """执行一次 RGB 推理并返回目标与烟雾摘要。"""
        result = RgbDetectResult()
        if self.backend is not None and image is not None:
            detections = self.backend.detect(image)
            result.detections = self._filter_detections(detections, env_quality, mode)
            result.smoke_coverage = float(getattr(self.backend, "smoke_coverage", 0.0))
            result.smoke_prob = float(getattr(self.backend, "smoke_prob", 0.0))
            result.smoke_centroid = getattr(self.backend, "smoke_centroid", None)
        if smoke_mask is not None:
            smoke_result = self.analyze_smoke_mask(smoke_mask)
            result.smoke_coverage = smoke_result.smoke_coverage
            result.smoke_centroid = smoke_result.smoke_centroid
            result.smoke_prob = smoke_result.smoke_prob
        return result
