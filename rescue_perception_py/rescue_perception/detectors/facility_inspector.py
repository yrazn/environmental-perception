"""固定设施异常巡检流程，模型推理通过可插拔后端提供。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import FacilityAnomaly


class FacilityBackend:
    """使用 RGB 和可选热图识别设施类别与异常类型。"""
    def inspect(self, rgb_image: np.ndarray,
                thermal_image: Optional[np.ndarray] = None) -> List[FacilityAnomaly]:
        raise NotImplementedError


class FacilityInspector:
    """为设施异常结果分配编号并统一输出数据结构。"""
    def __init__(self, config: Optional[PerceptionConfig] = None,
                 backend: Optional[FacilityBackend] = None):
        self.config = config or PerceptionConfig()
        self.backend = backend
        self.next_id = 1

    def set_backend(self, backend: FacilityBackend) -> None:
        self.backend = backend

    def detect(self,
               rgb_image: Optional[np.ndarray],
               thermal_image: Optional[np.ndarray] = None,
               odometry: Optional[dict] = None) -> List[FacilityAnomaly]:
        if self.backend is None or rgb_image is None:
            return []
        anomalies = self.backend.inspect(rgb_image, thermal_image)
        for anomaly in anomalies:
            anomaly.anomaly_id = self.next_id
            self.next_id += 1
        return anomalies
