"""积水和渗水检测流程，模型推理通过可插拔后端提供。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import WaterRegion


class WaterSegmentBackend:
    """RGB/热成像积水分割模型接口。"""
    def segment_water(self, rgb_image: np.ndarray,
                      thermal_image: Optional[np.ndarray] = None) -> List[WaterRegion]:
        raise NotImplementedError


class WaterDetector:
    """为后端水域结果分配系统内稳定递增编号。"""
    def __init__(self, config: Optional[PerceptionConfig] = None,
                 backend: Optional[WaterSegmentBackend] = None):
        self.config = config or PerceptionConfig()
        self.backend = backend
        self.next_id = 1

    def set_backend(self, backend: WaterSegmentBackend) -> None:
        self.backend = backend

    def detect(self,
               rgb_image: Optional[np.ndarray],
               ground_cloud: Optional[np.ndarray] = None,
               thermal_image: Optional[np.ndarray] = None,
               scans: Optional[dict] = None) -> List[WaterRegion]:
        if self.backend is None or rgb_image is None:
            return []
        regions = self.backend.segment_water(rgb_image, thermal_image)
        for region in regions:
            region.region_id = self.next_id
            self.next_id += 1
        return regions
