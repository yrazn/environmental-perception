"""巡检模式裂缝分割流程，模型推理通过可插拔后端提供。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import CrackDefect


class SegmentationBackend:
    """裂缝分割模型接口，返回与输入图像同尺寸的二值掩膜。"""
    def segment(self, image: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class CrackSegmenter:
    """把裂缝掩膜转换为带编号、位置和尺寸的结构化缺陷。"""
    def __init__(self, config: Optional[PerceptionConfig] = None,
                 backend: Optional[SegmentationBackend] = None):
        self.config = config or PerceptionConfig()
        self.backend = backend
        self.next_id = 1

    def set_backend(self, backend: SegmentationBackend) -> None:
        self.backend = backend

    def detect(self,
               rgb_image: Optional[np.ndarray],
               ground_cloud: Optional[np.ndarray] = None,
               odometry: Optional[dict] = None) -> List[CrackDefect]:
        if self.backend is None or rgb_image is None:
            return []
        mask = self.backend.segment(rgb_image)
        if mask is None or mask.sum() == 0:
            return []
        mask = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        length_px = len(ys)
        # 当前为演示标尺；生产系统应由相机标定和 LiDAR 表面距离动态计算。
        scale_mm_per_px = 0.8
        width_px = float(np.mean(mask.sum(axis=0)[mask.sum(axis=0) > 0]))
        defect = CrackDefect(
            crack_id=self.next_id,
            position=[float(xs.mean()), float(ys.mean()), 0.0],
            length_m=length_px * scale_mm_per_px / 1000.0,
            width_mm=width_px * scale_mm_per_px,
            width_max_mm=max(1.0, width_px * scale_mm_per_px * 1.2),
            confidence=0.75,
        )
        self.next_id += 1
        return [defect]
