"""传感器坐标到地图坐标的语义目标定位和协方差传播。"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..types import SourceMask, TrackedObject


class SemanticLocalizer:
    """维护各传感器外参，并把瞬时融合测量转换到统一 map 坐标。"""
    def __init__(self):
        self.transforms: Dict[str, np.ndarray] = {
            "rgb": np.eye(4),
            "thermal": np.eye(4),
            "lidar": np.eye(4),
            "radar": np.eye(4),
        }

    def register_transform(self, frame_id: str, matrix: np.ndarray) -> None:
        """注册指定传感器到 map 的 4×4 齐次变换矩阵。"""
        self.transforms[frame_id] = np.asarray(matrix, dtype=float).reshape(4, 4)

    def _frame_for(self, tr: TrackedObject) -> str:
        if tr.source_mask & int(SourceMask.LIDAR):
            return "lidar"
        if tr.source_mask & int(SourceMask.THERMAL):
            return "thermal"
        if tr.source_mask & int(SourceMask.RADAR):
            return "radar"
        return "rgb"

    def transform(self, tracks: List[TrackedObject]) -> List[TrackedObject]:
        """原地把瞬时测量转换到 map 坐标。

        必须在测量进入持久跟踪器前调用，使轨迹始终处于同一坐标系，
        避免对仅预测未更新的轨迹重复施加传感器外参。
        """
        for tr in tracks:
            frame = self._frame_for(tr)
            t = self.transforms.get(frame, np.eye(4))
            local = np.ones(4)
            local[:3] = tr.position
            world = t @ local
            tr.pose_map = t.copy()
            tr.pose_map[:3, 3] = world[:3]
            rotation = t[:3, :3]
            # 用旋转块构造一阶雅可比，传播位置/姿态近似协方差。
            j = np.zeros((6, 6))
            j[:3, :3] = rotation
            j[3:, 3:] = rotation
            tr.pose_cov_map = j @ tr.cov[:6, :6] @ j.T
            tr.state[:3] = world[:3]
        return tracks

    def markers(self, tracks: List[TrackedObject]) -> List[Dict]:
        """生成与具体可视化框架无关的轻量 Marker 字典。"""
        return [
            {
                "track_id": tr.track_id,
                "class_name": tr.class_name,
                "position": tr.position.tolist(),
                "confidence": tr.confidence,
            }
            for tr in tracks
        ]
