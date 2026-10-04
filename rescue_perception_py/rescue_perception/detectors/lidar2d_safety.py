"""前后二维激光雷达的扇区测距、近障碍分级和安全冗余。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..types import Lidar2DSafetyResult


class Lidar2DSafety:
    """把一个或多个二维扫描压缩成可直接进入安全决策的距离摘要。

    扫描值既可直接是 ``numpy`` 距离数组，也可为包含 ``ranges``、
    ``angle_min`` 和 ``angle_increment`` 的字典。直接数组默认覆盖以传感器
    正前方为中心的配置视场角。
    """

    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        params = self.config.detectors["lidar2d_safety"]
        self.range_min = float(params.get("range_min", 0.05))
        self.range_max = float(params.get("range_max", 30.0))
        self.default_fov = np.deg2rad(float(params.get("default_fov_deg", 180.0)))
        self.forward_half_angle = np.deg2rad(
            float(params.get("forward_sector_half_angle_deg", 35.0)))
        self.warning_distance = float(params.get("warning_distance_m", 2.0))
        self.stop_distance = float(params.get("stop_distance_m", 1.0))
        self.emergency_distance = float(params.get("emergency_distance_m", 0.6))

    def _parse_scan(self, payload) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(payload, dict):
            ranges = np.asarray(payload.get("ranges", []), dtype=float).reshape(-1)
            angle_min = float(payload.get("angle_min", -self.default_fov / 2.0))
            angle_increment = payload.get("angle_increment")
            if angle_increment is None:
                angles = np.linspace(-self.default_fov / 2.0,
                                     self.default_fov / 2.0,
                                     len(ranges), endpoint=True)
            else:
                angles = angle_min + np.arange(len(ranges)) * float(angle_increment)
        else:
            ranges = np.asarray(payload, dtype=float).reshape(-1)
            angles = np.linspace(-self.default_fov / 2.0,
                                 self.default_fov / 2.0,
                                 len(ranges), endpoint=True)
        return ranges, angles

    def _scan_distance(self, payload) -> Tuple[float, float]:
        ranges, angles = self._parse_scan(payload)
        if len(ranges) == 0:
            return float("inf"), 0.0
        valid = (np.isfinite(ranges) & (ranges >= self.range_min)
                 & (ranges <= self.range_max))
        valid_ratio = float(np.mean(valid))
        forward = valid & (np.abs(angles) <= self.forward_half_angle)
        if not np.any(forward):
            return float("inf"), valid_ratio
        return float(np.min(ranges[forward])), valid_ratio

    def analyze(self, scans: Optional[Dict]) -> Lidar2DSafetyResult:
        """分析前后扫描；未知名称的扫描仍作为全向近场安全来源。"""
        result = Lidar2DSafetyResult()
        if not scans:
            return result

        named_ranges: Dict[str, float] = {}
        valid_ratios = []
        for name, payload in scans.items():
            distance, valid_ratio = self._scan_distance(payload)
            named_ranges[str(name).lower()] = distance
            valid_ratios.append(valid_ratio)

        front_candidates = [distance for name, distance in named_ranges.items()
                            if "front" in name or "forward" in name]
        rear_candidates = [distance for name, distance in named_ranges.items()
                           if "rear" in name or "back" in name]
        result.front_range = min(front_candidates, default=float("inf"))
        result.rear_range = min(rear_candidates, default=float("inf"))
        result.nearest_range = min(named_ranges.values(), default=float("inf"))
        result.valid_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0

        for name, distance in named_ranges.items():
            if distance < self.warning_distance:
                result.blocked_sectors.append(name)

        if result.nearest_range < self.emergency_distance:
            result.warning_level = 3
        elif result.nearest_range < self.stop_distance:
            result.warning_level = 2
        elif result.nearest_range < self.warning_distance:
            result.warning_level = 1
        return result
