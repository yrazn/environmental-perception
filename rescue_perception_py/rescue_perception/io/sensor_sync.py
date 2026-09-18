"""多传感器近似时间同步缓冲器。"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..types import GasReadings, SyncedSensorData


class SensorSync:
    """以最新消息为参考时间，在各传感器容差内选择最近样本。"""
    def __init__(self, key_tolerance_ms: Optional[Dict[str, float]] = None):
        self.tolerance = dict(key_tolerance_ms or {})
        self.buffers: Dict[str, Deque[Tuple[float, Any]]] = {}

    def register(self, key: str, tolerance_ms: float = 10.0) -> None:
        """注册传感器键和允许的最大时间差。"""
        self.tolerance[key] = tolerance_ms
        self.buffers.setdefault(key, deque(maxlen=50))

    def push(self, key: str, stamp: float, data: Any) -> None:
        """写入带时间戳的消息；每类最多保留最近 50 条。"""
        self.buffers.setdefault(key, deque(maxlen=50)).append((stamp, data))

    def _nearest(self, key: str, ref_stamp: float) -> Optional[Tuple[float, Any]]:
        buf = self.buffers.get(key)
        if not buf:
            return None
        tol = self.tolerance.get(key, 10.0) / 1000.0
        best = None
        best_delta = tol
        for stamp, data in buf:
            delta = abs(stamp - ref_stamp)
            if delta <= best_delta:
                best_delta = delta
                best = (stamp, data)
        return best

    def pull(self, keys: Optional[List[str]] = None) -> Optional[SyncedSensorData]:
        """组装同步帧；容差外的输入留空并把同步误差记录为无穷大。"""
        keys = keys or list(self.buffers.keys())
        available = [k for k in keys if self.buffers.get(k)]
        if not available:
            return None
        ref_stamp = max(self.buffers[k][-1][0] for k in available)
        synced = SyncedSensorData(stamp=ref_stamp)
        for key in keys:
            nearest = self._nearest(key, ref_stamp)
            if nearest is None:
                synced.sync_status[key] = float("inf")
                continue
            selected_stamp, data = nearest
            if key == "rgb":
                synced.rgb_image = data
            elif key == "thermal":
                synced.thermal_image = data
            elif key == "temperature":
                synced.temperature_matrix = data
            elif key == "lidar":
                synced.lidar_points = data
            elif key == "radar":
                synced.radar_targets = data
            elif key == "scans":
                synced.scans = data
            elif key == "gas":
                synced.gas = data
            elif key == "pm":
                synced.pm = data
            elif key == "env":
                synced.temperature_humidity = data
            elif key == "odometry":
                synced.odometry = data
            synced.sync_status[key] = abs(selected_stamp - ref_stamp) * 1000.0
        return synced
