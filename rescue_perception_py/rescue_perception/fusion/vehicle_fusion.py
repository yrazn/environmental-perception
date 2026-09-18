"""RGB、LiDAR、毫米波车辆融合与隧道占道比例评估。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..types import (
    DegradationMode,
    Detection3D,
    EnvironmentRisk,
    ObjectClass,
    SensorType,
    SensorWeights,
    SourceMask,
    TrackedObject,
)


class VehicleFusion:
    """融合视觉类别、LiDAR 尺寸和雷达 RCS，并保留事故属性。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        params = self.config.fusion["vehicle_fusion"]
        self.lane_width = float(params.get("lane_width_m", 3.5))
        self.full_block = float(params.get("blocking_ratio_full", 0.7))
        self.partial_block = float(params.get("blocking_ratio_partial", 0.3))
        self.next_id = 1

    @staticmethod
    def _is_vehicle_rgb(det: Detection3D) -> bool:
        return ObjectClass.VEHICLE_CAR <= det.class_id <= ObjectClass.VEHICLE_UNKNOWN

    @staticmethod
    def _is_vehicle_lidar(det: Detection3D) -> bool:
        if det.class_id == ObjectClass.VEHICLE_UNKNOWN:
            return True
        return (det.dimensions[0] > 2.0 and det.dimensions[1] > 1.2
                and det.dimensions[2] > 1.0)

    @staticmethod
    def _is_vehicle_radar(det: Detection3D) -> bool:
        return (float(det.extra.get("rcs", -100.0)) > 10.0
                or det.extra.get("radar_class") == "large_object"
                or det.class_id == ObjectClass.VEHICLE_UNKNOWN)

    def fuse(self,
             rgb_detections: List[Detection3D],
             lidar_clusters: List[Detection3D],
             radar_tracks: List[Detection3D],
             weights: SensorWeights,
             mode: DegradationMode) -> List[TrackedObject]:
        """关联三类车辆证据并输出统一车辆目标。"""
        rgb = [d for d in rgb_detections if self._is_vehicle_rgb(d)]
        lidar = [d for d in lidar_clusters if self._is_vehicle_lidar(d)]
        radar = [d for d in radar_tracks if self._is_vehicle_radar(d)]

        results = []
        used_rgb = set()
        used_lidar = set()
        used_radar = set()

        def near(cloud: List[Detection3D], pos: np.ndarray,
                 tol: float = 2.0) -> List[Tuple[int, Detection3D]]:
            return [(i, d) for i, d in enumerate(cloud)
                    if np.linalg.norm(d.position[:2] - pos[:2]) < tol]

        for i, rd in enumerate(rgb):
            best_l = near(lidar, rd.position)
            best_r = near(radar, rd.position)
            lidar_det = best_l[0][1] if best_l else None
            radar_det = best_r[0][1] if best_r else None
            if lidar_det:
                used_lidar.add(best_l[0][0])
            if radar_det:
                used_radar.add(best_r[0][0])
            used_rgb.add(i)
            conf = 0.80 if lidar_det else 0.65
            width = lidar_det.dimensions[1] if lidar_det else 2.0
            tr = TrackedObject(
                track_id=self.next_id,
                class_id=rd.class_id,
                confidence=min(conf, rd.confidence + 0.1),
                source_mask=int(SourceMask.RGB)
                | (int(SourceMask.LIDAR) if lidar_det else 0)
                | (int(SourceMask.RADAR) if radar_det else 0),
                confirmed=bool(lidar_det or radar_det),
                state=np.concatenate([rd.position, rd.dimensions, [0, 0]]),
                temperature_max=rd.temperature_max,
                depth_valid=(rd.depth_valid
                             or bool(lidar_det and lidar_det.depth_valid)
                             or bool(radar_det and radar_det.depth_valid)),
                accident_state=str(rd.extra.get("accident_state", "unknown")),
            )
            self._apply_blockage(tr, width)
            self.next_id += 1
            results.append(tr)

        for j, ld in enumerate(lidar):
            if j in used_lidar:
                continue
            best_r = near(radar, ld.position)
            radar_det = best_r[0][1] if best_r else None
            if radar_det:
                used_radar.add(best_r[0][0])
            if mode >= DegradationMode.HEAVY_SMOKE:
                class_id = ObjectClass.VEHICLE_UNKNOWN
            else:
                class_id = ObjectClass.VEHICLE_UNKNOWN
            tr = TrackedObject(
                track_id=self.next_id,
                class_id=class_id,
                confidence=0.55 if radar_det else 0.45,
                source_mask=int(SourceMask.LIDAR)
                | (int(SourceMask.RADAR) if radar_det else 0),
                confirmed=bool(radar_det),
                state=np.concatenate([ld.position, ld.dimensions, [0, 0]]),
                depth_valid=(ld.depth_valid
                             or bool(radar_det and radar_det.depth_valid)),
                accident_state="unknown_accident_state",
            )
            if mode >= DegradationMode.HEAVY_SMOKE:
                tr.extra["class_name_override"] = "vehicle_like_obstacle"
            self._apply_blockage(tr, ld.dimensions[1])
            self.next_id += 1
            results.append(tr)

        for k, rdr in enumerate(radar):
            if k in used_radar:
                continue
            tr = TrackedObject(
                track_id=self.next_id,
                class_id=ObjectClass.VEHICLE_UNKNOWN,
                confidence=0.40,
                source_mask=int(SourceMask.RADAR),
                confirmed=False,
                state=np.concatenate([rdr.position, np.zeros(5)]),
                depth_valid=rdr.depth_valid,
                accident_state="unknown_accident_state",
            )
            if mode >= DegradationMode.HEAVY_SMOKE:
                tr.extra["class_name_override"] = "large_obstacle"
            self._apply_blockage(tr, 1.8)
            self.next_id += 1
            results.append(tr)
        return results

    def _apply_blockage(self, tr: TrackedObject, width: float) -> None:
        """用车辆宽度与车道宽度之比估计部分或完全阻塞。"""
        tr.blockage_ratio = min(1.0, max(0.0, width / self.lane_width))
        if tr.blockage_ratio >= self.full_block:
            tr.risk = EnvironmentRisk.HIGH
        elif tr.blockage_ratio >= self.partial_block:
            tr.risk = EnvironmentRisk.MEDIUM
