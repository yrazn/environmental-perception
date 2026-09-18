"""热成像坏点修复、热点分割、时序分类和热风险评估。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..config import PerceptionConfig
from ..types import (
    Detection3D,
    HotspotType,
    ObjectClass,
    SensorType,
    ThermalHotspot,
    clamp,
)


@dataclass
class ThermalDetectResult:
    detections: List[Detection3D] = field(default_factory=list)
    hotspots: List[ThermalHotspot] = field(default_factory=list)
    risk_level: int = 0
    max_temperature: float = 0.0
    confidence: float = 0.0


class ThermalDetector:
    """从辐射温度矩阵中分割并分类高温区域。

    Model-based person/pose classification is delegated to an inference
    backend; the geometry/temperature checks are implemented directly here.
    """

    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        params = self.config.detectors["thermal_detector"]
        self.min_area_px = int(params.get("hotspot_min_area_pixels", 9))
        self.bg_window = int(params.get("bg_window_size", 30))
        self.fire_temp_min = float(params.get("fire_temp_min", 200.0))
        self.suspected_temp_min = float(params.get("suspected_temp_min", 150.0))
        self.circ_fire_max = float(params.get("circularity_fire_max", 0.70))
        self.circ_equip_min = float(params.get("circularity_equipment_min", 0.80))
        self.temp_rise_threshold = float(params.get("temp_rise_threshold", 5.0))
        self.area_growth_threshold = float(params.get("area_growth_threshold", 0.10))
        self.person_min_temp = float(params.get("person_min_temperature", 24.0))
        self.person_max_temp = float(params.get("person_max_temperature", 44.0))
        self.person_bg_delta = float(params.get("person_background_delta", 3.0))
        self.person_min_area_px = int(params.get("person_min_area_pixels", 12))
        self.person_aspect_min = float(params.get("person_aspect_ratio_min", 0.4))
        self.person_aspect_max = float(params.get("person_aspect_ratio_max", 4.0))
        self.hotspot_bg_delta = float(params.get("hotspot_background_delta", 30.0))
        self.hotspot_sigma = float(params.get("hotspot_sigma_multiplier", 3.0))
        self.hot_exclusion_radius = int(params.get("hot_exclusion_radius_pixels", 2))
        self.pixel_scale_m = float(params.get("pixel_scale_m", 0.01))
        self.depth_default_m = float(params.get("depth_default_m", 6.0))
        self.frame_interval = float(params.get("frame_interval", 0.1))

        self.temp_history: List[np.ndarray] = []
        self.hotspot_history: List[Tuple[List[ThermalHotspot], float]] = []

    def _median_fix_outliers(self, matrix: np.ndarray) -> np.ndarray:
        """只替换相对 3×3 中值明显跳变的坏点，保留真实热边缘。"""
        if matrix.size < 9:
            return matrix.copy()
        padded = np.pad(matrix, 1, mode="edge")
        windows = sliding_window_view(padded, (3, 3)).reshape(matrix.shape[0],
                                                              matrix.shape[1], 9)
        medians = np.median(windows, axis=-1)
        fixed = matrix.copy()
        bad = np.abs(matrix - medians) > 20.0
        fixed[bad] = medians[bad]
        return fixed

    def _update_baseline(self, fixed: np.ndarray) -> Tuple[float, float]:
        """用近期逐像素最小值估计不易受短时热点污染的背景。"""
        self.temp_history.append(fixed)
        if len(self.temp_history) > self.bg_window:
            self.temp_history.pop(0)
        if len(self.temp_history) < 2:
            return float(np.mean(fixed)), float(np.std(fixed))
        bg_image = np.minimum.reduce(self.temp_history)
        h, w = bg_image.shape
        roi = bg_image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        return float(np.mean(roi)), float(np.std(roi))

    def _connected_components(self, mask: np.ndarray) -> List[np.ndarray]:
        """提取二值热点掩膜的八连通区域。"""
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        components: List[np.ndarray] = []
        current = 0
        for r in range(h):
            for c in range(w):
                if mask[r, c] and labels[r, c] == 0:
                    current += 1
                    stack = [(r, c)]
                    labels[r, c] = current
                    pixels = []
                    while stack:
                        rr, cc = stack.pop()
                        pixels.append((rr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = rr + dr, cc + dc
                                if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                                        and labels[nr, nc] == 0):
                                    labels[nr, nc] = current
                                    stack.append((nr, nc))
                    components.append(np.array(pixels))
        return components

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        """使用方形邻域扩张高温区，用于排除火焰周围的人体温区像素。"""
        if radius <= 0 or not np.any(mask):
            return mask.copy()
        padded = np.pad(mask.astype(bool), radius, mode="constant")
        windows = sliding_window_view(
            padded, (2 * radius + 1, 2 * radius + 1))
        return np.any(windows, axis=(-2, -1))

    @staticmethod
    def _perimeter(component: np.ndarray, shape: Tuple[int, int]) -> int:
        comp = set(map(tuple, component))
        boundary = 0
        for r, c in component:
            if ((r - 1, c) not in comp or (r + 1, c) not in comp
                    or (r, c - 1) not in comp or (r, c + 1) not in comp):
                boundary += 1
        return boundary

    def _classify(self, hs: ThermalHotspot, human_like: bool) -> None:
        if human_like and self.person_min_temp <= hs.t_mean <= self.person_max_temp:
            hs.htype = HotspotType.HUMAN_THERMAL
            hs.type_confidence = 0.60
        elif (hs.t_max > self.fire_temp_min and hs.dt_dt > self.temp_rise_threshold
              and hs.da_dt > self.area_growth_threshold
              and hs.circularity < self.circ_fire_max):
            hs.htype = HotspotType.FIRE_SOURCE
            hs.type_confidence = 0.85
        elif (hs.t_max > self.suspected_temp_min and hs.dt_dt > 3.0
              and hs.duration_s > 3.0):
            hs.htype = HotspotType.FIRE_SUSPECTED
            hs.type_confidence = 0.60
        elif (hs.t_max > 80.0 and hs.circularity > self.circ_equip_min
              and hs.dt_dt < 1.0):
            hs.htype = HotspotType.HOT_EQUIPMENT
            hs.type_confidence = 0.75
        elif hs.t_max > 50.0 and hs.dt_dt < 0.5 and hs.circularity > 0.85:
            hs.htype = HotspotType.WARM_SURFACE
            hs.type_confidence = 0.70
        else:
            hs.htype = HotspotType.HOT_ANOMALY
            hs.type_confidence = 0.50

    def _match_history(self, hs: ThermalHotspot) -> Optional[Tuple[ThermalHotspot, float]]:
        if not self.hotspot_history:
            return None
        # 优先匹配最近一帧，避免用更早但偶然更近的热点计算错误变化率。
        for frame, stamp in reversed(self.hotspot_history):
            best: Optional[Tuple[ThermalHotspot, float]] = None
            best_dist = 25.0
            for prev in frame:
                dist = float(np.linalg.norm(prev.centroid - hs.centroid))
                if dist < best_dist:
                    best_dist = dist
                    best = (prev, stamp)
            if best is not None:
                return best
        return None

    def _human_like(self, component: np.ndarray, hs: ThermalHotspot) -> bool:
        """用面积、宽高比和平均温度筛选人体形状候选。"""
        if len(component) < self.person_min_area_px:
            return False
        rows = component[:, 0]
        cols = component[:, 1]
        height = float(rows.max() - rows.min() + 1)
        width = float(cols.max() - cols.min() + 1)
        aspect = height / max(width, 1.0)
        return (self.person_aspect_min <= aspect <= self.person_aspect_max
                and self.person_min_temp <= hs.t_mean <= self.person_max_temp)

    def _hotspot_to_detection(self, hs: ThermalHotspot,
                              shape: Tuple[int, int]) -> Detection3D:
        h, w = shape
        # ThermalHotspot质心统一保存为图像坐标[x, y]。
        cx, cy = hs.centroid
        if hs.htype == HotspotType.HUMAN_THERMAL:
            class_id = ObjectClass.PERSON_STANDING
            depth = self.depth_default_m
        elif hs.htype == HotspotType.FIRE_SOURCE:
            class_id = ObjectClass.FLAME
            depth = self.depth_default_m
        elif hs.htype == HotspotType.FIRE_SUSPECTED:
            class_id = ObjectClass.FIRE_CANDIDATE
            depth = self.depth_default_m
        elif hs.htype == HotspotType.HOT_EQUIPMENT:
            class_id = ObjectClass.HOT_ZONE
            depth = self.depth_default_m
        else:
            class_id = ObjectClass.HOTSPOT
            depth = self.depth_default_m
        x = depth
        y = (cx - w / 2.0) * self.pixel_scale_m * depth
        z = (h / 2.0 - cy) * self.pixel_scale_m * depth
        return Detection3D(
            class_id=class_id,
            confidence=hs.type_confidence,
            source=SensorType.THERMAL,
            position=np.array([x, y, z]),
            depth_valid=False,
            bearing=float(np.arctan2(y, max(x, 1e-6))),
            temperature_max=hs.t_max,
            temperature_min=hs.t_min,
            extra={"hotspot_type": int(hs.htype),
                   "centroid_px": hs.centroid.tolist(),
                   "depth_method": "configured_default_depth"},
        )

    def detect(self,
               temperature_matrix: Optional[np.ndarray],
               thermal_image: Optional[np.ndarray] = None,
               now: float = 0.0,
               frame_interval: Optional[float] = None) -> ThermalDetectResult:
        """检测热点并返回目标候选、最高温度和热风险等级。"""
        result = ThermalDetectResult()
        if temperature_matrix is None or temperature_matrix.size == 0:
            return result

        matrix = np.asarray(temperature_matrix, dtype=float)
        fixed = self._median_fix_outliers(matrix)
        bg_mean, bg_std = self._update_baseline(fixed)
        # 人体表面通常只比隧道背景高数摄氏度，不能与火源共用背景+30℃
        # 阈值。两个通道并行提取，再排除高温核心附近的人体温区像素。
        person_threshold = max(self.person_min_temp,
                               bg_mean + self.person_bg_delta)
        hotspot_threshold = bg_mean + max(
            self.hotspot_bg_delta, self.hotspot_sigma * bg_std)
        hot_mask = fixed > hotspot_threshold
        # 连通域使用完整的升温区域，不能先挖掉火焰核心，否则核心周围的
        # 人体温区环带会被切成多个区域并误判成人。高温影响区只作为分类
        # 禁止标记，不破坏原始连通关系。
        warm_mask = fixed >= person_threshold
        candidate_mask = warm_mask | hot_mask
        hot_influence = self._dilate(hot_mask, self.hot_exclusion_radius)
        h, w = fixed.shape

        hotspots: List[ThermalHotspot] = []
        for component in self._connected_components(candidate_mask):
            component_is_hot = bool(np.any(hot_influence[component[:, 0],
                                                         component[:, 1]]))
            min_area = self.min_area_px if component_is_hot else self.person_min_area_px
            if len(component) < min_area:
                continue
            temps = fixed[component[:, 0], component[:, 1]]
            hs = ThermalHotspot(
                # 连通域内部顺序是[row, col]，输出统一为[x, y]=[col, row]。
                centroid=np.array([component[:, 1].mean(),
                                   component[:, 0].mean()]),
                t_max=float(temps.max()),
                t_mean=float(temps.mean()),
                t_min=float(temps.min()),
                area_px=len(component),
                perimeter=self._perimeter(component, (h, w)),
            )
            hs.circularity = (4.0 * np.pi * hs.area_px / (hs.perimeter ** 2)
                              if hs.perimeter > 0 else 1.0)
            grad_y, grad_x = np.gradient(fixed)
            grad = np.hypot(grad_x, grad_y)
            hs.edge_gradient = float(grad[component[:, 0], component[:, 1]].max())

            interval = frame_interval if frame_interval is not None else self.frame_interval
            match = self._match_history(hs)
            if match is not None:
                prev, prev_stamp = match
                measured_dt = now - prev_stamp
                dt = max(measured_dt if measured_dt > 0.0 else interval, 1e-6)
                hs.dt_dt = (hs.t_max - prev.t_max) / dt
                hs.da_dt = (hs.area_px - prev.area_px) / (max(prev.area_px, 1) * dt)
                hs.duration_s = prev.duration_s + dt
            self._classify(
                hs, not component_is_hot and self._human_like(component, hs))
            hotspots.append(hs)

        self.hotspot_history.append((hotspots, now))
        if len(self.hotspot_history) > self.bg_window:
            self.hotspot_history.pop(0)

        max_temp = max([hs.t_max for hs in hotspots] + [bg_mean])
        fire_count = sum(hs.htype in (HotspotType.FIRE_SOURCE,
                                      HotspotType.FIRE_SUSPECTED)
                         for hs in hotspots)
        if max_temp > 500.0 or fire_count >= 3:
            risk_level = 4
        elif max_temp > 300.0 or fire_count >= 2:
            risk_level = 3
        elif max_temp > 150.0 or fire_count >= 1:
            risk_level = 2
        elif max_temp > bg_mean + 30.0:
            risk_level = 1
        else:
            risk_level = 0

        result.risk_level = risk_level
        result.hotspots = hotspots
        result.max_temperature = max_temp
        result.confidence = clamp(fire_count * 0.3 + len(hotspots) * 0.1)
        result.detections = [self._hotspot_to_detection(hs, (h, w))
                             for hs in hotspots]
        return result
