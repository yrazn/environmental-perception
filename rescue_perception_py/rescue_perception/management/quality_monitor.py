"""环境质量与传感器可信度监控。

该模块只评估“本帧数据是否值得信任”，不负责检测目标。输出可信度会
同时影响退化状态机和后续融合权重。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from ..types import (
    EnvironmentQuality,
    SensorCredibility,
    SensorHealth,
    SensorType,
    SyncedSensorData,
    VisibilityLevel,
    clamp,
)


class QualityMonitor:
    """从图像、点云、雷达和环境数据估计传感器可信度。"""
    def __init__(self):
        self.radar_target_counts: Deque[int] = deque(maxlen=10)
        self.lidar_history: Deque[float] = deque(maxlen=30)

    @staticmethod
    def _rgb_metrics(image: Optional[np.ndarray]) -> Dict[str, float]:
        """计算亮度、对比度和归一化梯度清晰度。"""
        if image is None or image.size == 0:
            return {"credibility": 0.0, "sharpness": 0.0, "brightness": 0.0}
        gray = np.asarray(image, dtype=float)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
        mean_brightness = float(gray.mean())
        brightness_score = 1.0 - abs(mean_brightness - 128.0) / 128.0
        rms = float(np.sqrt(np.mean((gray - mean_brightness) ** 2)))
        contrast_score = clamp(rms / 80.0)
        gy, gx = np.gradient(gray)
        sharpness = clamp(float(np.var(gx + gy)) / 500.0)
        credibility = clamp(sharpness * 0.35 + brightness_score * 0.25
                            + contrast_score * 0.25 + 0.15)
        return {"credibility": credibility, "sharpness": sharpness,
                "brightness": mean_brightness}

    @staticmethod
    def _thermal_metrics(matrix: Optional[np.ndarray]) -> float:
        """根据有效温度比例、均匀性和窗口梯度评估热成像。"""
        if matrix is None or matrix.size == 0:
            return 0.0
        m = np.asarray(matrix, dtype=float)
        valid_ratio = clamp(float(np.mean((m > -40.0) & (m < 500.0))))
        h, w = m.shape
        roi = m[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        uniformity = clamp(1.0 - float(np.std(roi)) / 50.0)
        gy, gx = np.gradient(m)
        grad_mean = float(np.mean(np.hypot(gx, gy)))
        window_score = clamp(grad_mean / 10.0)
        return clamp(valid_ratio * 0.35 + uniformity * 0.35 + window_score * 0.30)

    @staticmethod
    def _lidar_metrics(points: Optional[np.ndarray],
                       quality: Optional[Dict[str, float]]) -> Dict[str, float]:
        """优先使用聚类器质量统计，缺失时直接从原始点云估计。"""
        if quality:
            effective = clamp(float(quality.get("effective_ratio", 1.0)))
            scatter = clamp(float(quality.get("near_field_scatter_ratio", 0.0)))
            max_range = float(quality.get("max_range", 80.0))
            attenuation = clamp(float(quality.get("attenuation", 0.0)))
            credibility = clamp(effective * 0.30 + (1.0 - scatter) * 0.25
                                + clamp(max_range / 80.0) * 0.25
                                + (1.0 - attenuation) * 0.20)
            return {"credibility": credibility, "effective_ratio": effective,
                    "max_range": max_range, "scatter": scatter}
        if points is None or len(points) == 0:
            return {"credibility": 0.0, "effective_ratio": 0.0,
                    "max_range": 0.0, "scatter": 0.0}
        pts = np.asarray(points, dtype=float)
        ranges = np.linalg.norm(pts[:, :3], axis=1)
        effective = clamp(float(np.mean((ranges >= 0.5) & (ranges <= 80.0))))
        return {"credibility": clamp(effective * 0.8), "effective_ratio": effective,
                "max_range": float(np.percentile(ranges, 95)), "scatter": 0.0}

    def _radar_metrics(self, targets: List[Dict]) -> float:
        count = len(targets)
        self.radar_target_counts.append(count)
        if not self.radar_target_counts:
            return 0.5
        mean_count = max(float(np.mean(self.radar_target_counts)), 1.0)
        stability = clamp(1.0 - abs(count - mean_count) / mean_count)
        rcs_vals = [float(t.get("rcs", 0.0)) for t in targets]
        rcs_score = 1.0 if not rcs_vals else clamp(1.0 - float(np.var(rcs_vals)) / 100.0)
        return clamp(stability * 0.5 + rcs_score * 0.5)

    @staticmethod
    def _gas_metrics(synced: SyncedSensorData) -> float:
        humidity = 50.0
        if synced.temperature_humidity:
            humidity = float(synced.temperature_humidity.get("humidity", 50.0))
        humidity_factor = 1.0 if humidity <= 70.0 else (0.85 if humidity <= 90.0 else 0.7)
        return clamp(humidity_factor)

    def assess(self,
               synced: SyncedSensorData,
               smoke_score: float = 0.0,
               lidar_quality: Optional[Dict[str, float]] = None,
               radar_targets: Optional[List[Dict]] = None) -> EnvironmentQuality:
        """汇总当前同步帧的环境质量和传感器可信度。"""
        env = EnvironmentQuality(stamp=synced.stamp)
        rgb = self._rgb_metrics(synced.rgb_image)
        thermal_c = self._thermal_metrics(synced.temperature_matrix)
        lidar = self._lidar_metrics(synced.lidar_points, lidar_quality)
        radar_c = self._radar_metrics(radar_targets if radar_targets is not None
                                      else synced.radar_targets)
        gas_c = self._gas_metrics(synced) if synced.gas is not None else 0.0
        pm_available = (synced.pm is not None or
                        (synced.gas is not None
                         and (synced.gas.pm25 > 0 or synced.gas.pm10 > 0)))

        env.credibility[SensorType.RGB] = SensorCredibility(
            rgb["credibility"], rgb["credibility"], cause="normal")
        env.credibility[SensorType.THERMAL] = SensorCredibility(thermal_c, thermal_c)
        env.credibility[SensorType.LIDAR_3D] = SensorCredibility(
            lidar["credibility"], lidar["effective_ratio"])
        env.credibility[SensorType.LIDAR_2D] = SensorCredibility(
            clamp(lidar["credibility"] * 0.8), lidar["effective_ratio"])
        env.credibility[SensorType.RADAR_4D] = SensorCredibility(radar_c)
        env.credibility[SensorType.GAS] = SensorCredibility(gas_c, gas_c)
        env.credibility[SensorType.PM] = SensorCredibility(
            1.0 if pm_available else 0.0, 1.0 if pm_available else 0.0)

        brightness = rgb["brightness"]
        sharpness = rgb["sharpness"]
        if smoke_score < 0.15 and brightness > 50 and sharpness > 0.08:
            visibility = VisibilityLevel.CLEAR
        elif smoke_score < 0.15 and brightness < 50:
            visibility = VisibilityLevel.LOW_LIGHT
        elif smoke_score < 0.35:
            visibility = VisibilityLevel.LIGHT_SMOKE
        elif smoke_score < 0.65:
            visibility = VisibilityLevel.HEAVY_SMOKE
        else:
            visibility = VisibilityLevel.BLIND
        env.visibility_level = visibility
        env.visibility_score = clamp(1.0 - smoke_score)
        env.smoke_density = clamp(smoke_score)
        env.illuminance_lux = brightness * 4.0
        if synced.temperature_humidity:
            env.ambient_temperature = float(synced.temperature_humidity.get("temperature", 25.0))
            env.ambient_humidity = float(synced.temperature_humidity.get("humidity", 50.0))
        # 独立 PM 设备优先；复合气体设备中的 PM 字段作为兼容回退。
        if synced.pm is not None:
            pm25 = float(synced.pm.get("pm25", 0.0))
            pm10 = float(synced.pm.get("pm10", 0.0))
            env.smoke_density = max(env.smoke_density,
                                    clamp((pm25 + pm10) / 1000.0))
        elif synced.gas is not None:
            env.smoke_density = max(env.smoke_density,
                                    clamp((synced.gas.pm25 + synced.gas.pm10) / 1000.0))
        return env
