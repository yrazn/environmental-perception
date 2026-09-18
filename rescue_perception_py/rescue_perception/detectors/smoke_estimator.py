"""图像、颗粒物、LiDAR 衰减和 CO 的综合烟雾指数估计器。"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import (
    DegradationMode,
    SensorHealth,
    SmokeEstimateResult,
    VisibilityLevel,
    clamp,
)


class SmokeEstimator:
    """维护动态基线和 EMA，在传感器退化时重新分配证据权重。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        params = self.config.detectors["smoke_estimator"]
        self.weights = dict(params["weights"])
        self.degrade = dict(params["dynamic_degradation"])
        self.thresholds = params["thresholds"]
        self.alpha = float(params.get("smoothing_alpha", 0.3))
        self.pm_window_s = float(params["pm_baseline_window_s"])
        self.co_window_s = float(params["co_baseline_window_s"])
        self.pm_history: Deque[float] = deque()
        self.co_history: Deque[float] = deque()
        self.smoke_centroids: Deque[np.ndarray] = deque(maxlen=5)
        self.previous_score = 0.0

    @staticmethod
    def _bad(health: Optional[SensorHealth]) -> bool:
        return health is not None and health.status >= 1

    def _baseline(self, history: Deque[float], window_s: float, now: float) -> float:
        if not history:
            return 0.0
        # rolling minimum; doc keeps the minimum within the baseline window
        return min(history)

    def _push_with_window(self, history: Deque[float], value: float,
                          window_s: float, now: float, dt: float = 1.0) -> None:
        history.append(value)
        # approximate: keep at most ceil(window / dt) samples
        maxlen = max(2, int(window_s / max(dt, 0.01)))
        while len(history) > maxlen:
            history.popleft()

    def estimate(self,
                 image_smoke_prob: float = 0.0,
                 coverage_ratio: float = 0.0,
                 pm_now: float = 0.0,
                 co_now: float = 0.0,
                 lidar_effective_ratio: float = 1.0,
                 lidar_max_range: float = 80.0,
                 near_field_scatter_ratio: float = 0.0,
                 sensor_health: Optional[Dict[str, Optional[SensorHealth]]] = None,
                 mode: DegradationMode = DegradationMode.CLEAR,
                 smoke_centroid: Optional[np.ndarray] = None,
                 stamp: float = 0.0,
                 dt: float = 1.0) -> SmokeEstimateResult:
        """计算本帧烟雾分数、可见度等级和烟雾运动方向。"""
        health = sensor_health or {}

        # 四项证据先各自归一化到 0～1，再按健康状态动态加权。
        f_image = clamp(image_smoke_prob * coverage_ratio)
        if pm_now <= 0:
            f_pm = 0.0
        elif self.pm_history:
            pm_base = self._baseline(self.pm_history, self.pm_window_s, stamp)
            f_pm = clamp((pm_now - pm_base) / max(pm_base, 1e-6)) if pm_base > 0 else 0.0
        else:
            f_pm = clamp(pm_now / 500.0)

        f_lidar = clamp(
            ((1.0 - lidar_effective_ratio)
             + 0.5 * near_field_scatter_ratio
             + 0.3 * max(0.0, 1.0 - lidar_max_range / 80.0)) / 1.8
        )
        if co_now <= 0:
            f_co = 0.0
        elif self.co_history:
            co_base = self._baseline(self.co_history, self.co_window_s, stamp)
            f_co = clamp((co_now - co_base) / max(co_base, 1e-6)) if co_base > 0 else 0.0
        else:
            f_co = clamp(co_now / 500.0)

        self._push_with_window(self.pm_history, pm_now, self.pm_window_s, stamp, dt)
        self._push_with_window(self.co_history, co_now, self.co_window_s, stamp, dt)
        if smoke_centroid is not None:
            self.smoke_centroids.append(np.asarray(smoke_centroid, dtype=float))

        w = dict(self.weights)
        if self._bad(health.get("rgb")):
            w["image"] *= self.degrade["image_degraded_factor"]
        if self._bad(health.get("pm")):
            w["pm"] *= self.degrade["pm_degraded_factor"]
        if self._bad(health.get("lidar")):
            w["lidar"] *= self.degrade["lidar_degraded_factor"]
        if self._bad(health.get("gas")):
            w["co"] *= self.degrade["gas_degraded_factor"]

        if mode >= DegradationMode.HEAVY_SMOKE:
            w["image"] *= 0.3
            w["lidar"] *= 0.5
            w["pm"] *= 1.5
            w["co"] *= 1.3

        total = sum(w.values())
        if total > 1e-6:
            w = {k: v / total for k, v in w.items()}

        score = clamp(
            w["image"] * f_image + w["pm"] * f_pm
            + w["lidar"] * f_lidar + w["co"] * f_co
        )
        # EMA 抑制单帧噪声，避免退化状态因瞬时误检频繁切换。
        score = self.alpha * score + (1.0 - self.alpha) * self.previous_score
        self.previous_score = score

        if score < self.thresholds["clear"]:
            visibility = VisibilityLevel.CLEAR
        elif score < self.thresholds["light_smoke"]:
            visibility = VisibilityLevel.LIGHT_SMOKE
        elif score < self.thresholds["heavy_smoke"]:
            visibility = VisibilityLevel.HEAVY_SMOKE
        else:
            visibility = VisibilityLevel.BLIND

        motion = np.zeros(2)
        if len(self.smoke_centroids) >= 3:
            delta = self.smoke_centroids[-1] - self.smoke_centroids[-3]
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                motion = delta / norm

        return SmokeEstimateResult(
            stamp=stamp,
            smoke_score=float(score),
            visibility_level=visibility,
            motion_direction=motion,
            individual_scores={"image": f_image, "pm": f_pm,
                               "lidar": f_lidar, "co": f_co},
            image_smoke_prob=image_smoke_prob,
            pm_rise_rate=f_pm,
            lidar_attenuation=f_lidar,
            co_rise_rate=f_co,
            coverage_ratio=coverage_ratio,
        )
