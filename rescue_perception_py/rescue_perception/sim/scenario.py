"""供演示和回归测试使用的合成隧道火灾场景。"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..types import GasReadings, SyncedSensorData


class TunnelScenario:
    """生成时间一致的 RGB、热、点云、雷达、气体和环境帧。"""

    def __init__(self,
                 objects: Optional[List[Dict]] = None,
                 smoke_curve: Optional[callable] = None):
        self.objects = objects if objects is not None else [
            {"kind": "fire", "x": 6.0, "y": 0.4, "z": 0.8,
             "temperature": 420.0, "rcs": 5.0, "moving": False},
            {"kind": "person", "x": 7.5, "y": -1.2, "z": 1.0,
             "temperature": 36.5, "rcs": -2.0, "moving": True,
             "micro_doppler": True, "speed": 0.4},
            {"kind": "vehicle", "x": 12.0, "y": 0.0, "z": 1.4,
             "temperature": 45.0, "rcs": 18.0, "moving": False,
             "dims": [4.2, 1.8, 1.5]},
            {"kind": "equipment", "x": 9.0, "y": 1.6, "z": 1.0,
             "temperature": 92.0, "rcs": -5.0, "moving": False,
             "dims": [0.6, 0.4, 1.2]},
        ]
        self.smoke_curve = smoke_curve or (
            lambda t: min(1.0, max(0.0, (t - 15.0) / 65.0))
        )
        self.current_t = 0.0

    def smoke_level(self, t: float) -> float:
        return float(self.smoke_curve(t))

    def _thermal_matrix(self, t: float) -> np.ndarray:
        h, w = 120, 160
        smoke = self.smoke_level(t)
        matrix = np.full((h, w), 25.0)
        rng = np.random.default_rng(int(t * 100) % (2 ** 32 - 1))
        matrix += rng.normal(0.0, 0.4, matrix.shape)

        def blob(cy: int, cx: int, radius: int, temp: float, peak: float = 1.0):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    dist = np.hypot(dy, dx)
                    if dist <= radius:
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            matrix[yy, xx] = max(
                                matrix[yy, xx],
                                temp * (1.0 - 0.5 * dist / max(radius, 1)) * peak,
                            )

        for obj in self.objects:
            if obj["kind"] == "fire":
                blob(60, 40, 7, obj["temperature"],
                     peak=1.0 if smoke < 0.6 else 0.9)
            elif obj["kind"] == "person":
                blob(50, 75, 4, obj["temperature"])
            elif obj["kind"] == "equipment":
                blob(45, 105, 4, obj["temperature"])
            elif obj["kind"] == "vehicle":
                blob(70, 120, 6, obj["temperature"])
        if smoke > 0.35:
            matrix += rng.normal(0.0, smoke * 1.5, matrix.shape)
        return matrix

    def _rgb_image(self, t: float) -> np.ndarray:
        h, w = 240, 320
        smoke = self.smoke_level(t)
        rng = np.random.default_rng(int(t * 100) % (2 ** 32 - 1) + 1)
        img = np.full((h, w), 100.0)
        img += rng.normal(0.0, 6.0, img.shape)
        for obj in self.objects:
            cx = int(obj["x"] * 20.0)
            cy = int(obj["y"] * 20.0) + h // 2
            size = 16 if obj["kind"] == "vehicle" else 8
            color = 220.0 if obj["kind"] == "fire" else 160.0
            y0, y1 = max(0, cy - size), min(h, cy + size)
            x0, x1 = max(0, cx - size), min(w, cx + size)
            img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], color)
        if smoke > 0.01:
            img = img * (1.0 - smoke) + np.full_like(img, 185.0) * smoke
        return np.clip(img, 0, 255).astype(np.float32)

    def _lidar_points(self, t: float) -> np.ndarray:
        smoke = self.smoke_level(t)
        rng = np.random.default_rng(int(t * 100) % (2 ** 32 - 1) + 2)
        ground_n = int(5000 * (1.0 - 0.6 * smoke))
        xs = rng.uniform(0.5, 40.0, ground_n)
        ys = rng.uniform(-2.5, 2.5, ground_n)
        zs = rng.normal(0.0, 0.01, ground_n)
        pts = [np.column_stack([xs, ys, zs])]

        for obj in self.objects:
            dims = obj.get("dims", [0.8, 0.5, 1.0])
            n = int(400 * (1.0 - 0.5 * smoke))
            pts.append(np.column_stack([
                rng.uniform(obj["x"] - dims[0] / 2, obj["x"] + dims[0] / 2, n),
                rng.uniform(obj["y"] - dims[1] / 2, obj["y"] + dims[1] / 2, n),
                rng.uniform(0.05, dims[2], n),
            ]))
        cloud = np.vstack(pts)
        # 用随机丢点和近场散射点近似模拟烟雾对激光雷达的影响。
        if smoke > 0.1:
            keep = rng.random(len(cloud)) > 0.5 * smoke
            cloud = cloud[keep]
            scatter = np.column_stack([
                rng.uniform(0.5, 1.0, 300),
                rng.uniform(-1.0, 1.0, 300),
                rng.uniform(0.0, 0.3, 300),
            ])
            cloud = np.vstack([cloud, scatter])
        ranges = np.linalg.norm(cloud, axis=1)
        intensity = 120.0 - ranges * 0.8 + rng.normal(0.0, 8.0, len(cloud))
        return np.column_stack([cloud, intensity])

    def _radar_targets(self, t: float) -> List[Dict[str, float]]:
        smoke = self.smoke_level(t)
        targets = []
        for obj in self.objects:
            if obj["kind"] in ("person", "vehicle"):
                x, y = obj["x"], obj["y"]
                rng = np.random.default_rng(int(t * 100) % (2 ** 32 - 1) + 3)
                targets.append({
                    "range": np.hypot(x, y) + rng.normal(0.0, 0.1),
                    "azimuth": np.arctan2(y, x) + rng.normal(0.0, 0.01),
                    "elevation": 0.0,
                    "doppler": obj.get("speed", 0.0) + rng.normal(0.0, 0.05),
                    "rcs": obj["rcs"] + rng.normal(0.0, 0.5),
                    "micro_doppler": bool(obj.get("micro_doppler", False)),
                })
        return targets

    def _gas(self, t: float) -> GasReadings:
        smoke = self.smoke_level(t)
        return GasReadings(
            co_ppm=10.0 + smoke ** 2 * 900.0,
            co2_ppm=400.0 + smoke * 3500.0,
            o2_pct=20.9 - smoke * 4.0,
            ch4_lel=0.0 + smoke * 4.0,
            h2s_ppm=0.0 + smoke * 12.0,
            voc_ppm=0.0 + smoke * 200.0,
            pm25=smoke * 420.0,
            pm10=smoke * 500.0,
        )

    def frame(self, t: float) -> SyncedSensorData:
        """生成可直接输入主管线的完整同步帧。"""
        self.current_t = t
        smoke = self.smoke_level(t)
        return SyncedSensorData(
            stamp=t,
            rgb_image=self._rgb_image(t),
            thermal_image=np.clip((self._thermal_matrix(t) - 20.0) * 3.0,
                                  0, 255).astype(np.float32),
            temperature_matrix=self._thermal_matrix(t),
            lidar_points=self._lidar_points(t),
            radar_targets=self._radar_targets(t),
            gas=self._gas(t),
            pm={"pm25": smoke * 420.0, "pm10": smoke * 500.0},
            temperature_humidity={"temperature": 25.0 + smoke * 35.0,
                                  "humidity": 50.0 + smoke * 40.0},
            odometry={"x": 0.0, "y": 0.0, "yaw": 0.0},
        )
