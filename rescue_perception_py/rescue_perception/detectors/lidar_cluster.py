"""LiDAR 地面分割、欧氏聚类、三维包围盒和点云质量评估。"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..types import DegradationMode, Detection3D, ObjectClass, SensorType, clamp


@dataclass
class LidarClusterResult:
    detections: List[Detection3D] = field(default_factory=list)
    ground_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    quality: Dict[str, float] = field(default_factory=dict)


class LidarCluster:
    """把原始点云转换为障碍物候选及地面点集合。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        params = self.config.detectors["lidar_cluster"]
        self.min_range = float(params["passthrough_min"])
        self.max_range = float(params["passthrough_max"])
        self.voxel_normal = float(params["voxel_leaf_normal"])
        self.voxel_degraded = float(params["voxel_leaf_degraded"])
        self.ransac_thresh = float(params["ransac_dist_thresh"])
        self.ransac_iter = int(params["ransac_max_iter"])
        self.tol_normal = float(params["cluster_tolerance_normal"])
        self.tol_degraded = float(params["cluster_tolerance_degraded"])
        self.min_cluster_size = int(params["min_cluster_size"])
        self.max_cluster_size = int(params["max_cluster_size"])
        self.nominal_points = float(params.get("nominal_point_count", 30000))

    def _passthrough(self, points: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(points[:, :3], axis=1)
        keep = (r >= self.min_range) & (r <= self.max_range)
        return points[keep]

    def _voxel_downsample(self, points: np.ndarray, leaf: float) -> np.ndarray:
        if len(points) == 0:
            return points
        keys = np.floor(points[:, :3] / leaf).astype(np.int64)
        buckets: Dict[Tuple[int, int, int], List[int]] = {}
        for i, key in enumerate(keys):
            buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(i)
        out = []
        for idxs in buckets.values():
            out.append(points[idxs].mean(axis=0))
        return np.asarray(out)

    def _ransac_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """用 RANSAC 寻找最大地面内点集合，返回平面和点集划分。"""
        if len(points) < 3:
            return np.zeros(3), points, np.zeros(0)
        best_inliers = np.zeros(len(points), dtype=bool)
        best_params = np.array([0.0, 0.0, float(points[:, 2].mean())])
        for _ in range(self.ransac_iter):
            idx = np.random.choice(len(points), 3, replace=False)
            a = points[idx]
            try:
                A = np.column_stack([a[:, 0], a[:, 1], np.ones(3)])
                params, _, _, _ = np.linalg.lstsq(A, a[:, 2], rcond=None)
            except np.linalg.LinAlgError:
                continue
            pred = points[:, 0] * params[0] + points[:, 1] * params[1] + params[2]
            res = np.abs(pred - points[:, 2])
            inliers = res < self.ransac_thresh
            if inliers.sum() > best_inliers.sum():
                best_inliers = inliers
                best_params = params
        return best_params, points[~best_inliers], points[best_inliers]

    def _euclidean_cluster(self, points: np.ndarray, tol: float) -> List[List[int]]:
        """使用空间哈希限制邻域搜索范围，避免全量两两距离计算。"""
        n = len(points)
        if n == 0:
            return []
        origin = points[:, :3].min(axis=0)
        cells: Dict[Tuple[int, int, int], List[int]] = {}
        for i, p in enumerate(points):
            key = tuple(np.floor((p[:3] - origin) / tol).astype(np.int64))
            cells.setdefault(key, []).append(i)

        visited = set()
        clusters: List[List[int]] = []
        for start in range(n):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            cluster = []
            while stack:
                j = stack.pop()
                cluster.append(j)
                key = tuple(np.floor((points[j][:3] - origin) / tol).astype(np.int64))
                for offset in product((-1, 0, 1), repeat=3):
                    nk = tuple(k + o for k, o in zip(key, offset))
                    for k in cells.get(nk, []):
                        if k not in visited and np.linalg.norm(points[k][:3] - points[j][:3]) <= tol:
                            visited.add(k)
                            stack.append(k)
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
        return clusters

    @staticmethod
    def _fit_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        center = points.mean(axis=0)
        if len(points) < 3:
            return center, np.ones(3) * 0.1, 0.0
        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        # keep a right-handed frame
        if np.cross(eigvecs[:, 0], eigvecs[:, 1]) @ eigvecs[:, 2] < 0:
            eigvecs[:, 2] *= -1
        projected = (points - center) @ eigvecs
        extents = projected.max(axis=0) - projected.min(axis=0)

        up = abs(eigvecs[:, 2])
        if up.argmax() != 2:
            # vertical axis should be z; swap axes if necessary
            horizontal_idx = [i for i in range(3) if i != int(up.argmax())]
            new_order = horizontal_idx + [int(up.argmax())]
            eigvecs = eigvecs[:, new_order]
            extents = extents[new_order]
        dims = np.maximum(extents, 1e-3)
        yaw = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        return center, dims, yaw

    def _classify(self, dims: np.ndarray, area: float) -> ObjectClass:
        length, width, height = dims
        if length > 2.0 and width > 1.2 and height > 1.0:
            return ObjectClass.VEHICLE_UNKNOWN
        if height < 0.3 and area < 1.0:
            return ObjectClass.OBSTACLE_GENERIC
        if height > 0.5 and area < 0.5:
            return ObjectClass.PERSON_CANDIDATE
        return ObjectClass.OBSTACLE_GENERIC

    def _quality(self, points: np.ndarray, original_count: int) -> Dict[str, float]:
        if len(points) == 0:
            return {"effective_ratio": 0.0, "near_field_scatter_ratio": 0.0,
                    "max_range": 0.0, "attenuation": 1.0}
        ranges = np.linalg.norm(points[:, :3], axis=1)
        effective_ratio = clamp(len(points) / max(self.nominal_points, 1.0))
        near = points[ranges < 1.0]
        scatter = 0.0
        if len(near) > 0 and points.shape[1] > 3:
            intensity = points[:, 3]
            mean_i = float(np.mean(intensity))
            std_i = max(float(np.std(intensity)), 1e-6)
            scatter = clamp(float(np.mean(intensity[np.where(ranges < 1.0)[0]] < mean_i - 2 * std_i)))
        max_range = float(np.percentile(ranges, 95))
        attenuation = 0.0
        if points.shape[1] > 3 and len(points) > 2:
            log_i = np.log(np.maximum(points[:, 3], 1e-3))
            if np.std(ranges) > 1e-6:
                slope = float(np.polyfit(ranges, log_i, 1)[0])
                attenuation = clamp(-slope / 0.05)
        return {
            "effective_ratio": effective_ratio,
            "near_field_scatter_ratio": scatter,
            "max_range": max_range,
            "attenuation": attenuation,
        }

    def cluster(self,
                points: Optional[np.ndarray],
                mode: DegradationMode = DegradationMode.CLEAR) -> LidarClusterResult:
        """完成滤波、降采样、地面移除、聚类和尺寸规则分类。"""
        result = LidarClusterResult()
        if points is None or len(points) == 0:
            return result
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        original_count = len(pts)
        pts = self._passthrough(pts)
        if len(pts) == 0:
            result.quality = self._quality(np.zeros((0, 3)), original_count)
            return result

        leaf = self.voxel_degraded if mode >= DegradationMode.HEAVY_SMOKE else self.voxel_normal
        pts = self._voxel_downsample(pts, leaf)
        if len(pts) == 0:
            result.quality = self._quality(np.zeros((0, 3)), original_count)
            return result

        _, non_ground, ground = self._ransac_ground(pts)
        result.ground_points = ground
        tol = self.tol_degraded if mode >= DegradationMode.HEAVY_SMOKE else self.tol_normal
        clusters = self._euclidean_cluster(non_ground, tol)
        for cluster in clusters[:100]:
            cpts = non_ground[cluster]
            center, dims, yaw = self._fit_box(cpts[:, :3])
            area = dims[0] * dims[1]
            class_id = self._classify(dims, area)
            det = Detection3D(
                class_id=class_id,
                confidence=0.65 if class_id == ObjectClass.VEHICLE_UNKNOWN else 0.45,
                source=SensorType.LIDAR_3D,
                position=center,
                dimensions=dims,
                yaw=yaw,
                extra={"cluster_size": len(cluster)},
            )
            result.detections.append(det)
        result.quality = self._quality(pts, original_count)
        return result
