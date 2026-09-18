"""RGB、热成像、LiDAR 和毫米波的人员多源融合。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..types import (
    DegradationMode,
    Detection3D,
    EnvironmentQuality,
    ObjectClass,
    SensorType,
    SensorWeights,
    SourceMask,
    TrackedObject,
    clamp,
)
from .association import gated_hungarian


class PersonFusion:
    """通过位置关联、人体温度、尺寸和微多普勒验证人员候选。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        assoc = self.config.fusion["association"]
        self.max_dist = float(assoc["max_association_dist"])
        self.max_cost = float(assoc["max_association_cost"])
        self.w_pos = float(assoc["w_pos"])
        self.w_class = float(assoc["w_class"])
        self.w_temp = float(assoc["w_temp"])
        self.w_time = float(assoc["w_time"])
        params = self.config.fusion["person_fusion"]
        self.max_coast = int(params.get("max_coast_frames", 15))
        self.min_confirm = int(params.get("min_confirm_frames", 3))
        self.active_tracks: List[TrackedObject] = []
        self.next_id = 1

    @staticmethod
    def _is_person(det: Detection3D) -> bool:
        return ObjectClass.PERSON_STANDING <= det.class_id <= ObjectClass.PERSON_OCCLUDED

    def _collect(self, rgb: List[Detection3D], thermal: List[Detection3D],
                 lidar: List[Detection3D], radar: List[Detection3D],
                 env: EnvironmentQuality) -> List[Detection3D]:
        candidates = []
        if env.rgb_credibility() > 0.1:
            candidates += [d for d in rgb if self._is_person(d) and d.confidence > 0.3]
        if env.thermal_credibility() > 0.1:
            candidates += [d for d in thermal if self._is_person(d) and d.confidence > 0.2]
        if env.lidar_credibility() > 0.1:
            for d in lidar:
                if (d.class_id in (ObjectClass.OBSTACLE_GENERIC,
                                   ObjectClass.PERSON_CANDIDATE)
                        and 0.3 <= d.dimensions[2] <= 2.0
                        and d.dimensions[1] <= 1.0):
                    candidates.append(d)
        if env.radar_credibility() > 0.1:
            candidates += [d for d in radar if d.confidence > 0.3]
        return candidates

    def _predict(self, now: float) -> None:
        for tr in self.active_tracks:
            dt = max(min(now - tr.last_seen, 1.0), 0.0)
            if dt <= 0:
                continue
            f = np.eye(8)
            f[0, 6] = dt
            f[1, 7] = dt
            tr.state = f @ tr.state
            q = np.eye(8) * 0.1
            q[6, 6] = 0.5
            q[7, 7] = 0.5
            tr.cov = f @ tr.cov @ f.T + q

    def _update_kalman(self, tr: TrackedObject, position: np.ndarray) -> None:
        h = np.zeros((3, 8))
        h[0, 0] = h[1, 1] = h[2, 2] = 1.0
        r = np.eye(3) * 0.15
        s = h @ tr.cov @ h.T + r
        k = tr.cov @ h.T @ np.linalg.inv(s)
        tr.state += k @ (position - h @ tr.state)
        tr.cov = (np.eye(8) - k @ h) @ tr.cov

    def _association_cost(self, cand: Detection3D, tr: TrackedObject,
                          now: float) -> float:
        """组合空间、类别、温度和时间差异作为关联代价。"""
        dist = float(np.linalg.norm(cand.position - tr.position))
        if dist > self.max_dist:
            return float("inf")
        class_diff = 0.0 if cand.class_id == tr.class_id else 1.0
        temp_diff = abs(cand.temperature_max - tr.temperature_max) / 10.0
        dt = min(max(now - tr.last_seen, 0.0), 2.0)
        return (self.w_pos * dist + self.w_class * class_diff
                + self.w_temp * temp_diff + self.w_time * dt)

    @staticmethod
    def _sources_near(candidates: List[Detection3D], position: np.ndarray,
                      radius: float) -> Tuple[Optional[Detection3D],
                                              Optional[Detection3D],
                                              Optional[Detection3D],
                                              Optional[Detection3D]]:
        rgb = thermal = lidar = radar = None
        for c in candidates:
            if np.linalg.norm(c.position - position) > radius:
                continue
            if c.source == SensorType.RGB and rgb is None:
                rgb = c
            elif c.source == SensorType.THERMAL and thermal is None:
                thermal = c
            elif c.source == SensorType.LIDAR_3D and lidar is None:
                lidar = c
            elif c.source == SensorType.RADAR_4D and radar is None:
                radar = c
        return rgb, thermal, lidar, radar

    def fuse_confidence(self,
                        rgb: Optional[Detection3D],
                        thermal: Optional[Detection3D],
                        lidar: Optional[Detection3D],
                        radar: Optional[Detection3D],
                        weights: SensorWeights,
                        mode: DegradationMode) -> Tuple[float, ObjectClass, bool, int]:
        """融合邻近多源证据并返回置信度、类别、确认状态和来源掩码。"""
        fused = 0.0
        weight_sum = 0.0
        positive = 0
        source_mask = 0

        if rgb is not None and rgb.confidence > 0.3:
            w = weights.rgb
            if mode >= DegradationMode.HEAVY_SMOKE:
                w *= 0.1
            fused += w * rgb.confidence
            weight_sum += w
            source_mask |= int(SourceMask.RGB)
            if rgb.confidence > 0.5:
                positive += 1

        if thermal is not None and thermal.confidence > 0.2:
            w = weights.thermal
            t = thermal.temperature_max
            if t < 24.0 or t > 44.0:
                validity = 0.3
            elif t < 28.0 or t > 42.0:
                validity = 0.7
            else:
                validity = 1.0
            conf = thermal.confidence * validity
            fused += w * conf
            weight_sum += w
            source_mask |= int(SourceMask.THERMAL)
            if conf > 0.3:
                positive += 1

        if lidar is not None and lidar.confidence > 0.1:
            width = lidar.dimensions[1] if len(lidar.dimensions) > 1 else 0.0
            depth = lidar.dimensions[0] if len(lidar.dimensions) > 0 else 0.0
            height = lidar.dimensions[2] if len(lidar.dimensions) > 2 else 0.0
            size_ok = (0.3 <= width <= 1.0 and 0.2 <= depth <= 0.8
                       and 0.3 <= height <= 2.0)
            aspect_ok = height / max(width, 0.1) > 1.2
            shape_score = 1.0 if (size_ok and aspect_ok) else (0.7 if size_ok else 0.3)
            conf = 0.4 * shape_score
            fused += weights.lidar * conf
            weight_sum += weights.lidar
            source_mask |= int(SourceMask.LIDAR)
            if conf > 0.3 and shape_score > 0.5:
                positive += 1

        if radar is not None and radar.confidence > 0.3:
            speed = float(np.linalg.norm(radar.velocity))
            vel_ok = speed < 2.0
            vel_score = 1.0 if vel_ok else max(0.0, 1.0 - (speed - 2.0) / 3.0)
            rcs = float(radar.extra.get("rcs", 0.0))
            micro = bool(radar.extra.get("micro_doppler", False))
            score = 0.0
            if -5.0 <= rcs <= 10.0:
                score += 0.4
            else:
                score += 0.1
            score += 0.3 if vel_ok else 0.1
            score += 0.3 if micro else 0.0
            conf = 0.5 * score
            fused += weights.radar * conf
            weight_sum += weights.radar
            source_mask |= int(SourceMask.RADAR)
            if conf > 0.3 and vel_ok:
                positive += 1

        if weight_sum > 1e-6:
            fused /= weight_sum
        confirmed = positive >= 2
        if confirmed and fused >= 0.70:
            out_class = ObjectClass.PERSON_STANDING
        elif fused >= 0.40:
            out_class = ObjectClass.PERSON_CANDIDATE
        else:
            out_class = ObjectClass.OBSTACLE_GENERIC

        # 重烟时可见光不可靠，热成像与雷达联合证据可升格为人员候选。
        if mode >= DegradationMode.HEAVY_SMOKE:
            if (thermal is not None and radar is not None
                    and np.linalg.norm(radar.velocity) > 0.3):
                if out_class < ObjectClass.PERSON_CANDIDATE:
                    out_class = ObjectClass.PERSON_CANDIDATE
                fused = max(fused, 0.45)
            if (thermal is not None and thermal.confidence > 0.4
                    and 28.0 <= thermal.temperature_max <= 42.0):
                if out_class < ObjectClass.PERSON_CANDIDATE:
                    out_class = ObjectClass.PERSON_CANDIDATE
                fused = max(fused, 0.40)
        return clamp(fused), out_class, confirmed, source_mask

    def fuse(self,
             rgb_detections: List[Detection3D],
             thermal_detections: List[Detection3D],
             lidar_clusters: List[Detection3D],
             radar_tracks: List[Detection3D],
             weights: SensorWeights,
             mode: DegradationMode,
             env_quality: EnvironmentQuality,
             now: float = 0.0) -> List[TrackedObject]:
        """关联本帧候选与人员轨迹，输出供全局跟踪器使用的融合测量。"""
        if all(w < 0.01 for w in weights.as_array()[:4]):
            return []

        candidates = self._collect(rgb_detections, thermal_detections,
                                   lidar_clusters, radar_tracks, env_quality)
        self._predict(now)

        cost = np.full((len(candidates), len(self.active_tracks)),
                       float("inf"))
        for i, cand in enumerate(candidates):
            for j, tr in enumerate(self.active_tracks):
                cost[i, j] = self._association_cost(cand, tr, now)
        pairs = gated_hungarian(cost, self.max_cost)
        matched_cands = set(i for i, _ in pairs)
        matched_tracks = set(j for _, j in pairs)

        results = []
        for i, j in pairs:
            tr = self.active_tracks[j]
            cand = candidates[i]
            self._update_kalman(tr, cand.position)
            rgb, thermal, lidar, radar = self._sources_near(
                candidates, tr.position, radius=1.5)
            fused, out_class, confirmed, source_mask = self.fuse_confidence(
                rgb, thermal, lidar, radar, weights, mode)
            tr.confidence = fused
            tr.class_id = out_class
            tr.confirmed = confirmed
            tr.source_mask = source_mask
            tr.coast_frames = 0
            tr.confirmed_frames += 1
            tr.last_seen = now
            tr.temperature_max = max((d.temperature_max for d in
                                      (rgb, thermal, lidar, radar) if d is not None),
                                     default=tr.temperature_max)
            nearby = [d for d in (rgb, thermal, lidar, radar) if d is not None]
            tr.depth_valid = any(d.depth_valid for d in nearby)
            results.append(tr)

        for i in range(len(candidates)):
            if i in matched_cands:
                continue
            cand = candidates[i]
            threshold = 0.35 if mode >= DegradationMode.HEAVY_SMOKE else 0.50
            if cand.confidence < threshold:
                continue
            new_track = TrackedObject(
                track_id=self.next_id,
                class_id=cand.class_id,
                confidence=cand.confidence,
                source_mask={SensorType.RGB: SourceMask.RGB,
                             SensorType.THERMAL: SourceMask.THERMAL,
                             SensorType.LIDAR_3D: SourceMask.LIDAR,
                             SensorType.RADAR_4D: SourceMask.RADAR}[cand.source],
                state=np.concatenate([cand.position, cand.dimensions, [0, 0]]),
                confirmed_frames=0,
                first_seen=now,
                last_seen=now,
                temperature_max=cand.temperature_max,
                depth_valid=cand.depth_valid,
                bearing=cand.bearing,
                elevation=cand.elevation,
            )
            self.next_id += 1
            self.active_tracks.append(new_track)
            results.append(new_track)

        for j, tr in enumerate(self.active_tracks):
            if j not in matched_tracks:
                tr.coast_frames += 1
                if tr.coast_frames <= self.max_coast:
                    results.append(tr)
        self.active_tracks = [tr for tr in self.active_tracks
                              if tr.coast_frames <= self.max_coast]
        return results
