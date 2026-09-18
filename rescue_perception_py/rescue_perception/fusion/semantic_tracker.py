"""统一语义多目标跟踪器及轨迹生命周期管理。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import ObjectClass, TrackedObject


class SemanticTracker:
    """把人员、火源和车辆融合结果放入同一个全局轨迹池。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        params = self.config.fusion["semantic_tracker"]
        self.max_tracks = int(params.get("max_tracks", 256))
        self.max_coast = int(params.get("max_coast_frames", 15))
        self.min_confirm = int(params.get("min_confirm_frames", 3))
        self.merge_distance = float(params.get("merge_distance", 0.5))
        self.tracks: List[TrackedObject] = []
        self.next_id = 1

    @staticmethod
    def _group(class_id: ObjectClass) -> str:
        """按语义大类关联，避免车辆轨迹被火源或人员测量更新。"""
        if (ObjectClass.PERSON_STANDING <= class_id <= ObjectClass.PERSON_OCCLUDED
                or class_id == ObjectClass.PERSON_CANDIDATE):
            return "person"
        if class_id in (ObjectClass.FLAME, ObjectClass.FIRE_CANDIDATE,
                        ObjectClass.HOTSPOT, ObjectClass.HOT_ZONE):
            return "fire"
        if ObjectClass.VEHICLE_CAR <= class_id <= ObjectClass.VEHICLE_UNKNOWN:
            return "vehicle"
        return "obstacle"

    def _predict(self, now: float) -> None:
        """使用平面匀速模型预测未到达新测量前的轨迹状态。"""
        for tr in self.tracks:
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

    def _update_kalman(self, tr: TrackedObject, measurement: TrackedObject) -> None:
        h = np.zeros((3, 8))
        h[0, 0] = h[1, 1] = h[2, 2] = 1.0
        r = np.eye(3) * 0.15
        s = h @ tr.cov @ h.T + r
        k = tr.cov @ h.T @ np.linalg.inv(s)
        innovation = measurement.position - h @ tr.state
        tr.state += k @ innovation
        tr.cov = (np.eye(8) - k @ h) @ tr.cov

    def _match(self, measurement: TrackedObject) -> Optional[TrackedObject]:
        best = None
        best_dist = 3.0
        group = self._group(measurement.class_id)
        for tr in self.tracks:
            if tr.coast_frames > 0 or self._group(tr.class_id) != group:
                continue
            dist = float(np.linalg.norm(tr.position - measurement.position))
            if dist < best_dist:
                best_dist = dist
                best = tr
        return best

    def _merge(self) -> None:
        """合并过近的同语义轨迹，减少多个融合器造成的重复目标。"""
        changed = True
        while changed:
            changed = False
            for i in range(len(self.tracks)):
                for j in range(i + 1, len(self.tracks)):
                    a, b = self.tracks[i], self.tracks[j]
                    if (self._group(a.class_id) == self._group(b.class_id)
                            and np.linalg.norm(a.position - b.position) < self.merge_distance):
                        if a.confidence >= b.confidence:
                            a.confirmed = a.confirmed or b.confirmed
                            a.source_mask |= b.source_mask
                            a.last_seen = max(a.last_seen, b.last_seen)
                            self.tracks.pop(j)
                        else:
                            b.confirmed = a.confirmed or b.confirmed
                            b.source_mask |= a.source_mask
                            b.last_seen = max(a.last_seen, b.last_seen)
                            self.tracks.pop(i)
                        changed = True
                        break
                if changed:
                    break

    def update(self, fused: List[TrackedObject], now: float = 0.0) -> List[TrackedObject]:
        """完成预测、关联、更新、新建、coast、删除和合并。"""
        self._predict(now)
        matched = set()
        for measurement in fused:
            if len(self.tracks) >= self.max_tracks:
                break
            track = self._match(measurement)
            if track is None:
                # 低置信度单帧候选不创建轨迹，避免轨迹池被噪声占满。
                if measurement.confidence < 0.3:
                    continue
                track = TrackedObject(
                    track_id=self.next_id,
                    class_id=measurement.class_id,
                    confidence=measurement.confidence,
                    source_mask=measurement.source_mask,
                    confirmed=False,
                    state=measurement.state.copy(),
                    cov=measurement.cov.copy(),
                    pose_map=measurement.pose_map.copy(),
                    pose_cov_map=measurement.pose_cov_map.copy(),
                    first_seen=now,
                    last_seen=now,
                    temperature_max=measurement.temperature_max,
                    depth_valid=measurement.depth_valid,
                    bearing=measurement.bearing,
                    elevation=measurement.elevation,
                    hot_zone_radius_m=measurement.hot_zone_radius_m,
                    blockage_ratio=measurement.blockage_ratio,
                    accident_state=measurement.accident_state,
                    extra=dict(measurement.extra),
                )
                self.next_id += 1
                self.tracks.append(track)
                matched.add(id(track))
                continue
            self._update_kalman(track, measurement)
            track.confidence = measurement.confidence
            track.class_id = measurement.class_id
            track.source_mask |= measurement.source_mask
            track.coast_frames = 0
            track.confirmed_frames += 1
            track.confirmed = track.confirmed_frames >= self.min_confirm
            track.last_seen = now
            track.temperature_max = max(track.temperature_max,
                                        measurement.temperature_max)
            track.depth_valid = measurement.depth_valid
            track.bearing = measurement.bearing
            track.elevation = measurement.elevation
            track.hot_zone_radius_m = measurement.hot_zone_radius_m
            track.blockage_ratio = measurement.blockage_ratio
            track.accident_state = measurement.accident_state
            track.extra.update(measurement.extra)
            track.pose_map = measurement.pose_map.copy()
            track.pose_map[:3, 3] = track.position
            track.pose_cov_map = measurement.pose_cov_map.copy()
            matched.add(id(track))

        for tr in self.tracks:
            if id(tr) not in matched:
                tr.coast_frames += 1
        self.tracks = [tr for tr in self.tracks if tr.coast_frames <= self.max_coast]
        self._merge()
        return self.tracks
