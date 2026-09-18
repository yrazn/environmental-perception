"""4D 毫米波目标过滤、卡尔曼跟踪和人员/车辆粗分类。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..types import DegradationMode, Detection3D, ObjectClass, SensorType


@dataclass
class _RadarTrack:
    """雷达内部六维轨迹：x、y、vx、vy、ax、ay。"""
    track_id: int = 0
    state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    cov: np.ndarray = field(default_factory=lambda: np.eye(6))
    coast_frames: int = 0
    seen_frames: int = 0
    last_seen: float = 0.0
    rcs: float = 0.0
    doppler: float = 0.0
    micro_doppler: bool = False


class RadarTracker:
    """维护不受烟雾显著影响的毫米波目标轨迹和最近障碍距离。"""
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        params = self.config.detectors["radar_tracker"]
        self.doppler_thresh = float(params["static_clutter_doppler_thresh"])
        self.static_min_frames = int(params["static_object_min_frames"])
        self.rcs_min = float(params["rcs_min"])
        self.max_coast = int(params["max_coast_frames"])
        self.vehicle_rcs_min = float(params["vehicle_rcs_min"])
        self.person_rcs_min = float(params["person_rcs_min"])
        self.person_rcs_max = float(params["person_rcs_max"])
        self.micro_amp = float(params["micro_doppler_amplitude"])
        self.q = np.diag(params["process_noise"])
        self.r = np.diag(params["observation_noise"][:2])
        self.tracks: List[_RadarTrack] = []
        self.next_id = 1
        self.static_seen: Dict[Tuple[int, int], int] = {}
        self._nearest_range = float("inf")

    @staticmethod
    def _to_xy(target: Dict) -> np.ndarray:
        if "x" in target and "y" in target:
            return np.array([target["x"], target["y"]], dtype=float)
        r = float(target["range"])
        az = float(target.get("azimuth", 0.0))
        return np.array([r * np.cos(az), r * np.sin(az)])

    def _filter(self, targets: List[Dict[str, float]], now: float) -> List[Dict[str, float]]:
        """保留运动目标、高 RCS 目标和持续出现的静态目标。"""
        out = []
        for t in targets:
            r = float(t["range"])
            doppler = float(t.get("doppler", 0.0))
            rcs = float(t.get("rcs", -100.0))
            if abs(doppler) > self.doppler_thresh or rcs > 0.0:
                out.append(t)
                continue
            key = (int(r * 10), int(np.rad2deg(float(t.get("azimuth", 0.0))) * 10))
            self.static_seen[key] = self.static_seen.get(key, 0) + 1
            if self.static_seen[key] >= self.static_min_frames:
                out.append(t)
        return out

    def _predict(self, dt: float) -> None:
        for tr in self.tracks:
            x, y, vx, vy, ax, ay = tr.state
            tr.state += np.array([vx * dt, vy * dt, ax * dt, ay * dt, 0, 0])
            f = np.eye(6)
            f[0, 2] = dt
            f[1, 3] = dt
            f[2, 4] = dt
            f[3, 5] = dt
            tr.cov = f @ tr.cov @ f.T + self.q * dt

    def _update(self, tr: _RadarTrack, xy: np.ndarray) -> None:
        h = np.zeros((2, 6))
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        innovation = xy - h @ tr.state
        s = h @ tr.cov @ h.T + self.r
        k = tr.cov @ h.T @ np.linalg.inv(s)
        tr.state += k @ innovation
        tr.cov = (np.eye(6) - k @ h) @ tr.cov

    def _associate(self, xy_list: List[np.ndarray]) -> List[Tuple[int, int]]:
        """在三米门限内进行一对一最近邻关联。"""
        matches = []
        used_tracks = set()
        used_cands = set()
        for i, xy in enumerate(xy_list):
            best = None
            best_d = 1e9
            for j, tr in enumerate(self.tracks):
                if j in used_tracks:
                    continue
                d = float(np.linalg.norm(xy - tr.state[:2]))
                if d < 3.0 and d < best_d:
                    best_d = d
                    best = j
            if best is not None:
                matches.append((i, best))
                used_tracks.add(best)
                used_cands.add(i)
        return matches

    def _classify(self, rcs: float, speed: float,
                  micro: bool) -> Tuple[ObjectClass, str]:
        if micro and self.person_rcs_min <= rcs <= self.person_rcs_max:
            return ObjectClass.PERSON_CANDIDATE, "person_candidate"
        if rcs > self.vehicle_rcs_min:
            return ObjectClass.VEHICLE_UNKNOWN, "vehicle_like_obstacle"
        if rcs >= 0.0 and speed < 2.0:
            return ObjectClass.OBSTACLE_GENERIC, "medium_object"
        if speed < 0.5:
            return ObjectClass.OBSTACLE_GENERIC, "small_object"
        return ObjectClass.OBSTACLE_GENERIC, "radar_object"

    def track(self,
              targets: Optional[List[Dict[str, float]]],
              mode: DegradationMode = DegradationMode.CLEAR,
              now: float = 0.0,
              dt: float = 0.1) -> List[Detection3D]:
        """更新雷达轨迹池并输出统一 Detection3D 候选。"""
        if not targets:
            for tr in self.tracks:
                tr.coast_frames += 1
            self.tracks = [tr for tr in self.tracks if tr.coast_frames <= self.max_coast]
            self._nearest_range = float("inf")
            return []

        filtered = self._filter(targets, now)
        xy_list = [self._to_xy(t) for t in filtered]
        self._predict(max(dt, 1e-3))

        matched = self._associate(xy_list)
        matched_cands = set(i for i, _ in matched)
        matched_tracks = set(j for _, j in matched)
        for i, j in matched:
            tr = self.tracks[j]
            self._update(tr, xy_list[i])
            t = filtered[i]
            tr.coast_frames = 0
            tr.seen_frames += 1
            tr.last_seen = now
            tr.rcs = float(t.get("rcs", tr.rcs))
            tr.doppler = float(t.get("doppler", tr.doppler))
            tr.micro_doppler = bool(t.get("micro_doppler", False))

        for i, t in enumerate(filtered):
            if i in matched_cands:
                continue
            rcs = float(t.get("rcs", -100.0))
            if rcs < self.rcs_min:
                continue
            tr = _RadarTrack(
                track_id=self.next_id,
                state=np.array([xy_list[i][0], xy_list[i][1], 0, 0, 0, 0]),
                last_seen=now,
                seen_frames=1,
                rcs=rcs,
                doppler=float(t.get("doppler", 0.0)),
                micro_doppler=bool(t.get("micro_doppler", False)),
            )
            self.next_id += 1
            self.tracks.append(tr)

        for j, tr in enumerate(self.tracks):
            if j not in matched_tracks:
                tr.coast_frames += 1
        self.tracks = [tr for tr in self.tracks if tr.coast_frames <= self.max_coast]

        detections = []
        ranges = []
        for tr in self.tracks:
            speed = float(np.linalg.norm(tr.state[2:4]))
            class_id, extra_class = self._classify(tr.rcs, speed, tr.micro_doppler)
            confidence = 0.7 if mode >= DegradationMode.HEAVY_SMOKE else 0.5
            if tr.coast_frames > 0:
                confidence *= 0.8
            rng = float(np.linalg.norm(tr.state[:2]))
            ranges.append(rng)
            detections.append(Detection3D(
                class_id=class_id,
                confidence=confidence,
                source=SensorType.RADAR_4D,
                position=np.array([tr.state[0], tr.state[1], 0.0]),
                velocity=np.array([tr.state[2], tr.state[3], 0.0]),
                extra={"rcs": tr.rcs, "doppler": tr.doppler,
                       "radar_class": extra_class,
                       "micro_doppler": tr.micro_doppler},
            ))
        self._nearest_range = min(ranges) if ranges else float("inf")
        return detections

    def nearest_range(self) -> float:
        return self._nearest_range
