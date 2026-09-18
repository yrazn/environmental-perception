"""救援感知主管线共享的数据类型。

数据模型遵循目标级后融合、动态传感器权重、四级退化模式的设计。
检测层统一输出 ``Detection3D``，融合和跟踪层统一使用
``TrackedObject``，主管线最后汇总为 ``PipelineOutput``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把分数限制在指定范围，默认用于归一化可信度和风险值。"""
    return low if value < low else high if value > high else float(value)


class SensorType(IntEnum):
    RGB = 0
    THERMAL = 1
    LIDAR_3D = 2
    RADAR_4D = 3
    LIDAR_2D = 4
    GAS = 5
    PM = 6
    TEMP_HUMIDITY = 7


class SourceMask(IntEnum):
    """目标证据来源位掩码；同一目标可以同时包含多个来源位。"""
    RGB = 0x01
    THERMAL = 0x02
    LIDAR = 0x04
    RADAR = 0x08
    GAS = 0x10


class DegradationMode(IntEnum):
    """环境和传感器退化等级，数值越大表示感知能力越弱。"""
    CLEAR = 0
    LOW_VISIBILITY = 1
    HEAVY_SMOKE = 2
    PERCEPTION_DEGRADED = 3


class PerceptionStatus(IntEnum):
    OK = 0
    DEGRADED = 1
    LOST = 2


class SafetyLevel(IntEnum):
    SAFETY_OK = 0
    SAFETY_WARNING = 1
    SAFETY_DEGRADED = 2
    SAFETY_STOP_REQUIRED = 3
    SAFETY_ESTOP = 4


class EnvironmentRisk(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class VisibilityLevel(IntEnum):
    CLEAR = 0
    LOW_LIGHT = 1
    LIGHT_SMOKE = 2
    HEAVY_SMOKE = 3
    BLIND = 4


class ObjectClass(IntEnum):
    """系统统一目标类别编号，供所有检测器、融合器和页面共同使用。"""
    FLAME = 1
    SMOKE = 2
    PERSON_STANDING = 3
    PERSON_WALKING = 4
    PERSON_CROUCHING = 5
    PERSON_LYING = 6
    PERSON_OCCLUDED = 7
    VEHICLE_CAR = 8
    VEHICLE_TRUCK = 9
    VEHICLE_BUS = 10
    VEHICLE_MOTORCYCLE = 11
    VEHICLE_UNKNOWN = 12
    CRACK = 13
    WATER_PUDDLE = 14
    WATER_SEEPAGE = 15
    EQUIPMENT_DAMAGED = 16
    HOT_ZONE = 17
    OBSTACLE_GENERIC = 18
    PERSON_CANDIDATE = 19
    FIRE_CANDIDATE = 20
    HOTSPOT = 21
    UNKNOWN_HOT_OBJECT = 22


CLASS_NAMES: Dict[ObjectClass, str] = {
    ObjectClass.FLAME: "flame",
    ObjectClass.SMOKE: "smoke",
    ObjectClass.PERSON_STANDING: "person_standing",
    ObjectClass.PERSON_WALKING: "person_walking",
    ObjectClass.PERSON_CROUCHING: "person_crouching",
    ObjectClass.PERSON_LYING: "person_lying",
    ObjectClass.PERSON_OCCLUDED: "person_occluded",
    ObjectClass.VEHICLE_CAR: "vehicle_car",
    ObjectClass.VEHICLE_TRUCK: "vehicle_truck",
    ObjectClass.VEHICLE_BUS: "vehicle_bus",
    ObjectClass.VEHICLE_MOTORCYCLE: "vehicle_motorcycle",
    ObjectClass.VEHICLE_UNKNOWN: "vehicle_unknown",
    ObjectClass.CRACK: "crack",
    ObjectClass.WATER_PUDDLE: "water_puddle",
    ObjectClass.WATER_SEEPAGE: "water_seepage",
    ObjectClass.EQUIPMENT_DAMAGED: "equipment_damaged",
    ObjectClass.HOT_ZONE: "hot_zone",
    ObjectClass.OBSTACLE_GENERIC: "obstacle_generic",
    ObjectClass.PERSON_CANDIDATE: "person_candidate",
    ObjectClass.FIRE_CANDIDATE: "fire_candidate",
    ObjectClass.HOTSPOT: "hotspot",
    ObjectClass.UNKNOWN_HOT_OBJECT: "unknown_hot_object",
}


def class_name(class_id: ObjectClass) -> str:
    return CLASS_NAMES.get(ObjectClass(class_id), "unknown")


@dataclass
class SensorWeights:
    """当前帧六类传感器的融合权重。"""
    rgb: float = 0.0
    thermal: float = 0.0
    lidar: float = 0.0
    radar: float = 0.0
    lidar2d: float = 0.0
    gas: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.rgb, self.thermal, self.lidar,
                         self.radar, self.lidar2d, self.gas], dtype=float)

    def normalize(self) -> "SensorWeights":
        """原地归一化；全部为零时保持零权重，避免除零。"""
        total = float(np.sum(self.as_array()))
        if total > 1e-6:
            self.rgb /= total
            self.thermal /= total
            self.lidar /= total
            self.radar /= total
            self.lidar2d /= total
            self.gas /= total
        return self

    def to_dict(self) -> Dict[str, float]:
        return {
            "rgb": self.rgb, "thermal": self.thermal, "lidar": self.lidar,
            "radar": self.radar, "lidar2d": self.lidar2d, "gas": self.gas,
        }


@dataclass
class BehaviorRecommendation:
    max_speed_pct: float = 100.0
    require_confirmation: bool = False
    allow_autonomous: bool = True
    suggest_teleop: bool = False


@dataclass
class SensorHealth:
    sensor_name: str = ""
    data_quality: float = 1.0
    effective_ratio: float = 1.0
    noise_level: float = 0.0
    window_contaminated: bool = False
    internal_temperature: float = 25.0
    sync_error_ms: float = 0.0
    last_update_age: float = 0.0
    status: int = 0          # 0=OK 1=DEGRADED 2=FAILED 3=TIMEOUT
    cause: str = "normal"


@dataclass
class SensorCredibility:
    credibility: float = 1.0
    effective_ratio: float = 1.0
    cause: str = "normal"
    is_failed: bool = False


@dataclass
class EnvironmentQuality:
    """一帧环境状态和各传感器可信度快照。"""
    stamp: float = 0.0
    visibility_level: VisibilityLevel = VisibilityLevel.CLEAR
    visibility_score: float = 1.0
    credibility: Dict[SensorType, SensorCredibility] = field(
        default_factory=lambda: {
            SensorType.RGB: SensorCredibility(),
            SensorType.THERMAL: SensorCredibility(),
            SensorType.LIDAR_3D: SensorCredibility(),
            SensorType.RADAR_4D: SensorCredibility(),
            SensorType.LIDAR_2D: SensorCredibility(),
            SensorType.GAS: SensorCredibility(),
            SensorType.PM: SensorCredibility(),
        }
    )
    smoke_density: float = 0.0
    smoke_coverage: float = 0.0
    ambient_temperature: float = 25.0
    ambient_humidity: float = 50.0
    illuminance_lux: float = 500.0

    def rgb_credibility(self) -> float:
        return self.credibility[SensorType.RGB].credibility

    def thermal_credibility(self) -> float:
        return self.credibility[SensorType.THERMAL].credibility

    def lidar_credibility(self) -> float:
        return self.credibility[SensorType.LIDAR_3D].credibility

    def radar_credibility(self) -> float:
        return self.credibility[SensorType.RADAR_4D].credibility

    def lidar2d_credibility(self) -> float:
        return self.credibility[SensorType.LIDAR_2D].credibility

    def gas_credibility(self) -> float:
        return self.credibility[SensorType.GAS].credibility


@dataclass
class Detection3D:
    """单传感器检测候选的统一三维表示。"""
    stamp: float = 0.0
    track_id: int = 0
    class_id: ObjectClass = ObjectClass.OBSTACLE_GENERIC
    confidence: float = 0.0
    source: SensorType = SensorType.RGB
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dimensions: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    temperature_max: float = 0.0
    temperature_min: float = 0.0
    depth_valid: bool = True
    bearing: float = 0.0
    elevation: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        override = self.extra.get("class_name_override")
        return override if override else class_name(self.class_id)

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    @property
    def z(self) -> float:
        return float(self.position[2])


@dataclass
class TrackedObject:
    """完成融合和时序跟踪后的语义目标。

    ``state`` 顺序为 ``x,y,z,length,width,height,vx,vy``；位置和速度
    属性返回的是该数组视图，修改属性内容会同步修改内部状态。
    """
    track_id: int = 0
    class_id: ObjectClass = ObjectClass.OBSTACLE_GENERIC
    confidence: float = 0.0
    source_mask: int = 0
    confirmed: bool = False
    state: np.ndarray = field(default_factory=lambda: np.zeros(8))  # x,y,z,l,w,h,vx,vy
    cov: np.ndarray = field(default_factory=lambda: np.eye(8))
    pose_map: np.ndarray = field(default_factory=lambda: np.eye(4))
    pose_cov_map: np.ndarray = field(default_factory=lambda: np.eye(6))
    coast_frames: int = 0
    confirmed_frames: int = 0
    last_seen: float = 0.0
    first_seen: float = 0.0
    risk: EnvironmentRisk = EnvironmentRisk.LOW
    temperature_max: float = 0.0
    depth_valid: bool = True
    bearing: float = 0.0
    elevation: float = 0.0
    hot_zone_radius_m: float = 1.5
    blockage_ratio: float = 0.0
    accident_state: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        override = self.extra.get("class_name_override")
        return override if override else class_name(self.class_id)

    @property
    def position(self) -> np.ndarray:
        return self.state[:3]

    @property
    def dimensions(self) -> np.ndarray:
        return self.state[3:6]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[6:8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "class_name_override": self.extra.get("class_name_override", None),
            "confidence": round(self.confidence, 3),
            "source_mask": self.source_mask,
            "confirmed": self.confirmed,
            "position": self.position.tolist(),
            "dimensions": self.dimensions.tolist(),
            "velocity": self.velocity.tolist(),
            "risk_level": int(self.risk),
            "temperature_max": round(self.temperature_max, 1),
            "depth_valid": self.depth_valid,
            "blockage_ratio": round(self.blockage_ratio, 3),
            "accident_state": self.accident_state,
        }


class HotspotType(IntEnum):
    FIRE_SOURCE = 0
    FIRE_SUSPECTED = 1
    HOT_EQUIPMENT = 2
    WARM_SURFACE = 3
    HOT_ANOMALY = 4
    HUMAN_THERMAL = 5


@dataclass
class ThermalHotspot:
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t_max: float = 0.0
    t_mean: float = 0.0
    t_min: float = 0.0
    area_px: int = 0
    perimeter: int = 0
    circularity: float = 1.0
    dt_dt: float = 0.0
    da_dt: float = 0.0
    duration_s: float = 0.0
    edge_gradient: float = 0.0
    htype: HotspotType = HotspotType.HOT_ANOMALY
    type_confidence: float = 0.5
    center_px: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class SmokeEstimateResult:
    stamp: float = 0.0
    smoke_score: float = 0.0
    visibility_level: VisibilityLevel = VisibilityLevel.CLEAR
    motion_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    individual_scores: Dict[str, float] = field(default_factory=dict)
    image_smoke_prob: float = 0.0
    pm_rise_rate: float = 0.0
    lidar_attenuation: float = 0.0
    co_rise_rate: float = 0.0
    coverage_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smoke_score": round(self.smoke_score, 3),
            "visibility_level": int(self.visibility_level),
            "motion_direction": [round(v, 3) for v in self.motion_direction],
            "individual_scores": {k: round(v, 3)
                                  for k, v in self.individual_scores.items()},
        }


@dataclass
class GasReadings:
    co_ppm: float = 0.0
    co2_ppm: float = 400.0
    o2_pct: float = 20.9
    ch4_lel: float = 0.0
    h2s_ppm: float = 0.0
    voc_ppm: float = 0.0
    pm25: float = 0.0
    pm10: float = 0.0
    stamp: float = 0.0


@dataclass
class GasRiskResult:
    risk_level: int = 0            # 0=SAFE 1=CAUTION 2=DANGER 3=DEADLY
    individual_levels: Dict[str, int] = field(default_factory=dict)
    fire_signature: bool = False
    explosion_risk: bool = False
    asphyxiation_risk: bool = False
    alert: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "individual": self.individual_levels,
            "fire_signature": self.fire_signature,
            "explosion_risk": self.explosion_risk,
            "asphyxiation_risk": self.asphyxiation_risk,
            "alert": self.alert,
        }


@dataclass
class SyncedSensorData:
    """近似时间同步后的多传感器帧，也是主管线的输入边界。"""
    stamp: float = 0.0
    rgb_image: Optional[np.ndarray] = None
    thermal_image: Optional[np.ndarray] = None
    temperature_matrix: Optional[np.ndarray] = None
    lidar_points: Optional[np.ndarray] = None
    radar_targets: List[Dict[str, Any]] = field(default_factory=list)
    scans: Dict[str, np.ndarray] = field(default_factory=dict)
    gas: Optional[GasReadings] = None
    pm: Optional[Dict[str, float]] = None
    temperature_humidity: Optional[Dict[str, float]] = None
    odometry: Optional[Dict[str, Any]] = None
    sync_status: Dict[str, float] = field(default_factory=dict)


@dataclass
class SafetyDecision:
    safety_level: SafetyLevel = SafetyLevel.SAFETY_OK
    speed_limit: float = 1.0
    required_action: str = "NORMAL_OPERATION"
    require_remote_confirm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safety_level": int(self.safety_level),
            "speed_limit": round(self.speed_limit, 2),
            "required_action": self.required_action,
            "require_remote_confirm": self.require_remote_confirm,
        }


@dataclass
class RiskAssessment:
    risk_level: EnvironmentRisk = EnvironmentRisk.LOW
    combined_score: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    recommended_action: str = "NORMAL_OPERATION"
    hot_zones: List[Dict[str, Any]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": int(self.risk_level),
            "combined_score": round(self.combined_score, 3),
            "factors": {k: round(v, 3) for k, v in self.factors.items()},
            "recommended_action": self.recommended_action,
            "hot_zones": self.hot_zones,
            "events": self.events,
        }


@dataclass
class PipelineOutput:
    """单帧主管线最终输出，供终端、Web 和 ROS 2 适配器使用。"""
    stamp: float = 0.0
    targets: List[TrackedObject] = field(default_factory=list)
    status: PerceptionStatus = PerceptionStatus.OK
    mode: DegradationMode = DegradationMode.CLEAR
    weights: SensorWeights = field(default_factory=SensorWeights)
    env_quality: EnvironmentQuality = field(default_factory=EnvironmentQuality)
    smoke: SmokeEstimateResult = field(default_factory=SmokeEstimateResult)
    gas: GasRiskResult = field(default_factory=GasRiskResult)
    risk: RiskAssessment = field(default_factory=RiskAssessment)
    safety: SafetyDecision = field(default_factory=SafetyDecision)
    thermal_risk_level: int = 0
    cracks: List["CrackDefect"] = field(default_factory=list)
    water_regions: List["WaterRegion"] = field(default_factory=list)
    facility_anomalies: List["FacilityAnomaly"] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """生成可 JSON 序列化的紧凑摘要。"""
        return {
            "stamp": round(self.stamp, 3),
            "mode": int(self.mode),
            "status": int(self.status),
            "weights": self.weights.to_dict(),
            "env_risk": int(self.risk.risk_level),
            "safety": self.safety.to_dict(),
            "smoke": self.smoke.to_dict(),
            "gas": self.gas.to_dict(),
            "targets": [t.to_dict() for t in self.targets],
            "hot_zones": self.risk.hot_zones,
            "events": self.risk.events,
            "cracks": [item.to_dict() for item in self.cracks],
            "water_regions": [item.to_dict() for item in self.water_regions],
            "facility_anomalies": [item.to_dict()
                                   for item in self.facility_anomalies],
        }


@dataclass
class HotZone:
    polygon: List[List[float]] = field(default_factory=list)
    max_temperature: float = 0.0
    avg_temperature: float = 0.0
    area_m2: float = 0.0
    severity: int = 0


@dataclass
class CrackDefect:
    crack_id: int = 0
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wall_face: str = "unknown"
    length_m: float = 0.0
    width_mm: float = 0.0
    width_max_mm: float = 0.0
    orientation: float = 0.0
    num_branches: int = 0
    confidence: float = 0.0
    is_growing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crack_id": self.crack_id,
            "position": list(self.position),
            "wall_face": self.wall_face,
            "length_m": round(self.length_m, 4),
            "width_mm": round(self.width_mm, 3),
            "width_max_mm": round(self.width_max_mm, 3),
            "orientation": round(self.orientation, 3),
            "num_branches": self.num_branches,
            "confidence": round(self.confidence, 3),
            "is_growing": self.is_growing,
        }


@dataclass
class WaterRegion:
    region_id: int = 0
    kind: str = "puddle"            # puddle | seepage
    polygon: List[List[float]] = field(default_factory=list)
    depth_level: str = "shallow"
    wall_face: str = "ground"
    confidence: float = 0.0
    if_condensation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "polygon": self.polygon,
            "depth_level": self.depth_level,
            "wall_face": self.wall_face,
            "confidence": round(self.confidence, 3),
            "if_condensation": self.if_condensation,
        }


@dataclass
class FacilityAnomaly:
    anomaly_id: int = 0
    facility_class: str = "unknown"
    anomaly_type: str = "unknown"
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    confidence: float = 0.0
    is_new: bool = True
    thermal_temperature: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "facility_class": self.facility_class,
            "anomaly_type": self.anomaly_type,
            "position": list(self.position),
            "confidence": round(self.confidence, 3),
            "is_new": self.is_new,
            "thermal_temperature": round(self.thermal_temperature, 1),
        }
