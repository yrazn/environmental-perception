"""环境感知主管线控制器。

该模块把一个完成时间同步的多传感器帧依次送入质量评估、独立检测、
目标级融合、统一跟踪、地图定位、风险评估和安全决策，是系统端到端
调用链的唯一核心入口。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .config import PerceptionConfig
from .detectors.crack_segmenter import CrackSegmenter
from .detectors.facility_inspector import FacilityInspector
from .detectors.gas_risk import GasRiskAssessor
from .detectors.lidar_cluster import LidarCluster, LidarClusterResult
from .detectors.radar_tracker import RadarTracker
from .detectors.rgb_detector import RgbDetectResult, RgbDetector
from .detectors.smoke_estimator import SmokeEstimator
from .detectors.thermal_detector import ThermalDetectResult, ThermalDetector
from .detectors.water_detector import WaterDetector
from .fusion.fire_fusion import FireFusion
from .fusion.person_fusion import PersonFusion
from .fusion.semantic_localizer import SemanticLocalizer
from .fusion.semantic_tracker import SemanticTracker
from .fusion.vehicle_fusion import VehicleFusion
from .management.degradation_manager import DegradationManager
from .management.quality_monitor import QualityMonitor
from .management.risk_assessor import RiskAssessor
from .io.sensor_sync import SensorSync
from .types import (
    DegradationMode,
    EnvironmentQuality,
    GasRiskResult,
    PerceptionStatus,
    PipelineOutput,
    SensorHealth,
    SensorType,
    SmokeEstimateResult,
    SyncedSensorData,
)


class PerceptionPipeline:
    """对一个同步传感器帧执行完整感知流程。

    检测器和融合器都是有状态对象，例如烟雾 EMA、热点历史、卡尔曼
    轨迹和退化状态机会跨帧保留信息，因此生产代码不应每帧重新创建
    ``PerceptionPipeline``。
    """

    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        # 管理层：先判断数据是否可信，再决定本帧各传感器权重。
        self.quality_monitor = QualityMonitor()
        self.degradation_manager = DegradationManager(self.config)
        # 检测层：每种传感器先独立产生统一 Detection3D 候选。
        self.rgb_detector = RgbDetector(self.config)
        self.thermal_detector = ThermalDetector(self.config)
        self.lidar_cluster = LidarCluster(self.config)
        self.radar_tracker = RadarTracker(self.config)
        self.smoke_estimator = SmokeEstimator(self.config)
        self.gas_risk = GasRiskAssessor(self.config)
        # 融合层：按目标语义分别融合，最后进入全局统一跟踪器。
        self.person_fusion = PersonFusion(self.config)
        self.fire_fusion = FireFusion()
        self.vehicle_fusion = VehicleFusion(self.config)
        self.semantic_tracker = SemanticTracker(self.config)
        self.semantic_localizer = SemanticLocalizer()
        self.risk_assessor = RiskAssessor(self.config)
        # 巡检任务仅在当前运行模式允许时执行。
        self.crack_segmenter = CrackSegmenter(self.config)
        self.water_detector = WaterDetector(self.config)
        self.facility_inspector = FacilityInspector(self.config)

        self.env_quality = EnvironmentQuality()
        self.smoke_estimate = SmokeEstimateResult()
        self.gas_risk_result = GasRiskResult()
        self.thermal_risk_level = 0
        self.thermal_result = None
        self.mode = DegradationMode.CLEAR
        # 阶段0需要 LiDAR 质量，但本帧点云尚未处理，因此使用上一帧缓存。
        self._last_lidar_quality_cache: dict = {}

    def initialize(self) -> None:
        """初始化主管线的对外状态。

        完整清除所有算法历史应重新创建主管线；Web 页面的“重置”正是
        通过重建对象完成，避免旧轨迹或烟雾平滑值残留。
        """
        self.env_quality = EnvironmentQuality()
        self.smoke_estimate = SmokeEstimateResult()
        self.gas_risk_result = GasRiskResult()
        self.thermal_risk_level = 0
        self.thermal_result = None
        self.mode = DegradationMode.CLEAR

    @staticmethod
    def _health_from_quality(env: EnvironmentQuality,
                             synced: Optional[SyncedSensorData] = None) -> dict:
        """将连续可信度和同步误差转换为离散传感器健康状态。"""
        health = {}
        aliases = {
            SensorType.LIDAR_3D: "lidar",
            SensorType.RADAR_4D: "radar",
            SensorType.LIDAR_2D: "lidar2d",
        }
        sync_keys = {
            SensorType.RGB: "rgb",
            SensorType.THERMAL: "thermal",
            SensorType.LIDAR_3D: "lidar",
            SensorType.RADAR_4D: "radar",
            SensorType.GAS: "gas",
            SensorType.PM: "pm",
        }
        for sensor_type, cred in env.credibility.items():
            if cred.credibility < 0.10:
                status = 2
                cause = "failed"
            elif cred.credibility < 0.30:
                status = 1
                cause = "degraded"
            else:
                status = 0
                cause = "normal"
            sync_error = 0.0
            # 同步误差超过 100 ms 时降级，超过 500 ms 时按超时处理。
            if synced is not None and sync_keys.get(sensor_type) in synced.sync_status:
                sync_error = float(synced.sync_status[sync_keys[sensor_type]])
                if sync_error > 500.0:
                    status, cause = 3, "timeout"
                elif sync_error > 100.0 and status == 0:
                    status, cause = 1, "sync_error"
            name = aliases.get(sensor_type, sensor_type.name.lower())
            health[name] = SensorHealth(
                sensor_name=name,
                data_quality=cred.credibility,
                effective_ratio=cred.effective_ratio,
                sync_error_ms=sync_error,
                status=status,
                cause=cause,
            )
        return health

    def _apply_sensor_switches(self, weights) -> None:
        """把配置中禁用的传感器从融合权重中剔除并重新归一化。"""
        if not (self.config.sensor_enabled("rgb_front")
                or self.config.sensor_enabled("rgb_ptz")):
            weights.rgb = 0.0
        if not self.config.sensor_enabled("thermal"):
            weights.thermal = 0.0
        if not self.config.sensor_enabled("lidar_3d"):
            weights.lidar = 0.0
        if not self.config.sensor_enabled("radar_4d"):
            weights.radar = 0.0
        if not (self.config.sensor_enabled("lidar_2d_front")
                or self.config.sensor_enabled("lidar_2d_rear")):
            weights.lidar2d = 0.0
        if not self.config.sensor_enabled("gas_sensors"):
            weights.gas = 0.0
        weights.normalize()

    def _apply_sensor_credibility_switches(self) -> None:
        """让外部状态明确反映“配置禁用”，而不是显示虚假的可信度。"""
        disabled = []
        if not (self.config.sensor_enabled("rgb_front")
                or self.config.sensor_enabled("rgb_ptz")):
            disabled.append(SensorType.RGB)
        if not self.config.sensor_enabled("thermal"):
            disabled.append(SensorType.THERMAL)
        if not self.config.sensor_enabled("lidar_3d"):
            disabled.append(SensorType.LIDAR_3D)
        if not self.config.sensor_enabled("radar_4d"):
            disabled.append(SensorType.RADAR_4D)
        if not (self.config.sensor_enabled("lidar_2d_front")
                or self.config.sensor_enabled("lidar_2d_rear")):
            disabled.append(SensorType.LIDAR_2D)
        if not self.config.sensor_enabled("gas_sensors"):
            disabled.append(SensorType.GAS)
        if not self.config.sensor_enabled("pm_sensors"):
            disabled.append(SensorType.PM)
        for sensor_type in disabled:
            credibility = self.env_quality.credibility[sensor_type]
            credibility.credibility = 0.0
            credibility.effective_ratio = 0.0
            credibility.is_failed = True
            credibility.cause = "disabled"

    def spin_from_sync(self,
                       sensor_sync: SensorSync,
                       keys: Optional[Sequence[str]] = None,
                       now: Optional[float] = None) -> Optional[PipelineOutput]:
        """从近似时间同步器拉取一帧并运行主管线。

        当同步器中没有任何数据时返回 ``None``；调用方可在下一轮继续
        等待，不需要构造空的 ``SyncedSensorData``。
        """
        synced = sensor_sync.pull(list(keys) if keys is not None else None)
        if synced is None:
            return None
        return self.spin_once(synced, now=now)

    def spin_once(self,
                  synced: SyncedSensorData,
                  now: Optional[float] = None) -> PipelineOutput:
        """处理一个同步帧并返回可发布、记录和可视化的统一输出。"""
        stamp = now if now is not None else synced.stamp

        # 阶段0：质量检查 -> 退化模式 -> 动态权重。
        # 烟雾和 LiDAR 质量采用上一帧结果，从而打破当前帧的循环依赖。
        self.env_quality = self.quality_monitor.assess(
            synced,
            smoke_score=self.smoke_estimate.smoke_score,
            lidar_quality=self._last_lidar_quality_cache,
            radar_targets=synced.radar_targets,
        )
        self._apply_sensor_credibility_switches()
        health = self._health_from_quality(self.env_quality, synced)
        gas_failed = (self.config.sensor_enabled("gas_sensors")
                      and health.get("gas", SensorHealth()).status in (2, 3))
        self.mode = self.degradation_manager.update(
            self.env_quality, self.smoke_estimate, gas_failed=gas_failed, now=stamp)
        weights = self.degradation_manager.get_weights(self.mode, self.env_quality)
        self._apply_sensor_switches(weights)

        # 阶段1：各传感器独立检测。配置开关在调用前生效，禁用模块不会
        # 消耗推理或点云处理算力。
        rgb_enabled = (self.config.sensor_enabled("rgb_front")
                       or self.config.sensor_enabled("rgb_ptz"))
        any_rgb_task = any(self.config.detector_enabled(name) for name in (
            "flame_detection", "smoke_detection", "person_detection",
            "vehicle_detection", "crack_detection", "water_detection",
            "facility_inspection"))
        rgb_result = (self.rgb_detector.detect(
            synced.rgb_image, self.env_quality, self.mode)
            if rgb_enabled and any_rgb_task else RgbDetectResult())
        thermal_result = (self.thermal_detector.detect(
            synced.temperature_matrix, synced.thermal_image, now=stamp)
            if self.config.sensor_enabled("thermal") else ThermalDetectResult())
        self.thermal_result = thermal_result
        self.thermal_risk_level = thermal_result.risk_level
        lidar_result = (self.lidar_cluster.cluster(synced.lidar_points, self.mode)
                        if self.config.sensor_enabled("lidar_3d")
                        else LidarClusterResult())
        self._last_lidar_quality_cache = lidar_result.quality
        radar_tracks = (self.radar_tracker.track(
            synced.radar_targets, self.mode, now=stamp)
            if self.config.sensor_enabled("radar_4d") else [])

        # 独立 PM 传感器优先；缺失时兼容复合气体设备中的 PM 字段。
        pm_now = 0.0
        co_now = 0.0
        if self.config.sensor_enabled("pm_sensors") and synced.pm is not None:
            pm_now = (float(synced.pm.get("pm25", 0.0))
                      + float(synced.pm.get("pm10", 0.0))) / 2.0
        elif synced.gas is not None:
            pm_now = (synced.gas.pm25 + synced.gas.pm10) / 2.0
        if synced.gas is not None:
            co_now = synced.gas.co_ppm
        lidar_q = lidar_result.quality or {}
        self.smoke_estimate = self.smoke_estimator.estimate(
            image_smoke_prob=rgb_result.smoke_prob,
            coverage_ratio=rgb_result.smoke_coverage,
            pm_now=pm_now,
            co_now=co_now,
            lidar_effective_ratio=float(lidar_q.get("effective_ratio", 1.0)),
            lidar_max_range=float(lidar_q.get("max_range", 80.0)),
            near_field_scatter_ratio=float(lidar_q.get("near_field_scatter_ratio", 0.0)),
            sensor_health=health,
            mode=self.mode,
            smoke_centroid=rgb_result.smoke_centroid,
            stamp=stamp,
        ) if self.config.detector_enabled("smoke_detection") else SmokeEstimateResult(
            stamp=stamp)
        self.gas_risk_result = self.gas_risk.assess(
            synced.gas, self.env_quality.ambient_temperature,
            self.env_quality.ambient_humidity, now=stamp
        ) if (self.config.sensor_enabled("gas_sensors") and synced.gas) else GasRiskResult()

        # 阶段2：人员、火源、车辆分别执行目标级后融合。
        person_fused = self.person_fusion.fuse(
            rgb_result.detections, thermal_result.detections,
            lidar_result.detections, radar_tracks, weights, self.mode,
            self.env_quality, now=stamp
        ) if self.config.detector_enabled("person_detection") else []
        fire_fused = self.fire_fusion.fuse(
            rgb_result.detections, thermal_result.detections,
            self.gas_risk_result, weights, thermal_result.hotspots
        ) if self.config.detector_enabled("flame_detection") else []
        vehicle_fused = self.vehicle_fusion.fuse(
            rgb_result.detections, lidar_result.detections, radar_tracks,
            weights, self.mode
        ) if self.config.detector_enabled("vehicle_detection") else []

        # 巡检输出与救援目标分开保存，不进入移动目标跟踪池。
        cracks = self.crack_segmenter.detect(
            synced.rgb_image, lidar_result.ground_points, synced.odometry
        ) if (rgb_enabled and self.config.detector_enabled("crack_detection")) else []
        water_regions = self.water_detector.detect(
            synced.rgb_image, lidar_result.ground_points,
            synced.thermal_image, synced.scans
        ) if (rgb_enabled and self.config.detector_enabled("water_detection")) else []
        facility_anomalies = self.facility_inspector.detect(
            synced.rgb_image, synced.thermal_image, synced.odometry
        ) if (rgb_enabled
              and self.config.detector_enabled("facility_inspection")) else []

        # 阶段3+4：先把瞬时融合测量转换到 map 坐标，再进入持久跟踪器。
        # 顺序不能颠倒，否则非单位外参可能被重复作用到 coasted 轨迹。
        all_fused = person_fused + fire_fused + vehicle_fused
        all_fused = self.semantic_localizer.transform(all_fused)
        all_tracks = self.semantic_tracker.update(all_fused, now=stamp)

        # 阶段5+6：融合场景风险，并将风险映射为停车/限速/人工确认策略。
        radar_nearest = (self.radar_tracker.nearest_range()
                         if self.config.sensor_enabled("radar_4d")
                         else float("inf"))
        risk = self.risk_assessor.assess(
            all_tracks, self.env_quality, self.gas_risk_result,
            self.smoke_estimate, self.thermal_risk_level, self.mode,
            radar_nearest)
        safety = self.risk_assessor.map_to_safety(
            risk.risk_level, self.thermal_risk_level, self.mode,
            radar_nearest)

        # 对外状态是退化模式的简化表达，便于 ROS/Web 快速判断系统可用性。
        if self.mode == DegradationMode.CLEAR:
            status = PerceptionStatus.OK
        elif self.mode == DegradationMode.PERCEPTION_DEGRADED:
            status = PerceptionStatus.LOST
        else:
            status = PerceptionStatus.DEGRADED

        output = PipelineOutput(
            stamp=stamp,
            targets=all_tracks,
            status=status,
            mode=self.mode,
            weights=weights,
            env_quality=self.env_quality,
            smoke=self.smoke_estimate,
            gas=self.gas_risk_result,
            risk=risk,
            safety=safety,
            thermal_risk_level=self.thermal_risk_level,
            cracks=cracks,
            water_regions=water_regions,
            facility_anomalies=facility_anomalies,
        )
        return output
