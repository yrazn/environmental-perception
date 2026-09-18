"""系统配置对象及 YAML 加载入口。

Python 默认值保证系统在没有配置文件时仍可运行；YAML 只覆盖显式提供
的字段，未配置的嵌套参数继续沿用安全默认值。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """递归合并配置，避免局部 YAML 覆盖整个默认参数块。"""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


@dataclass
class DegradationConfig:
    """退化状态对应的权重、行为建议和深度源优先级。"""
    hysteresis_factor: float = 0.2
    base_weights: Dict[str, List[float]] = field(default_factory=lambda: {
        # rgb, thermal, lidar, radar, lidar2d, gas
        "clear":        [0.90, 0.40, 0.85, 0.30, 0.20, 0.20],
        "low_vis":      [0.50, 0.80, 0.60, 0.50, 0.30, 0.40],
        "heavy_smoke":  [0.10, 0.85, 0.25, 0.85, 0.50, 0.60],
        "degraded":     [0.00, 0.30, 0.00, 0.80, 0.60, 0.60],
    })
    behaviors: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "clear":        {"max_speed_pct": 100, "require_confirmation": False,
                         "allow_autonomous": True, "suggest_teleop": False},
        "low_vis":      {"max_speed_pct": 80, "require_confirmation": False,
                         "allow_autonomous": True, "suggest_teleop": False},
        "heavy_smoke":  {"max_speed_pct": 30, "require_confirmation": True,
                         "allow_autonomous": True, "suggest_teleop": False},
        "degraded":     {"max_speed_pct": 10, "require_confirmation": True,
                         "allow_autonomous": False, "suggest_teleop": True},
    })
    depth_priority: List[List[str]] = field(default_factory=lambda: [
        ["lidar", "stereo", "laser_range", "mmwave", "ground_project"],
        ["lidar", "mmwave", "laser_range", "stereo", "ground_project"],
        ["mmwave", "thermal_lidar_proj", "laser_range", "ground_project"],
        ["mmwave", "tof", "uwb_position", "ground_project"],
    ])


@dataclass
class PerceptionConfig:
    """主管线配置，包含传感器、检测器、融合、风险和运行模式。"""
    system: Dict[str, Any] = field(default_factory=lambda: {
        "pipeline_rate": 100.0,
        "frame_id_map": "map",
        "frame_id_odom": "odom",
        "frame_id_base": "base_link",
        "target_hardware": "jetson_agx_orin",
    })
    sensors: Dict[str, bool] = field(default_factory=lambda: {
        "rgb_front": True,
        "rgb_ptz": True,
        "thermal": True,
        "lidar_3d": True,
        "radar_4d": True,
        "lidar_2d_front": True,
        "lidar_2d_rear": True,
        "gas_sensors": True,
        "pm_sensors": True,
        "temperature_humidity": True,
    })
    enabled_detectors: Dict[str, bool] = field(default_factory=lambda: {
        "flame_detection": True,
        "smoke_detection": True,
        "person_detection": True,
        "vehicle_detection": True,
        "crack_detection": True,
        "water_detection": True,
        "facility_inspection": True,
    })
    modes: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        "inspection": {
            "crack_detection": True,
            "water_detection": True,
            "facility_inspection": True,
            "person_detection": False,
        },
        "rescue": {
            "crack_detection": False,
            "water_detection": False,
            "facility_inspection": False,
            "person_detection": True,
        },
        "minimal": {
            "crack_detection": False,
            "water_detection": False,
            "facility_inspection": False,
            "person_detection": True,
        },
    })
    detectors: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "rgb_detector": {
            "backend": "synthetic",
            "model_path": "yolo11n.pt",
            "device": "cpu",
            "image_size": 640,
            "inference_confidence": 0.25,
            "nms_iou_threshold": 0.50,
            "focal_length_px": 640.0,
            "camera_height_m": 1.0,
            "flame_conf_threshold": 0.35,
            "vehicle_conf_threshold": 0.45,
            "person_conf_threshold": 0.40,
            "person_min_confidence_output": 0.50,
            "smoke_coverage_update_rate": 5.0,
            "temporal_window": 5,
            "flicker_threshold": 0.15,
        },
        "thermal_detector": {
            "person_min_temperature": 24.0,
            "person_max_temperature": 44.0,
            "person_background_delta": 3.0,
            "person_min_area_pixels": 12,
            "person_aspect_ratio_min": 0.4,
            "person_aspect_ratio_max": 4.0,
            "hotspot_background_delta": 30.0,
            "hotspot_sigma_multiplier": 3.0,
            "hot_exclusion_radius_pixels": 2,
            "min_height_m": 0.3,
            "max_aspect_ratio_lying": 1.5,
            "temporal_consistency": 3,
            "hotspot_min_area_pixels": 9,
            "bg_window_size": 30,
            "temp_rise_threshold": 5.0,
            "area_growth_threshold": 0.10,
            "fire_temp_min": 200.0,
            "suspected_temp_min": 150.0,
            "circularity_fire_max": 0.70,
            "circularity_equipment_min": 0.80,
        },
        "lidar_cluster": {
            "passthrough_min": 0.5,
            "passthrough_max": 80.0,
            "voxel_leaf_normal": 0.1,
            "voxel_leaf_degraded": 0.3,
            "ransac_dist_thresh": 0.05,
            "ransac_max_iter": 100,
            "cluster_tolerance_normal": 0.3,
            "cluster_tolerance_degraded": 0.5,
            "min_cluster_size": 20,
            "max_cluster_size": 50000,
            "nominal_point_count": 30000,
        },
        "radar_tracker": {
            "static_clutter_doppler_thresh": 0.1,
            "static_object_min_frames": 5,
            "rcs_min": -10.0,
            "max_coast_frames": 10,
            "vehicle_rcs_min": 10.0,
            "person_rcs_max": 10.0,
            "person_rcs_min": -5.0,
            "micro_doppler_amplitude": 0.5,
            "micro_doppler_period_min": 0.3,
            "micro_doppler_period_max": 2.0,
            "process_noise": [0.1, 0.1, 0.5, 0.5, 1.0, 1.0],
            "observation_noise": [0.15, 0.02, 0.1],
        },
        "smoke_estimator": {
            "weights": {"image": 0.30, "pm": 0.25, "lidar": 0.30, "co": 0.15},
            "dynamic_degradation": {
                "image_degraded_factor": 0.5,
                "pm_degraded_factor": 0.3,
                "lidar_degraded_factor": 0.3,
                "gas_degraded_factor": 0.2,
            },
            "pm_baseline_window_s": 30.0,
            "co_baseline_window_s": 60.0,
            "thresholds": {"clear": 0.15, "light_smoke": 0.35, "heavy_smoke": 0.65},
            "smoothing_alpha": 0.3,
        },
        "gas_risk": {
            "thresholds": {
                "co": [50, 200, 1200],
                "co2": [1000, 5000, 10000],
                "o2": [19.5, 16.0, 10.0],
                "ch4_lel": [1.0, 10.0, 25.0],
                "h2s": [10, 50, 100],
            },
            "fire_combo": {
                "co_rise_rate_threshold": 5.0,
                "co2_rise_rate_threshold": 50.0,
                "o2_drop_rate_threshold": 0.1,
                "temp_rise_threshold": 2.0,
            },
            "history_size": 60,
        },
    })
    fusion: Dict[str, Any] = field(default_factory=lambda: {
        "association": {
            "max_association_dist": 3.0,
            "max_association_cost": 2.0,
            "w_pos": 0.4,
            "w_class": 0.3,
            "w_temp": 0.2,
            "w_time": 0.1,
        },
        "person_fusion": {"max_coast_frames": 15, "min_confirm_frames": 3},
        "semantic_tracker": {
            "max_tracks": 256,
            "max_coast_frames": 15,
            "min_confirm_frames": 3,
            "merge_distance": 0.5,
        },
        "vehicle_fusion": {
            "lane_width_m": 3.5,
            "blocking_ratio_full": 0.7,
            "blocking_ratio_partial": 0.3,
        },
    })
    risk: Dict[str, Any] = field(default_factory=lambda: {
        "weights": {
            "clear": {"thermal": 0.20, "gas": 0.15, "obstacle": 0.10,
                      "blockage": 0.10, "depth": 0.15, "rgb": 0.20,
                      "motion": 0.10},
            "low_vis": {"thermal": 0.25, "gas": 0.15, "obstacle": 0.15,
                        "blockage": 0.10, "depth": 0.15, "rgb": 0.15,
                        "motion": 0.05},
            "heavy": {"thermal": 0.35, "gas": 0.20, "obstacle": 0.20,
                      "blockage": 0.10, "depth": 0.05, "rgb": 0.05,
                      "motion": 0.05},
        },
        "escalation": {"critical": 0.85, "high": 0.60, "medium": 0.30},
    })
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    @property
    def run_mode(self) -> str:
        """当前任务模式：rescue、inspection 或 minimal。"""
        return str(self.system.get("run_mode", "rescue"))

    def detector_enabled(self, name: str) -> bool:
        """返回检测任务开关；任务模式覆盖全局开关，但不能强制开启全局禁用项。"""
        enabled = bool(self.enabled_detectors.get(name, True))
        mode_overrides = self.modes.get(self.run_mode, {})
        return enabled and bool(mode_overrides.get(name, True))

    def sensor_enabled(self, name: str) -> bool:
        """查询物理传感器是否在配置中启用。"""
        return bool(self.sensors.get(name, False))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionConfig":
        """从字典构建配置，兼容顶层 ``perception`` 包装。"""
        if "perception" in data:
            data = data["perception"]
        cfg = cls()
        for key in ("system", "sensors", "fusion", "risk", "modes"):
            if key in data:
                _deep_merge(getattr(cfg, key), data[key])
        if "detectors" in data:
            parameter_blocks = {}
            # YAML 的 detectors 同时允许布尔任务开关和算法参数块。
            for key, value in data["detectors"].items():
                if isinstance(value, bool):
                    cfg.enabled_detectors[key] = value
                else:
                    parameter_blocks[key] = value
            _deep_merge(cfg.detectors, parameter_blocks)
        if "enabled_detectors" in data:
            _deep_merge(cfg.enabled_detectors, data["enabled_detectors"])
        if "degradation" in data:
            deg = data["degradation"]
            aliases = {
                "weights": "base_weights",
                "behavior": "behaviors",
            }
            deg = {aliases.get(key, key): value for key, value in deg.items()}
            for key in ("hysteresis_factor", "base_weights", "behaviors",
                        "depth_priority"):
                if key in deg:
                    current = getattr(cfg.degradation, key)
                    if isinstance(current, dict) and isinstance(deg[key], dict):
                        _deep_merge(current, deg[key])
                    else:
                        setattr(cfg.degradation, key, deepcopy(deg[key]))
            transitions = deg.get("transitions", {})
            if "hysteresis_factor" in transitions:
                cfg.degradation.hysteresis_factor = float(
                    transitions["hysteresis_factor"])
        return cfg

    @classmethod
    def from_yaml(cls,
                  path: Union[str, Path],
                  degradation_path: Optional[Union[str, Path]] = None,
                  run_mode: Optional[str] = None) -> "PerceptionConfig":
        """加载主配置和可选退化配置。

        同时兼容仓库使用的顶层 ``perception``/``degradation`` 结构和直接
        参数映射。``run_mode`` 参数优先于 YAML 中的任务模式。
        """
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - installation guard
            raise RuntimeError(
                "PyYAML is required for PerceptionConfig.from_yaml()") from exc

        def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
            with Path(file_path).expanduser().open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"configuration root must be a mapping: {file_path}")
            return loaded

        cfg = cls.from_dict(load_yaml(path))
        if degradation_path is not None:
            degradation_data = load_yaml(degradation_path)
            normalized = degradation_data.get("degradation", degradation_data)
            overlay = cls.from_dict({"degradation": normalized})
            cfg.degradation = overlay.degradation
        if run_mode is not None:
            if run_mode not in cfg.modes:
                raise ValueError(f"unknown run mode: {run_mode}")
            cfg.system["run_mode"] = run_mode
        elif "run_mode" not in cfg.system:
            cfg.system["run_mode"] = "rescue"
        return cfg
