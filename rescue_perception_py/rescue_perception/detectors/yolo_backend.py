"""Ultralytics YOLO 可见光目标检测后端。

后端允许同帧运行两个职责不同的模型：官方 COCO 模型检测人员/车辆，
项目联合训练模型检测火焰/烟雾。两个模型的框最终都转换成统一
``Detection3D``，避免用火情专用权重替换 COCO 权重后丢失人员和车辆能力。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .rgb_detector import DetectionBackend
from ..config import PerceptionConfig
from ..types import Detection3D, ObjectClass, SensorType


class YoloBackend(DetectionBackend):
    """将 Ultralytics ``Results`` 转换成系统统一的 ``Detection3D``。

    官方 COCO 模型只输出二维框。若未接入真实深度，本类使用目标框高度和
    类别先验估算距离，使人员/车辆结果可以先进入现有三维融合链路；此时
    ``depth_valid`` 为 ``False``，提醒下游该位置不是深度传感器实测值。
    """

    DEFAULT_CLASS_MAP: Dict[str, ObjectClass] = {
        "fire": ObjectClass.FLAME,
        "flame": ObjectClass.FLAME,
        "smoke": ObjectClass.SMOKE,
        "person": ObjectClass.PERSON_STANDING,
        "car": ObjectClass.VEHICLE_CAR,
        "truck": ObjectClass.VEHICLE_TRUCK,
        "bus": ObjectClass.VEHICLE_BUS,
        "motorcycle": ObjectClass.VEHICLE_MOTORCYCLE,
    }

    # 用于单目框高测距的粗略实际高度（米），仅供基准模型联调。
    DEFAULT_HEIGHTS: Dict[ObjectClass, float] = {
        # 火焰和烟雾没有固定物理高度，下列值只用于未接入真实深度时
        # 生成可进入融合门控的粗略距离，结果仍标记 depth_valid=False。
        ObjectClass.FLAME: 1.00,
        ObjectClass.SMOKE: 2.00,
        ObjectClass.PERSON_STANDING: 1.70,
        ObjectClass.VEHICLE_CAR: 1.50,
        ObjectClass.VEHICLE_TRUCK: 3.20,
        ObjectClass.VEHICLE_BUS: 3.20,
        ObjectClass.VEHICLE_MOTORCYCLE: 1.30,
    }

    # 无 LiDAR/RGB-D 尺寸时提供的目标长、宽、高先验（米）。
    DEFAULT_DIMENSIONS: Dict[ObjectClass, Tuple[float, float, float]] = {
        ObjectClass.FLAME: (0.8, 0.8, 1.0),
        ObjectClass.SMOKE: (1.5, 1.5, 2.0),
        ObjectClass.PERSON_STANDING: (0.5, 0.5, 1.70),
        ObjectClass.VEHICLE_CAR: (4.5, 1.8, 1.50),
        ObjectClass.VEHICLE_TRUCK: (8.0, 2.5, 3.20),
        ObjectClass.VEHICLE_BUS: (10.0, 2.5, 3.20),
        ObjectClass.VEHICLE_MOTORCYCLE: (2.0, 0.8, 1.30),
    }

    def __init__(self, config: Optional[PerceptionConfig] = None,
                 model: Optional[Any] = None,
                 hazard_model: Optional[Any] = None):
        self.config = config or PerceptionConfig()
        params = self.config.detectors["rgb_detector"]
        self.model_path = str(params.get("model_path", "yolo11n.pt"))
        self.fire_smoke_model_path = str(
            params.get("fire_smoke_model_path", "") or "")
        self.device = str(params.get("device", "cpu"))
        self.imgsz = int(params.get("image_size", 640))
        self.inference_conf = float(params.get("inference_confidence", 0.25))
        self.hazard_inference_conf = float(
            params.get("fire_smoke_inference_confidence", self.inference_conf))
        self.nms_iou = float(params.get("nms_iou_threshold", 0.50))
        self.focal_length_px = float(params.get("focal_length_px", 640.0))
        self.camera_height_m = float(params.get("camera_height_m", 1.0))
        self.class_map = dict(self.DEFAULT_CLASS_MAP)

        # 允许测试注入轻量假模型；生产运行时才延迟导入重型依赖。
        yolo_class = None
        if model is None or (hazard_model is None and self.fire_smoke_model_path):
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO backend requires ultralytics; install the 'yolo' extra"
                ) from exc
            yolo_class = YOLO
        if model is None:
            model = yolo_class(self.model_path)
        self.model = model

        # 路径为空时保持单模型兼容；相同路径不重复加载或推理。
        if hazard_model is None and self.fire_smoke_model_path:
            if self.fire_smoke_model_path == self.model_path:
                hazard_model = self.model
            else:
                hazard_model = yolo_class(self.fire_smoke_model_path)
        self.hazard_model = hazard_model

        # 检测框会在每帧更新成烟雾区域摘要，供 SmokeEstimator 使用。
        self.smoke_coverage = 0.0
        self.smoke_prob = 0.0
        self.smoke_centroid: Optional[np.ndarray] = None

    def detect(self, image: Optional[np.ndarray]) -> List[Detection3D]:
        """运行可见光模型并输出项目已定义的目标类别与烟雾摘要。"""
        # 每次调用先清空上一帧摘要，避免空帧沿用旧的烟雾状态。
        self.smoke_coverage = 0.0
        self.smoke_prob = 0.0
        self.smoke_centroid = None
        if image is None or image.size == 0:
            return []

        detections: List[Detection3D] = []
        smoke_boxes: List[np.ndarray] = []
        smoke_confidences: List[float] = []
        model_specs = [(self.model, self.inference_conf, "general", None)]
        if self.hazard_model is not None and self.hazard_model is not self.model:
            model_specs.append((
                self.hazard_model,
                self.hazard_inference_conf,
                "fire_smoke",
                {ObjectClass.FLAME, ObjectClass.SMOKE},
            ))

        for inference_model, confidence_threshold, role, allowed in model_specs:
            results = inference_model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=confidence_threshold,
                iou=self.nms_iou,
                device=self.device,
                verbose=False,
            )
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                xyxy = self._to_numpy(boxes.xyxy)
                confidences = self._to_numpy(boxes.conf)
                classes = self._to_numpy(boxes.cls).astype(int)
                names = result.names

                for box, confidence, model_class_id in zip(
                        xyxy, confidences, classes):
                    model_class_name = self._class_name(
                        names, int(model_class_id)).strip().lower()
                    class_id = self.class_map.get(model_class_name)
                    if class_id is None or (allowed is not None and class_id not in allowed):
                        continue
                    detections.append(self._make_detection(
                        image, box, float(confidence), int(model_class_id),
                        model_class_name, class_id, role))
                    if class_id == ObjectClass.SMOKE:
                        smoke_boxes.append(np.asarray(box, dtype=float))
                        smoke_confidences.append(float(confidence))

        self._update_smoke_summary(image.shape, smoke_boxes, smoke_confidences)
        return detections

    def _update_smoke_summary(self, image_shape: Tuple[int, ...],
                              boxes: List[np.ndarray],
                              confidences: List[float]) -> None:
        """由烟雾检测框计算去重覆盖率、最高概率和像素质心。"""
        if not boxes:
            return
        image_h, image_w = image_shape[:2]
        mask = np.zeros((image_h, image_w), dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            left = max(0, min(image_w, int(np.floor(x1))))
            top = max(0, min(image_h, int(np.floor(y1))))
            right = max(0, min(image_w, int(np.ceil(x2))))
            bottom = max(0, min(image_h, int(np.ceil(y2))))
            if right > left and bottom > top:
                mask[top:bottom, left:right] = True
        self.smoke_coverage = float(np.mean(mask))
        self.smoke_prob = float(max(confidences, default=0.0))
        if np.any(mask):
            ys, xs = np.nonzero(mask)
            self.smoke_centroid = np.array([xs.mean(), ys.mean()], dtype=float)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """兼容 PyTorch Tensor、Ultralytics wrapper 和测试 NumPy 数组。"""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        """兼容 Ultralytics 使用的字典或列表类别表。"""
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _make_detection(self, image: np.ndarray, box: np.ndarray,
                        confidence: float, model_class_id: int,
                        model_class_name: str,
                        class_id: ObjectClass,
                        model_role: str = "general") -> Detection3D:
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        position, bearing = self._estimate_position(
            image.shape, (x1, y1, x2, y2), class_id)
        dimensions = np.asarray(
            self.DEFAULT_DIMENSIONS.get(class_id, (0.0, 0.0, 0.0)),
            dtype=float,
        )
        return Detection3D(
            class_id=class_id,
            confidence=confidence,
            source=SensorType.RGB,
            position=position,
            dimensions=dimensions,
            depth_valid=False,
            bearing=bearing,
            extra={
                "bbox_xyxy": [x1, y1, x2, y2],
                "model_class_id": model_class_id,
                "model_class_name": model_class_name,
                "model_role": model_role,
                "depth_method": "monocular_bbox_height",
            },
        )

    def _estimate_position(self, image_shape: Tuple[int, ...],
                           box: Tuple[float, float, float, float],
                           class_id: ObjectClass) -> Tuple[np.ndarray, float]:
        """由框高估计 ``base_link`` 风格的前向、左向和高度坐标。

        该方法只用于基准模型联调。正式部署应由 RGB-D 深度或 LiDAR 投影
        覆盖，并把 ``depth_valid`` 设为 ``True``。
        """
        image_h, image_w = image_shape[:2]
        x1, y1, x2, y2 = box
        pixel_height = max(y2 - y1, 1.0)
        real_height = self.DEFAULT_HEIGHTS.get(class_id, 1.7)
        forward = self.focal_length_px * real_height / pixel_height
        center_u = 0.5 * (x1 + x2)
        center_v = 0.5 * (y1 + y2)
        cx = 0.5 * image_w
        cy = 0.5 * image_h
        bearing = float(np.arctan2(center_u - cx, self.focal_length_px))
        # 相机光学坐标右为正；机器人坐标通常左为正，因此横向取反。
        left = -(center_u - cx) * forward / self.focal_length_px
        vertical = self.camera_height_m - (
            (center_v - cy) * forward / self.focal_length_px)
        return np.array([forward, left, vertical], dtype=float), bearing
