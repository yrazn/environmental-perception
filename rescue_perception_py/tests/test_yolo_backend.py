import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescue_perception.config import PerceptionConfig
from rescue_perception.detectors.rgb_detector import RgbDetector
from rescue_perception.detectors.yolo_backend import YoloBackend
from rescue_perception.types import (
    DegradationMode,
    EnvironmentQuality,
    ObjectClass,
    SensorType,
)


class FakeYoloModel:
    """不依赖 torch/ultralytics 的后端适配测试替身。"""

    def __init__(self):
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        boxes = SimpleNamespace(
            xyxy=np.array([
                [100.0, 80.0, 200.0, 380.0],
                [300.0, 180.0, 560.0, 380.0],
                [10.0, 10.0, 50.0, 50.0],
            ]),
            conf=np.array([0.91, 0.82, 0.99]),
            cls=np.array([0.0, 2.0, 15.0]),
        )
        return [SimpleNamespace(
            boxes=boxes,
            names={0: "person", 2: "car", 15: "cat"},
        )]


class FakeFireSmokeModel:
    """模拟联合模型的 smoke/fire 类别和二维检测框。"""

    def __init__(self):
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        boxes = SimpleNamespace(
            xyxy=np.array([
                [64.0, 48.0, 320.0, 240.0],
                [400.0, 100.0, 520.0, 360.0],
            ]),
            conf=np.array([0.80, 0.90]),
            cls=np.array([0.0, 1.0]),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "smoke", 1: "fire"})]


class YoloBackendTest(unittest.TestCase):
    def test_maps_official_coco_classes(self):
        model = FakeYoloModel()
        config = PerceptionConfig()
        backend = YoloBackend(config, model=model)

        detections = backend.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].class_id, ObjectClass.PERSON_STANDING)
        self.assertEqual(detections[1].class_id, ObjectClass.VEHICLE_CAR)
        self.assertTrue(all(d.source == SensorType.RGB for d in detections))
        self.assertTrue(all(not d.depth_valid for d in detections))
        self.assertTrue(all(d.position[0] > 0.0 for d in detections))
        self.assertEqual(detections[0].extra["model_class_name"], "person")
        self.assertEqual(model.kwargs["source"].shape, (480, 640, 3))

    def test_empty_image_does_not_run_model(self):
        model = FakeYoloModel()
        backend = YoloBackend(model=model)
        self.assertEqual(backend.detect(None), [])
        self.assertIsNone(model.kwargs)

    def test_combines_general_and_fire_smoke_models(self):
        general_model = FakeYoloModel()
        hazard_model = FakeFireSmokeModel()
        config = PerceptionConfig.from_dict({
            "detectors": {"rgb_detector": {
                "fire_smoke_inference_confidence": 0.20,
            }},
        })
        backend = YoloBackend(
            config, model=general_model, hazard_model=hazard_model)

        detections = backend.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(
            [item.class_id for item in detections],
            [ObjectClass.PERSON_STANDING, ObjectClass.VEHICLE_CAR,
             ObjectClass.SMOKE, ObjectClass.FLAME],
        )
        self.assertEqual(detections[2].extra["model_role"], "fire_smoke")
        self.assertEqual(detections[3].extra["model_class_name"], "fire")
        self.assertAlmostEqual(backend.smoke_coverage, 0.16, places=3)
        self.assertAlmostEqual(backend.smoke_prob, 0.80)
        np.testing.assert_allclose(backend.smoke_centroid, [191.5, 143.5])
        self.assertAlmostEqual(hazard_model.kwargs["conf"], 0.20)

    def test_rgb_detector_keeps_fire_and_smoke_detections(self):
        backend = YoloBackend(
            model=FakeYoloModel(), hazard_model=FakeFireSmokeModel())
        detector = RgbDetector(backend=backend)

        result = detector.detect(
            np.zeros((480, 640, 3), dtype=np.uint8),
            EnvironmentQuality(),
            DegradationMode.CLEAR,
        )

        classes = {item.class_id for item in result.detections}
        self.assertIn(ObjectClass.FLAME, classes)
        self.assertIn(ObjectClass.SMOKE, classes)
        self.assertAlmostEqual(result.smoke_coverage, 0.16, places=3)
        self.assertAlmostEqual(result.smoke_prob, 0.80)


if __name__ == "__main__":
    unittest.main()
