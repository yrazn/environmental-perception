import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescue_perception.config import PerceptionConfig
from rescue_perception.detectors.yolo_backend import YoloBackend
from rescue_perception.types import ObjectClass, SensorType


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


if __name__ == "__main__":
    unittest.main()
