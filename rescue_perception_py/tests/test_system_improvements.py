import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescue_perception.config import PerceptionConfig
from rescue_perception.fusion.semantic_localizer import SemanticLocalizer
from rescue_perception.fusion.semantic_tracker import SemanticTracker
from rescue_perception.io.sensor_sync import SensorSync
from rescue_perception.management.quality_monitor import QualityMonitor
from rescue_perception.perception_pipeline import PerceptionPipeline
from rescue_perception.sim.backends import (
    SyntheticCrackBackend,
    SyntheticFacilityBackend,
    SyntheticRgbBackend,
    SyntheticWaterBackend,
)
from rescue_perception.sim.scenario import TunnelScenario
from rescue_perception.types import (
    ObjectClass,
    SourceMask,
    TrackedObject,
    VisibilityLevel,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class ConfigurationTest(unittest.TestCase):
    def test_yaml_and_mode_overrides_are_applied(self):
        cfg = PerceptionConfig.from_yaml(
            PROJECT_DIR / "config/perception_master.yaml",
            PROJECT_DIR / "config/degradation.yaml",
            run_mode="inspection",
        )
        self.assertEqual(cfg.run_mode, "inspection")
        self.assertTrue(cfg.detector_enabled("crack_detection"))
        self.assertFalse(cfg.detector_enabled("person_detection"))
        self.assertEqual(cfg.degradation.base_weights["clear"][0], 0.90)


class QualityAndSyncTest(unittest.TestCase):
    def test_clear_visibility_uses_normalized_sharpness(self):
        rng = np.random.default_rng(42)
        image = rng.uniform(40.0, 216.0, (80, 100)).astype(np.float32)
        from rescue_perception.types import SyncedSensorData
        env = QualityMonitor().assess(SyncedSensorData(rgb_image=image), 0.0)
        self.assertEqual(env.visibility_level, VisibilityLevel.CLEAR)

    def test_sync_error_uses_selected_sample_stamp(self):
        sync = SensorSync({"rgb": 100.0, "radar": 10.0})
        sync.push("rgb", 0.70, "old")
        sync.push("rgb", 0.95, "selected")
        sync.push("radar", 1.00, [])
        frame = sync.pull(["rgb", "radar"])
        self.assertIsNotNone(frame)
        self.assertEqual(frame.rgb_image, "selected")
        self.assertAlmostEqual(frame.sync_status["rgb"], 50.0)


class LocalizationAndClassesTest(unittest.TestCase):
    def test_non_identity_transform_updates_target_pose_and_marker(self):
        localizer = SemanticLocalizer()
        transform = np.eye(4)
        transform[:3, 3] = [10.0, -2.0, 1.0]
        localizer.register_transform("rgb", transform)
        target = TrackedObject(source_mask=int(SourceMask.RGB))
        target.state[:3] = [1.0, 2.0, 3.0]
        localizer.transform([target])
        np.testing.assert_allclose(target.position, [11.0, 0.0, 4.0])
        np.testing.assert_allclose(
            localizer.markers([target])[0]["position"], [11.0, 0.0, 4.0])

    def test_vehicle_is_not_grouped_as_person(self):
        self.assertEqual(SemanticTracker._group(ObjectClass.VEHICLE_CAR), "vehicle")
        self.assertEqual(SemanticTracker._group(ObjectClass.PERSON_CANDIDATE), "person")


class PipelineIntegrationTest(unittest.TestCase):
    def test_independent_pm_input_has_priority(self):
        scenario = TunnelScenario(smoke_curve=lambda _: 0.0)
        frame = scenario.frame(0.0)
        frame.gas.pm25 = 0.0
        frame.gas.pm10 = 0.0
        frame.pm = {"pm25": 500.0, "pm10": 500.0}
        output = PerceptionPipeline().spin_once(frame, now=0.0)
        self.assertGreater(output.smoke.individual_scores["pm"], 0.5)

    def test_inspection_mode_outputs_are_in_pipeline_result(self):
        cfg = PerceptionConfig.from_dict({
            "system": {"run_mode": "inspection"},
        })
        scenario = TunnelScenario(smoke_curve=lambda _: 0.0)
        pipeline = PerceptionPipeline(cfg)
        pipeline.rgb_detector.set_backend(SyntheticRgbBackend(scenario))
        pipeline.crack_segmenter.set_backend(SyntheticCrackBackend())
        pipeline.water_detector.set_backend(SyntheticWaterBackend())
        pipeline.facility_inspector.set_backend(SyntheticFacilityBackend())
        output = pipeline.spin_once(scenario.frame(0.0), now=0.0)
        self.assertEqual(len(output.cracks), 1)
        self.assertEqual(len(output.water_regions), 1)
        self.assertEqual(len(output.facility_anomalies), 1)
        self.assertFalse(any(event.startswith("person_found")
                             for event in output.risk.events))

    def test_pipeline_can_pull_from_sensor_sync(self):
        scenario = TunnelScenario(smoke_curve=lambda _: 0.0)
        frame = scenario.frame(0.0)
        sync = SensorSync()
        sync.push("rgb", 0.0, frame.rgb_image)
        sync.push("thermal", 0.0, frame.thermal_image)
        sync.push("temperature", 0.0, frame.temperature_matrix)
        sync.push("lidar", 0.0, frame.lidar_points)
        sync.push("radar", 0.0, frame.radar_targets)
        sync.push("gas", 0.0, frame.gas)
        sync.push("pm", 0.0, frame.pm)
        sync.push("env", 0.0, frame.temperature_humidity)
        output = PerceptionPipeline().spin_from_sync(sync, now=0.0)
        self.assertIsNotNone(output)
        self.assertEqual(output.stamp, 0.0)

    def test_disabled_rgb_sensor_suppresses_inspection_backends(self):
        cfg = PerceptionConfig.from_dict({
            "system": {"run_mode": "inspection"},
            "sensors": {"rgb_front": False, "rgb_ptz": False},
        })
        scenario = TunnelScenario(smoke_curve=lambda _: 0.0)
        pipeline = PerceptionPipeline(cfg)
        pipeline.crack_segmenter.set_backend(SyntheticCrackBackend())
        pipeline.water_detector.set_backend(SyntheticWaterBackend())
        pipeline.facility_inspector.set_backend(SyntheticFacilityBackend())
        output = pipeline.spin_once(scenario.frame(0.0), now=0.0)
        self.assertFalse(output.cracks)
        self.assertFalse(output.water_regions)
        self.assertFalse(output.facility_anomalies)


if __name__ == "__main__":
    unittest.main()
