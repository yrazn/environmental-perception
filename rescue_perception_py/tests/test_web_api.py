import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescue_perception.types import PipelineOutput  # noqa: E402
from web.app import VisualizerServer, build_payload  # noqa: E402


class WebApiTest(unittest.TestCase):
    def test_build_payload_has_required_keys(self):
        output = PipelineOutput()
        payload = build_payload(output, thermal_result=None, stamp=1.5)
        for key in ("mode", "mode_name", "status_name", "env_risk_name",
                    "safety_action", "speed_limit", "smoke_score",
                    "thermal_risk_level", "weights", "credibilities",
                    "targets", "hot_zones", "events", "cracks",
                    "water_regions", "facility_anomalies", "risk_score"):
            self.assertIn(key, payload)
        self.assertEqual(payload["time"], 1.5)

    def test_control_actions(self):
        server = VisualizerServer(scenario=object(), pipeline=object())
        self.assertTrue(server.handle_control({"action": "pause"})["ok"])
        self.assertTrue(server.handle_control({"action": "resume"})["ok"])
        self.assertTrue(server.handle_control({"action": "step"})["ok"])
        self.assertTrue(server.handle_control({"action": "reset"})["ok"])
        self.assertTrue(server.handle_control(
            {"action": "speed", "value": 2.0})["ok"])
        self.assertFalse(server.handle_control({"action": "unknown"})["ok"])

    def test_step_advances_and_reset_rebuilds_state(self):
        class Scenario:
            def frame(self, stamp):
                from rescue_perception.types import SyncedSensorData
                return SyncedSensorData(stamp=stamp)

        from rescue_perception.perception_pipeline import PerceptionPipeline
        server = VisualizerServer(Scenario(), PerceptionPipeline(), dt=0.1)
        server.handle_control({"action": "pause"})
        server.handle_control({"action": "step"})
        server._advance_and_compute()
        self.assertAlmostEqual(server.t, 0.1)
        old_pipeline = server.pipeline
        server.handle_control({"action": "reset"})
        server._advance_and_compute()
        self.assertEqual(server.t, 0.0)
        self.assertIsNot(server.pipeline, old_pipeline)

    def test_static_page_exists(self):
        index = Path(__file__).resolve().parents[1] / "web/static/index.html"
        self.assertTrue(index.is_file())


if __name__ == "__main__":
    unittest.main()
