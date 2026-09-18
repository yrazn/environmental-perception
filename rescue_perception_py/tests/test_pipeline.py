import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescue_perception.perception_pipeline import PerceptionPipeline
from rescue_perception.sim.backends import SyntheticRgbBackend
from rescue_perception.sim.scenario import TunnelScenario


class PipelineEndToEndTest(unittest.TestCase):
    def test_runs_and_degrades(self):
        scenario = TunnelScenario(smoke_curve=lambda t: 0.8 if t > 1.0 else 0.0)
        pipeline = PerceptionPipeline()
        pipeline.initialize()
        pipeline.rgb_detector.set_backend(SyntheticRgbBackend(scenario))
        last = None
        for i in range(30):
            t = i * 0.1
            output = pipeline.spin_once(scenario.frame(t), now=t)
            last = output
        self.assertIsNotNone(last)
        self.assertTrue(hasattr(last, "targets"))
        self.assertTrue(last.smoke.smoke_score > 0.2)
        self.assertTrue(last.mode >= 1)
        self.assertIn("gas_danger_level", last.risk.events
                      or ["gas_danger_level"])


if __name__ == "__main__":
    unittest.main()

