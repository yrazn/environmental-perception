import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescue_perception.config import PerceptionConfig
from rescue_perception.detectors.gas_risk import GasRiskAssessor
from rescue_perception.detectors.smoke_estimator import SmokeEstimator
from rescue_perception.detectors.thermal_detector import ThermalDetector
from rescue_perception.fusion.association import gated_hungarian, hungarian
from rescue_perception.management.degradation_manager import DegradationManager
from rescue_perception.types import (
    DegradationMode,
    EnvironmentQuality,
    GasReadings,
    ObjectClass,
    SmokeEstimateResult,
)


class AssociationTest(unittest.TestCase):
    def test_hungarian_min_cost(self):
        cost = np.array([[1.0, 4.0], [2.0, 1.0]])
        pairs = hungarian(cost)
        self.assertEqual(sum(cost[i, j] for i, j in pairs), 2.0)

    def test_gated_hungarian(self):
        cost = np.array([[0.2, 10.0], [10.0, 0.3]])
        pairs = gated_hungarian(cost, 1.0)
        self.assertEqual(len(pairs), 2)


class DegradationTest(unittest.TestCase):
    def test_clear_to_low_visibility(self):
        mgr = DegradationManager(PerceptionConfig())
        env = EnvironmentQuality()
        env.credibility[0].credibility = 0.4
        smoke = SmokeEstimateResult(smoke_score=0.2)
        mode = mgr.update(env, smoke, now=1.0)
        self.assertEqual(mode, DegradationMode.LOW_VISIBILITY)

    def test_heavy_smoke_reduces_rgb_weight(self):
        mgr = DegradationManager(PerceptionConfig())
        env = EnvironmentQuality()
        env.credibility[0].credibility = 0.4
        env.credibility[1].credibility = 0.9
        env.credibility[2].credibility = 0.2
        env.credibility[3].credibility = 0.9
        smoke = SmokeEstimateResult(smoke_score=0.6)
        mgr.update(env, smoke, now=1.0)
        mgr.current_mode = DegradationMode.HEAVY_SMOKE
        mode = mgr.current_mode
        weights = mgr.get_weights(mode, env)
        self.assertLess(weights.rgb, 0.2)
        self.assertGreater(weights.radar, weights.rgb)
        self.assertGreater(weights.radar, weights.lidar)


class GasRiskTest(unittest.TestCase):
    def test_high_co_is_danger(self):
        assessor = GasRiskAssessor()
        readings = GasReadings(co_ppm=400.0, o2_pct=20.9)
        result = assessor.assess(readings, now=0.0)
        self.assertGreaterEqual(result.risk_level, 2)


class SmokeEstimatorTest(unittest.TestCase):
    def test_smoke_score_rises_with_inputs(self):
        estimator = SmokeEstimator()
        r1 = estimator.estimate(image_smoke_prob=0.9, coverage_ratio=0.9,
                                pm_now=400.0, co_now=200.0,
                                lidar_effective_ratio=0.2,
                                lidar_max_range=20.0,
                                near_field_scatter_ratio=0.5,
                                stamp=0.0)
        self.assertGreater(r1.smoke_score, 0.15)


class ThermalDetectorTest(unittest.TestCase):
    def test_hotspot_risk(self):
        matrix = np.full((80, 100), 25.0)
        matrix[30:38, 40:48] = 420.0
        det = ThermalDetector()
        result = det.detect(matrix, now=0.0)
        self.assertGreaterEqual(result.risk_level, 2)
        self.assertTrue(result.hotspots)

    def test_human_channel_detects_body_temperature_below_hotspot_threshold(self):
        matrix = np.full((80, 100), 25.0)
        matrix[25:55, 45:55] = 36.0
        det = ThermalDetector()

        result = det.detect(matrix, now=0.0)

        people = [d for d in result.detections
                  if d.class_id == ObjectClass.PERSON_STANDING]
        self.assertEqual(len(people), 1)
        self.assertAlmostEqual(people[0].temperature_max, 36.0)
        self.assertFalse(people[0].depth_valid)
        self.assertAlmostEqual(people[0].extra["centroid_px"][0], 49.5)
        self.assertAlmostEqual(people[0].extra["centroid_px"][1], 39.5)

    def test_fire_core_is_not_classified_as_person(self):
        matrix = np.full((80, 100), 25.0)
        matrix[25:55, 45:55] = 36.0
        matrix[34:46, 47:53] = 420.0
        det = ThermalDetector()

        result = det.detect(matrix, now=0.0)

        people = [d for d in result.detections
                  if d.class_id == ObjectClass.PERSON_STANDING]
        self.assertFalse(people)


if __name__ == "__main__":
    unittest.main()
