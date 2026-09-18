#!/usr/bin/env python3
"""在合成隧道火灾场景上运行感知主管线并输出终端摘要。"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescue_perception.perception_pipeline import PerceptionPipeline  # noqa: E402
from rescue_perception.config import PerceptionConfig  # noqa: E402
from rescue_perception.detectors.yolo_backend import YoloBackend  # noqa: E402
from rescue_perception.sim.backends import (  # noqa: E402
    SyntheticCrackBackend,
    SyntheticFacilityBackend,
    SyntheticRgbBackend,
    SyntheticWaterBackend,
)
from rescue_perception.sim.scenario import TunnelScenario  # noqa: E402


def main() -> int:
    """加载配置、注入合成模型后端，并以 10 Hz 仿真 12 秒。"""
    parser = argparse.ArgumentParser(description="Run the synthetic perception demo")
    project_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", default=str(project_dir / "config/perception_master.yaml"))
    parser.add_argument("--degradation", default=str(project_dir / "config/degradation.yaml"))
    parser.add_argument("--mode", choices=("rescue", "inspection", "minimal"),
                        default=None)
    parser.add_argument("--rgb-backend", choices=("synthetic", "yolo"),
                        default=None,
                        help="override the configured RGB inference backend")
    parser.add_argument("--yolo-model", default=None,
                        help="Ultralytics model path/name, e.g. yolo11n.pt")
    parser.add_argument("--yolo-device", default=None,
                        help="Ultralytics device, e.g. cpu, 0 or 0,1")
    args = parser.parse_args()

    # 分段烟雾曲线用于观察 CLEAR/LOW_VISIBILITY/HEAVY_SMOKE 状态切换。
    def smoke_curve(t: float) -> float:
        if t < 2.0:
            return 0.0
        if t < 5.0:
            return 0.25
        if t < 9.0:
            return 0.55
        return 0.85

    scenario = TunnelScenario(smoke_curve=smoke_curve)
    config = PerceptionConfig.from_yaml(
        args.config, degradation_path=args.degradation, run_mode=args.mode)
    pipeline = PerceptionPipeline(config)
    pipeline.initialize()
    rgb_params = config.detectors["rgb_detector"]
    if args.yolo_model:
        rgb_params["model_path"] = args.yolo_model
    if args.yolo_device:
        rgb_params["device"] = args.yolo_device
    rgb_backend = args.rgb_backend or str(rgb_params.get("backend", "synthetic"))
    if rgb_backend == "yolo":
        pipeline.rgb_detector.set_backend(YoloBackend(config))
    else:
        pipeline.rgb_detector.set_backend(SyntheticRgbBackend(scenario))
    pipeline.crack_segmenter.set_backend(SyntheticCrackBackend())
    pipeline.water_detector.set_backend(SyntheticWaterBackend())
    pipeline.facility_inspector.set_backend(SyntheticFacilityBackend())

    print("=" * 72)
    print("Rescue perception pipeline demo (synthetic tunnel fire scenario)")
    print(f"run mode: {config.run_mode}")
    print(f"RGB backend: {rgb_backend}")
    print("=" * 72)
    last_summary = None
    for i in range(121):
        t = i * 0.1
        synced = scenario.frame(t)
        output = pipeline.spin_once(synced, now=t)
        last_summary = output.summary()
        if i % 10 == 0 or i == 120:
            print(f"\n[t={t:5.1f}s] mode={output.mode.name:>16s} "
                  f"status={output.status.name:>8s} "
                  f"smoke={output.smoke.smoke_score:.2f} "
                  f"thermal_risk={output.thermal_risk_level} "
                  f"env_risk={output.risk.risk_level.name}")
            print(f"  weights RGB/Therm/Lidar/Radar = "
                  f"{output.weights.rgb:.2f}/{output.weights.thermal:.2f}/"
                  f"{output.weights.lidar:.2f}/{output.weights.radar:.2f}")
            print(f"  safety={output.safety.required_action} "
                  f"speed={output.safety.speed_limit:.0%} "
                  f"events={output.risk.events}")
            for tr in output.targets:
                print(f"    track {tr.track_id}: {tr.class_name:>18s} "
                      f"conf={tr.confidence:.2f} "
                      f"pos=({tr.position[0]:.1f},{tr.position[1]:.1f}) "
                      f"mask={tr.source_mask:02x} confirmed={int(tr.confirmed)}")
            if output.cracks or output.water_regions or output.facility_anomalies:
                print(f"  inspection cracks/water/facilities = "
                      f"{len(output.cracks)}/{len(output.water_regions)}/"
                      f"{len(output.facility_anomalies)}")

    print("\nFinal summary:")
    print(f"mode={last_summary['mode']} status={last_summary['status']}")
    print(f"smoke={last_summary['smoke']}")
    print(f"gas={last_summary['gas']}")
    print(f"risk={last_summary['env_risk']} safety={last_summary['safety']}")
    print(f"hot_zones={last_summary['hot_zones']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
