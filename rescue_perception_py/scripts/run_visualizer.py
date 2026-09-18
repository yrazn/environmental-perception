#!/usr/bin/env python3
"""启动基于 HTTP、SSE 和 Canvas 的环境感知可视化仪表盘。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web.app import create_server  # noqa: E402

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
    """加载任务模式和合成后端，自动选择可用端口并运行 Web 服务。"""
    parser = argparse.ArgumentParser(description="Perception visualization server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-time", type=float, default=120.0)
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

    httpd = None
    for port in range(args.port, args.port + 10):
        try:
            httpd = create_server(scenario, pipeline, host=args.host,
                                  port=port, max_time=args.max_time)
            break
        except OSError:
            continue
    if httpd is None:
        print(f"ERROR: no free port from {args.port} to {args.port + 9}")
        return 1

    httpd.visualizer.start()
    url = f"http://{args.host}:{httpd.server_address[1]}"
    print(f"Perception dashboard running at {url} "
          f"(mode={config.run_mode}, rgb={rgb_backend})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.visualizer.stop()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
