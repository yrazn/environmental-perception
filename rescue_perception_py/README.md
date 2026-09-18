# rescue_perception_py

Pure Python implementation of the environment perception framework in
`floofy-dazzling-cosmos.md`. It follows the same architecture:

Detailed code documentation and the future-work roadmap are in
[CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) and
[PROJECT_ANALYSIS_AND_ROADMAP.md](PROJECT_ANALYSIS_AND_ROADMAP.md).
The consolidated subsystem implementation report is in
[ENVIRONMENT_PERCEPTION_IMPLEMENTATION_REPORT.md](ENVIRONMENT_PERCEPTION_IMPLEMENTATION_REPORT.md).

- sensor-level detectors: RGB, thermal, LiDAR, 4D mmWave, smoke, gas
- target-level fusion: person / fire / vehicle
- unified tracking and map localization
- scene-level management: quality monitor, degradation state machine, risk assessor

The package has no ROS dependency. ONNX/TensorRT model inference is behind
pluggable backends; `sim.backends.SyntheticRgbBackend` provides a synthetic
backend for demos and tests. An optional `io.ros2_bridge` wrapper imports
`rclpy` lazily.

## Use the official YOLO baseline

Install the optional inference dependency:

```bash
python3 -m pip install -e '.[yolo]'
```

Run either entry point with the official lightweight COCO checkpoint:

```bash
python3 scripts/run_demo.py --rgb-backend yolo --yolo-model yolo11n.pt
python3 scripts/run_visualizer.py --rgb-backend yolo --yolo-model yolo11n.pt --port 9100
```

The first run downloads the checkpoint through Ultralytics. The official COCO
model is mapped to this project's person, car, truck, bus and motorcycle
classes. It does not detect flame or smoke; those classes require a custom
tunnel dataset. Until calibrated RGB-D or LiDAR projection is connected,
`YoloBackend` estimates range from bounding-box height and marks every result
with `depth_valid=False`.

## Run the demo

```bash
python3 scripts/run_demo.py
```

Run the inspection pipeline, including crack, water and facility outputs:

```bash
python3 scripts/run_demo.py --mode inspection
```

Both demo entry points load `config/perception_master.yaml` and
`config/degradation.yaml` by default. Use `--config`, `--degradation` and
`--mode` to override them.

## Run the visualization dashboard

```bash
python3 scripts/run_visualizer.py --port 9100
```

The dashboard also accepts `--mode inspection` and displays detailed crack,
water and facility anomaly cards. It includes risk-colored status chips,
sensor weight/credibility bars, a tunnel situation map, deduplicated events,
and smoke/fusion-weight trends. Pause, single-step and reset operate on the
stateful perception pipeline. Its state API includes `risk_score`, `cracks`,
`water_regions` and `facility_anomalies`.

Then open `http://127.0.0.1:9100`. The dashboard uses only the Python
standard library and browser-native Canvas, with SSE for real-time updates.

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## Directory map

| Directory | Role |
| --- | --- |
| `rescue_perception/detectors` | sensor detection and estimation |
| `rescue_perception/fusion` | association, fusion, tracking, localization |
| `rescue_perception/management` | quality, degradation, risk |
| `rescue_perception/io` | time sync and optional ROS 2 bridge |
| `rescue_perception/sim` | synthetic scenario and inference backends |

## Configuration and synchronized input

```python
from rescue_perception.config import PerceptionConfig
from rescue_perception.perception_pipeline import PerceptionPipeline

config = PerceptionConfig.from_yaml(
    "config/perception_master.yaml",
    "config/degradation.yaml",
    run_mode="rescue",
)
pipeline = PerceptionPipeline(config)
```

For live adapters, push timestamped messages into `io.SensorSync` and call
`pipeline.spin_from_sync(sync)`. Missing or out-of-tolerance messages are
reported through `SyncedSensorData.sync_status` and sensor health.
