"""基于标准库 HTTP 与 SSE 的零前端框架可视化服务。"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from rescue_perception.perception_pipeline import PerceptionPipeline
from rescue_perception.types import PipelineOutput, SensorType


STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_payload(output: PipelineOutput,
                  thermal_result: Any = None,
                  stamp: float = 0.0) -> Dict[str, Any]:
    """把包含 NumPy/枚举对象的主管线输出转换为浏览器 JSON。"""
    env = output.env_quality
    cred = {}
    for sensor_type in (SensorType.RGB, SensorType.THERMAL, SensorType.LIDAR_3D,
                        SensorType.RADAR_4D, SensorType.LIDAR_2D, SensorType.GAS):
        cred[sensor_type.name.lower()] = round(
            env.credibility[sensor_type].credibility, 3)

    hotspot_count = 0
    thermal_max = 0.0
    thermal_confidence = 0.0
    hotspots = []
    if thermal_result is not None:
        hotspot_count = len(thermal_result.hotspots)
        thermal_max = round(thermal_result.max_temperature, 1)
        thermal_confidence = round(thermal_result.confidence, 3)
        hotspots = [
            {
                "centroid": [round(float(h.centroid[0]), 1),
                             round(float(h.centroid[1]), 1)],
                "t_max": round(h.t_max, 1),
                "t_mean": round(h.t_mean, 1),
                "area_px": h.area_px,
                "htype": int(h.htype),
                "confidence": round(h.type_confidence, 3),
            }
            for h in thermal_result.hotspots
        ]

    return {
        "time": round(stamp, 3),
        "mode": int(output.mode),
        "mode_name": output.mode.name,
        "status": int(output.status),
        "status_name": output.status.name,
        "env_risk": int(output.risk.risk_level),
        "env_risk_name": output.risk.risk_level.name,
        "risk_score": round(output.risk.combined_score, 3),
        "safety_action": output.safety.required_action,
        "safety_level": int(output.safety.safety_level),
        "speed_limit": round(output.safety.speed_limit, 3),
        "smoke_score": round(output.smoke.smoke_score, 3),
        "smoke_visibility": int(output.smoke.visibility_level),
        "smoke_visibility_name": output.smoke.visibility_level.name,
        "smoke_individual": {k: round(v, 3)
                             for k, v in output.smoke.individual_scores.items()},
        "gas_risk_level": output.gas.risk_level,
        "gas_individual": dict(output.gas.individual_levels),
        "gas_alert": output.gas.alert,
        "thermal_risk_level": output.thermal_risk_level,
        "thermal_max_temperature": thermal_max,
        "thermal_hotspot_count": hotspot_count,
        "thermal_confidence": thermal_confidence,
        "thermal_hotspots": hotspots,
        "weights": output.weights.to_dict(),
        "credibilities": cred,
        "targets": [t.to_dict() for t in output.targets],
        "hot_zones": output.risk.hot_zones,
        "events": output.risk.events,
        "cracks": [item.to_dict() for item in output.cracks],
        "water_regions": [item.to_dict() for item in output.water_regions],
        "facility_anomalies": [item.to_dict()
                               for item in output.facility_anomalies],
        "risk_factors": {k: round(v, 3)
                         for k, v in output.risk.factors.items()},
        "visibility_score": round(env.visibility_score, 3),
        "ambient_temperature": round(env.ambient_temperature, 1),
        "ambient_humidity": round(env.ambient_humidity, 1),
    }


class VisualizerServer:
    """后台推进仿真和主管线，并向多个 HTTP 客户端提供最新状态。"""
    def __init__(self,
                 scenario,
                 pipeline: PerceptionPipeline,
                 dt: float = 0.1,
                 max_time: float = 120.0):
        self.scenario = scenario
        self.pipeline = pipeline
        self.dt = dt
        self.max_time = max_time
        self.lock = threading.Lock()
        self.paused = False
        self.step_pending = False
        self.reset_pending = False
        self.speed = 1.0
        self.t = 0.0
        self.state: Optional[Dict[str, Any]] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def _fresh_pipeline(self) -> PerceptionPipeline:
        """重建有状态算法，同时保留配置、推理后端和传感器外参。"""
        old = self.pipeline
        pipeline = PerceptionPipeline(old.config)
        pipeline.rgb_detector.backend = old.rgb_detector.backend
        pipeline.crack_segmenter.backend = old.crack_segmenter.backend
        pipeline.water_detector.backend = old.water_detector.backend
        pipeline.facility_inspector.backend = old.facility_inspector.backend
        pipeline.semantic_localizer.transforms = {
            key: value.copy()
            for key, value in old.semantic_localizer.transforms.items()
        }
        pipeline.initialize()
        return pipeline

    def _advance_and_compute(self) -> None:
        """根据暂停、单步和重置信号推进一次仿真并计算页面状态。"""
        with self.lock:
            if self.reset_pending:
                self.reset_pending = False
                self.pipeline = self._fresh_pipeline()
                stamp = self.t
            elif self.step_pending:
                self.step_pending = False
                self.t = (self.t + self.dt * self.speed) % self.max_time
                stamp = self.t
            elif self.paused:
                return
            else:
                self.t = (self.t + self.dt * self.speed) % self.max_time
                stamp = self.t
        synced = self.scenario.frame(stamp)
        output = self.pipeline.spin_once(synced, now=stamp)
        payload = build_payload(output, self.pipeline.thermal_result, stamp)
        payload["run_mode"] = self.pipeline.config.run_mode
        payload["paused"] = self.paused
        payload["speed"] = self.speed
        with self.lock:
            self.state = payload

    def run_loop(self) -> None:
        while not self._stop:
            self._advance_and_compute()
            time.sleep(self.dt)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self.run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def current_payload(self) -> Dict[str, Any]:
        with self.lock:
            return self.state if self.state is not None else {}

    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理暂停、继续、单步、重置和速度控制。"""
        action = data.get("action", "")
        with self.lock:
            if action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
                self.step_pending = False
            elif action == "step":
                self.paused = True
                self.step_pending = True
            elif action == "reset":
                self.t = 0.0
                self.paused = True
                self.step_pending = False
                self.reset_pending = True
            elif action == "speed":
                try:
                    self.speed = max(0.1, min(float(data.get("value", 1.0)), 20.0))
                except (TypeError, ValueError):
                    return {"ok": False, "error": "invalid speed"}
            else:
                return {"ok": False, "error": "unknown action"}
        return {"ok": True, "paused": self.paused, "speed": self.speed,
                "time": self.t}


def make_handler(server: VisualizerServer, static_dir: Path = STATIC_DIR):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._serve_file(static_dir / "index.html")
            elif path == "/api/state":
                self._send_json(server.current_payload())
            elif path == "/events":
                self._stream_events()
            elif path.startswith("/static/"):
                rel = path[len("/static/"):]
                target = (static_dir / rel).resolve()
                if not str(target).startswith(str(static_dir.resolve())):
                    self.send_error(403)
                else:
                    self._serve_file(target)
            else:
                self.send_error(404)

        def _stream_events(self) -> None:
            """以 10 Hz 推送最新完整状态；断开由浏览器自动重连。"""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while not server._stop:
                    payload = server.current_payload()
                    line = json.dumps(payload, ensure_ascii=False)
                    self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            if self.path != "/api/control":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self._send_json({"ok": False, "error": "empty body"}, 400)
                return
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "invalid json"}, 400)
                return
            self._send_json(server.handle_control(data))

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def create_server(scenario,
                  pipeline: PerceptionPipeline,
                  host: str = "127.0.0.1",
                  port: int = 8000,
                  dt: float = 0.1,
                  max_time: float = 120.0) -> ThreadingHTTPServer:
    server = VisualizerServer(scenario, pipeline, dt=dt, max_time=max_time)
    httpd = ThreadingHTTPServer((host, port), make_handler(server))
    httpd.daemon_threads = True
    httpd.visualizer = server
    return httpd
