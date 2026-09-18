"""可选 ROS 2 适配层。

``rclpy`` 采用惰性导入，未安装 ROS 2 时核心算法包仍可独立运行。
当前类是最小包装，真实部署仍需补充订阅、TF、QoS 和自定义消息。
"""

from __future__ import annotations

from typing import Optional

from ..perception_pipeline import PerceptionPipeline
from ..types import SyncedSensorData


class Ros2PerceptionNode:
    """主管线的最小 rclpy 包装示例。"""

    def __init__(self, node_name: str = "perception_node"):
        import rclpy  # noqa: F401
        from rclpy.node import Node

        rclpy.init()
        self.node = Node(node_name)
        self.pipeline = PerceptionPipeline()
        self.pipeline.initialize()
        self.pub_targets = self.node.create_publisher(
            "std_msgs/msg/String", "/perception/target_observations", 10)
        self.pub_status = self.node.create_publisher(
            "std_msgs/msg/String", "/perception/status", 10)

    def spin_once(self, synced: SyncedSensorData, now: Optional[float] = None) -> None:
        output = self.pipeline.spin_once(synced, now)
        summary = output.summary()
        self.pub_targets.publish(
            self._string_message(str(summary["targets"])))
        self.pub_status.publish(
            self._string_message(str(summary["status"])))

    @staticmethod
    def _string_message(text: str):
        from std_msgs.msg import String
        msg = String()
        msg.data = text
        return msg
