#!/usr/bin/env python3
"""Repair the TMR odometry frame contract for Nav2 without touching the robot host.

The TMR controller currently publishes an Odometry message with ``world`` as
the parent frame and an empty child frame.  This node republishes the same
measurement on a separate topic, labels the parent ``odom`` (configurable),
sets the child to ``base_link`` (configurable), and optionally broadcasts the
matching dynamic TF.  The source topic is never republished in place.
"""

from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomFrameAdapter(Node):
    def __init__(self) -> None:
        super().__init__("tmr_odom_frame_adapter")
        self.declare_parameter("input_topic", "/swerve_drive_controller/odom")
        self.declare_parameter("output_topic", "/navigation/odom")
        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("publish_tf", True)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._parent_frame = str(self.get_parameter("parent_frame").value)
        self._child_frame = str(self.get_parameter("child_frame").value)
        self._publish_tf = bool(self.get_parameter("publish_tf").value)

        self._publisher = self.create_publisher(Odometry, output_topic, 10)
        self._subscription = self.create_subscription(
            Odometry, input_topic, self._on_odom, 10
        )
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None
        self.get_logger().info(
            f"{input_topic} -> {output_topic}; TF {self._parent_frame} -> {self._child_frame}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        out = Odometry()
        out.header = msg.header
        out.header.frame_id = self._parent_frame
        out.child_frame_id = self._child_frame
        out.pose = msg.pose
        out.twist = msg.twist
        self._publisher.publish(out)

        if self._tf_broadcaster is None:
            return
        transform = __import__("geometry_msgs.msg", fromlist=["TransformStamped"]).TransformStamped()
        transform.header = out.header
        transform.child_frame_id = self._child_frame
        transform.transform.translation.x = out.pose.pose.position.x
        transform.transform.translation.y = out.pose.pose.position.y
        transform.transform.translation.z = out.pose.pose.position.z
        transform.transform.rotation = out.pose.pose.orientation
        self._tf_broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomFrameAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
