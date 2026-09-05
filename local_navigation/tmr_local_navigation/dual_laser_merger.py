#!/usr/bin/env python3
"""Merge front and rear TMR LaserScan streams in the base_link frame."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


def _stamp_seconds(scan: LaserScan) -> float:
    return float(scan.header.stamp.sec) + float(scan.header.stamp.nanosec) * 1e-9


def _rotation_xy(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    """Return the XY rows of Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cy * cp,
        cy * sp * sr - sy * cr,
        sy * cp,
        sy * sp * sr + cy * cr,
    )


class DualLaserMerger(Node):
    def __init__(self) -> None:
        super().__init__("tmr_dual_laser_merger")
        self.declare_parameter("front_topic", "/lidar_front/scan")
        self.declare_parameter("rear_topic", "/lidar_rear/scan")
        self.declare_parameter("output_topic", "/navigation/scan")
        self.declare_parameter("output_frame", "base_link")
        self.declare_parameter("angle_increment", 0.0029088794253766537)
        self.declare_parameter("max_pair_age", 0.08)
        # Bounds measured from the 1:1 TMRv0.2 collision mesh.
        self.declare_parameter("footprint_min_x", -0.40)
        self.declare_parameter("footprint_max_x", 0.40)
        self.declare_parameter("footprint_min_y", -0.29)
        self.declare_parameter("footprint_max_y", 0.29)

        self._output_frame = str(self.get_parameter("output_frame").value)
        self._angle_increment = float(self.get_parameter("angle_increment").value)
        self._max_pair_age = float(self.get_parameter("max_pair_age").value)
        self._footprint = (
            float(self.get_parameter("footprint_min_x").value),
            float(self.get_parameter("footprint_max_x").value),
            float(self.get_parameter("footprint_min_y").value),
            float(self.get_parameter("footprint_max_y").value),
        )
        self._latest: dict[str, LaserScan | None] = {"front": None, "rear": None}
        self._last_used_stamps: dict[str, tuple[int, int] | None] = {
            "front": None,
            "rear": None,
        }

        # Exact composed base_link -> scan-frame transforms from the TMR and
        # nanoScan Xacros. Values are (x, y, roll, pitch, yaw).
        self._transforms = {
            "front": (0.3275, 0.2175, -math.pi, 0.0, 0.7846018366025517),
            "rear": (-0.3275, -0.2175, math.pi, 0.0, -2.3569908169872414),
        }

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._publisher = self.create_publisher(
            LaserScan, str(self.get_parameter("output_topic").value), sensor_qos
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("front_topic").value),
            lambda msg: self._on_scan("front", msg),
            sensor_qos,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("rear_topic").value),
            lambda msg: self._on_scan("rear", msg),
            sensor_qos,
        )

    def _on_scan(self, side: str, scan: LaserScan) -> None:
        self._latest[side] = scan
        front, rear = self._latest["front"], self._latest["rear"]
        if front is None or rear is None:
            return
        if abs(_stamp_seconds(front) - _stamp_seconds(rear)) > self._max_pair_age:
            return
        stamps = {
            "front": (front.header.stamp.sec, front.header.stamp.nanosec),
            "rear": (rear.header.stamp.sec, rear.header.stamp.nanosec),
        }
        # Publish at the sensor rate rather than once for every individual
        # callback. Both scanners must have supplied a new frame since the
        # previous merge, so no physical scan is reused in two outputs.
        if any(stamps[key] == self._last_used_stamps[key] for key in stamps):
            return
        self._last_used_stamps = stamps
        self._publisher.publish(self._merge(front, rear))

    def _merge(self, front: LaserScan, rear: LaserScan) -> LaserScan:
        out = LaserScan()
        out.header.stamp = front.header.stamp
        if _stamp_seconds(rear) > _stamp_seconds(front):
            out.header.stamp = rear.header.stamp
        out.header.frame_id = self._output_frame
        out.angle_min = -math.pi
        out.angle_max = math.pi
        out.angle_increment = self._angle_increment
        count = int(math.ceil((out.angle_max - out.angle_min) / out.angle_increment))
        out.angle_max = out.angle_min + (count - 1) * out.angle_increment
        out.time_increment = 0.0
        out.scan_time = max(front.scan_time, rear.scan_time)
        out.range_min = min(front.range_min, rear.range_min)
        out.range_max = max(front.range_max, rear.range_max)
        ranges = [math.inf] * count

        for side, scan in (("front", front), ("rear", rear)):
            tx, ty, roll, pitch, yaw = self._transforms[side]
            r00, r01, r10, r11 = _rotation_xy(roll, pitch, yaw)
            for index, distance in enumerate(scan.ranges):
                if not math.isfinite(distance):
                    continue
                if distance < scan.range_min or distance > scan.range_max:
                    continue
                angle = scan.angle_min + index * scan.angle_increment
                sx, sy = distance * math.cos(angle), distance * math.sin(angle)
                bx = tx + r00 * sx + r01 * sy
                by = ty + r10 * sx + r11 * sy
                min_x, max_x, min_y, max_y = self._footprint
                # The nanoScans can see the chassis edges and lower robot
                # structure. Those returns move with base_link and otherwise
                # leave false occupied trails in an empty room. An obstacle
                # endpoint inside the physical footprint cannot be external,
                # so discard it before it reaches SLAM.
                if min_x <= bx <= max_x and min_y <= by <= max_y:
                    continue
                merged_range = math.hypot(bx, by)
                merged_angle = math.atan2(by, bx)
                bin_index = int(round((merged_angle - out.angle_min) / out.angle_increment))
                if 0 <= bin_index < count and merged_range < ranges[bin_index]:
                    ranges[bin_index] = merged_range

        out.ranges = ranges
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DualLaserMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
