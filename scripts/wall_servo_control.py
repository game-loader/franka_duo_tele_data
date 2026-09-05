#!/usr/bin/env python3
"""基于激光雷达的墙壁伺服控制脚本

控制流程：
1. 向前运动直到前墙距离0.5m
2. 底盘顺时针旋转90°
3. 向后运动直到前方激光雷达显示和前墙距离0.5m
4. 向左直线运动
"""

from __future__ import annotations

import math
import time
from enum import Enum

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class ControlState(Enum):
    """控制状态机"""
    MOVE_FORWARD = 1
    ROTATE_CLOCKWISE = 2
    MOVE_BACKWARD = 3
    MOVE_LEFT = 4
    COMPLETED = 5


class WallServoController(Node):
    def __init__(self) -> None:
        super().__init__("wall_servo_controller")

        # 声明参数
        self.declare_parameter("scan_topic", "/navigation/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/navigation/odom")
        self.declare_parameter("target_wall_distance", 0.5)  # 目标墙壁距离（米）
        self.declare_parameter("distance_tolerance", 0.05)  # 距离容差（米）
        self.declare_parameter("rotation_angle", 90.0)  # 旋转角度（度）
        self.declare_parameter("angle_tolerance", 5.0)  # 角度容差（度）
        self.declare_parameter("linear_speed", 0.15)  # 线速度（m/s）
        self.declare_parameter("angular_speed", 0.3)  # 角速度（rad/s）
        self.declare_parameter("left_movement_duration", 5.0)  # 向左运动时长（秒）

        # 获取参数
        self._target_distance = float(self.get_parameter("target_wall_distance").value)
        self._distance_tolerance = float(self.get_parameter("distance_tolerance").value)
        self._rotation_angle = math.radians(float(self.get_parameter("rotation_angle").value))
        self._angle_tolerance = math.radians(float(self.get_parameter("angle_tolerance").value))
        self._linear_speed = float(self.get_parameter("linear_speed").value)
        self._angular_speed = float(self.get_parameter("angular_speed").value)
        self._left_duration = float(self.get_parameter("left_movement_duration").value)

        # 状态变量
        self._state = ControlState.MOVE_FORWARD
        self._latest_scan: LaserScan | None = None
        self._latest_odom: Odometry | None = None
        self._rotation_start_yaw: float | None = None
        self._left_move_start_time: float | None = None

        # QoS配置
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # 订阅激光雷达
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._on_scan,
            sensor_qos,
        )

        # 订阅里程计
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            10,
        )

        # 发布速度命令
        self._cmd_vel_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            10,
        )

        # 创建控制循环定时器（50Hz）
        self._control_timer = self.create_timer(0.02, self._control_loop)

        self.get_logger().info("墙壁伺服控制器已启动")
        self.get_logger().info(f"目标距离: {self._target_distance}m, 旋转角度: {math.degrees(self._rotation_angle)}°")

    def _on_scan(self, msg: LaserScan) -> None:
        """激光雷达回调"""
        self._latest_scan = msg

    def _on_odom(self, msg: Odometry) -> None:
        """里程计回调"""
        self._latest_odom = msg

    def _get_front_distance(self) -> float | None:
        """获取前方最小距离（-30°到+30°扇区）"""
        if self._latest_scan is None:
            return None

        scan = self._latest_scan
        front_distances = []

        # 搜索前方60度扇区内的有效距离
        for i, distance in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            # 限制在前方±30度（±0.52 rad）
            if abs(angle) <= 0.52 and math.isfinite(distance):
                if scan.range_min <= distance <= scan.range_max:
                    front_distances.append(distance)

        if not front_distances:
            return None

        # 返回最小距离
        return min(front_distances)

    def _get_current_yaw(self) -> float | None:
        """从里程计获取当前偏航角"""
        if self._latest_odom is None:
            return None

        # 从四元数转换为欧拉角
        q = self._latest_odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到[-π, π]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _publish_velocity(self, linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0) -> None:
        """发布速度命令"""
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.linear.y = linear_y
        cmd.angular.z = angular_z
        self._cmd_vel_pub.publish(cmd)

    def _stop(self) -> None:
        """停止机器人"""
        self._publish_velocity(0.0, 0.0, 0.0)

    def _control_loop(self) -> None:
        """主控制循环"""
        if self._state == ControlState.COMPLETED:
            return

        # 状态1: 向前运动直到距离前墙0.5m
        if self._state == ControlState.MOVE_FORWARD:
            front_dist = self._get_front_distance()
            if front_dist is None:
                self.get_logger().warn("无法获取前方距离，等待激光雷达数据...")
                self._stop()
                return

            self.get_logger().info(f"[向前] 前方距离: {front_dist:.3f}m", throttle_duration_sec=0.5)

            if front_dist <= self._target_distance + self._distance_tolerance:
                # 到达目标距离，停止并切换到旋转状态
                self._stop()
                self.get_logger().info(f"到达目标距离 {front_dist:.3f}m，开始旋转90°")
                self._state = ControlState.ROTATE_CLOCKWISE
                self._rotation_start_yaw = self._get_current_yaw()
                if self._rotation_start_yaw is None:
                    self.get_logger().error("无法获取当前偏航角！")
                time.sleep(0.5)  # 短暂停顿
            else:
                # 继续向前
                self._publish_velocity(linear_x=self._linear_speed)

        # 状态2: 顺时针旋转90°
        elif self._state == ControlState.ROTATE_CLOCKWISE:
            current_yaw = self._get_current_yaw()
            if current_yaw is None or self._rotation_start_yaw is None:
                self.get_logger().warn("无法获取偏航角，等待里程计数据...")
                self._stop()
                return

            # 计算旋转角度（顺时针为负）
            angle_rotated = self._normalize_angle(current_yaw - self._rotation_start_yaw)
            angle_remaining = -self._rotation_angle - angle_rotated

            self.get_logger().info(
                f"[旋转] 已旋转: {math.degrees(angle_rotated):.1f}°, "
                f"目标: {math.degrees(-self._rotation_angle):.1f}°",
                throttle_duration_sec=0.5
            )

            if abs(angle_remaining) <= self._angle_tolerance:
                # 旋转完成
                self._stop()
                self.get_logger().info("旋转90°完成，开始向后运动")
                self._state = ControlState.MOVE_BACKWARD
                time.sleep(0.5)
            else:
                # 继续旋转（顺时针为负角速度）
                self._publish_velocity(angular_z=-self._angular_speed)

        # 状态3: 向后运动直到前方激光雷达显示距离0.5m
        elif self._state == ControlState.MOVE_BACKWARD:
            front_dist = self._get_front_distance()
            if front_dist is None:
                self.get_logger().warn("无法获取前方距离，等待激光雷达数据...")
                self._stop()
                return

            self.get_logger().info(f"[向后] 前方距离: {front_dist:.3f}m", throttle_duration_sec=0.5)

            if front_dist <= self._target_distance + self._distance_tolerance:
                # 到达目标距离
                self._stop()
                self.get_logger().info(f"到达目标距离 {front_dist:.3f}m，开始向左运动")
                self._state = ControlState.MOVE_LEFT
                self._left_move_start_time = time.time()
                time.sleep(0.5)
            else:
                # 继续向后（负x速度）
                self._publish_velocity(linear_x=-self._linear_speed)

        # 状态4: 向左直线运动
        elif self._state == ControlState.MOVE_LEFT:
            if self._left_move_start_time is None:
                self._left_move_start_time = time.time()

            elapsed = time.time() - self._left_move_start_time
            self.get_logger().info(
                f"[向左] 运动时间: {elapsed:.1f}s / {self._left_duration:.1f}s",
                throttle_duration_sec=0.5
            )

            if elapsed >= self._left_duration:
                # 完成向左运动
                self._stop()
                self.get_logger().info("向左运动完成，控制流程结束")
                self._state = ControlState.COMPLETED
            else:
                # 继续向左（正y速度）
                self._publish_velocity(linear_y=self._linear_speed)


def main(args=None) -> None:
    rclpy.init(args=args)
    controller = WallServoController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("收到中断信号，停止控制器")
    finally:
        controller._stop()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
