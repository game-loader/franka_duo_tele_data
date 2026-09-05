"""启动墙壁伺服控制节点的Launch文件"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            # 声明启动参数
            DeclareLaunchArgument(
                "scan_topic",
                default_value="/navigation/scan",
                description="激光雷达话题名称",
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic",
                default_value="/cmd_vel",
                description="速度控制话题名称",
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/navigation/odom",
                description="里程计话题名称",
            ),
            DeclareLaunchArgument(
                "target_wall_distance",
                default_value="0.5",
                description="目标墙壁距离（米）",
            ),
            DeclareLaunchArgument(
                "distance_tolerance",
                default_value="0.05",
                description="距离容差（米）",
            ),
            DeclareLaunchArgument(
                "rotation_angle",
                default_value="90.0",
                description="旋转角度（度）",
            ),
            DeclareLaunchArgument(
                "angle_tolerance",
                default_value="5.0",
                description="角度容差（度）",
            ),
            DeclareLaunchArgument(
                "linear_speed",
                default_value="0.15",
                description="线速度（m/s）",
            ),
            DeclareLaunchArgument(
                "angular_speed",
                default_value="0.3",
                description="角速度（rad/s）",
            ),
            DeclareLaunchArgument(
                "left_movement_duration",
                default_value="5.0",
                description="向左运动持续时间（秒）",
            ),
            # 墙壁伺服控制节点
            Node(
                package="tmr_local_navigation",  # 根据实际包名调整
                executable="wall_servo_control.py",
                name="wall_servo_controller",
                output="screen",
                parameters=[
                    {
                        "scan_topic": LaunchConfiguration("scan_topic"),
                        "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "target_wall_distance": LaunchConfiguration("target_wall_distance"),
                        "distance_tolerance": LaunchConfiguration("distance_tolerance"),
                        "rotation_angle": LaunchConfiguration("rotation_angle"),
                        "angle_tolerance": LaunchConfiguration("angle_tolerance"),
                        "linear_speed": LaunchConfiguration("linear_speed"),
                        "angular_speed": LaunchConfiguration("angular_speed"),
                        "left_movement_duration": LaunchConfiguration("left_movement_duration"),
                    }
                ],
            ),
        ]
    )
