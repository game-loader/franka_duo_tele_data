"""Start local-only odometry and LiDAR TF adapters for TMR navigation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_odom", default_value="/swerve_drive_controller/odom"
            ),
            DeclareLaunchArgument("output_odom", default_value="/navigation/odom"),
            Node(
                package="tmr_local_navigation",
                executable="odom_frame_adapter",
                name="tmr_odom_frame_adapter",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_odom"),
                        "output_topic": LaunchConfiguration("output_odom"),
                        "parent_frame": "odom",
                        "child_frame": "base_link",
                        "publish_tf": True,
                    }
                ],
            ),
            # Mount positions come from the TMR mobile FR3 URDF. The physical
            # nanoScan Xacro adds a -1.57 rad sensor-frame yaw relative to its
            # mounting point; the rotations below are the composed transforms
            # to the scan driver's lidar_front/lidar_rear frame IDs.
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="lidar_front_static_tf",
                arguments=[
                    "--x", "0.3275", "--y", "0.2175", "--z", "0.19065",
                    "--roll", "-3.141592653589793", "--pitch", "0.0",
                    "--yaw", "0.7846018366025517",
                    "--frame-id", "base_link", "--child-frame-id", "lidar_front",
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="lidar_rear_static_tf",
                arguments=[
                    "--x", "-0.3275", "--y", "-0.2175", "--z", "0.19065",
                    "--roll", "3.141592653589793", "--pitch", "0.0",
                    "--yaw", "-2.3569908169872414",
                    "--frame-id", "base_link", "--child-frame-id", "lidar_rear",
                ],
            ),
        ]
    )
