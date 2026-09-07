"""Start only the RGB20D relay; never switch controllers or recover hardware."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_robot", default_value="false"),
            DeclareLaunchArgument("enable_gripper", default_value="false"),
            Node(
                package="franka_duo_policy_control",
                executable="policy_action_relay",
                name="franka_duo_rgb20d_relay",
                output="screen",
                parameters=[
                    {
                        "input_topic": "/franka_duo/rgb20d/action",
                        "action_frame": "link0",
                        "enable_robot": ParameterValue(LaunchConfiguration("enable_robot"), value_type=bool),
                        "enable_gripper": ParameterValue(
                            LaunchConfiguration("enable_gripper"), value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
