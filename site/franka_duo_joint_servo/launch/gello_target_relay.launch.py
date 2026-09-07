"""Gated relays from the joint servo to each arm's joint_impedance_controller input."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    enable_robot = ParameterValue(LaunchConfiguration("enable_robot"), value_type=bool)
    enable_gripper = ParameterValue(LaunchConfiguration("enable_gripper"), value_type=bool)
    nodes = []
    for side in ("left", "right"):
        nodes.append(
            Node(
                package="franka_duo_joint_servo",
                executable="gello_target_relay",
                name=f"{side}_gello_target_relay",
                output="screen",
                parameters=[
                    {
                        "input_topic": f"/franka_duo/joint_servo/{side}/target",
                        "output_topic": f"/{side}/gello/joint_states",
                        "gripper_input_topic": f"/franka_duo/joint_servo/{side}/gripper",
                        "gripper_output_topic": f"/{side}/gripper/gripper_client/target_gripper_width_percent",
                        "expected_joint_names": [f"{side}_fr3v2_joint{i}" for i in range(1, 8)],
                        "enable_robot": enable_robot,
                        "enable_gripper": enable_gripper,
                    }
                ],
            )
        )
    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_robot", default_value="false"),
            DeclareLaunchArgument("enable_gripper", default_value="false"),
            *nodes,
        ]
    )
