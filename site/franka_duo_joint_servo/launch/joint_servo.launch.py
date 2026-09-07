"""Start only the joint servo; it never switches controllers or touches hardware state.

The servo publishes targets to site-owned topics.  Robot motion additionally
requires gello_target_relay.launch.py enable_robot:=true and an active
joint_impedance_controller on each arm.
"""

from franka_mobile_fr3_duo_moveit_config.description import get_robot_descriptions
from franka_mobile_fr3_duo_moveit_config.parameters import get_parameters
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _launch_node(context):
    robot_description, robot_description_semantic = get_robot_descriptions("mobile_fr3_duo_v0_2", "false")
    kinematics, joint_limits, _, _ = get_parameters()
    moveit_parameters = {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": kinematics,
        "robot_description_planning": joint_limits,
    }

    def number(name):
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    def integer(name):
        return ParameterValue(LaunchConfiguration(name), value_type=int)

    node = Node(
        package="franka_duo_joint_servo",
        executable="policy_chunk_joint_servo",
        name="franka_duo_joint_servo",
        output="screen",
        parameters=[
            moveit_parameters,
            {
                "action_frame": LaunchConfiguration("action_frame"),
                "tool_offset_z_m": number("tool_offset_z_m"),
                "playback_speed": number("playback_speed"),
                "action_rate_hz": number("action_rate_hz"),
                "servo_rate_hz": number("servo_rate_hz"),
                "commit_lead_steps": integer("commit_lead_steps"),
                "blend_steps": integer("blend_steps"),
                "max_joint_velocity_rad_s": [float(LaunchConfiguration("max_joint_velocity_rad_s").perform(context))],
                "max_joint_acceleration_rad_s2": [
                    float(LaunchConfiguration("max_joint_acceleration_rad_s2").perform(context))
                ],
                "max_joint_jerk_rad_s3": [float(LaunchConfiguration("max_joint_jerk_rad_s3").perform(context))],
                "max_joint_delta_rad": number("max_joint_delta_rad"),
                "max_tracking_error_rad": number("max_tracking_error_rad"),
                "joint_state_timeout_s": number("joint_state_timeout_s"),
                "idle_follow_timeout_s": number("idle_follow_timeout_s"),
                "enable_gripper": ParameterValue(LaunchConfiguration("enable_gripper"), value_type=bool),
            },
        ],
    )
    return [node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            SetEnvironmentVariable(name="RMW_IMPLEMENTATION", value="rmw_cyclonedds_cpp"),
            DeclareLaunchArgument("action_frame", default_value="link0"),
            DeclareLaunchArgument("tool_offset_z_m", default_value="0.174"),
            DeclareLaunchArgument("playback_speed", default_value="0.1"),
            DeclareLaunchArgument("action_rate_hz", default_value="30.0"),
            DeclareLaunchArgument("servo_rate_hz", default_value="1000.0"),
            DeclareLaunchArgument("commit_lead_steps", default_value="3"),
            DeclareLaunchArgument("blend_steps", default_value="4"),
            DeclareLaunchArgument("max_joint_velocity_rad_s", default_value="0.8"),
            DeclareLaunchArgument("max_joint_acceleration_rad_s2", default_value="2.0"),
            DeclareLaunchArgument("max_joint_jerk_rad_s3", default_value="20.0"),
            DeclareLaunchArgument("max_joint_delta_rad", default_value="0.35"),
            DeclareLaunchArgument("max_tracking_error_rad", default_value="0.15"),
            DeclareLaunchArgument("joint_state_timeout_s", default_value="0.2"),
            DeclareLaunchArgument("idle_follow_timeout_s", default_value="20.0"),
            DeclareLaunchArgument("enable_gripper", default_value="false"),
            OpaqueFunction(function=_launch_node),
        ]
    )
