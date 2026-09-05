from franka_mobile_fr3_duo_moveit_config.description import get_robot_descriptions
from franka_mobile_fr3_duo_moveit_config.parameters import get_parameters
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    robot_description, robot_description_semantic = get_robot_descriptions(
        "mobile_fr3_duo_v0_2",
        "false",
    )
    kinematics, joint_limits, _, _ = get_parameters()

    moveit_parameters = {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": kinematics,
        "robot_description_planning": joint_limits,
    }

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="RMW_IMPLEMENTATION",
                value="rmw_cyclonedds_cpp",
            ),
            DeclareLaunchArgument("offset_m", default_value="0.01"),
            DeclareLaunchArgument("tool_offset_z_m", default_value="0.174"),
            DeclareLaunchArgument("ik_timeout_s", default_value="0.1"),
            DeclareLaunchArgument("max_joint_delta_rad", default_value="0.35"),
            DeclareLaunchArgument("group_name", default_value="left_arm"),
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("confirm", default_value="false"),
            Node(
                package="franka_duo_ptp_step",
                executable="left_ptp_step",
                name="franka_duo_left_ptp_step",
                output="screen",
                parameters=[
                    moveit_parameters,
                    {
                        "offset_m": LaunchConfiguration("offset_m"),
                        "tool_offset_z_m": LaunchConfiguration("tool_offset_z_m"),
                        "ik_timeout_s": LaunchConfiguration("ik_timeout_s"),
                        "max_joint_delta_rad": LaunchConfiguration("max_joint_delta_rad"),
                        "group_name": LaunchConfiguration("group_name"),
                        "execute": LaunchConfiguration("execute"),
                        "confirm": LaunchConfiguration("confirm"),
                    },
                ],
            ),
        ]
    )
