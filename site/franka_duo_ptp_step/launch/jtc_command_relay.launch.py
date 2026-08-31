from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    enable_robot = LaunchConfiguration("enable_robot")
    left_input_topic = LaunchConfiguration("left_input_topic")
    right_input_topic = LaunchConfiguration("right_input_topic")
    left_output_topic = LaunchConfiguration("left_output_topic")
    right_output_topic = LaunchConfiguration("right_output_topic")
    left_joints = [
        "left_fr3v2_joint1",
        "left_fr3v2_joint2",
        "left_fr3v2_joint3",
        "left_fr3v2_joint4",
        "left_fr3v2_joint5",
        "left_fr3v2_joint6",
        "left_fr3v2_joint7",
    ]
    right_joints = [
        "right_fr3v2_joint1",
        "right_fr3v2_joint2",
        "right_fr3v2_joint3",
        "right_fr3v2_joint4",
        "right_fr3v2_joint5",
        "right_fr3v2_joint6",
        "right_fr3v2_joint7",
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_robot", default_value="false"),
            DeclareLaunchArgument(
                "left_input_topic",
                default_value="/franka_duo/eval/left/joint_trajectory",
            ),
            DeclareLaunchArgument(
                "right_input_topic",
                default_value="/franka_duo/eval/right/joint_trajectory",
            ),
            DeclareLaunchArgument(
                "left_output_topic",
                default_value="/left/joint_trajectory_controller/joint_trajectory",
            ),
            DeclareLaunchArgument(
                "right_output_topic",
                default_value="/right/joint_trajectory_controller/joint_trajectory",
            ),
            Node(
                package="franka_duo_ptp_step",
                executable="jtc_command_relay",
                name="left_jtc_command_relay",
                output="screen",
                parameters=[
                    {
                        "input_topic": left_input_topic,
                        "output_topic": left_output_topic,
                        "expected_joint_names": left_joints,
                        "enable_robot": enable_robot,
                    }
                ],
            ),
            Node(
                package="franka_duo_ptp_step",
                executable="jtc_command_relay",
                name="right_jtc_command_relay",
                output="screen",
                parameters=[
                    {
                        "input_topic": right_input_topic,
                        "output_topic": right_output_topic,
                        "expected_joint_names": right_joints,
                        "enable_robot": enable_robot,
                    }
                ],
            ),
        ]
    )
