"""Live action-chunk executor: evaluator chunk -> MoveIt KDL IK -> per-arm JTC.

Start order on the robot host:
  1. franka_bringup for both arms (per-arm controller managers)
  2. deactivate joint_impedance_controller, spawn joint_trajectory_controller
     on /left and /right with the config/*_joint_trajectory_controller.yaml
  3. ros2 launch franka_duo_ptp_step jtc_command_relay.launch.py enable_robot:=true
  4. this launch with execute:=true confirm:=true (dry-run without them)
  5. franka-duo-eval with control_mode: chunk --publish --enable-robot
"""

from franka_mobile_fr3_duo_moveit_config.description import get_robot_descriptions
from franka_mobile_fr3_duo_moveit_config.parameters import get_parameters
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_node(context):
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
    node = Node(
        package="franka_duo_ptp_step",
        executable="policy_chunk_jtc_stream",
        name="policy_chunk_jtc_stream",
        output="screen",
        parameters=[
            moveit_parameters,
            {
                "chunk_topic": LaunchConfiguration("chunk_topic"),
                "left_trajectory_topic": LaunchConfiguration("left_trajectory_topic"),
                "right_trajectory_topic": LaunchConfiguration("right_trajectory_topic"),
                "action_frame": LaunchConfiguration("action_frame"),
                "tool_offset_z_m": LaunchConfiguration("tool_offset_z_m"),
                "ik_timeout_s": LaunchConfiguration("ik_timeout_s"),
                "max_joint_delta_rad": LaunchConfiguration("max_joint_delta_rad"),
                "input_action_rate_hz": LaunchConfiguration("input_action_rate_hz"),
                "stream_rate_hz": LaunchConfiguration("stream_rate_hz"),
                "jtc_initial_delay_s": LaunchConfiguration("jtc_initial_delay_s"),
                "max_joint_velocity_rad_s": LaunchConfiguration("max_joint_velocity_rad_s"),
                "max_chunk_age_s": LaunchConfiguration("max_chunk_age_s"),
                "joint_state_timeout_s": LaunchConfiguration("joint_state_timeout_s"),
                "wait_timeout_s": LaunchConfiguration("wait_timeout_s"),
                "execute": LaunchConfiguration("execute"),
                "confirm": LaunchConfiguration("confirm"),
                "enable_gripper": LaunchConfiguration("enable_gripper"),
            },
        ],
    )
    return [node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            SetEnvironmentVariable(name="RMW_IMPLEMENTATION", value="rmw_cyclonedds_cpp"),
            DeclareLaunchArgument("chunk_topic", default_value="/franka_duo/policy_action_chunk"),
            DeclareLaunchArgument(
                "left_trajectory_topic", default_value="/franka_duo/eval/left/joint_trajectory"
            ),
            DeclareLaunchArgument(
                "right_trajectory_topic", default_value="/franka_duo/eval/right/joint_trajectory"
            ),
            DeclareLaunchArgument(
                "action_frame",
                default_value="link0",
                description="link0 (evaluator applied manifest transforms) or midpoint",
            ),
            DeclareLaunchArgument("tool_offset_z_m", default_value="0.174"),
            DeclareLaunchArgument("ik_timeout_s", default_value="0.02"),
            DeclareLaunchArgument("max_joint_delta_rad", default_value="0.5"),
            DeclareLaunchArgument(
                "input_action_rate_hz",
                default_value="15.0",
                description="Must equal the evaluator fps",
            ),
            DeclareLaunchArgument("stream_rate_hz", default_value="50.0"),
            DeclareLaunchArgument("jtc_initial_delay_s", default_value="0.1"),
            DeclareLaunchArgument("max_joint_velocity_rad_s", default_value="1.5"),
            DeclareLaunchArgument("max_chunk_age_s", default_value="0.5"),
            DeclareLaunchArgument("joint_state_timeout_s", default_value="0.5"),
            DeclareLaunchArgument("wait_timeout_s", default_value="10.0"),
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("confirm", default_value="false"),
            DeclareLaunchArgument("enable_gripper", default_value="false"),
            OpaqueFunction(function=_launch_node),
        ]
    )
