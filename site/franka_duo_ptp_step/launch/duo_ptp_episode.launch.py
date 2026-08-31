import json
import math
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from franka_mobile_fr3_duo_moveit_config.description import get_robot_descriptions
from franka_mobile_fr3_duo_moveit_config.parameters import get_parameters


def _load_action(context):
    action_file = Path(
        LaunchConfiguration("action_file").perform(context)
    ).expanduser()
    action_index = int(LaunchConfiguration("action_index").perform(context))
    action_start_index = int(LaunchConfiguration("action_start_index").perform(context))
    action_end_index = int(LaunchConfiguration("action_end_index").perform(context))
    payload = json.loads(action_file.read_text(encoding="utf-8"))
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise RuntimeError(f"no actions found in {action_file}")
    if action_index >= 0:
        action_start_index = action_index
        action_end_index = action_index + 1
    if action_end_index < 0:
        action_end_index = len(actions)
    if (
        action_start_index < 0
        or action_end_index <= action_start_index
        or action_end_index > len(actions)
    ):
        raise RuntimeError(
            f"invalid action range [{action_start_index}, {action_end_index}) in {action_file}"
        )
    values = []
    for action in actions[action_start_index:action_end_index]:
        if not isinstance(action, list) or len(action) != 18:
            raise RuntimeError(f"expected an 18D action in {action_file}")
        action_values = [float(value) for value in action]
        if not all(math.isfinite(value) for value in action_values):
            raise RuntimeError(f"action contains non-finite values: {action_file}")
        values.extend(action_values)

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
        executable="duo_ptp_episode",
        name="franka_duo_episode_ptp",
        output="screen",
        parameters=[
            moveit_parameters,
            {
                "action_trajectory": values,
                "tool_offset_z_m": LaunchConfiguration("tool_offset_z_m"),
                "ik_timeout_s": LaunchConfiguration("ik_timeout_s"),
                "max_joint_delta_rad": LaunchConfiguration("max_joint_delta_rad"),
                "max_joint_velocity": LaunchConfiguration("max_joint_velocity"),
                "goal_tolerance": LaunchConfiguration("goal_tolerance"),
                "wait_timeout_s": LaunchConfiguration("wait_timeout_s"),
                "log_file": LaunchConfiguration("log_file"),
                "mode": LaunchConfiguration("mode"),
                "left_trajectory_topic": LaunchConfiguration("left_trajectory_topic"),
                "right_trajectory_topic": LaunchConfiguration("right_trajectory_topic"),
                "input_action_rate_hz": LaunchConfiguration("input_action_rate_hz"),
                "stream_rate_hz": LaunchConfiguration("stream_rate_hz"),
                "chunk_size": LaunchConfiguration("chunk_size"),
                "chunk_publish_period_s": LaunchConfiguration("chunk_publish_period_s"),
                "jtc_initial_delay_s": LaunchConfiguration("jtc_initial_delay_s"),
                "stream_log_file": LaunchConfiguration("stream_log_file"),
                "execute": LaunchConfiguration("execute"),
                "confirm": LaunchConfiguration("confirm"),
            },
        ],
    )
    return [node]


def generate_launch_description() -> LaunchDescription:
    default_action_file = os.environ.get(
        "FRANKA_DUO_ACTION_FILE",
        "/home/aup/franka_duo_tele_data_capture/artifacts/"
        "franka_duo_tmr_lerobot_v6_episode_000000_first_1s_action.json",
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="RMW_IMPLEMENTATION",
                value="rmw_cyclonedds_cpp",
            ),
            DeclareLaunchArgument("action_file", default_value=default_action_file),
            DeclareLaunchArgument(
                "action_index",
                default_value="-1",
                description="Run one action index; -1 runs the selected trajectory range",
            ),
            DeclareLaunchArgument("action_start_index", default_value="0"),
            DeclareLaunchArgument(
                "action_end_index",
                default_value="-1",
                description="Exclusive action end index; -1 means the end of the file",
            ),
            DeclareLaunchArgument("tool_offset_z_m", default_value="0.174"),
            DeclareLaunchArgument("ik_timeout_s", default_value="0.1"),
            DeclareLaunchArgument("max_joint_delta_rad", default_value="1.0"),
            DeclareLaunchArgument("max_joint_velocity", default_value="0.2"),
            DeclareLaunchArgument("goal_tolerance", default_value="0.01"),
            DeclareLaunchArgument("wait_timeout_s", default_value="10.0"),
            DeclareLaunchArgument(
                "log_file",
                default_value=os.environ.get(
                    "FRANKA_DUO_PTP_LOG_FILE",
                    "/tmp/franka_duo_ptp_episode.csv",
                ),
            ),
            DeclareLaunchArgument("mode", default_value="ptp"),
            DeclareLaunchArgument(
                "left_trajectory_topic",
                default_value="/franka_duo/eval/left/joint_trajectory",
            ),
            DeclareLaunchArgument(
                "right_trajectory_topic",
                default_value="/franka_duo/eval/right/joint_trajectory",
            ),
            DeclareLaunchArgument("input_action_rate_hz", default_value="15.0"),
            DeclareLaunchArgument("stream_rate_hz", default_value="50.0"),
            DeclareLaunchArgument("chunk_size", default_value="16"),
            DeclareLaunchArgument("chunk_publish_period_s", default_value="0.1"),
            DeclareLaunchArgument("jtc_initial_delay_s", default_value="1.0"),
            DeclareLaunchArgument(
                "stream_log_file",
                default_value=os.environ.get(
                    "FRANKA_DUO_JTC_LOG_FILE",
                    "/tmp/franka_duo_jtc_chunk_stream.csv",
                ),
            ),
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("confirm", default_value="false"),
            OpaqueFunction(function=_load_action),
        ]
    )
