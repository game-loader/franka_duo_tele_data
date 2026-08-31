from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _set_hardware_active(controller_manager, hardware_name):
    return ExecuteProcess(
        cmd=[
            "ros2",
            "control",
            "set_hardware_component_state",
            hardware_name,
            "active",
            "--controller-manager",
            controller_manager,
        ],
        output="screen",
    )


def _deactivate_builtin_controller(controller_manager):
    return ExecuteProcess(
        cmd=[
            "ros2",
            "control",
            "switch_controllers",
            "--controller-manager",
            controller_manager,
            "--deactivate",
            "joint_impedance_controller",
        ],
        output="screen",
    )


def _activate_policy_controller(controller_manager, policy_controller):
    return ExecuteProcess(
        cmd=[
            "ros2",
            "control",
            "switch_controllers",
            "--controller-manager",
            controller_manager,
            "--strict",
            "--activate",
            policy_controller,
        ],
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    config = LaunchConfiguration("config")
    left_controller_manager = LaunchConfiguration("left_controller_manager")
    right_controller_manager = LaunchConfiguration("right_controller_manager")
    enable_robot = LaunchConfiguration("enable_robot")

    left_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "left_policy_cartesian_pose_controller",
            "--inactive",
            "--controller-manager",
            left_controller_manager,
            "--param-file",
            config,
        ],
        parameters=[config],
        output="screen",
    )
    right_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "right_policy_cartesian_pose_controller",
            "--inactive",
            "--controller-manager",
            right_controller_manager,
            "--param-file",
            config,
        ],
        parameters=[config],
        output="screen",
    )
    left_set_hardware_active = _set_hardware_active(
        left_controller_manager, "left_FrankaHardwareInterface"
    )
    left_deactivate_builtin = _deactivate_builtin_controller(left_controller_manager)
    left_activate_policy = _activate_policy_controller(
        left_controller_manager, "left_policy_cartesian_pose_controller"
    )
    right_set_hardware_active = _set_hardware_active(
        right_controller_manager, "right_FrankaHardwareInterface"
    )
    right_deactivate_builtin = _deactivate_builtin_controller(right_controller_manager)
    right_activate_policy = _activate_policy_controller(
        right_controller_manager, "right_policy_cartesian_pose_controller"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("franka_duo_policy_control"),
                        "config",
                        "policy_control.example.yaml",
                    ]
                ),
                description="Relay and always-on policy controller parameter file",
            ),
            DeclareLaunchArgument(
                "left_controller_manager",
                default_value="/left/controller_manager",
                description="Existing left-arm controller_manager node",
            ),
            DeclareLaunchArgument(
                "right_controller_manager",
                default_value="/right/controller_manager",
                description="Existing right-arm controller_manager node",
            ),
            DeclareLaunchArgument(
                "enable_robot",
                default_value="false",
                description="Explicitly enable relay output to the robot",
            ),
            Node(
                package="franka_duo_policy_control",
                executable="policy_action_relay",
                name="franka_duo_policy_action_relay",
                parameters=[config, {"enable_robot": enable_robot}],
                output="screen",
            ),
            left_spawner,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=left_spawner,
                    on_exit=[left_set_hardware_active],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=left_set_hardware_active,
                    on_exit=[left_deactivate_builtin],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=left_deactivate_builtin,
                    on_exit=[left_activate_policy],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=left_activate_policy,
                    on_exit=[right_spawner],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=right_spawner,
                    on_exit=[right_set_hardware_active],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=right_set_hardware_active,
                    on_exit=[right_deactivate_builtin],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=right_deactivate_builtin,
                    on_exit=[right_activate_policy],
                )
            ),
        ]
    )
