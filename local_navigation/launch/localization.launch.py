"""Start the clean static map, dual-LiDAR merger and AMCL localization."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("tmr_local_navigation"))
    nav2_share = Path(get_package_share_directory("nav2_bringup"))
    default_map = Path.home() / "tmr_navigation/maps/final_clean_20260828_144206.yaml"
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value=str(default_map)),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(package_share / "config/nav2_params.yaml"),
            ),
            Node(
                package="tmr_local_navigation",
                executable="dual_laser_merger",
                name="tmr_dual_laser_merger",
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(nav2_share / "launch/localization_launch.py")
                ),
                launch_arguments={
                    "map": LaunchConfiguration("map"),
                    "params_file": LaunchConfiguration("params_file"),
                    "use_sim_time": "false",
                    "autostart": "true",
                    "use_composition": "False",
                }.items(),
            ),
        ]
    )
