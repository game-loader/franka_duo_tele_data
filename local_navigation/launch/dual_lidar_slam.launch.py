"""Merge both TMR lidars and start online asynchronous SLAM Toolbox."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("tmr_local_navigation"))
    slam_share = Path(get_package_share_directory("slam_toolbox"))
    return LaunchDescription(
        [
            Node(
                package="tmr_local_navigation",
                executable="dual_laser_merger",
                name="tmr_dual_laser_merger",
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(slam_share / "launch" / "online_async_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": "false",
                    "slam_params_file": str(package_share / "config" / "slam_toolbox.yaml"),
                }.items(),
            ),
        ]
    )
