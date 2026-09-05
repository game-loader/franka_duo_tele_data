# Local TMR navigation adapters

This package is intentionally local-only. It does not alter the TMR host or
replace any remote topic.

It adapts:

- `/swerve_drive_controller/odom` -> `/navigation/odom`
- `frame_id=world`, empty `child_frame_id` -> `odom` -> `base_link`
- broadcasts the dynamic `odom -> base_link` transform
- broadcasts static `base_link -> lidar_front` and `base_link -> lidar_rear`
  transforms from the mobile FR3 URDF

Build and run from a ROS 2 Humble environment:

```bash
cd local_navigation
colcon build --symlink-install
source install/setup.bash
ros2 launch tmr_local_navigation navigation_adapter.launch.py
```

Then configure SLAM Toolbox/Nav2 to consume `/navigation/odom` and either
`/lidar_front/scan` (fastest first test) or a locally merged scan topic.

The source odometry remains available unchanged for rollback and comparison.

On the TMR host, keep this as a separate workspace (for example
`~/tmr_navigation`) and use the directory-scoped management scripts:

```bash
./start_adapter.sh
./status_adapter.sh
./stop_adapter.sh
```

They manage only the adapter's own process group and do not stop the existing
TMR, SICK or ZED processes.

To merge both physical lidars and start SLAM Toolbox:

```bash
./start_slam.sh
./status_slam.sh
./stop_slam.sh
```

The merger publishes `/navigation/scan` in `base_link`; SLAM Toolbox consumes
that topic and publishes `/map` plus `map -> odom`.

The merger also rejects endpoints inside the TMR footprint measured from the
1:1 collision mesh (`x=-0.40..0.40 m`, `y=-0.29..0.29 m`) so the front/rear scanners cannot draw
the chassis into the occupancy map. The original `/lidar_*` topics remain
unchanged.

After mapping, start static-map localization using the cleaned map:

```bash
./start_localization.sh
./status_localization.sh
./stop_localization.sh
```

AMCL requires an initial estimate through RViz **2D Pose Estimate** or the
`/initialpose` topic. Physical velocity output remains disconnected while
localization and costmaps are validated.
