from pathlib import Path


def test_adapter_keeps_source_and_uses_separate_output_topic():
    source = Path(__file__).parents[1].joinpath(
        "tmr_local_navigation", "odom_frame_adapter.py"
    ).read_text()
    assert '"/swerve_drive_controller/odom"' in source
    assert '"/navigation/odom"' in source
    assert 'out.child_frame_id = self._child_frame' in source


def test_launch_contains_both_lidar_static_transforms():
    launch = Path(__file__).parents[1].joinpath(
        "launch", "navigation_adapter.launch.py"
    ).read_text()
    assert '"lidar_front"' in launch
    assert '"lidar_rear"' in launch
    assert '"base_link"' in launch


def test_dual_lidar_slam_uses_both_scans_and_navigation_output():
    merger = Path(__file__).parents[1].joinpath(
        "tmr_local_navigation", "dual_laser_merger.py"
    ).read_text()
    assert '"/lidar_front/scan"' in merger
    assert '"/lidar_rear/scan"' in merger
    assert '"/navigation/scan"' in merger
    assert '"footprint_min_x", -0.40' in merger
    assert "if min_x <= bx <= max_x and min_y <= by <= max_y" in merger

    slam = Path(__file__).parents[1].joinpath("config", "slam_toolbox.yaml").read_text()
    assert "scan_topic: /navigation/scan" in slam
    assert "odom_frame: odom" in slam


def test_localization_uses_clean_map_and_omni_amcl():
    launch = Path(__file__).parents[1].joinpath(
        "launch", "localization.launch.py"
    ).read_text()
    params = Path(__file__).parents[1].joinpath("config", "nav2_params.yaml").read_text()
    assert "final_clean_20260828_144206.yaml" in launch
    assert "nav2_amcl::OmniMotionModel" in params
    assert "scan_topic: /navigation/scan" in params
    assert 'footprint: "[[-0.40, -0.29]' in params
