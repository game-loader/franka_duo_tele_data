"""Explicitly opt-in ROS wire test, in a domain with no robot participants."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from franka_duo_tele_data.mcap_to_lerobot import pose_to_transform, pose_vector
from franka_duo_tele_data.replay_rgb20d import DatasetEpisodePolicy
from franka_duo_tele_data.rgb20d_io import RGB20DContract


@pytest.mark.skipif(os.environ.get("FRANKA_RGB20D_ROS_TEST") != "1", reason="requires isolated ROS domain")
def test_recorded_actions_through_compiled_link0_relay():
    # This test cannot be redirected to the production ROS graph.
    assert os.environ.get("ROS_DOMAIN_ID") == "221"
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32, Float32MultiArray

    contract = RGB20DContract(Path(os.environ["FRANKA_RGB20D_TEST_DATASET"]))
    policy = DatasetEpisodePolicy(contract, 0)
    executable = Path("site/install/lib/franka_duo_policy_control/policy_action_relay").resolve()
    process = subprocess.Popen(
        [
            str(executable),
            "--ros-args",
            "-p",
            "action_frame:=link0",
            "-p",
            "enable_robot:=true",
            "-p",
            "enable_gripper:=true",
            "-p",
            "input_topic:=/rgb20d_wire_test/action",
            "-p",
            "left_pose_topic:=/rgb20d_wire_test/left_pose",
            "-p",
            "right_pose_topic:=/rgb20d_wire_test/right_pose",
            "-p",
            "left_gripper_topic:=/rgb20d_wire_test/left_gripper",
            "-p",
            "right_gripper_topic:=/rgb20d_wire_test/right_gripper",
        ]
    )
    rclpy.init()
    node = rclpy.create_node("rgb20d_wire_test")
    received = {}
    try:
        publisher = node.create_publisher(Float32MultiArray, "/rgb20d_wire_test/action", 1)
        for side in ("left", "right"):
            node.create_subscription(
                PoseStamped,
                f"/rgb20d_wire_test/{side}_pose",
                lambda msg, key=f"{side}_pose": received.__setitem__(key, msg),
                10,
            )
            node.create_subscription(
                Float32,
                f"/rgb20d_wire_test/{side}_gripper",
                lambda msg, key=f"{side}_gripper": received.__setitem__(key, msg),
                10,
            )
        deadline = time.monotonic() + 5
        while publisher.get_subscription_count() != 1:
            assert time.monotonic() < deadline, "relay discovery timed out"
            rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.2)
        # More than one chunk including a changing gripper state. Commands
        # are recorded dataset targets; no simulated dynamics or hardware.
        selected = list(policy.actions[:65])
        transition = next(
            (row for row in policy.actions if not np.array_equal(row[18:], selected[0][18:])), None
        )
        if transition is not None:
            selected.append(transition)
        for row in selected:
            received.clear()
            link0 = contract.action_spec.to_link0_action(row)
            publisher.publish(Float32MultiArray(data=link0.tolist()))
            deadline = time.monotonic() + 2
            while len(received) != 4:
                assert time.monotonic() < deadline, "relay output timed out"
                rclpy.spin_once(node, timeout_sec=0.01)
            for offset, side, gripper_index in ((0, "left", 18), (9, "right", 19)):
                message = received[f"{side}_pose"]
                assert message.header.frame_id == f"{side}_fr3v2_link0"
                np.testing.assert_allclose(
                    pose_vector(pose_to_transform(message.pose)), link0[offset : offset + 9], atol=1e-6
                )
                assert received[f"{side}_gripper"].data == row[gripper_index]
    finally:
        node.destroy_node()
        rclpy.shutdown()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
