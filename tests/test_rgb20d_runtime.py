from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from franka_duo_tele_data.action_spec import matrix_to_rot6d
from franka_duo_tele_data.cartesian_chunk import ActionChunk, CartesianChunkBuffer, check_tracking
from franka_duo_tele_data.mcap_to_lerobot import compose_transform, pose_to_transform, pose_vector
from franka_duo_tele_data.replay_rgb20d import optional_recorder
from franka_duo_tele_data.rgb20d_io import (
    CAMERAS,
    RGB20DCache,
    RGB20DContract,
    RGB20DReader,
    RobotStateReader,
)


@pytest.fixture
def contract(tmp_path):
    left = np.eye(4)
    left[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    left[1, 3] = 0.05
    right = np.eye(4)
    right[:3, :3] = left[:3, :3].T
    right[1, 3] = -0.05
    (tmp_path / "meta").mkdir()
    (tmp_path / "franka_duo_extras").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.state": {"shape": [20]},
                    "action": {"shape": [20]},
                    **{
                        f"observation.images.{key}": {"dtype": "video", "shape": [4, 6, 3]} for key in CAMERAS
                    },
                },
            }
        )
    )
    (tmp_path / "franka_duo_extras/derived_manifest.json").write_text(
        json.dumps(
            {
                "schema": "franka_duo_tele_data.mcap_to_lerobot.rgb20d.v1",
                "coordinate_transforms": {
                    "T_newbase_from_left_arm_base": left.tolist(),
                    "T_newbase_from_right_arm_base": right.tolist(),
                },
                "action_spec": {"dimension": 20, "ee_dimension": 9, "ee_rotation": "rot6d_rows"},
                "gripper_calibration": {
                    "closed_position": 0.8,
                    "open_position": 0,
                    "threshold": 0.5,
                    "encoding": "0=closed, 1=open",
                },
                "sync": {"rgb_tolerance_ms": 45, "state_tolerance_ms": 50},
            }
        )
    )
    return RGB20DContract(tmp_path)


def pose():
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.4, y=0.1, z=0.3), orientation=SimpleNamespace(x=0, y=0, z=0, w=1)
        )
    )


def actions(count=8):
    value = np.zeros((count, 20), dtype=np.float32)
    value[:, 0] = np.arange(count) * 0.001
    value[:, 3:9] = matrix_to_rot6d(np.eye(3))
    value[:, 12:18] = matrix_to_rot6d(np.eye(3))
    value[:, 18] = np.arange(count) % 2
    return value


def test_state_matches_conversion_and_arm_roundtrip(contract):
    selected = {
        "left_pose": pose(),
        "right_pose": pose(),
        "left_gripper": SimpleNamespace(position=[0.8]),
        "right_gripper": SimpleNamespace(position=[0.0]),
    }
    state = contract.state_from_messages(selected)
    expected = np.concatenate(
        [
            *(
                pose_vector(
                    compose_transform(
                        contract.transforms[side], pose_to_transform(selected[f"{side}_pose"].pose)
                    )
                )
                for side in ("left", "right")
            ),
            [0, 1],
        ]
    )
    np.testing.assert_array_equal(state, expected)
    assert state.shape == (20,)
    arm = contract.action_spec.to_link0_action(state)
    for offset in (0, 9):
        np.testing.assert_allclose(arm[offset : offset + 3], [0.4, 0.1, 0.3], atol=1e-6)
        np.testing.assert_allclose(arm[offset + 3 : offset + 9], matrix_to_rot6d(np.eye(3)), atol=1e-6)
    np.testing.assert_array_equal(arm[18:], [0, 1])


def test_late_chunk_trims_consumed_steps_and_preserves_binary(contract):
    buffer = CartesianChunkBuffer(contract)
    sequence = actions()
    buffer.submit(ActionChunk(0, 100, sequence[:4]))
    np.testing.assert_array_equal(buffer.pop(), sequence[0])
    np.testing.assert_array_equal(buffer.pop(), sequence[1])
    assert buffer.submit(ActionChunk(1, 101, sequence[1:6])) == 1
    for expected in sequence[2:6]:
        np.testing.assert_array_equal(buffer.pop(), expected)
    with pytest.raises(TimeoutError, match="underrun"):
        buffer.pop()


def test_invalid_chunk_does_not_replace_valid_buffer(contract):
    buffer = CartesianChunkBuffer(contract)
    sequence = actions()
    buffer.submit(ActionChunk(0, 100, sequence[:4]))
    buffer.pop()
    for invalid in (float("nan"), 0.5, 2.0):
        bad = sequence[1:4].copy()
        bad[-1, 18] = invalid
        with pytest.raises(ValueError):
            buffer.submit(ActionChunk(1, 101, bad))
    np.testing.assert_array_equal(buffer.pop(), sequence[1])


def test_expired_out_of_order_and_jump_chunks_are_rejected(contract):
    buffer = CartesianChunkBuffer(contract)
    sequence = actions()
    buffer.submit(ActionChunk(0, 100, sequence[:4]))
    for _ in range(4):
        buffer.pop()
    with pytest.raises(TimeoutError, match="expired"):
        buffer.submit(ActionChunk(1, 101, sequence[:2]))
    with pytest.raises(ValueError, match="out-of-order"):
        buffer.submit(ActionChunk(0, 100, sequence))
    with pytest.raises(ValueError, match="after"):
        buffer.submit(ActionChunk(5, 101, sequence))
    bad = sequence[4:].copy()
    bad[0, 0] = 1.0
    with pytest.raises(ValueError, match="jump"):
        buffer.submit(ActionChunk(4, 102, bad))


def test_start_guard_uses_both_translation_and_rotation():
    first, second = actions(2)
    check_tracking(first, second, max_m=0.03, max_rad=0.2)
    second[12:18] = matrix_to_rot6d(np.diag([-1, -1, 1]))
    with pytest.raises(ValueError, match="too far"):
        check_tracking(first, second, max_m=0.03, max_rad=0.2)


def test_rgb_reader_needs_no_joints_or_depth_and_rejects_stale(contract):
    cache = RGB20DCache()
    stamp = time.time_ns()
    header = SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9))
    for key in CAMERAS:
        # BGR payload lets the test check channel conversion as well as HWC.
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[..., 0] = 255
        cache.store_image(
            key,
            SimpleNamespace(
                header=header,
                width=6,
                height=4,
                encoding="bgr8",
                step=18,
                is_bigendian=False,
                data=image.tobytes(),
            ),
        )
    for side in ("left", "right"):
        msg = pose()
        msg.header = header
        getattr(cache, f"store_{side}_pose")(msg)
        # Headerless messages must get ROS-compatible receipt timestamps.
        getattr(cache, f"store_{side}_gripper_states")(SimpleNamespace(position=[0.8]))
    reader = RGB20DReader(cache, contract)
    observation = reader.next(timeout_s=0.5)
    assert observation.state.shape == (20,)
    assert len(observation.source_stamps_ns) == 7
    np.testing.assert_array_equal(observation.images["head"][0, 0], [0, 0, 255])
    assert set(observation.policy_input()) == {
        "observation.state",
        *(f"observation.images.camera{i}" for i in (1, 2, 3)),
    }
    with pytest.raises(TimeoutError):
        reader.next(timeout_s=0.02)
    old = time.monotonic_ns() - 1_000_000_000
    for collection in (
        *cache.images.values(),
        cache.left_pose,
        cache.right_pose,
        cache.left_gripper_states,
        cache.right_gripper_states,
    ):
        collection[0] = type(collection[0])(collection[0].message, old, collection[0].stamp_ns)
    with pytest.raises(TimeoutError):
        RGB20DReader(cache, contract).next(timeout_s=0.02)


def test_improper_geometry_fails_before_live_input(contract):
    path = contract.root / "franka_duo_extras/derived_manifest.json"
    document = json.loads(path.read_text())
    document["coordinate_transforms"]["T_newbase_from_left_arm_base"][2][2] = -1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="right handed"):
        RGB20DContract(contract.root)


def test_live_inference_does_not_require_recorder_config_or_rosbag():
    assert optional_recorder(SimpleNamespace(record_mcap=False), {}) is None


def test_recorded_action_feedback_needs_real_poses_and_grippers_without_images(contract):
    cache = RGB20DCache()
    stamp = time.time_ns()
    header = SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9))
    for side in ("left", "right"):
        msg = pose()
        msg.header = header
        getattr(cache, f"store_{side}_pose")(msg)
        getattr(cache, f"store_{side}_gripper_states")(SimpleNamespace(position=[0.0]))
    feedback = RobotStateReader(cache, contract).next()
    assert feedback.images == {}
    assert feedback.state.shape == (20,)
    np.testing.assert_array_equal(feedback.state[18:], [1, 1])
    cache.right_gripper_states.clear()
    with pytest.raises(TimeoutError, match="paired"):
        RobotStateReader(cache, contract).next(timeout_s=0.01)
