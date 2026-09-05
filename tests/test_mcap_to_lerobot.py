from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from franka_duo_tele_data.mcap_to_lerobot import (
    DerivedFrame,
    FixedRateGate,
    LeRobotV3Writer,
    SyncStats,
    TimedBuffer,
    compose_transform,
    invert_transform,
    load_usd_geometry,
    midpoint_base,
    pose_vector,
    quaternion_to_matrix,
    resize_rgb,
    transform_points,
)
from franka_duo_tele_data.pointcloud import adaptive_voxel_sample, farthest_point_sample
from franka_duo_tele_data.ros_utils import depth_msg_to_meters


def test_fps_is_deterministic_and_preserves_extra_channels() -> None:
    xyz = np.stack(np.meshgrid(np.arange(5), np.arange(4), indexing="ij"), axis=-1).reshape(-1, 2)
    points = np.column_stack((xyz, np.ones(len(xyz)), np.arange(len(xyz), dtype=np.float32)))
    first = farthest_point_sample(points, 8, seed=7, candidate_limit=16)
    second = farthest_point_sample(points, 8, seed=7, candidate_limit=16)
    assert first.shape == (8, 4)
    np.testing.assert_array_equal(first, second)
    assert set(first[:, 3]).issubset(set(points[:, 3]))


def test_transform_composition_inverse_and_pose_vector() -> None:
    rotation = quaternion_to_matrix((0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)))
    first = np.eye(4, dtype=np.float32)
    first[:3, :3] = rotation
    first[:3, 3] = [1.0, 2.0, 3.0]
    second = np.eye(4, dtype=np.float32)
    second[:3, 3] = [0.5, 0.0, 0.0]
    composed = compose_transform(first, second)
    np.testing.assert_allclose(composed[:3, 3], [1.0, 2.5, 3.0], atol=1e-6)
    np.testing.assert_allclose(compose_transform(composed, invert_transform(composed)), np.eye(4), atol=1e-6)
    transformed = transform_points(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), composed)
    np.testing.assert_allclose(transformed[0], [1.0, 2.5, 3.0], atol=1e-6)
    assert pose_vector(composed).shape == (9,)


def test_adaptive_voxel_sample_returns_xyz_only_fixed_size() -> None:
    grid = (
        np.stack(
            np.meshgrid(
                np.linspace(0.4, 1.2, 80),
                np.linspace(-0.3, 0.3, 40),
                np.linspace(-0.3, 0.3, 30),
                indexing="ij",
            ),
            axis=-1,
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )
    sampled = adaptive_voxel_sample(grid, 2048, seed=5)
    assert sampled.shape == (2048, 3)
    assert sampled.dtype == np.float32
    assert np.isfinite(sampled).all()
    assert np.unique(sampled, axis=0).shape[0] == 2048
    assert np.all(sampled[:, 0] >= 0.4)
    assert np.all(sampled[:, 0] <= 1.2)


def test_resize_rgb_is_256_square() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    resized = resize_rgb(image)
    assert resized.shape == (256, 256, 3)


def test_midpoint_base_does_not_average_mirrored_rotations() -> None:
    left = np.eye(4, dtype=np.float32)
    right = np.eye(4, dtype=np.float32)
    left[:3, 3] = [0.0, 0.2, 0.5]
    right[:3, 3] = [0.0, -0.2, 0.5]
    result = midpoint_base(left, right)
    np.testing.assert_allclose(result[:3, 3], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(result[:3, :3], np.eye(3))


def test_nearest_buffer_uses_receipt_for_headerless_message() -> None:
    buffer = TimedBuffer(2)
    value = SimpleNamespace()
    buffer.append(value, 100)
    assert buffer.nearest(105, 5).message is value
    assert buffer.nearest(106, 5) is None


def test_fixed_rate_gate_keeps_source_frames_on_output_grid() -> None:
    gate = FixedRateGate(10)
    assert gate.keep(1_000_000_000)
    assert not gate.keep(1_050_000_000)
    assert gate.keep(1_100_000_000)
    assert not gate.keep(1_150_000_000)
    assert gate.keep(1_200_000_000)


def test_offline_depth_decode_can_preserve_float32_precision() -> None:
    message = SimpleNamespace(
        height=1,
        width=2,
        step=8,
        encoding="32FC1",
        is_bigendian=False,
        data=np.asarray([[0.1234567, 1.2345678]], dtype=np.float32).tobytes(),
    )
    precise = depth_msg_to_meters(message, dtype=np.float32)
    compact = depth_msg_to_meters(message)
    assert precise.dtype == np.float32
    assert compact.dtype == np.float16
    np.testing.assert_array_equal(precise, np.asarray([[0.1234567, 1.2345678]], dtype=np.float32))


def test_writer_emits_video_metadata_for_v3_reader(tmp_path) -> None:
    pytest.importorskip("av")
    pytest.importorskip("pyarrow")
    writer = LeRobotV3Writer(tmp_path, fps=15, task="test task", num_points=2, channels=3)
    pose = np.zeros(18, dtype=np.float32)
    frame = DerivedFrame(
        source_stamp_ns=123,
        source_skew_ns=np.zeros(9, dtype=np.int64),
        head_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_left_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_right_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        point_cloud=np.zeros((2, 3), dtype=np.float32),
        state=np.zeros(16, dtype=np.float32),
        ee_pose=pose,
        gripper=np.zeros(2, dtype=np.float32),
    )
    action = np.concatenate((pose, np.zeros(2, dtype=np.float32)))
    writer.add_frame(frame, action, episode_index=0, frame_index=0)
    writer.finish_episode(0, "episode_000000", SyncStats(frames_written=1))
    writer.finalize()
    info = json.loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.ee_pose"]["shape"] == [18]
    assert info["features"]["action"]["shape"] == [20]
    assert info["features"]["action"]["names"][-2:] == [
        "left_gripper_open_fraction",
        "right_gripper_open_fraction",
    ]
    assert info["features"]["observation.point_cloud"]["names"] == ["x", "y", "z"]
    video_info = info["features"]["observation.images.head"]["info"]
    assert video_info["video.height"] == 4
    assert video_info["video.width"] == 4
    assert video_info["video.channels"] == 3
    assert video_info["video.fps"] == 15
    assert video_info["video.pix_fmt"] == "yuv420p"
    assert video_info["video.is_depth_map"] is False
    import pyarrow.parquet as pq

    episodes = pq.read_table(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert "meta/episodes/chunk_index" in episodes.column_names
    assert "meta/episodes/file_index" in episodes.column_names


def test_usd_geometry_contains_arm_midpoint_and_mount() -> None:
    pytest.importorskip("pxr")
    usd_path = Path("/tmp/franka-usd/mobile_fr3_duo_v0_2.usd")
    if not usd_path.is_file():
        pytest.skip("benchmark USD asset is not available")
    geometry = load_usd_geometry(usd_path)
    np.testing.assert_allclose(geometry.new_base_world[:3, 3], [0.4419, 0.0, 0.500885], atol=2e-5)
    np.testing.assert_allclose(geometry.left_arm_world[:3, 3], [0.4419, 0.05018, 0.500885], atol=2e-5)
    np.testing.assert_allclose(geometry.right_arm_world[:3, 3], [0.4419, -0.05018, 0.500885], atol=2e-5)
    assert geometry.mount_prim.endswith("/head_camera_mounting_point")
    np.testing.assert_allclose(geometry.zed_mount_world[:3, 3], [0.4548, -0.02, 0.8515], atol=2e-5)
