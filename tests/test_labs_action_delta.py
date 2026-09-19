import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from franka_duo_tele_data.action_spec import matrix_to_rot6d, rot6d_to_matrix
from franka_duo_tele_data.labs_action_delta import (
    CONVENTION,
    absolute20_to_delta14,
    convert_dataset,
    delta14_to_absolute20,
)
from franka_duo_tele_data.labs_mcap_to_lerobot import ACTION_NAMES, STATE_NAMES


def poses(n=3):
    state = np.zeros((n, 34), dtype=np.float32)
    state[:, 3:9] = matrix_to_rot6d(Rotation.from_euler("xyz", [0.4, -0.7, 0.9]).as_matrix())
    state[:, 12:18] = matrix_to_rot6d(Rotation.from_euler("xyz", [-0.6, 0.5, -0.2]).as_matrix())
    state[:, :3] = [0.3, 0.1, 0.4]
    state[:, 9:12] = [0.2, -0.1, 0.5]
    state[:, 20:] = np.arange(14) / 20
    action = state[:, :20].copy()
    action[:, :3] += [0.012, -0.02, 0.03]
    action[:, 9:12] += [-0.01, 0.015, -0.025]
    for offset in (0, 9):
        reference = rot6d_to_matrix(state[:, offset + 3 : offset + 9])
        increments = Rotation.from_rotvec(np.tile([0.08, -0.12, 0.04], (n, 1))).as_matrix()
        action[:, offset + 3 : offset + 9] = matrix_to_rot6d(increments @ reference)
    action[:, 18:20] = [0.37, 1]
    return state, action


def test_delta_link0_sign_order_and_no_gripper_difference():
    state, action = poses()
    delta = absolute20_to_delta14(action, state)
    np.testing.assert_allclose(delta[:, :3], np.tile([0.012, -0.02, 0.03], (3, 1)), atol=3e-8)
    np.testing.assert_allclose(delta[:, 3:6], np.tile([0.08, -0.12, 0.04], (3, 1)), atol=2e-7)
    np.testing.assert_allclose(delta[:, 6:9], np.tile([-0.01, 0.015, -0.025], (3, 1)), atol=3e-8)
    np.testing.assert_array_equal(delta[:, 12:14], action[:, 18:20])
    np.testing.assert_allclose(delta14_to_absolute20(delta, state), action, atol=2e-7)
    # Single-row and arbitrary leading batch dimensions use the same contract.
    np.testing.assert_allclose(absolute20_to_delta14(action[0], state[0]), delta[0])
    np.testing.assert_allclose(absolute20_to_delta14(action[None], state[None]), delta[None])


@pytest.mark.parametrize("angle", [0.0, 1e-9, np.pi - 1e-6, np.pi, np.pi + 1e-6])
def test_roundtrip_at_small_angle_and_pi_branch(angle):
    state, action = poses(1)
    axis = np.array([1, -2, 3]) / np.sqrt(14)
    for offset in (0, 9):
        reference = rot6d_to_matrix(state[:, offset + 3 : offset + 9])
        action[:, offset + 3 : offset + 9] = matrix_to_rot6d(
            Rotation.from_rotvec(axis * angle).as_matrix() @ reference
        )
    restored = delta14_to_absolute20(absolute20_to_delta14(action, state), state)
    np.testing.assert_allclose(restored, action, atol=3e-7)


def test_invalid_dimensions_and_nonfinite_values():
    state, action = poses()
    with pytest.raises(ValueError, match="batch"):
        absolute20_to_delta14(action, state[:1])
    with pytest.raises(ValueError, match="finite"):
        absolute20_to_delta14(action * np.nan, state)
    bad = action.copy()
    bad[:, 3:9] = 0
    with pytest.raises(ValueError, match="degenerate"):
        absolute20_to_delta14(bad, state)


def test_dataset_roundtrip_preserves_state_videos_source_and_recomputes_stats(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, delta, restored = [tmp_path / name for name in ("source", "delta", "restored")]
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    (source / "videos").mkdir()
    state, action = poses()
    # Deliberate discontinuity at episode boundary must not affect same-row deltas.
    state[-1, :3] += 10
    action[-1, :3] += 10
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 34)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 20)),
            "episode_index": [0, 0, 1],
            "timestamp": [0.0, 1 / 30, 0.0],
        }
    )
    path = "data/chunk-000/file-000.parquet"
    pq.write_table(table, source / path)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_frames": 3,
        "total_episodes": 2,
        "features": {
            "observation.state": {"names": STATE_NAMES, "shape": [34]},
            "action": {"names": ACTION_NAMES, "shape": [20]},
        },
    }
    manifest = {
        "frames": {"left": "left_fr3_link0", "right": "right_fr3_link0"},
        "tips": {"left": "left_fr3_link8", "right": "right_fr3_link8"},
        "rotation_representation": "rot6d_columns",
        "task": "test",
    }
    for name, content in (
        ("info.json", info),
        ("conversion_manifest.json", manifest),
        ("stats.json", {"action": {"mean": [99] * 20}}),
        ("validation.json", {"source_only": True}),
    ):
        (source / "meta" / name).write_text(json.dumps(content))
    (source / "meta/tasks.parquet").write_bytes(b"unchanged tasks")
    (source / "videos/camera.mp4").write_bytes(b"unchanged video")
    source_bytes = (source / path).read_bytes()
    forward = convert_dataset(source, delta, "to-delta")
    assert forward["frames"] == 3 and forward["episodes"] == 2
    assert not (delta / "meta/validation.json").exists()
    assert (delta / "meta/source_metadata/validation.json").exists()
    assert (delta / "videos/camera.mp4").read_bytes() == b"unchanged video"
    stats = json.loads((delta / "meta/stats.json").read_text())
    assert len(stats["action"]["mean"]) == 14 and stats["action"]["count"] == [3]
    convert_dataset(delta, restored, "to-absolute")
    recovered = pq.read_table(restored / path)
    np.testing.assert_allclose(np.array(recovered["action"].to_pylist()), action, atol=3e-7)
    assert recovered["observation.state"].equals(table["observation.state"])
    assert (source / path).read_bytes() == source_bytes
    wrong = json.loads((delta / "meta/conversion_manifest.json").read_text())
    wrong["delta_convention"] = {**CONVENTION, "reference": "previous action"}
    (delta / "meta/conversion_manifest.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="convention"):
        convert_dataset(delta, tmp_path / "wrong", "to-absolute")
