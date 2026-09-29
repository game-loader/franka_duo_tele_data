import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from franka_duo_tele_data.labs_joint_dataset import JOINT_NAMES, convert_dataset, joint16
from franka_duo_tele_data.labs_mcap_to_lerobot import ACTION_NAMES, STATE_NAMES


def source_dataset(tmp_path):
    root = tmp_path / "source"
    work = tmp_path / "source.work/episode-a"
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "videos/camera/chunk-000").mkdir(parents=True)
    work.mkdir(parents=True)
    measured = np.arange(42, dtype=np.float64).reshape(3, 14) / 30
    targets = measured + 0.125
    stamps = np.array([100, 200, 300], dtype=np.int64)
    state = np.zeros((3, 34), dtype=np.float32)
    state[:, 20:] = measured
    state[:, 18:20] = [[0, 1], [1, 0], [0, 0]]
    action = np.zeros((3, 20), dtype=np.float32)
    action[:, 18:20] = [[1, 0], [0, 1], [1, 1]]
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 34)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 20)),
            "episode_index": [0, 0, 0],
            "index": [0, 1, 2],
            "frame_index": [0, 1, 2],
            "observation.source_timestamp_ns": stamps,
            "observation.sync_skew_ns": [[0] * 6 + [-5] * 4] * 3,
        }
    )
    pq.write_table(table, root / "data/chunk-000/file-000.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "length": 3,
                    "dataset_from_index": 0,
                    "dataset_to_index": 3,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "original_episode": "episode-a",
                    "videos/camera/chunk_index": 0,
                    "videos/camera/file_index": 0,
                }
            ]
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    np.savez(work / "joint_provenance.npz", measured=measured, recorded_targets=targets, timestamps_ns=stamps)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_frames": 3,
        "total_episodes": 1,
        "total_videos": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"names": STATE_NAMES, "shape": [34], "dtype": "float32"},
            "action": {"names": ACTION_NAMES, "shape": [20], "dtype": "float32"},
            "camera": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    manifest = {
        "schema": "labs_fr3_link8_columns_state34_action20_v1",
        "task": "test",
        "gripper": {},
        "sync": {},
    }
    for filename, content in (
        ("info.json", info),
        ("conversion_manifest.json", manifest),
        ("stats.json", {"action": {"mean": [99] * 20}, "camera": {"mean": [0.5]}}),
        ("validation.json", {"source_only": True}),
    ):
        (root / "meta" / filename).write_text(json.dumps(content))
    (root / "meta/tasks.parquet").write_bytes(b"unchanged task")
    (root / "videos/camera/chunk-000/file-000.mp4").write_bytes(b"unchanged video")
    return root, work, measured, targets


def test_layout_retains_joint7_and_uses_per_arm_gripper():
    result = joint16(np.arange(14)[None, :], [[1, 0]])
    np.testing.assert_array_equal(result[0], [0, 1, 2, 3, 4, 5, 6, 1, 7, 8, 9, 10, 11, 12, 13, 0])
    assert result.dtype == np.float32
    assert JOINT_NAMES[6:9] == ["left_joint7_position", "left_gripper_open", "right_joint1_position"]
    with pytest.raises(ValueError, match="finite"):
        joint16(np.full((1, 14), np.nan), [[1, 0]])
    with pytest.raises(ValueError, match="binary"):
        joint16(np.zeros((1, 14)), [[0.5, 0]])


def test_dataset_uses_targets_preserves_source_and_rebuilds_both_statistics(tmp_path):
    root, work, measured, targets = source_dataset(tmp_path)
    output = tmp_path / "joint16"
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = convert_dataset(root, output)
    assert report["frames"] == 3 and report["rows_with_measured_target_difference_gt_1e_6_rad"] == 3
    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    state, action = [np.asarray(table[k].to_pylist()) for k in ("observation.state", "action")]
    np.testing.assert_array_equal(state[:, :7], measured[:, :7].astype(np.float32))
    np.testing.assert_array_equal(state[:, 8:15], measured[:, 7:].astype(np.float32))
    np.testing.assert_array_equal(action[:, :7], targets[:, :7].astype(np.float32))
    np.testing.assert_array_equal(action[:, 8:15], targets[:, 7:].astype(np.float32))
    np.testing.assert_array_equal(state[:, [7, 15]], [[0, 1], [1, 0], [0, 0]])
    np.testing.assert_array_equal(action[:, [7, 15]], [[1, 0], [0, 1], [1, 1]])
    stats = json.loads((output / "meta/stats.json").read_text())
    episode = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    for key, values in (("observation.state", state), ("action", action)):
        np.testing.assert_allclose(stats[key]["mean"], values.mean(axis=0))
        np.testing.assert_allclose(stats[key]["std"], values.std(axis=0))
        assert stats[key]["count"] == [3]
        assert episode[f"stats/{key}/mean"] == stats[key]["mean"]
    assert stats["camera"] == {"mean": [0.5]}
    assert not (output / "meta/validation.json").exists()
    assert (output / "meta/source_metadata/validation.json").exists()
    assert (output / "meta/joint_provenance/episode_000000.npz").read_bytes() == (
        work / "joint_provenance.npz"
    ).read_bytes()
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        convert_dataset(root, output)


@pytest.mark.parametrize("corruption", ["timestamps", "measured", "missing_targets", "future_target"])
def test_rejects_unaligned_or_missing_provenance(tmp_path, corruption):
    root, work, measured, targets = source_dataset(tmp_path)
    content = {"measured": measured, "recorded_targets": targets, "timestamps_ns": [100, 200, 300]}
    if corruption == "timestamps":
        content["timestamps_ns"] = [101, 200, 300]
    elif corruption == "measured":
        content["measured"] = measured + 1
    elif corruption == "missing_targets":
        del content["recorded_targets"]
    else:
        path = root / "data/chunk-000/file-000.parquet"
        table = pq.read_table(path)
        table = table.set_column(
            table.schema.get_field_index("observation.sync_skew_ns"),
            "observation.sync_skew_ns",
            pa.array([[0] * 6 + [1] * 4] * 3),
        )
        pq.write_table(table, path)
    np.savez(work / "joint_provenance.npz", **content)
    with pytest.raises((AssertionError, KeyError, ValueError)):
        convert_dataset(root, tmp_path / "joint16")
    assert not (tmp_path / "joint16").exists()
