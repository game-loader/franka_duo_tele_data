import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from franka_duo_tele_data.labs_joint_dataset import digest
from franka_duo_tele_data.labs_kinematics import URDFFK
from franka_duo_tele_data.labs_mcap_to_lerobot import ACTION_NAMES, SCHEMA, STATE_NAMES
from franka_duo_tele_data.labs_next_state_dataset import convert_dataset, measured_pose20
from franka_duo_tele_data.mcap_to_lerobot import VideoWriter

CONFIG = Path(__file__).resolve().parents[1] / "configs/labs_fr3_31"


def make_source(tmp_path):
    root = tmp_path / "source"
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    fk = {s: URDFFK(CONFIG / f"{s}.urdf", s) for s in ("left", "right")}
    episodes, all_poses = [], []
    start = 0
    for index, n in enumerate((3, 2)):
        # Two distinct episodes exercise boundary handling and global reindexing.
        joints = np.tile([0, -0.2, 0, -1.5, 0, 1.2, 0] * 2, (n, 1))
        joints[:, 0] += np.arange(n) * 0.1 + index
        grippers = np.array([[i % 2, (i + 1) % 2] for i in range(n)])
        poses = measured_pose20(joints, grippers, fk)
        all_poses.append(poses)
        stamps = 1_000_000_000 + index * 10_000_000_000 + np.arange(n) * 33_333_333
        table = pa.table(
            {
                "observation.state": pa.array(
                    np.column_stack([poses, joints]).tolist(), type=pa.list_(pa.float32(), 34)
                ),
                "action": pa.array(np.full((n, 20), 99).tolist(), type=pa.list_(pa.float32(), 20)),
                "index": np.arange(start, start + n),
                "episode_index": [index] * n,
                "frame_index": np.arange(n),
                "task_index": [0] * n,
                "timestamp": np.arange(n, dtype=np.float32) / 30,
                "observation.source_timestamp_ns": stamps,
                "observation.sync_skew_ns": [[1, 2, 3, 4, 5, 6, -7, -8, -9, -10]] * n,
            }
        )
        pq.write_table(table, root / f"data/chunk-000/file-{index:03d}.parquet")
        work = tmp_path / f"source.work/episode-{index}"
        work.mkdir(parents=True)
        np.savez(
            work / "joint_provenance.npz", measured=joints, recorded_targets=joints + 10, timestamps_ns=stamps
        )
        video = VideoWriter(
            root / f"videos/observation.images.head/chunk-000/file-{index:03d}.mp4", 16, 16, 30
        )
        for i in range(n):
            # Only discarded terminal observations contain bright pixels.
            video.write(np.full((16, 16, 3), 255 if i == n - 1 else 0, dtype=np.uint8))
        video.close()
        episodes.append(
            {
                "episode_index": index,
                "length": n,
                "dataset_from_index": start,
                "dataset_to_index": start + n,
                "original_episode": f"episode-{index}",
                "data/chunk_index": 0,
                "data/file_index": index,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "tasks": ["test"],
                "sync_stats": "old source stats",
                "videos/observation.images.head/chunk_index": 0,
                "videos/observation.images.head/file_index": index,
                "videos/observation.images.head/from_timestamp": 0.0,
                "videos/observation.images.head/to_timestamp": n / 30,
            }
        )
        start += n
    features = {
        k: {"dtype": "int64", "shape": [1], "names": None}
        for k in ("index", "episode_index", "frame_index", "task_index", "observation.source_timestamp_ns")
    }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "observation.state": {"dtype": "float32", "shape": [34], "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": [20], "names": ACTION_NAMES},
            "observation.sync_skew_ns": {"dtype": "int64", "shape": [10], "names": None},
            "observation.images.head": {"dtype": "video", "shape": [16, 16, 3]},
        }
    )
    info = {
        "features": features,
        "total_episodes": 2,
        "total_frames": 5,
        "total_videos": 2,
        "codebase_version": "v3.0",
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    manifest = {
        "schema": SCHEMA,
        "urdf_sha256": {s: digest(CONFIG / f"{s}.urdf") for s in fk},
        "frames": {s: f"{s}_fr3_link0" for s in fk},
        "tips": {s: f"{s}_fr3_link8" for s in fk},
        "gripper": {"encoding": "binary", "open": 1, "closed": 0},
        "task": "test",
    }
    for name, content in (("info.json", info), ("conversion_manifest.json", manifest), ("stats.json", {})):
        (root / "meta" / name).write_text(json.dumps(content))
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "meta/tasks.parquet").write_bytes(b"unchanged task")
    return root, all_poses


def test_next_measured_labels_no_boundary_crossing_and_retained_image_stats(tmp_path):
    source, poses = make_source(tmp_path)
    hashes = {p: digest(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "next"
    report = convert_dataset(source, output, CONFIG)
    assert report["frames"] == 3 and report["dropped_terminal_observations"] == 2
    assert report["cross_episode_transitions"] == 0
    assert report["physical_video_frames_decoded"] == 5
    assert report["retained_image_frames_in_statistics"] == 3
    offset = 0
    for i, original in enumerate(poses):
        t = pq.read_table(output / f"data/chunk-000/file-{i:03d}.parquet")
        n = len(original) - 1
        np.testing.assert_array_equal(t["observation.state"].to_pylist(), original[:-1])
        np.testing.assert_array_equal(t["action"].to_pylist(), original[1:])
        np.testing.assert_array_equal(t["index"].to_numpy(), np.arange(offset, offset + n))
        np.testing.assert_array_equal(t["action_source_frame_index"].to_numpy(), np.arange(1, n + 1))
        assert np.all(
            t["action_source_timestamp_ns"].to_numpy() > t["observation.source_timestamp_ns"].to_numpy()
        )
        assert np.asarray(t["observation.sync_skew_ns"].to_pylist()).shape == (n, 6)
        np.testing.assert_array_equal(t["action_source_sync_skew_ns"].to_pylist(), [[3, 4, 5, 6]] * n)
        offset += n
    stats = json.loads((output / "meta/stats.json").read_text())
    assert all(v["count"] == [3] for v in stats.values())
    assert np.max(stats["observation.images.head"]["max"]) == 0
    np.testing.assert_allclose(
        stats["action"]["mean"], np.concatenate([p[1:] for p in poses]).astype(np.float64).mean(0)
    )
    entries = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [e["length"] for e in entries] == [2, 1]
    assert entries[1]["dataset_from_index"] == 2
    assert entries[0]["videos/observation.images.head/to_timestamp"] == 2 / 30
    manifest = json.loads((output / "meta/conversion_manifest.json").read_text())
    assert manifest["is_recorded_command_action"] is False
    assert {p: digest(p) for p in hashes} == hashes
    with pytest.raises(FileExistsError):
        convert_dataset(source, output, CONFIG)


def test_reject_wrong_fk_or_misaligned_provenance(tmp_path):
    source, _ = make_source(tmp_path)
    prov = tmp_path / "source.work/episode-0/joint_provenance.npz"
    with np.load(prov) as p:
        values = {k: p[k].copy() for k in p.files}
    values["timestamps_ns"][0] += 1
    np.savez(prov, **values)
    with pytest.raises(AssertionError):
        convert_dataset(source, tmp_path / "bad", CONFIG)
    assert not (tmp_path / "bad").exists()


def test_no_terminal_self_action_for_single_observation():
    with pytest.raises(ValueError, match="at least two"):
        measured_pose20(np.zeros((1, 14)), [[0, 1]], {})
