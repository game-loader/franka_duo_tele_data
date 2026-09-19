"""Validate every exported row and video against episode metadata and FK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from .action_spec import rot6d_to_matrix
from .labs_kinematics import URDFFK, pose_vector
from .labs_mcap_to_lerobot import ACTION_NAMES, STATE_NAMES
from .mcap_to_lerobot import ImageStatsAccumulator


def validate(root, config, *, decode_video=True):
    info = json.loads((root / "meta/info.json").read_text())
    manifest = json.loads((root / "meta/conversion_manifest.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["features"]["observation.state"]["names"] == STATE_NAMES
    assert info["features"]["action"]["names"] == ACTION_NAMES
    assert manifest["rotation_representation"] == "rot6d_columns"
    assert info["features"]["observation.state"]["shape"] == [34]
    assert info["features"]["action"]["shape"] == [20]
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    task = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    assert task.index.tolist() == [manifest["task"]]
    assert task["task_index"].tolist() == [0]
    fk = {s: URDFFK(config / f"{s}.urdf", s) for s in ("left", "right")}
    totals = 0
    max_state_error = max_action_error = 0.0
    video_frames = 0
    image_stats = {
        key: ImageStatsAccumulator() for key, spec in info["features"].items() if spec["dtype"] == "video"
    }
    for e in episodes:
        path = root / info["data_path"].format(
            chunk_index=e["data/chunk_index"], file_index=e["data/file_index"]
        )
        table = pq.read_table(path)
        n = len(table)
        assert n == e["length"] and e["dataset_from_index"] == totals and e["dataset_to_index"] == totals + n
        assert e["tasks"] == [manifest["task"]]
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(n, e["episode_index"]))
        np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(totals, totals + n))
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(n))
        np.testing.assert_array_equal(table["task_index"].to_numpy(), np.zeros(n, dtype=int))
        np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(n) / info["fps"], atol=2e-6)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        assert state.shape == (n, 34) and action.shape == (n, 20)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        assert np.isin(state[:, 18:20], [0, 1]).all() and np.isin(action[:, 18:20], [0, 1]).all()
        skew = np.asarray(table["observation.sync_skew_ns"].to_pylist())
        assert np.all(np.abs(skew[:, :2]) <= 45_000_000)
        assert np.all(np.abs(skew[:, 2:6]) <= 50_000_000)
        assert np.all(skew[:, 6:] <= 0) and np.all(skew[:, 6:] >= -100_000_000)
        source = np.asarray(table["observation.source_timestamp_ns"].to_pylist())
        assert np.all(np.diff(source) > 0)
        prov = np.load(root.with_name(root.name + ".work") / e["original_episode"] / "joint_provenance.npz")
        np.testing.assert_array_equal(state[:, 20:], prov["measured"].astype(np.float32))
        np.testing.assert_array_equal(source, prov["timestamps_ns"])
        for side, offset, joint_offset in (("left", 0, 0), ("right", 9, 7)):
            for i in range(n):
                measured = pose_vector(fk[side](prov["measured"][i, joint_offset : joint_offset + 7]))
                target = pose_vector(fk[side](prov["recorded_targets"][i, joint_offset : joint_offset + 7]))
                max_state_error = max(
                    max_state_error,
                    float(np.max(np.abs(state[i, offset : offset + 9] - measured))),
                )
                max_action_error = max(
                    max_action_error,
                    float(np.max(np.abs(action[i, offset : offset + 9] - target))),
                )
            for values in (state, action):
                rotation = rot6d_to_matrix(values[:, offset + 3 : offset + 9])
                np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-5)
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            path = root / info["video_path"].format(
                video_key=key,
                chunk_index=e[f"videos/{key}/chunk_index"],
                file_index=e[f"videos/{key}/file_index"],
            )
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                assert (stream.width, stream.height) == (640, 480)
                assert float(stream.average_rate) == info["fps"]
                assert abs(e[f"videos/{key}/from_timestamp"]) < 1e-9
                assert abs(e[f"videos/{key}/to_timestamp"] - n / info["fps"]) < 1e-9
                if decode_video:
                    count = 0
                    for frame in container.decode(stream):
                        image_stats[key].update(frame.to_ndarray(format="rgb24"))
                        count += 1
                    assert count == n, (path, count, n)
                    video_frames += count
                else:
                    assert stream.frames == n
        totals += n
        print(
            json.dumps({"validated_episode": e["episode_index"], "frames": n}),
            flush=True,
        )
    assert totals == info["total_frames"] and len(episodes) == info["total_episodes"]
    assert max_state_error < 1e-6 and max_action_error < 1e-6
    stats = json.loads((root / "meta/stats.json").read_text())
    for key, v in stats.items():
        assert v["count"] == [totals], key
        assert np.isfinite(v["mean"]).all() and np.isfinite(v["std"]).all()
    if decode_video:
        stats.update({key: value.finish() for key, value in image_stats.items()})
        (root / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    result = {
        "episodes": len(episodes),
        "frames": totals,
        "decoded_video_frames": video_frames,
        "max_state_fk_error": max_state_error,
        "max_action_fk_error": max_action_error,
        "task": manifest["task"],
        "status": "passed",
        "image_statistics": "per-channel decoded RGB pixels, every eighth row and column, every frame"
        if decode_video
        else "not recomputed",
    }
    (root / "meta/validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--metadata-only-videos", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(validate(args.dataset, args.config, decode_video=not args.metadata_only_videos)),
        flush=True,
    )


if __name__ == "__main__":
    main()
