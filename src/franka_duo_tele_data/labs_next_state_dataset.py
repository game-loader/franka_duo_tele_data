"""Versioned offline labels: measured pose20 at t -> measured pose20 at t+1.

Explicitly requested next-observation supervision, NOT recorded command labels.
Each episode loses its final observation row. Original videos retain an unused
terminal frame outside the new episode range, preserving encoded RGB exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .labs_joint_dataset import digest, statistics
from .labs_kinematics import URDFFK, pose_vector
from .labs_mcap_to_lerobot import ACTION_NAMES, SCHEMA as SOURCE_SCHEMA, STATE_NAMES
from .mcap_to_lerobot import ImageStatsAccumulator

SCHEMA = "labs_fr3_link8_columns_state20_next_measured_action20_v1"
ACTION_SYNC_NAMES = ["left_q", "right_q", "left_gripper", "right_gripper"]


def measured_pose20(joints, grippers, fk):
    joints, grippers = np.asarray(joints), np.asarray(grippers)
    if joints.ndim != 2 or joints.shape[1] != 14 or grippers.shape != (len(joints), 2):
        raise ValueError("Expected measured joints (N,14) and grippers (N,2)")
    if len(joints) < 2 or not np.isfinite(joints).all() or not np.isin(grippers, [0, 1]).all():
        raise ValueError("Need at least two finite observations with binary grippers")
    result = np.empty((len(joints), 20), dtype=np.float32)
    for i, q in enumerate(joints):
        result[i, :9] = pose_vector(fk["left"](q[:7]))
        result[i, 9:18] = pose_vector(fk["right"](q[7:]))
    result[:, 18:20] = grippers
    return result


def copy_video_and_statistics(source, destination, source_length, fps, shape):
    """Preserve the bitstream; calculate RGB statistics only for retained rows."""
    import av

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    sha = digest(source)
    if digest(destination) != sha:
        raise ValueError(f"Video copy mismatch: {source}")
    stats = ImageStatsAccumulator()
    count = 0
    with av.open(str(destination)) as container:
        stream = container.streams.video[0]
        if [stream.height, stream.width, 3] != shape or float(stream.average_rate) != fps:
            raise ValueError(f"Video dimensions/rate mismatch: {source}")
        for frame in container.decode(stream):
            if abs(float(frame.pts * stream.time_base) - count / fps) > 1e-5:
                raise ValueError(f"Video timestamp mismatch: {source}")
            if count < source_length - 1:
                stats.update(frame.to_ndarray(format="rgb24"))
            count += 1
    if count != source_length:
        raise ValueError(f"Video source frame count mismatch: {source}")
    return stats, sha


def convert_dataset(source, output, config, provenance=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, output, config = source.resolve(), output.resolve(), config.resolve()
    provenance = (provenance or source.with_name(source.name + ".work")).resolve()
    if any(output == p or p in output.parents for p in (source, provenance, config)):
        raise ValueError("Output must be independent of all inputs")
    building = output.with_name(output.name + ".assembling")
    if output.exists() or building.exists():
        raise FileExistsError(f"Output or unfinished assembly exists: {output}")
    info = json.loads((source / "meta/info.json").read_text())
    manifest = json.loads((source / "meta/conversion_manifest.json").read_text())
    if (
        info["codebase_version"] != "v3.0"
        or info["features"]["observation.state"]["names"] != STATE_NAMES
        or info["features"]["action"]["names"] != ACTION_NAMES
        or manifest["schema"] != SOURCE_SCHEMA
    ):
        raise ValueError("Expected original labs state34/action20 source")
    fk = {side: URDFFK(config / f"{side}.urdf", side) for side in ("left", "right")}
    urdf_hashes = {side: digest(config / f"{side}.urdf") for side in fk}
    if urdf_hashes != manifest["urdf_sha256"]:
        raise ValueError("URDF does not match source export")
    episodes = [
        e for p in sorted((source / "meta/episodes").rglob("*.parquet")) for e in pq.read_table(p).to_pylist()
    ]
    if [e["episode_index"] for e in episodes] != list(range(info["total_episodes"])):
        raise ValueError("Invalid episode metadata")
    source_hashes = {
        str(p.relative_to(source)): digest(p)
        for folder in ("data", "meta")
        for p in sorted((source / folder).rglob("*"))
        if p.is_file()
    }
    building.mkdir(parents=True)
    shutil.copytree(source / "meta", building / "meta/source_metadata")
    shutil.copy2(source / "meta/tasks.parquet", building / "meta/tasks.parquet")
    (building / "meta/kinematics").mkdir()
    (building / "meta/measured_provenance").mkdir()
    for side in fk:
        shutil.copy2(config / f"{side}.urdf", building / f"meta/kinematics/{side}.urdf")
    for key in ("observation.state", "action"):
        info["features"][key].update(dtype="float32", shape=[20], names=ACTION_NAMES)
    info["features"]["observation.sync_skew_ns"].update(
        shape=[6], names=manifest.get("sync_names", ["wrist_left", "wrist_right", *ACTION_SYNC_NAMES])[:6]
    )
    info["features"].update(
        {
            "action_source_timestamp_ns": {"dtype": "int64", "shape": [1], "names": None},
            "action_source_frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "action_source_sync_skew_ns": {"dtype": "int64", "shape": [4], "names": ACTION_SYNC_NAMES},
        }
    )
    numeric = {k: [] for k, f in info["features"].items() if f["dtype"] != "video"}
    images = {k: ImageStatsAccumulator() for k, f in info["features"].items() if f["dtype"] == "video"}
    frame_count = source_count = 0
    entries, evidence, used_files, intervals = [], [], set(), []
    max_fk_error = 0.0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for ep in episodes:
            index, n = ep["episode_index"], ep["length"]
            rel = info["data_path"].format(
                chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
            )
            if rel in used_files:
                raise ValueError("Expected one source data file per episode")
            used_files.add(rel)
            table = pq.read_table(source / rel)
            if (
                len(table) != n
                or n < 2
                or ep["dataset_from_index"] != source_count
                or ep["dataset_to_index"] != source_count + n
            ):
                raise ValueError("Invalid source episode length or indices")
            np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(n, index))
            np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(n))
            np.testing.assert_array_equal(
                table["index"].to_numpy(), np.arange(source_count, source_count + n)
            )
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            timestamps = table["observation.source_timestamp_ns"].to_numpy()
            skew = np.asarray(table["observation.sync_skew_ns"].to_pylist(), dtype=np.int64)
            if np.any(np.diff(timestamps) <= 0) or skew.shape != (n, 10):
                raise ValueError("Invalid source timestamps/alignment")
            path = provenance / ep["original_episode"] / "joint_provenance.npz"
            prov_hash = digest(path)
            with np.load(path, allow_pickle=False) as prov:
                measured = prov["measured"].copy()
                np.testing.assert_array_equal(timestamps, prov["timestamps_ns"])
                np.testing.assert_array_equal(state[:, 20:], measured.astype(np.float32))
            poses = measured_pose20(measured, state[:, 18:20], fk)
            error = float(np.max(np.abs(poses - state[:, :20])))
            if error > 1e-6:
                raise ValueError("Recomputed measured FK does not match source measured state")
            max_fk_error = max(max_fk_error, error)
            retained = table.slice(0, n - 1)
            updates = {
                "observation.state": pa.array(poses[:-1].tolist(), type=pa.list_(pa.float32(), 20)),
                "action": pa.array(poses[1:].tolist(), type=pa.list_(pa.float32(), 20)),
                "index": pa.array(np.arange(frame_count, frame_count + n - 1), type=pa.int64()),
                "observation.sync_skew_ns": pa.array(skew[:-1, :6].tolist(), type=pa.list_(pa.int64(), 6)),
                "action_source_timestamp_ns": pa.array(timestamps[1:], type=pa.int64()),
                "action_source_frame_index": pa.array(np.arange(1, n), type=pa.int64()),
                "action_source_sync_skew_ns": pa.array(skew[1:, 2:6].tolist(), type=pa.list_(pa.int64(), 4)),
            }
            for key, values in updates.items():
                column = retained.schema.get_field_index(key)
                retained = (
                    retained.append_column(key, values)
                    if column == -1
                    else retained.set_column(column, key, values)
                )
            dest = building / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(retained, dest, compression="zstd")
            stored = pq.read_table(dest)
            for key in retained.column_names:
                if not stored[key].equals(retained[key]):
                    raise ValueError(f"Parquet serialization mismatch: {key}")
            stored_state = np.asarray(stored["observation.state"].to_pylist(), dtype=np.float32)
            stored_action = np.asarray(stored["action"].to_pylist(), dtype=np.float32)
            np.testing.assert_array_equal(stored_state, poses[:-1])
            np.testing.assert_array_equal(stored_action, poses[1:])
            np.testing.assert_array_equal(stored_action[:-1], stored_state[1:])
            entry = {k: v for k, v in ep.items() if not k.startswith("stats/") and k != "sync_stats"}
            entry.update(
                length=n - 1,
                dataset_from_index=frame_count,
                dataset_to_index=frame_count + n - 1,
                source_length=n,
                removed_terminal_observations=1,
            )
            for key in numeric:
                values = np.asarray(stored[key].to_pylist())
                numeric[key].append(values)
                for stat, value in statistics(values).items():
                    if np.ndim(value) == 0:
                        value = [value]
                    entry[f"stats/{key}/{stat}"] = value
            futures = {}
            for key in images:
                relvideo = info["video_path"].format(
                    video_key=key,
                    chunk_index=ep[f"videos/{key}/chunk_index"],
                    file_index=ep[f"videos/{key}/file_index"],
                )
                if ep[f"videos/{key}/from_timestamp"] != 0:
                    raise ValueError("Expected one source video per episode")
                futures[key] = pool.submit(
                    copy_video_and_statistics,
                    source / relvideo,
                    building / relvideo,
                    n,
                    info["fps"],
                    info["features"][key]["shape"],
                )
                entry[f"videos/{key}/to_timestamp"] = (n - 1) / info["fps"]
            video_hashes = {}
            for key, future in futures.items():
                image_stats, sha = future.result()
                video_hashes[key] = sha
                accumulator = images[key]
                accumulator.minimum = np.minimum(accumulator.minimum, image_stats.minimum)
                accumulator.maximum = np.maximum(accumulator.maximum, image_stats.maximum)
                accumulator.total += image_stats.total
                accumulator.square += image_stats.square
                accumulator.pixels += image_stats.pixels
                accumulator.count += image_stats.count
                for stat, value in image_stats.finish().items():
                    entry[f"stats/{key}/{stat}"] = value
            provdest = building / f"meta/measured_provenance/episode_{index:06d}.npz"
            np.savez_compressed(
                provdest,
                measured_joints=measured,
                measured_pose20=poses,
                timestamps_ns=timestamps,
                measured_sync_skew_ns=skew[:, :6],
            )
            if digest(path) != prov_hash:
                raise ValueError("Source provenance changed")
            evidence.append(
                {
                    "episode_index": index,
                    "original_episode": ep["original_episode"],
                    "source_joint_provenance_sha256": prov_hash,
                    "measured_provenance": str(provdest.relative_to(building)),
                    "measured_provenance_sha256": digest(provdest),
                    "video_sha256": video_hashes,
                }
            )
            entries.append(entry)
            intervals.extend(np.diff(timestamps).tolist())
            frame_count += n - 1
            source_count += n
            print(
                json.dumps({"episode": index, "output_frames": frame_count, "source_frames": source_count}),
                flush=True,
            )
    if source_count != info["total_frames"] or len(entries) != info["total_episodes"]:
        raise ValueError("Source totals mismatch")
    if used_files != {str(p.relative_to(source)) for p in (source / "data").rglob("*.parquet")}:
        raise ValueError("Unaccounted source data files")
    for rel, sha in source_hashes.items():
        if digest(source / rel) != sha:
            raise ValueError(f"Source modified: {rel}")
    meta = building / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    for entry in entries:
        entry["meta/episodes/chunk_index"], entry["meta/episodes/file_index"] = 0, 0
    pq.write_table(
        pa.Table.from_pylist(entries), meta / "episodes/chunk-000/file-000.parquet", compression="zstd"
    )
    stats = {}
    for key, parts in numeric.items():
        values = np.concatenate(parts)
        stats[key] = statistics(values.reshape(-1, 1) if values.ndim == 1 else values)
    stats.update({key: a.finish() for key, a in images.items()})
    info["total_frames"] = frame_count
    new_manifest = {
        "schema": SCHEMA,
        "label_semantics": "next measured observation, explicitly requested offline trajectory supervision",
        "action_source": "FK of the NEXT retained same-episode measured joints plus NEXT measured binary grippers",
        "state_source": "FK of CURRENT measured joints plus CURRENT measured binary grippers",
        "is_recorded_command_action": False,
        "action_spec": {
            "dimension": 20,
            "ee_dimension": 9,
            "ee_rotation": "rot6d_columns",
            "gripper_range": [0, 1],
        },
        "state_names": ACTION_NAMES,
        "action_names": ACTION_NAMES,
        "rotation_representation": "rot6d_columns",
        "frames": manifest["frames"],
        "tips": manifest["tips"],
        "tool_offset_m": 0,
        "gripper": manifest["gripper"],
        "fps": info["fps"],
        "task": manifest["task"],
        "normalized": False,
        "absolute_pose": True,
        "delta_reference": None,
        "terminal_policy": "drop last observation row per episode; never cross episode boundaries",
        "next_frame_definition": "next retained synchronized source observation; actual interval saved by source timestamps",
        "video_policy": "bitwise original videos; one unused terminal image per episode outside metadata time range",
        "image_statistics": "decoded RGB grid every eighth row/column, every RETAINED observation frame",
        "sync": {
            "anchor": "head image header.stamp",
            "measured": "source nearest measured state, 50ms tolerance",
            "action": "next retained observation's measured state; next source timestamp and measured offsets saved separately",
        },
        "source_dataset": str(source),
        "source_metadata": "meta/source_metadata",
        "source_file_sha256": source_hashes,
        "urdf_sha256": urdf_hashes,
        "episodes": evidence,
        "code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "labs_kinematics.py",
                "action_spec.py",
                "labs_joint_dataset.py",
                "mcap_to_lerobot.py",
            )
        },
    }
    report = {
        "status": "passed",
        "episodes": len(entries),
        "source_frames": source_count,
        "frames": frame_count,
        "dropped_terminal_observations": len(entries),
        "state_shape": [20],
        "action_shape": [20],
        "all_actions_match_next_measured_pose_and_grippers": True,
        "cross_episode_transitions": 0,
        "max_recomputed_fk_error_vs_source": max_fk_error,
        "videos_copied_and_fully_decoded": len(entries) * len(images),
        "physical_video_frames_decoded": source_count * len(images),
        "retained_image_frames_in_statistics": frame_count * len(images),
        "all_feature_statistics_recomputed": True,
        "source_unchanged": True,
        "transition_interval_ms_min_median_max": (np.quantile(intervals, [0, 0.5, 1]) / 1e6).tolist(),
        "transition_intervals_over_50ms": int((np.asarray(intervals) > 50_000_000).sum()),
    }
    for name, value in (
        ("info.json", info),
        ("stats.json", stats),
        ("conversion_manifest.json", new_manifest),
        ("next_state_validation.json", report),
    ):
        (meta / name).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(building, output)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--provenance", type=Path)
    args = p.parse_args(argv)
    print(json.dumps(convert_dataset(args.input, args.output, args.config, args.provenance), indent=2))


if __name__ == "__main__":
    main()
