"""Export labs joint16 state/action from aligned measured/recorded-target provenance.

Offline only: seven absolute joint angles followed by the gripper for each arm.
Requires the original state34/action20 export and its .work checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .labs_mcap_to_lerobot import ACTION_NAMES, STATE_NAMES

SCHEMA = "labs_fr3_absolute_joint_state16_action16_v1"
JOINT_NAMES = [
    name
    for side in ("left", "right")
    for name in [*[f"{side}_joint{i}_position" for i in range(1, 8)], f"{side}_gripper_open"]
]


def joint16(joints, grippers):
    """Interleave original absolute joints and binary openings; never infer targets."""
    joints, grippers = np.asarray(joints), np.asarray(grippers)
    if joints.ndim != 2 or joints.shape[1] != 14 or grippers.shape != (len(joints), 2):
        raise ValueError("Expected joints (N, 14) and grippers (N, 2)")
    if not np.isfinite(joints).all() or not np.isin(grippers, [0, 1]).all():
        raise ValueError("Expected finite joints and binary grippers")
    return np.concatenate((joints[:, :7], grippers[:, :1], joints[:, 7:], grippers[:, 1:]), axis=1).astype(
        np.float32
    )


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot calculate statistics for empty/nonfinite values")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def convert_dataset(source: Path, output: Path, provenance: Path | None = None):
    """Write an independent, atomically finalized v3 dataset and validate every row."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, output = source.resolve(), output.resolve()
    provenance = (provenance or source.with_name(source.name + ".work")).resolve()
    if output == source or source in output.parents or provenance == output or provenance in output.parents:
        raise ValueError("Output must be separate from source and provenance")
    building = output.with_name(output.name + ".assembling")
    if output.exists() or building.exists():
        raise FileExistsError(f"Output or unfinished assembly exists: {output}")
    info = json.loads((source / "meta/info.json").read_text())
    if (
        info["codebase_version"] != "v3.0"
        or info["features"]["observation.state"]["names"] != STATE_NAMES
        or info["features"]["observation.state"]["shape"] != [34]
        or info["features"]["action"]["names"] != ACTION_NAMES
        or info["features"]["action"]["shape"] != [20]
    ):
        raise ValueError("Expected original labs state34/action20 dataset")
    manifest = json.loads((source / "meta/conversion_manifest.json").read_text())
    if manifest["schema"] != "labs_fr3_link8_columns_state34_action20_v1":
        raise ValueError("Expected original recorded-follower-target export")
    episode_paths = sorted((source / "meta/episodes").rglob("*.parquet"))
    episodes = [e for p in episode_paths for e in pq.read_table(p).to_pylist()]
    if [e["episode_index"] for e in episodes] != list(range(info["total_episodes"])):
        raise ValueError("Episode metadata is incomplete or out of order")
    # Numeric files and metadata are small; snapshot their hashes before processing.
    inputs = [p for folder in ("data", "meta") for p in sorted((source / folder).rglob("*")) if p.is_file()]
    source_hashes = {str(p.relative_to(source)): digest(p) for p in inputs}
    building.mkdir(parents=True)
    shutil.copytree(source / "meta", building / "meta/source_metadata")
    shutil.copy2(source / "meta/tasks.parquet", building / "meta/tasks.parquet")
    (building / "meta/joint_provenance").mkdir()
    collected = {"observation.state": [], "action": []}
    evidence = []
    frame_count = video_count = different_rows = 0
    used_files = set()
    for episode in episodes:
        index, n = episode["episode_index"], episode["length"]
        relative = info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
        )
        if relative in used_files:
            raise ValueError("This converter requires one source data file per episode")
        used_files.add(relative)
        table = pq.read_table(source / relative)
        if len(table) != n or episode["dataset_from_index"] != frame_count:
            raise ValueError("Episode length or start index mismatch")
        if episode["dataset_to_index"] != frame_count + n:
            raise ValueError("Episode end index mismatch")
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(n, index))
        np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(frame_count, frame_count + n))
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(n))
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        prov = provenance / episode["original_episode"] / "joint_provenance.npz"
        prov_hash = digest(prov)
        with np.load(prov, allow_pickle=False) as saved:
            np.testing.assert_array_equal(
                table["observation.source_timestamp_ns"].to_numpy(), saved["timestamps_ns"]
            )
            measured, targets = saved["measured"], saved["recorded_targets"]
            np.testing.assert_array_equal(state[:, 20:34], measured.astype(np.float32))
            transformed = {
                "observation.state": joint16(measured, state[:, 18:20]),
                "action": joint16(targets, action[:, 18:20]),
            }
            different_rows += int(np.any(np.abs(measured - targets) > 1e-6, axis=1).sum())
        skew = np.asarray(table["observation.sync_skew_ns"].to_pylist())
        if skew.shape != (n, 10) or np.any(skew[:, 6:] > 0) or np.any(skew[:, 6:] < -100_000_000):
            raise ValueError("Target alignment must be causal, with age <=100 ms")
        for key, values in transformed.items():
            table = table.set_column(
                table.schema.get_field_index(key),
                key,
                pa.array(values.tolist(), type=pa.list_(pa.float32(), 16)),
            )
            collected[key].append(values)
            for stat, value in statistics(values).items():
                episode[f"stats/{key}/{stat}"] = value
        destination = building / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")
        stored, original = pq.read_table(destination), pq.read_table(source / relative)
        for key in original.column_names:
            if key in transformed:
                np.testing.assert_array_equal(np.asarray(stored[key].to_pylist()), transformed[key])
            elif not stored[key].equals(original[key]):
                raise ValueError(f"Unexpected change in {key}")
        saved_prov = building / f"meta/joint_provenance/episode_{index:06d}.npz"
        shutil.copy2(prov, saved_prov)
        if digest(saved_prov) != prov_hash or digest(prov) != prov_hash:
            raise ValueError("Provenance changed during export")
        evidence.append(
            {
                "episode_index": index,
                "original_episode": episode["original_episode"],
                "path": str(saved_prov.relative_to(building)),
                "sha256": prov_hash,
            }
        )
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            video = info["video_path"].format(
                video_key=key,
                chunk_index=episode[f"videos/{key}/chunk_index"],
                file_index=episode[f"videos/{key}/file_index"],
            )
            dst = building / video
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / video, dst)
            if digest(source / video) != digest(dst):
                raise ValueError(f"Video copy mismatch: {video}")
            video_count += 1
        frame_count += n
        print(json.dumps({"episode": index, "frames": frame_count}), flush=True)
    if frame_count != info["total_frames"] or video_count != info["total_videos"]:
        raise ValueError("Dataset totals do not match source metadata")
    if used_files != {str(p.relative_to(source)) for p in (source / "data").rglob("*.parquet")}:
        raise ValueError("Source contains unaccounted data files")
    for relative, sha in source_hashes.items():
        if digest(source / relative) != sha:
            raise ValueError(f"Source changed during conversion: {relative}")
    for path in episode_paths:
        old = pq.read_table(path)
        indices = set(old["episode_index"].to_pylist())
        destination = building / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist([e for e in episodes if e["episode_index"] in indices]), destination
        )
    stats = json.loads((source / "meta/stats.json").read_text())
    for key, values in collected.items():
        info["features"][key].update(dtype="float32", shape=[16], names=JOINT_NAMES)
        stats[key] = statistics(np.concatenate(values))
    report = {
        "status": "passed",
        "episodes": len(episodes),
        "frames": frame_count,
        "state_dimension": 16,
        "action_dimension": 16,
        "rows_with_measured_target_difference_gt_1e_6_rad": different_rows,
        "all_rows_match_timestamped_joint_provenance": True,
        "all_other_data_columns_unchanged": True,
        "binary_grippers_unchanged": True,
        "video_files_verified_sha256": video_count,
        "state_and_action_statistics_recomputed": True,
        "source_numeric_and_metadata_hashes_unchanged": True,
    }
    new_manifest = {
        "schema": SCHEMA,
        "source_dataset": str(source),
        "source_provenance": str(provenance),
        "source_metadata": "meta/source_metadata",
        "source_file_sha256": source_hashes,
        "joint_provenance": evidence,
        "state_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "state_source": "recorded measured joints and measured binary gripper opening",
        "action_source": "recorded follower joint targets and recorded binary target gripper opening",
        "joint_units": "rad",
        "joint_representation": "absolute joint position",
        "gripper": manifest["gripper"],
        "sync": manifest["sync"],
        "fps": info["fps"],
        "task": manifest["task"],
        "normalized": False,
        "conversion_code_sha256": digest(Path(__file__)),
    }
    for name, value in (
        ("info.json", info),
        ("stats.json", stats),
        ("conversion_manifest.json", new_manifest),
        ("joint_conversion_validation.json", report),
    ):
        (building / "meta" / name).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(building, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(convert_dataset(args.input, args.output, args.provenance), indent=2))


if __name__ == "__main__":
    main()
