"""Bidirectional offline conversion: link0 absolute20 <-> same-frame-state delta14.

Delta rotation is a principal rotation vector (radians), not Euler subtraction.
Each row independently references its measured state; rows are NOT accumulated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .action_spec import matrix_to_rot6d, rot6d_to_matrix
from .labs_mcap_to_lerobot import ACTION_NAMES, STATE_NAMES

DELTA_REPRESENTATION = "labs_fr3_link8_state_relative_link0_delta14_v1"
DELTA_NAMES = [
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("delta_x", "delta_y", "delta_z", "delta_rotvec_x", "delta_rotvec_y", "delta_rotvec_z")
] + ["left_gripper_open", "right_gripper_open"]
CONVENTION = {
    "representation": DELTA_REPRESENTATION,
    "reference": "same-row observation.state measured pose",
    "translation": "p_target - p_state in each arm's link0",
    "rotation": "rotvec(R_target @ R_state.T), principal angle in [0, pi]",
    "reconstruction": "p_target = p_state + delta_xyz; R_target = Exp(delta_rotvec) @ R_state",
    "units": {"translation": "m", "rotation": "rad"},
    "gripper": "absolute left/right openness, copied without differencing",
    "normalized": False,
    "accumulate_across_rows": False,
    "order": DELTA_NAMES,
}


def _inputs(action, state, dimension):
    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if action.ndim < 1 or action.shape[-1] != dimension:
        raise ValueError(f"action must end in {dimension} values")
    if state.ndim < 1 or state.shape[-1] not in (20, 34):
        raise ValueError("reference state must end in 20 or 34 values")
    if action.shape[:-1] != state.shape[:-1]:
        raise ValueError("action and same-frame state must have matching batch dimensions")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("action/state must be finite")
    if np.any((action[..., -2:] < 0) | (action[..., -2:] > 1)):
        raise ValueError("absolute grippers must be in [0, 1]")
    return action, state


def absolute20_to_delta14(action, state):
    """Return float32 delta14, independently referenced to each supplied state."""
    action, state = _inputs(action, state, 20)
    result = np.empty(action.shape[:-1] + (14,), dtype=np.float32)
    for absolute_offset, delta_offset in ((0, 0), (9, 6)):
        a, d = absolute_offset, delta_offset
        result[..., d : d + 3] = action[..., a : a + 3] - state[..., a : a + 3]
        target = rot6d_to_matrix(action[..., a + 3 : a + 9]).astype(np.float64)
        measured = rot6d_to_matrix(state[..., a + 3 : a + 9]).astype(np.float64)
        relative = target @ np.swapaxes(measured, -1, -2)
        vector = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
        result[..., d + 3 : d + 6] = vector.reshape(action.shape[:-1] + (3,))
    result[..., 12:14] = action[..., 18:20]
    return result


def delta14_to_absolute20(action, state):
    """Restore column-6D absolute20 using the SAME state used for differencing."""
    action, state = _inputs(action, state, 14)
    result = np.empty(action.shape[:-1] + (20,), dtype=np.float32)
    for absolute_offset, delta_offset in ((0, 0), (9, 6)):
        a, d = absolute_offset, delta_offset
        result[..., a : a + 3] = state[..., a : a + 3] + action[..., d : d + 3]
        measured = rot6d_to_matrix(state[..., a + 3 : a + 9]).astype(np.float64)
        vector = action[..., d + 3 : d + 6]
        increment = Rotation.from_rotvec(vector.reshape(-1, 3)).as_matrix()
        target = increment.reshape(action.shape[:-1] + (3, 3)) @ measured
        result[..., a + 3 : a + 9] = matrix_to_rot6d(target)
    result[..., 18:20] = action[..., 12:14]
    return result


def _pose_errors(expected, actual):
    position = rotation = 0.0
    for offset in (0, 9):
        position = max(
            position,
            float(
                np.linalg.norm(
                    expected[:, offset : offset + 3] - actual[:, offset : offset + 3], axis=1
                ).max()
            ),
        )
        left = rot6d_to_matrix(expected[:, offset + 3 : offset + 9]).astype(np.float64)
        right = rot6d_to_matrix(actual[:, offset + 3 : offset + 9]).astype(np.float64)
        relative = left @ np.swapaxes(right, -1, -2)
        rotation = max(rotation, float(Rotation.from_matrix(relative).magnitude().max()))
    np.testing.assert_array_equal(expected[:, 18:20], actual[:, 18:20])
    if position > 1e-6 or rotation > 1e-6:
        raise ValueError(f"round-trip pose error too large: {position} m, {rotation} rad")
    return position, rotation


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def convert_dataset(source: Path, output: Path, direction: str):
    """Convert all Parquet rows, retain images/state, rebuild action stats atomically."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .mcap_to_lerobot import _StatsAccumulator

    source, output = source.resolve(), output.resolve()
    if direction not in ("to-delta", "to-absolute"):
        raise ValueError("direction must be to-delta or to-absolute")
    if output == source or source in output.parents:
        raise ValueError("output must be separate from the source dataset")
    building = output.with_name(output.name + ".assembling")
    if output.exists() or building.exists():
        raise FileExistsError(f"Output or unfinished assembly already exists: {output}")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["features"]["observation.state"]["names"] != STATE_NAMES:
        raise ValueError("Expected labs LeRobot v3 state34 schema")
    forward = direction == "to-delta"
    expected_names = ACTION_NAMES if forward else DELTA_NAMES
    if info["features"]["action"]["names"] != expected_names:
        raise ValueError("Source action schema does not match conversion direction")
    manifest = json.loads((source / "meta/conversion_manifest.json").read_text())
    if manifest.get("frames") != {"left": "left_fr3_link0", "right": "right_fr3_link0"} or manifest.get(
        "tips"
    ) != {"left": "left_fr3_link8", "right": "right_fr3_link8"}:
        raise ValueError("Expected per-arm link0 frames and link8 tips")
    if not forward and manifest.get("delta_convention") != CONVENTION:
        raise ValueError("Delta convention does not match same-frame-state link0 delta14")
    if forward and manifest.get("rotation_representation") != "rot6d_columns":
        raise ValueError("Expected absolute rotation columns")
    dimension, names = (14, DELTA_NAMES) if forward else (20, ACTION_NAMES)
    info = copy.deepcopy(info)
    info["features"]["action"].update(shape=[dimension], names=names, dtype="float32")
    # Archive original reports: they describe the source labels, not these new labels.
    building.mkdir(parents=True)
    (building / "meta").mkdir()
    shutil.copytree(source / "meta", building / "meta/source_metadata")
    shutil.copytree(source / "meta/episodes", building / "meta/episodes")
    shutil.copy2(source / "meta/tasks.parquet", building / "meta/tasks.parquet")
    shutil.copytree(source / "videos", building / "videos", copy_function=_link_or_copy)
    accumulator = _StatsAccumulator((dimension,), "float32")
    frames = 0
    max_position = max_rotation = 0.0
    episodes = set()
    action_max = np.zeros(dimension)
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if forward:
            transformed = absolute20_to_delta14(action, state)
            reconstructed = delta14_to_absolute20(transformed, state)
            pos, rot = _pose_errors(action, reconstructed)
        else:
            transformed = delta14_to_absolute20(action, state)
            # Compare physical poses, not non-unique rotvecs at the pi branch cut.
            redelta = absolute20_to_delta14(transformed, state)
            pos, rot = _pose_errors(transformed, delta14_to_absolute20(redelta, state))
            np.testing.assert_array_equal(action[:, 12:14], transformed[:, 18:20])
        max_position, max_rotation = max(max_position, pos), max(max_rotation, rot)
        for row in transformed:
            accumulator.update(row)
        action_max = np.maximum(action_max, np.abs(transformed).max(axis=0))
        array = pa.array(transformed.tolist(), type=pa.list_(pa.float32(), dimension))
        table = table.set_column(table.schema.get_field_index("action"), "action", array)
        destination = building / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")
        # Verify serialized action and bitwise preservation of all other columns.
        stored = pq.read_table(destination)
        np.testing.assert_array_equal(np.asarray(stored["action"].to_pylist(), dtype=np.float32), transformed)
        original = pq.read_table(path)
        for key in original.column_names:
            if key != "action" and not original[key].equals(stored[key]):
                raise ValueError(f"Non-action column changed: {path}: {key}")
        episodes.update(table["episode_index"].to_pylist())
        frames += len(table)
        print(json.dumps({"converted_file": str(path.relative_to(source)), "frames": frames}), flush=True)
    if frames != info["total_frames"] or len(episodes) != info["total_episodes"]:
        raise ValueError("Dataset frame/episode totals do not match metadata")
    stats = json.loads((source / "meta/stats.json").read_text())
    stats["action"] = accumulator.finish()
    new_manifest = {
        "schema": DELTA_REPRESENTATION if forward else "labs_fr3_link8_columns_state34_action20_v1",
        "frames": manifest["frames"],
        "tips": manifest["tips"],
        "tool_offset_m": 0,
        "task": manifest["task"],
        "fps": info["fps"],
        "image_size": [640, 480],
        "state_names": STATE_NAMES,
        "action_names": names,
        "state_rotation_representation": "rot6d_columns",
        "rotation_representation": "rotvec_link0_delta" if forward else "rot6d_columns",
        "action_spec": {
            "dimension": dimension,
            "ee_dimension": 6 if forward else 9,
            "ee_rotation": "rotvec_link0_delta" if forward else "rot6d_columns",
            "gripper_range": [0, 1],
        },
        "delta_convention": CONVENTION,
        "action_source": "recorded follower target FK relative to same-frame measured state"
        if forward
        else "restored recorded follower target FK from delta14 and same-frame measured state",
        "source_dataset": str(source),
        "source_metadata": "meta/source_metadata",
        "normalized": False,
        "conversion_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    report = {
        "status": "passed",
        "direction": direction,
        "episodes": len(episodes),
        "frames": frames,
        "state_dimension": 34,
        "action_dimension": dimension,
        "max_roundtrip_position_error_m": max_position,
        "max_roundtrip_rotation_error_rad": max_rotation,
        "all_non_action_columns_unchanged": True,
        "grippers_unchanged": True,
        "videos_reused_without_reencoding": True,
        "action_statistics_recomputed": True,
        "action_max_abs_per_dimension": action_max.tolist(),
        "source": str(source),
        "output": str(output),
    }
    for name, content in (
        ("info.json", info),
        ("stats.json", stats),
        ("conversion_manifest.json", new_manifest),
        ("action_conversion_validation.json", report),
    ):
        (building / "meta" / name).write_text(json.dumps(content, indent=2) + "\n")
    os.replace(building, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("to-delta", "to-absolute"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(convert_dataset(args.input, args.output, args.direction)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
