"""labs raw MCAP adapter for FR3 link8 / column-6D LeRobot v3 datasets.

Offline only. Reuses the existing synchronization primitives, RGB decoder,
video encoder, statistics and LeRobot v3 writer. Each episode is checkpointed.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import shutil
from collections import Counter, deque
from pathlib import Path

import numpy as np

from .labs_kinematics import URDFFK, joint_positions, pose_vector
from .mcap_to_lerobot import (
    DerivedFrame,
    LeRobotV3Writer,
    SyncStats,
    TimedBuffer,
    TimedMessage,
    message_stamp_ns,
)
from .ros_utils import image_msg_to_rgb

TASK = (
    "Use the left arm to place the square head into the yellow box on the left, "
    "and the right arm to place the screw into the green box on the right."
)
SCHEMA = "labs_fr3_link8_columns_state34_action20_v1"
CAMERAS = {
    "head": "/head_camera/zed_node/rgb/image_rect_color",
    "wrist_left": "/wrist_camera_left/camera/color/image_raw",
    "wrist_right": "/wrist_camera_right/camera/color/image_raw",
}
POSE_NAMES = [
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (
        "x",
        "y",
        "z",
        "rot6d_col0_x",
        "rot6d_col0_y",
        "rot6d_col0_z",
        "rot6d_col1_x",
        "rot6d_col1_y",
        "rot6d_col1_z",
    )
]
ACTION_NAMES = POSE_NAMES + ["left_gripper_open", "right_gripper_open"]
STATE_NAMES = ACTION_NAMES + [f"{s}_joint{i}_position" for s in ("left", "right") for i in range(1, 8)]
SYNC_NAMES = [
    "wrist_left",
    "wrist_right",
    "left_q",
    "right_q",
    "left_gripper",
    "right_gripper",
    "left_target",
    "right_target",
    "left_gripper_target",
    "right_gripper_target",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_topics():
    result = {}
    for side in ("left", "right"):
        for key, suffix in {
            "q": "franka_robot_state_broadcaster/measured_joint_states",
            "target": "follower/gello/joint_states",
            "gripper": "follower/gripper/joint_states",
            "gripper_target": "follower/gripper/gripper_client/target_gripper_width_percent",
        }.items():
            result[f"{side}_{key}"] = f"/{side}/{suffix}"
    return result


class HeadFrameGate:
    """Keep the first valid head frame in each nearest output-rate time slot.

    Rounding to the nearest slot tolerates sub-millisecond camera jitter;
    ceil-style gating can wrongly discard alternating ~30 Hz image samples.
    Missing camera slots are not filled with fabricated images.
    """

    def __init__(self, fps):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.period_ns = 1_000_000_000 / fps
        self.start = None
        self.last_slot = -1

    def keep(self, stamp):
        if self.start is None:
            self.start = stamp
        slot = int(np.floor((stamp - self.start) / self.period_ns + 0.5))
        if slot <= self.last_slot:
            return False
        self.last_slot = slot
        return True


class NumericSeries:
    def __init__(self, rows):
        rows.sort(key=lambda r: r[0])
        self.times = [r[0] for r in rows]
        self.values = [r[1] for r in rows]

    def match(self, timestamp, tolerance, *, previous=False):
        if previous:
            index = bisect.bisect_right(self.times, timestamp) - 1
            if index < 0 or timestamp - self.times[index] > tolerance:
                return None
        else:
            index = bisect.bisect_left(self.times, timestamp)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(self.times)]
            if not candidates:
                return None
            index = min(
                candidates,
                key=lambda i: (abs(self.times[i] - timestamp), self.times[i]),
            )
            if abs(self.times[index] - timestamp) > tolerance:
                return None
        return self.values[index], self.times[index] - timestamp


def decoded(paths, topics):
    import heapq
    from contextlib import ExitStack

    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    with ExitStack() as stack:
        streams = [stack.enter_context(p.open("rb")) for p in paths]
        readers = [make_reader(s, decoder_factories=[DecoderFactory()]) for s in streams]
        iterators = [r.iter_decoded_messages(topics=topics) for r in readers]
        yield from heapq.merge(*iterators, key=lambda item: item[2].log_time)


def read_numeric(paths):
    topics = numeric_topics()
    reverse = {v: k for k, v in topics.items()}
    data = {k: [] for k in topics}
    for _, channel, record, msg in decoded(paths, list(reverse)):
        key = reverse[channel.topic]
        side = key.split("_")[0]
        target = key.endswith("target")
        stamp = record.log_time if target else message_stamp_ns(msg, record.log_time)
        if key.endswith("_q") or key.endswith("_target") and not key.endswith("gripper_target"):
            value = joint_positions(msg.name, msg.position, side)
        elif key.endswith("gripper_target"):
            value = float(msg.data)
            if not np.isfinite(value) or not -1e-5 <= value <= 1.00001:
                raise ValueError(f"{key} is not a 0..1 opening fraction: {value}")
        else:
            names = list(msg.name)
            candidates = [i for i, n in enumerate(names) if n.endswith("robotiq_85_left_knuckle_joint")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one Robotiq knuckle in {names}")
            value = float(msg.position[candidates[0]])
            validate_knuckle(value)
        data[key].append((int(stamp), value))
    if any(not values for values in data.values()):
        raise ValueError(f"Missing numeric streams: {[k for k, v in data.items() if not v]}")
    return {key: NumericSeries(rows) for key, rows in data.items()}


def validate_knuckle(value):
    # Deployed Robotiq 2f_85 driver: 0.7929 * (feedback_byte - 3) / (230 - 3).
    # The byte spans 0..255; feedback is not clamped at nominal closed byte 230.
    low, high = -0.7929 * 3 / 227, 0.7929 * 252 / 227
    if not np.isfinite(value) or not low - 1e-6 <= value <= high + 1e-6:
        raise ValueError(f"Invalid knuckle position {value}: expected [{low}, {high}]")


def binary_gripper(value, *, target=False):
    opening = value if target else np.clip(1.0 - value / 0.8, 0.0, 1.0)
    return np.float32(opening >= 0.5)


def convert_episode(paths, destination, config, args):
    import av

    series = read_numeric(paths)
    fk = {s: URDFFK(config / f"{s}.urdf", s) for s in ("left", "right")}
    writer = LeRobotV3Writer(destination, args.fps, args.task, state_names=STATE_NAMES, sync_names=SYNC_NAMES)
    buffers = {key: TimedBuffer(128) for key in ("wrist_left", "wrist_right")}
    pending = deque()
    gate = HeadFrameGate(args.fps)
    stats = SyncStats()
    drops = Counter()
    dropped_frames = []
    originals = []
    shapes = {}
    rgb_tolerance = 45_000_000
    state_tolerance = 50_000_000
    target_tolerance = 100_000_000

    def flush(force=False):
        while pending:
            head = pending[0]
            ready = all(
                b.values and b.values[-1].stamp_ns >= head.stamp_ns + rgb_tolerance for b in buffers.values()
            )
            if not force and not ready:
                break
            pending.popleft()
            wrists = {k: b.nearest(head.stamp_ns, rgb_tolerance) for k, b in buffers.items()}
            if any(v is None for v in wrists.values()):
                drops["missing_wrist"] += 1
                stats.dropped_missing_wrist += 1
                continue
            matched = {}
            for key, values in series.items():
                target = key.endswith("target")
                matched[key] = values.match(
                    head.stamp_ns,
                    target_tolerance if target else state_tolerance,
                    previous=target,
                )
            missing = [k for k, v in matched.items() if v is None]
            if missing:
                drops["missing_numeric"] += 1
                boundary = any(head.stamp_ns < series[k].times[0] for k in missing)
                drops["start_boundary" if boundary else "interior_or_end_missing"] += 1
                dropped_frames.append(
                    {
                        "image_stamp_ns": head.stamp_ns,
                        "missing_streams": missing,
                        "start_boundary": boundary,
                    }
                )
                for key in missing:
                    drops[f"missing_{key}"] += 1
                stats.dropped_missing_state += 1
                continue
            stats.frames_ready += 1
            if not gate.keep(head.stamp_ns):
                stats.dropped_resampled += 1
                continue
            measured = [matched[f"{s}_q"][0] for s in ("left", "right")]
            targets = [matched[f"{s}_target"][0] for s in ("left", "right")]
            ee = np.concatenate(
                [pose_vector(fk[s](q)) for s, q in zip(("left", "right"), measured, strict=True)]
            )
            goal = np.concatenate(
                [pose_vector(fk[s](q)) for s, q in zip(("left", "right"), targets, strict=True)]
            )
            grippers = np.array([binary_gripper(matched[f"{s}_gripper"][0]) for s in ("left", "right")])
            goal_grippers = np.array(
                [binary_gripper(matched[f"{s}_gripper_target"][0], target=True) for s in ("left", "right")]
            )
            state = np.concatenate((ee, grippers, *measured)).astype(np.float32)
            action = np.concatenate((goal, goal_grippers)).astype(np.float32)
            skew = np.array(
                [wrists[k].stamp_ns - head.stamp_ns for k in ("wrist_left", "wrist_right")]
                + [matched[k][1] for k in SYNC_NAMES[2:]],
                dtype=np.int64,
            )
            frame = DerivedFrame(
                head.stamp_ns,
                skew,
                head.message,
                wrists["wrist_left"].message,
                wrists["wrist_right"].message,
                state,
                ee,
                gripper=grippers,
            )
            writer.add_frame(frame, action, 0, stats.frames_written)
            originals.append((head.stamp_ns, np.concatenate(measured), np.concatenate(targets)))
            stats.frames_written += 1

    reverse = {v: k for k, v in CAMERAS.items()}
    for _, channel, record, msg in decoded(paths, list(reverse)):
        key = reverse[channel.topic]
        stamp = message_stamp_ns(msg, record.log_time)
        rgb = image_msg_to_rgb(msg)
        shapes.setdefault(key, list(rgb.shape))
        rgb = (
            av.VideoFrame.from_ndarray(rgb, format="rgb24")
            .reformat(width=640, height=480)
            .to_ndarray(format="rgb24")
        )
        timed = TimedMessage(stamp, record.log_time, rgb)
        if key == "head":
            stats.head_seen += 1
            pending.append(timed)
            if len(pending) > 128:
                raise ValueError("Camera synchronization exceeded bounded pending queue")
        else:
            buffers[key].values.append(timed)
        flush()
    flush(force=True)
    if not stats.frames_written:
        raise ValueError("No synchronized frames")
    writer.finish_episode(0, paths[0].parent.parent.name, stats)
    writer.finalize()
    np.savez_compressed(
        destination / "joint_provenance.npz",
        timestamps_ns=np.array([r[0] for r in originals], dtype=np.int64),
        measured=np.array([r[1] for r in originals]),
        recorded_targets=np.array([r[2] for r in originals]),
    )
    return {
        "stats": stats.as_dict(),
        "drops": dict(drops),
        "dropped_frame_details": dropped_frames,
        "original_image_shapes": shapes,
        "numeric_message_counts": {k: len(v.times) for k, v in series.items()},
    }


def combine(shards, output, contract):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .mcap_to_lerobot import _StatsAccumulator

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    building = output.with_name(output.name + ".assembling")
    if building.exists():
        raise FileExistsError(f"Inspect unfinished assembly first: {building}")
    building.mkdir(parents=True)
    features = json.loads((shards[0] / "meta/info.json").read_text())["features"]
    accum = {
        k: _StatsAccumulator(tuple(v["shape"]), v["dtype"])
        for k, v in features.items()
        if v["dtype"] != "video"
    }
    episodes = []
    offset = 0
    for index, shard in enumerate(shards):
        chunk, file_index = divmod(index, 1000)
        table = pq.read_table(shard / "data/chunk-000/file-000.parquet")
        count = len(table)
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array([index] * count, type=pa.int64()),
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(range(offset, offset + count), type=pa.int64()),
        )
        path = building / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")
        for key, a in accum.items():
            for value in table[key].to_pylist():
                a.update(value)
        entry = pq.read_table(shard / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
        entry.update(
            {
                "episode_index": index,
                "data/chunk_index": chunk,
                "data/file_index": file_index,
                "dataset_from_index": offset,
                "dataset_to_index": offset + count,
            }
        )
        for key, feature in features.items():
            if feature["dtype"] != "video":
                continue
            src = shard / f"videos/{key}/chunk-000/file-000.mp4"
            dst = building / f"videos/{key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
            entry[f"videos/{key}/chunk_index"] = chunk
            entry[f"videos/{key}/file_index"] = file_index
        episodes.append(entry)
        offset += count
    meta = building / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(episodes),
        meta / "episodes/chunk-000/file-000.parquet",
        compression="zstd",
    )
    info = json.loads((shards[0] / "meta/info.json").read_text())
    info.update(
        total_episodes=len(shards),
        total_frames=offset,
        splits={"train": f"0:{len(shards)}"},
        robot_type="franka_fr3_duo_link8",
        total_videos=3 * len(shards),
    )
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    pd.DataFrame({"task_index": np.array([0], dtype=np.int64)}, index=[contract["task"]]).to_parquet(
        meta / "tasks.parquet"
    )
    stats = {k: a.finish() for k, a in accum.items()}
    for key, v in features.items():
        if v["dtype"] != "video":
            continue
        source = [json.loads((s / "meta/stats.json").read_text())[key] for s in shards]
        weights = np.array([x["count"][0] for x in source])
        means = np.array([x["mean"] for x in source])
        stds = np.array([x["std"] for x in source])
        mean = np.average(means, axis=0, weights=weights)
        var = np.maximum(np.average(stds**2 + means**2, axis=0, weights=weights) - mean**2, 0)
        stats[key] = {
            "min": np.min([x["min"] for x in source], axis=0).tolist(),
            "max": np.max([x["max"] for x in source], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [int(weights.sum())],
        }
    (meta / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (meta / "conversion_manifest.json").write_text(json.dumps(contract, indent=2) + "\n")
    os.replace(building, output)
    return {"episodes": len(shards), "frames": offset, "output": str(output)}


def compatible_producer_hash(previous, current):
    """Explicit caller opt-in permits code changes only, preserving producer provenance."""
    for key in set(previous) | set(current):
        if key != "code_sha256" and previous.get(key) != current.get(key):
            raise ValueError(f"Checkpoint conversion settings changed: {key}")
    return hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()


def export_one(episode, paths, root, work, identity, config, args):
    sources = [
        {
            "path": str(p.relative_to(root)),
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in paths
    ]
    shard = work / episode.name
    token = {"contract_hash": identity, "sources": sources}
    if (shard / "done.json").exists():
        done = json.loads((shard / "done.json").read_text())
        old = done["identity"]
        accepted = {identity, *getattr(args, "compatible_producer_hashes", [])}
        if old["sources"] != sources or old["contract_hash"] not in accepted:
            raise ValueError(f"Source/config changed for completed episode {episode.name}")
        producer_hash = old["contract_hash"]
        result = done["result"]
    else:
        partial = work / (episode.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        result = convert_episode(paths, partial, config, args)
        after = [
            {
                "path": str(p.relative_to(root)),
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in paths
        ]
        if after != sources:
            raise ValueError("Source changed during conversion")
        (partial / "done.json").write_text(json.dumps({"identity": token, "result": result}, indent=2))
        os.replace(partial, shard)
        producer_hash = identity
    return shard, {
        "episode_id": episode.name,
        "files": sources,
        "producer_contract_hash": producer_hash,
        **result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--reuse-compatible-contract",
        type=Path,
        action="append",
        default=[],
        help="Explicitly accept reviewed code-only changes; retains original checkpoint hashes",
    )
    args = parser.parse_args(argv)
    root, output, config = (
        args.input.resolve(),
        args.output.resolve(),
        args.config.resolve(),
    )
    if args.workers <= 0 or args.fps <= 0 or args.limit is not None and args.limit <= 0:
        parser.error("fps and limit must be positive")
    if output.exists():
        raise FileExistsError(output)
    all_paths = sorted(root.rglob("*.mcap"))
    groups = {}
    for p in all_paths:
        groups.setdefault(p.parent.parent, []).append(p)
    selected = list(groups.items())[: args.limit] if args.limit else list(groups.items())
    if not selected:
        raise ValueError("No MCAP episodes found")
    contract = {
        "schema": SCHEMA,
        "rotation_representation": "rot6d_columns",
        "rotation_order": ["R00", "R10", "R20", "R01", "R11", "R21"],
        "frames": {"left": "left_fr3_link0", "right": "right_fr3_link0"},
        "tips": {"left": "left_fr3_link8", "right": "right_fr3_link8"},
        "tool_offset_m": 0,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "action_source": "recorded follower joint targets, selected causally by MCAP log_time",
        "gripper": {
            "encoding": "binary",
            "closed": 0,
            "open": 1,
            "threshold": 0.5,
            "knuckle_open_rad": 0,
            "knuckle_closed_rad": 0.8,
        },
        "image_size": [640, 480],
        "resize": "direct resize without crop",
        "fps": args.fps,
        "task": args.task,
        "sync": {
            "anchor": "head image header.stamp",
            "rate_gate": "nearest output-rate slot; keep first valid image per slot",
            "measured": "nearest header timestamp, 50ms tolerance",
            "targets": "last log_time <= anchor, max age 100ms",
            "wrist": "nearest header timestamp, 45ms tolerance",
        },
        "urdf_sha256": {s: sha256(config / f"{s}.urdf") for s in ("left", "right")},
        "code_sha256": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("labs_kinematics.py"),
                Path(__file__).with_name("action_spec.py"),
                Path(__file__).with_name("mcap_to_lerobot.py"),
            )
        },
        "action_spec": {
            "dimension": 20,
            "ee_dimension": 9,
            "ee_rotation": "rot6d_columns",
            "gripper_range": [0, 1],
        },
        "input_root": str(root),
        "source_episodes": [],
        "directories_without_mcap": [
            p.name for p in sorted(root.iterdir()) if p.is_dir() and p not in groups
        ],
    }
    work = output.with_name(output.name + ".work")
    work.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    producers = {identity: dict(contract)}
    args.compatible_producer_hashes = []
    for path in args.reuse_compatible_contract:
        previous = json.loads(path.read_text())
        previous_hash = compatible_producer_hash(previous, contract)
        args.compatible_producer_hashes.append(previous_hash)
        producers[previous_hash] = previous
    (work / f"producer_contract_{identity}.json").write_text(json.dumps(contract, indent=2) + "\n")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    results = {}
    failures = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = {
            pool.submit(export_one, episode, paths, root, work, identity, config, args): episode.name
            for episode, paths in selected
        }
        for number, future in enumerate(as_completed(futures), 1):
            try:
                shard, result = future.result()
            except Exception as exc:
                failures[futures[future]] = f"{type(exc).__name__}: {exc}"
                print(
                    json.dumps(
                        {
                            "progress": f"{number}/{len(selected)}",
                            "failed_episode": futures[future],
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )
                continue
            results[result["episode_id"]] = (shard, result)
            print(json.dumps({"progress": f"{number}/{len(selected)}", **result}), flush=True)
    if failures:
        raise RuntimeError(f"Episodes failed; successful checkpoints preserved: {failures}")
    shards = [results[episode.name][0] for episode, _ in selected]
    contract["source_episodes"] = [results[episode.name][1] for episode, _ in selected]
    contract["files_at_finish"] = len(list(root.rglob("*.mcap")))
    contract["producer_contracts"] = producers
    print(json.dumps(combine(shards, output, contract)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
