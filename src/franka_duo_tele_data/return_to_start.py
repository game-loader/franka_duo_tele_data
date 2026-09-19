"""Return to archived episode 0's first observation through the running site servo.

No PTP, camera input or MCAP recording. Publication requires both robot gates.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from .action_spec import matrix_to_rot6d, rot6d_to_matrix
from .cartesian_chunk import pose_distance
from .config_io import load_mapping
from .joint_servo_client import ChunkPacer, chunk_payload, parse_status
from .replay_rgb20d import check_joint_servo_controllers
from .rgb20d_io import RGB20DCache, RGB20DContract, RobotStateReader

ROOT = Path(__file__).resolve().parents[2]


def load_target(path, contract, config):
    payload = json.loads(Path(path).read_text())
    if (
        payload.get("source_field") != "observation.state"
        or payload.get("episode_index") != 0
        or payload.get("frame_index") != 0
        or payload.get("timestamp") != 0.0
    ):
        raise ValueError("target must be episode 0 frame 0 observation.state")
    if payload.get("fps") != contract.fps:
        raise ValueError("target and dataset timing differ")
    for side in ("left", "right"):
        key = f"T_newbase_from_{side}_arm_base"
        archived = np.asarray(payload["coordinate_transforms"][key], dtype=float)
        if archived.shape != (4, 4) or not np.allclose(
            archived, contract.transforms[side], atol=1e-7, rtol=0
        ):
            raise ValueError("archive coordinate transforms differ from the live contract")
    for key in ("closed_position", "open_position", "encoding"):
        if payload["gripper_calibration"][key] != contract.gripper[key]:
            raise ValueError("archive gripper calibration differs from the live contract")
    target_spec = dataclasses.replace(
        contract.action_spec,
        workspace_min=tuple(config["workspace_min"]),
        workspace_max=tuple(config["workspace_max"]),
    )
    target = target_spec.validate(payload["state"], clip=False).astype(float)
    return target, payload, target_spec


def physical_opening(position, calibration):
    closed = float(calibration["closed_position"])
    opened = float(calibration["open_position"])
    if not np.isfinite([position, closed, opened]).all() or closed == opened:
        raise ValueError("invalid gripper encoder/calibration")
    if not min(closed, opened) - 0.01 <= position <= max(closed, opened) + 0.01:
        raise ValueError("gripper encoder position outside calibration")
    return float(np.clip((position - closed) / (opened - closed), 0, 1))


def plan_return(state, target, spec, hz):
    """Quintic translation and rotation interpolation with physical grippers held.

    The caller validates the endpoint workspace. The observed starting pose can
    be outside that policy workspace; the path stays within its endpoint box.
    """
    from scipy.spatial.transform import Rotation, Slerp

    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("action rate must be finite and positive")
    state = spec.validate(state, clip=False).astype(float)
    target = spec.validate(target, clip=False).astype(float)
    target[18:] = state[18:]
    distance, angle = pose_distance(state, target)
    # Quintic easing has peak derivative 1.875. Respect both physical speed
    # (25 mm/s, 0.10 rad/s) and per-row limits (3 mm, 0.012 rad).
    duration = max(
        5.0,
        1.875 * float(max(distance)) / min(0.025, 0.003 * hz),
        1.875 * float(max(angle)) / min(0.10, 0.012 * hz),
    )
    count = math.ceil(duration * hz) + 1
    t = np.linspace(0, 1, count)
    fraction = np.clip(10 * t**3 - 15 * t**4 + 6 * t**5, 0, 1)
    rows = np.tile(state, (count, 1))
    for offset in (0, 9):
        a, b = state[offset : offset + 3], target[offset : offset + 3]
        rows[:, offset : offset + 3] = a + fraction[:, None] * (b - a)
        rotation = Rotation.from_matrix(
            np.stack(
                [
                    rot6d_to_matrix(state[offset + 3 : offset + 9]),
                    rot6d_to_matrix(target[offset + 3 : offset + 9]),
                ]
            )
        )
        rows[:, offset + 3 : offset + 9] = matrix_to_rot6d(Slerp([0, 1], rotation)(fraction).as_matrix())
        if np.any(rows[:, offset : offset + 3] < np.minimum(a, b) - 1e-6) or np.any(
            rows[:, offset : offset + 3] > np.maximum(a, b) + 1e-6
        ):
            raise ValueError("reposition path left the endpoint corridor")
    rows = np.concatenate((rows, np.tile(target, (math.ceil(2 * hz), 1))))
    previous = state
    for row in rows:
        spec.validate(row, clip=False)
        pos, rot = pose_distance(previous, row)
        if max(pos) > 0.003 + 1e-7 or max(rot) > 0.012 + 1e-7:
            raise ValueError("reposition step limit exceeded")
        previous = row
    return rows


def start_step(status):
    hz = status.action_rate_hz * status.playback_speed
    return math.ceil(status.step) + math.ceil(hz) if status.started else 0


def check_ack(status, before, start, count):
    if (
        status.chunks != before + 1
        or status.last_chunk_start_step != start
        or status.last_step != start + count - 1
    ):
        raise RuntimeError("unexpected chunk acknowledgement or skipped trajectory prefix")


def run_ros(args, contract, config, target, payload, target_spec, output, record):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

    node = None
    thread = None
    rclpy.init()
    try:
        node = rclpy.create_node("servo_return_to_dataset_start")
        cache = RGB20DCache()
        latest = {}
        state_lock = threading.Lock()
        for side in ("left", "right"):
            node.create_subscription(
                PoseStamped,
                config["topics"][f"{side}_pose"],
                getattr(cache, f"store_{side}_pose"),
                qos_profile_sensor_data,
            )
            node.create_subscription(
                JointState,
                config["topics"][f"{side}_gripper"],
                getattr(cache, f"store_{side}_gripper_states"),
                qos_profile_sensor_data,
            )

        def on_status(message):
            with state_lock:
                latest["status"] = (message.data, time.monotonic())

        node.create_subscription(String, config["joint_servo_status_topic"], on_status, 10)
        reader = RobotStateReader(cache, contract, max_age_ms=config["max_input_age_ms"])
        thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        thread.start()

        def status():
            with state_lock:
                entry = latest.get("status")
            if entry is None:
                return None
            text, received = entry
            age = time.monotonic() - received
            if age > 0.25:
                raise TimeoutError("servo status stale")
            s = parse_status(text)
            if s.fault:
                raise RuntimeError(f"servo fault: {s.fault_reason}")
            if (
                not math.isclose(s.action_rate_hz, contract.fps)
                or not math.isclose(s.playback_speed, args.speed)
                or not math.isfinite(s.step)
                or s.commit_lead_steps != 0
                or s.blend_steps != 4
                or s.blend_mode != "quintic_hold_v1"
            ):
                raise ValueError("servo timing/blending must match the configured inference servo")
            if s.started:
                s = dataclasses.replace(s, step=s.step + age * s.action_rate_hz * s.playback_speed)
            return s

        def live_state():
            state = reader.next(timeout_s=5).state.astype(float)
            snapshot = cache.snapshot()
            for i, side in enumerate(("left", "right")):
                samples = snapshot[f"{side}_gripper_states"]
                if not samples or (time.monotonic_ns() - samples[-1].arrival_ns) / 1e9 > 0.2:
                    raise RuntimeError("gripper feedback stale")
                position = samples[-1].message.position
                if not position:
                    raise ValueError("gripper encoder position missing")
                state[18 + i] = physical_opening(float(position[0]), contract.gripper)
            return state

        deadline = time.monotonic() + 10
        while status() is None:
            if time.monotonic() > deadline:
                raise TimeoutError("servo unavailable; start the site servo first")
            time.sleep(0.05)
        if not status().holding:
            raise RuntimeError("servo must be holding before repositioning; stop inference first")
        check_joint_servo_controllers(node, config, args.speed)
        topic = config["joint_servo_chunk_topic"]
        if node.count_publishers(topic) != 0:
            raise RuntimeError("another command publisher exists; stop inference first")
        state = live_state()
        hz = contract.fps * args.speed
        rows = plan_return(state, target, contract.action_spec, hz)
        target = rows[-1]
        link0 = np.stack([contract.action_spec.to_link0_action(row) for row in rows])
        np.save(output / "midpoint_targets.npy", rows)
        (output / "target_source.json").write_text(json.dumps(payload, indent=2) + "\n")
        distance, angle = pose_distance(state, target)
        record(
            {
                "event": "plan",
                "published": args.publish,
                "duration_s": round(len(rows) / hz, 2),
                "row_count": len(rows),
                "left_start": state[:3].tolist(),
                "right_start": state[9:12].tolist(),
                "left_target": target[:3].tolist(),
                "right_target": target[9:12].tolist(),
                "distance_m": distance.tolist(),
                "rotation_rad": angle.tolist(),
                "grippers_held": state[18:].tolist(),
                "output": str(output),
            }
        )
        if not args.publish:
            return

        publisher = node.create_publisher(Float32MultiArray, topic, 10)

        def check_publishers():
            if node.count_publishers(topic) != 1 or publisher.get_subscription_count() != 1:
                raise RuntimeError("expected exactly one command publisher and one servo subscriber")

        deadline = time.monotonic() + 10
        while publisher.get_subscription_count() != 1:
            if time.monotonic() > deadline:
                raise TimeoutError("unique servo subscriber missing")
            time.sleep(0.05)
        check_publishers()
        fresh = live_state()
        pos, rot = pose_distance(fresh, state)
        s = status()
        if (
            max(pos) > 0.005
            or max(rot) > 0.03
            or not s.holding
            or np.max(np.abs(fresh[18:] - state[18:])) > 0.01
        ):
            raise RuntimeError("robot or grippers moved since planning")
        base_step = start_step(s)
        pacer = ChunkPacer(32, len(rows))
        deadline = time.monotonic() + 30 + len(rows) / hz * 1.5
        reported = 0
        max_tracking = 0.0
        while time.monotonic() < deadline:
            current = status()
            check_publishers()
            max_tracking = max(max_tracking, current.tracking_error_rad)
            relative = dataclasses.replace(
                current,
                started=pacer.last_start >= 0,
                step=current.step - base_step,
                last_step=current.last_step - base_step,
            )
            start = pacer.next_start(relative)
            if start is not None:
                data, dims, offset = chunk_payload(link0[start : start + 32], base_step + start)
                message = Float32MultiArray(data=data)
                message.layout.dim = [
                    MultiArrayDimension(label="rows", size=dims[0], stride=dims[0] * 20),
                    MultiArrayDimension(label="action", size=20, stride=20),
                ]
                message.layout.data_offset = offset
                before = current.chunks
                record({"event": "command", "start_step": offset, "row_count": dims[0]})
                publisher.publish(message)
                ack_deadline = time.monotonic() + 3
                ack = status()
                while ack.chunks == before:
                    check_publishers()
                    if time.monotonic() > ack_deadline:
                        raise TimeoutError("servo did not acknowledge chunk; inspect IK log")
                    time.sleep(0.02)
                    ack = status()
                check_ack(ack, before, offset, dims[0])
                pacer.record(start)
            if pacer.finished(relative):
                break
            if time.monotonic() - reported > 5:
                record(
                    {
                        "event": "progress",
                        "step": round(relative.step, 1),
                        "total_rows": len(rows),
                        "chunks": current.chunks,
                        "tracking_rad": current.tracking_error_rad,
                    }
                )
                reported = time.monotonic()
            time.sleep(0.03)
        else:
            raise TimeoutError("return to start timed out")
        time.sleep(1)
        final = live_state()
        check_publishers()
        final_status = status()
        if not final_status.holding or final_status.last_step != base_step + len(rows) - 1:
            raise RuntimeError("servo did not hold the final target")
        pos, rot = pose_distance(final, target)
        record(
            {
                "event": "final",
                "position_error_m": pos.tolist(),
                "rotation_error_rad": rot.tolist(),
                "max_tracking_rad": max_tracking,
                "servo": dataclasses.asdict(final_status),
            }
        )
        if max(pos) > 0.01 or max(rot) > 0.05:
            raise RuntimeError("final EE error exceeds tolerance")
        target_spec.validate(final, clip=False)
    finally:
        rclpy.shutdown()
        if thread is not None:
            thread.join(timeout=2)
        if node is not None:
            node.destroy_node()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/franka_duo_lerobot_rgb20d_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tmr_rgb20d.yaml")
    parser.add_argument("--target", type=Path, default=ROOT / "configs/dataset_initial_state.json")
    parser.add_argument("--speed", type=float, default=0.3, help="must match the running servo")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    return parser


def run(args):
    if args.publish != args.enable_robot:
        raise ValueError("both --publish and --enable-robot are required")
    if not math.isfinite(args.speed) or not 0 < args.speed <= 1:
        raise ValueError("speed must be finite and in (0, 1]")
    config = load_mapping(args.config)
    # This entrypoint only addresses the site-owned relay, never controller commands.
    for key, suffix in (("joint_servo_chunk_topic", "action_chunk"), ("joint_servo_status_topic", "status")):
        if config.get(key) != f"/franka_duo/joint_servo/{suffix}":
            raise ValueError(f"{key} must use the standard site joint_servo topic")
    contract = RGB20DContract(args.dataset)
    target, payload, target_spec = load_target(args.target, contract, config)
    lock_path = ROOT / "log/servo_start/.smolvla_loop.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another policy loop or return script is running; stop it first") from exc
        output = ROOT / "outputs" / f"return_to_start_{time.time_ns()}"
        output.mkdir(parents=True)
        with (output / "events.jsonl").open("w") as log:

            def record(event):
                line = json.dumps(event, allow_nan=False)
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)

            try:
                run_ros(args, contract, config, target, payload, target_spec, output, record)
            except BaseException as exc:
                record({"event": "error", "error": str(exc)})
                raise
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.publish != args.enable_robot:
        parser.error("both --publish and --enable-robot are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
