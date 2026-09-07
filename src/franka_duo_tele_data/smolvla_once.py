"""One-shot SmolVLA inference on live RGB20D observations, executed through the site joint servo.

Steps: read one synchronized head/wrist/state observation, send it to the
inference server, validate the returned [32][20] chunk, publish it once to the
joint servo with absolute step 0, then wait until the servo holds at the end.
Dry-run by default; robot publication requires --publish --enable-robot and
the same read-only controller checks as the recorded-action replay.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import io
import json
import threading
import time
from pathlib import Path

import numpy as np

from .action_spec import rot6d_to_matrix
from .cartesian_chunk import check_tracking, pose_distance
from .config_io import load_mapping
from .joint_servo_client import chunk_payload, parse_status
from .rgb20d_io import CAMERAS, RGB20DCache, RGB20DContract, RGB20DReader
from .smolvla_client import SmolVLAClient


def encode_png(image: np.ndarray) -> bytes:
    from PIL import Image

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be uint8 RGB HWC")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def sanitize_chunk(
    actions,
    contract: RGB20DContract,
    state: np.ndarray,
    *,
    max_step_m: float,
    max_step_rad: float,
    first_offset_m: float,
    first_offset_rad: float,
):
    """Re-orthonormalize rot6d rows, threshold grippers, and check jumps. Returns float32[H,20] and stats."""
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 20 or not np.isfinite(array).all():
        raise ValueError(f"server chunk must be finite [H,20], got {array.shape}")
    cleaned = array.copy()
    for offset in (0, 9):
        for row in cleaned:
            matrix = rot6d_to_matrix(row[offset + 3 : offset + 9])
            row[offset + 3 : offset + 9] = matrix[:2].reshape(-1)
    raw_grippers = cleaned[:, 18:].copy()
    cleaned[:, 18:] = (cleaned[:, 18:] >= 0.5).astype(np.float32)
    for row in cleaned:
        contract.validate_action(row)
    previous = state
    max_pos, max_rot = 0.0, 0.0
    for index, row in enumerate(cleaned):
        position, angle = pose_distance(previous, row)
        max_pos, max_rot = max(max_pos, float(position.max())), max(max_rot, float(angle.max()))
        # Row 0 is an offset from the live pose that the servo's Ruckig tracker
        # closes under its velocity limits; later rows are consecutive steps.
        limit_m, limit_rad = (first_offset_m, first_offset_rad) if index == 0 else (max_step_m, max_step_rad)
        if np.any(position > limit_m) or np.any(angle > limit_rad):
            raise ValueError(
                f"chunk row {index} exceeds limit: meters={position.tolist()} radians={angle.tolist()}"
            )
        previous = row
    return cleaned, {
        "rows": int(len(cleaned)),
        "max_step_m": max_pos,
        "max_step_rad": max_rot,
        "raw_gripper_min": raw_grippers.min(axis=0).tolist(),
        "raw_gripper_max": raw_grippers.max(axis=0).tolist(),
        "gripper_first": cleaned[0, 18:].tolist(),
        "gripper_last": cleaned[-1, 18:].tolist(),
    }


def run(args) -> int:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

    from .replay_rgb20d import check_joint_servo_controllers

    if args.publish != args.enable_robot:
        raise ValueError("robot publication requires both --publish and --enable-robot")
    contract = RGB20DContract(args.dataset)
    config = load_mapping(args.config)
    contract.action_spec = dataclasses.replace(
        contract.action_spec,
        workspace_min=tuple(config["workspace_min"]),
        workspace_max=tuple(config["workspace_max"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base_output = args.output
    node = None
    ros_thread = None
    latest = {"status": None, "status_ns": 0}
    lock = threading.Lock()
    rclpy.init()
    node = rclpy.create_node("franka_duo_smolvla_once")
    cache = RGB20DCache(history_size=120)
    reader = RGB20DReader(cache, contract, max_age_ms=config["max_input_age_ms"])
    for camera in CAMERAS:
        node.create_subscription(
            Image,
            config["topics"][camera],
            lambda msg, key=camera: cache.store_image(key, msg),
            qos_profile_sensor_data,
        )
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

    def on_status(msg):
        with lock:
            latest["status"], latest["status_ns"] = msg.data, time.monotonic_ns()

    node.create_subscription(String, config["joint_servo_status_topic"], on_status, 10)
    publisher = (
        node.create_publisher(Float32MultiArray, config["joint_servo_chunk_topic"], 10)
        if args.publish
        else None
    )
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    def status():
        with lock:
            text, stamp_ns = latest["status"], latest["status_ns"]
        if text is None:
            return None
        if time.monotonic_ns() - stamp_ns > 500_000_000:
            raise TimeoutError("joint servo status expired")
        return parse_status(text)

    if args.publish:
        check_joint_servo_controllers(node, config, args.speed)
        deadline = time.monotonic() + 15
        while publisher is not None and publisher.get_subscription_count() != 1:
            if time.monotonic() > deadline:
                raise TimeoutError("unique joint servo chunk subscriber is not present")
            time.sleep(0.02)
        initial = status()
        deadline = time.monotonic() + 5
        while initial is None:
            if time.monotonic() > deadline:
                raise TimeoutError("joint servo status not received")
            time.sleep(0.02)
            initial = status()
        if initial.fault:
            raise RuntimeError(f"joint servo is in fault hold: {initial.fault_reason}")
        if initial.started and not initial.holding:
            raise RuntimeError("joint servo is still executing a chunk")

    def one_round(round_index: int) -> None:
        observation = reader.next(timeout_s=5)
        images = {key: encode_png(observation.images[key]) for key in CAMERAS}
        state = observation.state
        print(
            json.dumps(
                {
                    "state": state.tolist(),
                    "image_shapes": {k: list(observation.images[k].shape) for k in CAMERAS},
                    "png_bytes": {k: len(v) for k, v in images.items()},
                }
            ),
            flush=True,
        )

        async def infer():
            async with SmolVLAClient(args.url, timeout=args.timeout) as client:
                started = time.perf_counter()
                result = await client.infer(state.tolist(), images, args.task)
                result["round_trip_ms"] = round((time.perf_counter() - started) * 1000, 1)
                return result

        result = asyncio.run(infer())
        raw = np.asarray(result["actions"], dtype=np.float32)
        args.output.write_text(
            json.dumps(
                {
                    "schema": "franka_duo_smolvla_once_raw_v1",
                    "inference_ms": result.get("inference_ms"),
                    "round_trip_ms": result.get("round_trip_ms"),
                    "state": state.tolist(),
                    "raw_actions": raw.tolist(),
                },
                indent=1,
            )
            + "\n"
        )
        # Per-row diagnostics before any guard can abort: offset from the
        # current state and step between consecutive rows, both arms.
        rows = []
        previous = state
        for index, row in enumerate(raw):
            try:
                offset_m, offset_rad = pose_distance(state, row)
                step_m, step_rad = pose_distance(previous, row)
                rows.append(
                    {
                        "row": index,
                        "offset_m": offset_m.round(4).tolist(),
                        "offset_rad": offset_rad.round(3).tolist(),
                        "step_m": step_m.round(4).tolist(),
                        "step_rad": step_rad.round(3).tolist(),
                        "grippers": row[18:].round(2).tolist(),
                    }
                )
            except ValueError as exc:
                rows.append({"row": index, "error": str(exc)})
            previous = row
        for entry in rows[:4] + rows[-2:]:
            print(json.dumps(entry), flush=True)
        print(
            json.dumps(
                {
                    "inference_ms": result.get("inference_ms"),
                    "round_trip_ms": result.get("round_trip_ms"),
                    "max_step_m": max(max(r["step_m"]) for r in rows if "step_m" in r),
                    "max_step_rad": max(max(r["step_rad"]) for r in rows if "step_rad" in r),
                    "final_offset_m": rows[-1].get("offset_m"),
                }
            ),
            flush=True,
        )
        actions, stats = sanitize_chunk(
            result["actions"],
            contract,
            state,
            max_step_m=config["max_target_step_m"],
            max_step_rad=config["max_target_step_rad"],
            first_offset_m=args.max_first_offset_m,
            first_offset_rad=args.max_first_offset_rad,
        )
        first_pos, first_rot = pose_distance(state, actions[0])
        record = {
            "schema": "franka_duo_smolvla_once_v1",
            "url": args.url,
            "task": args.task,
            "inference_ms": result.get("inference_ms"),
            "round_trip_ms": result.get("round_trip_ms"),
            "chunk_size": result.get("chunk_size"),
            "state": state.tolist(),
            "raw_actions": np.asarray(result["actions"], dtype=np.float32).tolist(),
            "actions": actions.tolist(),
            "first_row_offset_m": first_pos.tolist(),
            "first_row_offset_rad": first_rot.tolist(),
            **stats,
            "published": bool(args.publish),
        }
        args.output.write_text(json.dumps(record, indent=1) + "\n")
        print(
            json.dumps({k: v for k, v in record.items() if k not in ("raw_actions", "actions", "state")}),
            flush=True,
        )
        check_tracking(state, actions[0], max_m=args.max_first_offset_m, max_rad=args.max_first_offset_rad)
        if publisher is None:
            print(json.dumps({"dry_run": True, "robot_commands_published": 0}), flush=True)
            return

        # On a servo that already ran a chunk, append after its current step
        # plus a lead so the commit boundary is in the future; otherwise start at 0.
        current = status()
        start_step = 0
        if current is not None and current.started:
            start_step = int(np.ceil(current.step)) + args.append_lead_steps
        link0 = np.stack([contract.action_spec.to_link0_action(row) for row in actions])
        data, dims, offset = chunk_payload(link0, start_step)
        message = Float32MultiArray(data=data)
        message.layout.dim = [
            MultiArrayDimension(label="rows", size=dims[0], stride=dims[0] * dims[1]),
            MultiArrayDimension(label="action", size=dims[1], stride=dims[1]),
        ]
        message.layout.data_offset = offset
        publisher.publish(message)
        end_step = start_step + len(actions) - 1
        deadline = time.monotonic() + 20 + len(actions) / (contract.fps * args.speed) * 1.5
        max_track = 0.0
        while True:
            time.sleep(0.05)
            current = status()
            if current is not None and current.fault:
                raise RuntimeError(f"joint servo fault: {current.fault_reason}")
            if current is not None:
                max_track = max(max_track, current.tracking_error_rad)
            if (
                current is not None
                and current.started
                and current.holding
                and current.last_step >= end_step
                and current.step >= end_step
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("joint servo did not finish the chunk in time")
        final = reader.next(timeout_s=3)
        end_pos, end_rot = pose_distance(final.state, actions[-1])
        print(
            json.dumps(
                {
                    "round": round_index,
                    "start_step": start_step,
                    "executed_rows": len(actions),
                    "max_servo_tracking_error_rad": max_track,
                    "end_position_error_m": end_pos.tolist(),
                    "end_rotation_error_rad": end_rot.tolist(),
                    "end_grippers": final.state[18:].tolist(),
                }
            ),
            flush=True,
        )

    try:
        for round_index in range(args.repeat):
            if args.repeat > 1:
                args.output = base_output.with_name(
                    f"{base_output.stem}_r{round_index + 1}{base_output.suffix}"
                )
            one_round(round_index + 1)
        return 0
    except KeyboardInterrupt:
        # Stop requesting; the servo keeps holding the last commanded pose and the
        # relay keeps the impedance controller fed, so nothing else needs to happen.
        print(json.dumps({"stopped_by_user": True}), flush=True)
        return 130
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if ros_thread is not None:
            ros_thread.join(timeout=2)
        if node is not None:
            node.destroy_node()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/tmr_rgb20d.yaml"))
    parser.add_argument("--url", default="ws://100.86.181.61:8081/infer")
    parser.add_argument("--task", default="pick cup and bowl")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--speed", type=float, default=0.1, help="must equal the servo playback_speed")
    parser.add_argument("--repeat", type=int, default=1, help="observe/infer/execute rounds, sequential")
    parser.add_argument("--append-lead-steps", type=int, default=3)
    parser.add_argument("--max-first-offset-m", type=float, default=0.12)
    parser.add_argument("--max-first-offset-rad", type=float, default=0.3)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/smolvla_once.json"))
    args = parser.parse_args(argv)
    if not 0 < args.speed <= 1:
        parser.error("speed must be in (0,1]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
