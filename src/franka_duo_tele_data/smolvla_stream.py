"""Stream 32-row SmolVLA chunks at 9 Hz, requesting again with 400 ms left.

The policy keeps its 30 Hz dataset contract; playback_speed=0.3 makes the
execution timeline advance at 9 Hz. Requests use one persistent connection.
Rows retain an absolute time anchored to the observation; the servo discards
expired rows and blends the future suffix while its Ruckig state stays intact.
Robot publication requires both --publish and --enable-robot.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import io
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from .cartesian_chunk import pose_distance
from .config_io import load_mapping
from .joint_servo_client import chunk_payload, parse_status
from .rgb20d_io import CAMERAS, RGB20DCache, RGB20DContract, RGB20DReader
from .smolvla_client import SmolVLAClient
from .smolvla_once import sanitize_chunk


def encode_images(images, image_format: str) -> dict[str, bytes]:
    from PIL import Image

    encoded = {}
    for name, pixels in images.items():
        buffer = io.BytesIO()
        options = {"quality": 95, "subsampling": 0} if image_format == "jpeg" else {}
        Image.fromarray(pixels).save(buffer, format=image_format.upper(), **options)
        encoded[name] = buffer.getvalue()
    return encoded


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
    action_hz = contract.fps * args.speed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latest = {"status": None, "status_ns": 0}
    lock = threading.Lock()
    rclpy.init()
    node = rclpy.create_node("franka_duo_smolvla_stream")
    cache = RGB20DCache(history_size=120)
    reader = RGB20DReader(cache, contract, max_age_ms=config["max_input_age_ms"])
    for camera in CAMERAS:
        node.create_subscription(
            Image,
            config["topics"][camera],
            lambda m, k=camera: cache.store_image(k, m),
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
        if publisher is None:
            return None
        with lock:
            text, received_ns = latest["status"], latest["status_ns"]
        if text is None:
            return None
        age_s = (time.monotonic_ns() - received_ns) / 1e9
        if age_s > 0.25:
            raise TimeoutError("joint servo status expired")
        value = parse_status(text)
        if value.fault:
            raise RuntimeError(f"joint servo fault: {value.fault_reason}")
        # Account for the 30 Hz status sampling interval when applying a
        # millisecond trigger. The servo itself uses a steady absolute clock.
        if value.started:
            value = dataclasses.replace(
                value, step=value.step + age_s * value.action_rate_hz * value.playback_speed
            )
        return value

    def remaining_ms(value):
        if value is None or not value.started:
            return None
        return (value.last_step - value.step) / action_hz * 1000

    log = args.output.open("w")

    def record(value):
        log.write(json.dumps(value, allow_nan=False) + "\n")
        log.flush()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in value.items()
                    if k not in ("state", "actions", "raw_actions", "source_stamps_ns")
                },
                allow_nan=False,
            ),
            flush=True,
        )

    try:
        if publisher is not None:
            check_joint_servo_controllers(node, config, args.speed)
            deadline = time.monotonic() + 10
            while True:
                if node.count_publishers(config["joint_servo_chunk_topic"]) > 1:
                    raise RuntimeError("another action-chunk publisher is running")
                current = status()
                if current is not None and publisher.get_subscription_count() == 1:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("unique joint servo subscriber/status is not present")
                time.sleep(0.02)
            if (
                not math.isclose(current.playback_speed, args.speed, abs_tol=1e-9)
                or not math.isclose(current.action_rate_hz, contract.fps, abs_tol=1e-9)
                or current.commit_lead_steps != 0
                or current.blend_steps != 4
                or current.blend_mode != "quintic_hold_v1"
            ):
                raise ValueError(
                    "servo requires matching 30 Hz/speed, commit_lead_steps=0, "
                    "blend_steps=4 and quintic_hold_v1; restart it with start_servo_and_activate.sh"
                )
        record(
            {
                "event": "configuration",
                "action_hz": action_hz,
                "playback_speed": args.speed,
                "request_lead_ms": args.request_lead_ms,
                "image_format": args.image_format,
                "published": args.publish,
            }
        )

        async def session():
            sent = 0
            final_end = None
            started = time.monotonic()
            async with SmolVLAClient(args.url, timeout=args.timeout) as client:
                while sent < args.max_chunks and (
                    args.duration_s == 0 or time.monotonic() - started < args.duration_s
                ):
                    current = status()
                    if publisher is not None and current is None:
                        raise TimeoutError("joint servo status lost")
                    remaining = remaining_ms(current)
                    if publisher is not None and remaining is not None and remaining > args.request_lead_ms:
                        await asyncio.sleep(0.01)
                        continue
                    observation = reader.next(timeout_s=3)
                    state = observation.state
                    encode_start = time.perf_counter()
                    images = encode_images(observation.images, args.image_format)
                    encode_ms = (time.perf_counter() - encode_start) * 1000
                    requested = status()
                    source_age_s = (time.time_ns() - observation.stamp_ns) / 1e9
                    if not -0.05 <= source_age_s <= 1.0:
                        raise TimeoutError("observation timestamp is stale or camera/robot clocks differ")
                    start_step = 0
                    if requested is not None and requested.started:
                        observation_step = requested.step - max(0.0, source_age_s) * action_hz
                        start_step = max(0, math.floor(observation_step) + 1)
                    request_remaining = remaining_ms(requested)
                    record(
                        {
                            "event": "request",
                            "chunk": sent + 1,
                            "t_s": round(time.monotonic() - started, 3),
                            "remaining_ms_at_request": None
                            if request_remaining is None
                            else round(request_remaining, 1),
                            "observation_age_ms": round(source_age_s * 1000, 1),
                            "start_step": start_step,
                            "image_bytes": sum(map(len, images.values())),
                            "encode_ms": round(encode_ms, 1),
                            "source_stamps_ns": observation.source_stamps_ns,
                        }
                    )
                    t0 = time.perf_counter()
                    result = await client.infer(state.tolist(), images, args.task)
                    round_trip_ms = (time.perf_counter() - t0) * 1000
                    raw = np.asarray(result["actions"], dtype=np.float32)
                    if raw.shape != (args.horizon, 20):
                        raise ValueError(f"expected [{args.horizon},20] actions, received {raw.shape}")
                    # Persist the raw result before any motion validation can reject it.
                    record(
                        {
                            "event": "response",
                            "chunk": sent + 1,
                            "request_id": result.get("request_id"),
                            "inference_ms": result.get("inference_ms"),
                            "round_trip_ms": round(round_trip_ms, 1),
                            "state": state.tolist(),
                            "raw_actions": raw.tolist(),
                        }
                    )
                    actions, stats = sanitize_chunk(
                        raw,
                        contract,
                        state,
                        max_step_m=config["max_target_step_m"],
                        max_step_rad=config["max_target_step_rad"],
                        first_offset_m=args.max_first_offset_m,
                        first_offset_rad=args.max_first_offset_rad,
                    )
                    current = status()
                    end_step = start_step + len(actions) - 1
                    if publisher is not None:
                        if current is None:
                            raise TimeoutError("joint servo status lost after inference")
                        min_commit = (
                            math.ceil(current.step) + current.commit_lead_steps if current.started else 0
                        )
                        if end_step - min_commit < current.blend_steps:
                            raise TimeoutError(
                                "inference returned too late for a complete blend; keeping hold"
                            )
                    first_pos, first_rot = pose_distance(state, actions[0])
                    record(
                        {
                            "event": "publish" if publisher is not None else "dry_run",
                            "chunk": sent + 1,
                            "start_step": start_step,
                            "end_step": end_step,
                            "remaining_ms_on_response": remaining_ms(current),
                            "first_offset_m": first_pos.tolist(),
                            "first_offset_rad": first_rot.tolist(),
                            "max_step_m": stats["max_step_m"],
                            "actions": actions.tolist(),
                        }
                    )
                    if publisher is None:
                        record({"event": "finished", "dry_run": True, "robot_commands_published": 0})
                        return 1, None
                    before_chunks = current.chunks
                    link0 = np.stack([contract.action_spec.to_link0_action(row) for row in actions])
                    data, dims, offset = chunk_payload(link0, start_step)
                    message = Float32MultiArray(data=data)
                    message.layout.dim = [
                        MultiArrayDimension(label="rows", size=dims[0], stride=dims[0] * dims[1]),
                        MultiArrayDimension(label="action", size=dims[1], stride=dims[1]),
                    ]
                    message.layout.data_offset = offset
                    publisher.publish(message)
                    # Do not trigger again from the previous chunk's stale last_step.
                    # A rejected chunk must not be reported as accepted/executed.
                    ack_deadline = time.monotonic() + 2
                    while True:
                        current = status()
                        if current is not None and current.chunks > before_chunks:
                            if current.chunks != before_chunks + 1 or current.last_step != end_step:
                                raise RuntimeError("unexpected servo chunk acknowledgement")
                            break
                        if time.monotonic() > ack_deadline:
                            raise TimeoutError("servo did not accept the chunk; inspect its IK/velocity log")
                        await asyncio.sleep(0.01)
                    sent += 1
                    final_end = end_step
                    record(
                        {
                            "event": "accepted",
                            "chunk": sent,
                            "servo_chunks": current.chunks,
                            "servo_step": round(current.step, 3),
                            "last_step": current.last_step,
                            "committed_start_step": current.last_chunk_start_step,
                            "skipped_expired_rows": max(0, current.last_chunk_start_step - start_step),
                            "tracking_error_rad": current.tracking_error_rad,
                        }
                    )
            return sent, final_end

        sent, final_end = asyncio.run(session())
        if publisher is None:
            return 0
        deadline = time.monotonic() + 20 + args.horizon / action_hz * 1.5
        while final_end is not None:
            current = status()
            if (
                current is not None
                and current.holding
                and current.step >= final_end
                and current.last_step >= final_end
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("servo did not finish the final accepted chunk")
            time.sleep(0.05)
        final = reader.next(timeout_s=3)
        record(
            {
                "event": "finished",
                "chunks_sent": sent,
                "final_state": final.state.tolist(),
                "final_servo_step": None if current is None else round(current.step, 3),
            }
        )
        return 0
    except KeyboardInterrupt:
        record({"event": "stopped_by_user", "note": "accepted plan finishes; servo and relay keep holding"})
        return 130
    finally:
        log.close()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=2)
        node.destroy_node()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/tmr_rgb20d.yaml"))
    parser.add_argument("--url", default="ws://100.86.181.61:8081/infer")
    parser.add_argument("--task", default="pick cup and bowl")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--speed", type=float, default=0.3, help="must equal the servo playback_speed")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--request-lead-ms", type=float, default=400.0)
    parser.add_argument("--image-format", choices=("jpeg", "png"), default="jpeg")
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--duration-s", type=float, default=300.0, help="0 disables the duration limit")
    parser.add_argument("--max-first-offset-m", type=float, default=0.12)
    parser.add_argument("--max-first-offset-rad", type=float, default=0.3)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/smolvla_stream.jsonl"))
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.speed)
        or not 0 < args.speed <= 1
        or args.max_chunks < 1
        or args.horizon != 32
        or not math.isfinite(args.request_lead_ms)
        or args.request_lead_ms <= 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.duration_s)
        or args.duration_s < 0
    ):
        parser.error("speed in (0,1], max-chunks >= 1, horizon=32, positive timeouts, duration >= 0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
