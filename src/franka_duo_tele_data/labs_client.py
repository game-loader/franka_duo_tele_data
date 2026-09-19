"""Labs state34 / request-observation delta14 WebSocket client. Dry-run by default."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .labs_action_delta import DELTA_NAMES
from .labs_inference import (
    CAMERAS,
    CHUNK_REFERENCE,
    COMMAND_SCHEMA,
    COMMAND_TOPIC,
    DELTA_REPRESENTATION,
    FR3_REPRESENTATION,
    GRIPPER_POSTPROCESS,
    RETURN_SCHEMA,
    STATUS_SCHEMA,
    STATUS_TOPIC,
    TOPICS,
    LabsContract,
    LabsObservationCache,
    episode_start,
    reconstruct_chunk,
    start_identity,
    validate_response,
)
from .labs_mcap_to_lerobot import STATE_NAMES
from .smolvla_client import SmolVLAClient


class LabsClient(SmolVLAClient):
    """Reuse the binary transport with the FR3 FastWAM subprotocol."""

    action_representation = DELTA_REPRESENTATION
    subprotocol = "fastwam.msgpack.v1"

    def __init__(self, url, contract, timeout=10, *, server_profile="labs"):
        if server_profile not in ("labs", "c23"):
            raise ValueError("Unknown Labs server profile")
        self.server_profile = server_profile
        super().__init__(url, timeout=timeout)
        self.contract = contract
        self.health = None
        self.response_sink = None

    async def __aenter__(self):
        await super().__aenter__()
        try:
            if self.ws.protocol != self.subprotocol:
                raise ValueError(f"Server must negotiate {self.subprotocol}")
            url = urlsplit(self.url)
            health_url = urlunsplit(
                ("https" if url.scheme == "wss" else "http", url.netloc, "/health", "", "")
            )
            async with self.session.get(health_url, timeout=self.timeout) as response:
                response.raise_for_status()
                self.health = await response.json()
            expected = {
                "state_dim": 34,
                "action_dim": 14,
                "horizon": 32,
                "action_rate_hz": 30,
                "state_layout": STATE_NAMES,
                "action_layout": DELTA_NAMES,
            }
            if (
                self.health.get("ready") is not True
                or self.health.get("busy") is not False
                or self.health.get("normalized") is not False
                or self.health.get("action_representation") not in (DELTA_REPRESENTATION, FR3_REPRESENTATION)
                or any(self.health.get(k) != v for k, v in expected.items())
                or self.health.get("task") != self.contract.task
            ):
                raise ValueError("FastWAM health does not match the Labs state34/action14 contract")
            if self.server_profile == "c23" and (
                self.health.get("model") != "FastWAM-FR3-C23" or self.health.get("variant") != "c23"
            ):
                raise ValueError("C23 profile requires model=FastWAM-FR3-C23 and variant=c23")
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def infer(self, state, images, task=None):
        from PIL import Image

        state = self.contract.validate_state(state)
        if set(images) != set(CAMERAS):
            raise ValueError("Provide head, wrist_left and wrist_right images")
        for data in images.values():
            with Image.open(io.BytesIO(data)) as image:
                if image.mode != "RGB" or image.size != (640, 480):
                    raise ValueError("Labs wire images must be 640x480 RGB PNG/JPEG")
                image.verify()
        result = await super().infer(state.tolist(), images, task or self.contract.task)
        if self.response_sink is not None:
            self.response_sink(result)
        actions = validate_response(result)
        if result["action_representation"] != self.health["action_representation"] or len(actions) != 32:
            raise ValueError("Response representation/horizon disagrees with FastWAM health")
        return result


def encode_images(images):
    from PIL import Image

    result = {}
    for name, pixels in images.items():
        out = io.BytesIO()
        Image.fromarray(pixels).save(out, format="PNG", compress_level=1)
        result[name] = out.getvalue()
    return result


def command_for(result, observation, contract, *, speed=0.3):
    if not np.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("Require 0 < speed <= 1")
    return {
        "schema": COMMAND_SCHEMA,
        "mode": "policy_chunk",
        "command_id": uuid.uuid4().hex,
        "request_id": result["request_id"],
        "chunk_reference": CHUNK_REFERENCE,
        "gripper_postprocess": GRIPPER_POSTPROCESS,
        "model_hashes": contract.model_hashes,
        "created_ns": time.time_ns(),
        "observation_ns": observation.stamp_ns,
        "reference_state": observation.state.tolist(),
        "targets": reconstruct_chunk(result, observation.state).tolist(),
        "action_rate_hz": 30,
        "speed": speed,
        "execution_rate_hz": 30 * speed,
    }


def start_command(start_state, contract, episode, *, speed=0.3):
    # The relay reads current joints itself. No camera/FK/IK observation is
    # needed to execute this separate, measured-joint positioning operation.
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0 < speed <= 1:
        raise ValueError("Return requires 0 < speed <= 1")
    return {
        "schema": RETURN_SCHEMA,
        "mode": "return_to_start",
        "command_id": uuid.uuid4().hex,
        "created_ns": time.time_ns(),
        "model_hashes": contract.model_hashes,
        "start_episode": episode,
        "start_identity": start_identity(start_state, episode, contract),
        "speed": speed,
        "execution_rate_hz": 30 * speed,
    }


async def return_before_inference(command, *, publish, wait_hold, record):
    """Once per session; a failed return prevents all inference requests."""
    record({"event": "return_to_start", "command": command, "published": False})
    if publish is None:
        return
    await wait_hold()
    publish(command)
    record({"event": "published", "command_id": command["command_id"]})
    status = await wait_hold(command["command_id"], timeout=120)
    if status.get("completed_start_identity") != command["start_identity"]:
        raise RuntimeError("Relay did not verify the selected episode start")
    record({"event": "completed", "command_id": command["command_id"], "status": status})


def require_relay(status, contract, *, command_id=None):
    if status.get("schema") != STATUS_SCHEMA or status.get("model_hashes") != contract.model_hashes:
        raise ValueError("Incompatible Labs site relay/model")
    if status.get("fault"):
        raise RuntimeError(f"Labs relay fault: {status['fault']}")
    if status.get("enabled") is not True:
        raise RuntimeError("Labs relay robot output is disabled")
    if command_id is not None and status.get("command_id") != command_id:
        return False
    return status.get("ready") is True and status.get("phase") == "holding"


def restore_preview(args, contract, start_state):
    """Offline target inspection; do not require ROS, cameras, server or IK."""
    args.output.mkdir(parents=True, exist_ok=False)
    command = start_command(start_state, contract, args.start_episode, speed=args.speed)
    record = {
        "event": "return_to_start",
        "command": command,
        "published": False,
        "dataset": str(args.dataset),
        "target_joints": start_state[20:].tolist(),
    }
    (args.output / "trace.jsonl").write_text(json.dumps(record, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "restore": "dry-run",
                "episode": args.start_episode,
                "target_joints": start_state[20:].tolist(),
                "output": str(args.output),
            }
        )
    )
    return 0


def launch_relay(args):
    """Keep the site relay alive between client runs to hold its last target."""
    directory = Path("outputs/labs_relay")
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "flock",
        "--nonblock",
        "--conflict-exit-code",
        "75",
        str(directory / "process.lock"),
        sys.executable,
        "-m",
        "franka_duo_tele_data.labs_relay",
        "--config",
        str(args.config),
        "--dataset",
        str(args.dataset),
        "--start-episode",
        str(args.start_episode),
        "--publish",
        "--enable-robot",
        "--hold-current-on-start",
    ]
    if args.ik:
        command.extend(["--ik", str(args.ik)])
    with (directory / "relay.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    (directory / "launcher.pid").write_text(str(process.pid) + "\n")
    print(f"Labs relay started; log: {directory / 'relay.log'}; remains running to hold the endpoint.")
    return process


def run(args):
    if args.publish != args.enable_robot or args.capture_only and args.publish:
        raise ValueError("Robot publication requires both gates and cannot be capture-only")
    contract = LabsContract(args.config, args.dataset)
    start_state = episode_start(args.dataset, args.start_episode, contract) if args.restore else None
    if args.restore and not args.infer and not args.publish:
        return restore_preview(args, contract, start_state)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String

    cache = LabsObservationCache(contract)
    rclpy.init()
    node = rclpy.create_node("labs_policy_client")
    status_lock = threading.Lock()
    latest = {"value": None, "received": 0.0, "error": None}
    thread = None
    try:
        for key, topic in TOPICS.items() if args.infer or args.capture_only else []:
            node.create_subscription(
                Image if key in CAMERAS else JointState,
                topic,
                lambda message, k=key: cache.store(k, message),
                qos_profile_sensor_data,
            )

        def on_status(message):
            with status_lock:
                try:
                    latest.update(value=json.loads(message.data), received=time.monotonic())
                except Exception as exc:
                    latest["error"] = exc

        publisher = None
        if args.publish:
            node.create_subscription(String, STATUS_TOPIC, on_status, 10)
            publisher = node.create_publisher(String, COMMAND_TOPIC, 1)
        thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        thread.start()

        def relay_status():
            if node.count_publishers(COMMAND_TOPIC) != 1 or publisher.get_subscription_count() != 1:
                raise RuntimeError("Expected one client publisher and one Labs relay subscriber")
            with status_lock:
                if latest["error"]:
                    raise RuntimeError("Invalid relay status") from latest["error"]
                if latest["value"] is None or time.monotonic() - latest["received"] > 0.5:
                    raise TimeoutError("Labs relay status missing or stale")
                return latest["value"].copy()

        async def wait_hold(command_id=None, timeout=10):
            reason = "no completion acknowledgment"
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                value = relay_status()
                if require_relay(value, contract, command_id=command_id):
                    return value
                reason = value.get("reason") or value.get("phase")
                await asyncio.sleep(0.02)
            raise TimeoutError(f"Labs relay not ready/completed: {reason}")

        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "trace.jsonl").open("x") as log:

            def record(value):
                log.write(json.dumps(value, allow_nan=False) + "\n")
                log.flush()

            async def session():
                if publisher is not None:
                    # Allow DDS discovery only before the first command.
                    await asyncio.sleep(1)
                    if args.manage_relay and publisher.get_subscription_count() == 0:
                        process = launch_relay(args)
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            if process.poll() not in (None, 75):
                                raise RuntimeError("Relay exited; inspect outputs/labs_relay/relay.log")
                            with status_lock:
                                received = latest["value"] is not None
                            if received and publisher.get_subscription_count() == 1:
                                break
                            await asyncio.sleep(0.1)
                    await wait_hold()
                after_ns = time.time_ns()
                client = None
                try:
                    if args.infer:
                        client = await LabsClient(
                            args.url, contract, args.timeout, server_profile=args.server_profile
                        ).__aenter__()
                        record(
                            {
                                "event": "server_health",
                                "health": client.health,
                                "client_chunk_reference": CHUNK_REFERENCE,
                                "server_profile": args.server_profile,
                                "execution_rate_hz": 30 * args.speed,
                                "reference_authority": "user-confirmed Labs client contract",
                            }
                        )
                    if args.restore:
                        reset = start_command(start_state, contract, args.start_episode, speed=args.speed)
                        record(
                            {
                                "event": "episode_start",
                                "dataset": str(args.dataset),
                                "episode": args.start_episode,
                                "frame_index": 0,
                                "state34": start_state.tolist(),
                                "identity": reset["start_identity"],
                            }
                        )
                        await return_before_inference(
                            reset,
                            publish=(
                                lambda command: publisher.publish(
                                    String(data=json.dumps(command, allow_nan=False))
                                )
                            )
                            if publisher
                            else None,
                            wait_hold=wait_hold,
                            record=record,
                        )
                        print(
                            json.dumps(
                                {
                                    "restore": "completed" if args.publish else "dry-run",
                                    "episode": args.start_episode,
                                    "published": args.publish,
                                }
                            )
                        )
                        after_ns = time.time_ns()
                    if not args.infer and not args.capture_only:
                        return
                    for index in range(args.max_chunks):
                        deadline = time.monotonic() + 5
                        while True:
                            observation = cache.next(
                                after_ns=after_ns, timeout_s=max(0, deadline - time.monotonic())
                            )
                            images = encode_images(observation.images)
                            if time.time_ns() - observation.stamp_ns <= 200_000_000:
                                break
                            # Cold decoder/encoder imports can age the first frame.
                            # Discard it and capture again, without relaxing freshness.
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Observation expired during image encoding")
                        frame_dir = args.output / f"observation_{index:06d}"
                        frame_dir.mkdir()
                        (frame_dir / "state.json").write_text(json.dumps(observation.state.tolist()))
                        for name, data in images.items():
                            (frame_dir / f"{name}.png").write_bytes(data)
                        record(
                            {
                                "event": "observation",
                                "index": index,
                                "state": observation.state.tolist(),
                                "stamp_ns": observation.stamp_ns,
                                "source_stamps_ns": observation.source_stamps_ns,
                                "model_hashes": contract.model_hashes,
                                "task": contract.task,
                                "image_directory": frame_dir.name,
                            }
                        )
                        if args.capture_only:
                            print(json.dumps({"capture": "ok", "state_dim": 34, "images": [640, 480, 3]}))
                            break
                        started = time.monotonic()
                        # Retain the raw reply even when validation rejects it.
                        import msgpack

                        client.response_sink = lambda value, folder=frame_dir: (
                            folder / "response.msgpack"
                        ).write_bytes(msgpack.packb(value, use_bin_type=True))
                        result = await client.infer(observation.state, images)
                        record(
                            {
                                "event": "raw_response",
                                "response": result,
                                "round_trip_ms": (time.monotonic() - started) * 1000,
                            }
                        )
                        command = command_for(result, observation, contract, speed=args.speed)
                        record(
                            {
                                "event": "inference",
                                "response": result,
                                "command": command,
                                "round_trip_ms": (time.monotonic() - started) * 1000,
                                "published": False,
                            }
                        )
                        if publisher is not None:
                            if not require_relay(relay_status(), contract):
                                raise RuntimeError("Relay stopped holding during inference")
                            # The relay independently checks age, joint drift, limits and IK.
                            publisher.publish(String(data=json.dumps(command, allow_nan=False)))
                            record({"event": "published", "command_id": command["command_id"]})
                            final_status = await wait_hold(command["command_id"], timeout=120)
                            record(
                                {
                                    "event": "completed",
                                    "command_id": command["command_id"],
                                    "status": final_status,
                                }
                            )
                        print(
                            json.dumps(
                                {
                                    "chunk": index,
                                    "rows": len(command["targets"]),
                                    "execution_rate_hz": command["execution_rate_hz"],
                                    "published": args.publish,
                                    "chunk_reference": CHUNK_REFERENCE,
                                }
                            )
                        )
                        after_ns = time.time_ns()  # Discard delayed frames from before completion.
                finally:
                    if client is not None:
                        await client.__aexit__(None, None, None)

            try:
                asyncio.run(session())
            except BaseException as exc:
                record({"event": "stopped", "error": str(exc), "type": type(exc).__name__})
                raise
    finally:
        rclpy.shutdown()
        if thread is not None:
            thread.join(timeout=2)
        node.destroy_node()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="Return to the dataset episode start")
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Infer at the current pose; combine with --restore to return first",
    )
    parser.add_argument(
        "--manage-relay",
        action="store_true",
        help="Start a persistent site relay if none is discovered (robot gates required)",
    )
    parser.add_argument("--ik", type=Path, help="Host-built labs_fr3_ik executable for a new relay")
    parser.add_argument("--url", help="Existing fastwam.msgpack.v1 WebSocket /infer endpoint")
    parser.add_argument("--server-profile", choices=("labs", "c23"), default="labs")
    parser.add_argument("--config", type=Path, default=Path("configs/labs_fr3_31"))
    parser.add_argument("--dataset", type=Path, help="Labs dataset containing the measured episode start")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(f"outputs/labs_client_{time.time_ns()}"))
    parser.add_argument("--capture-only", action="store_true", help="Validate live input without a server")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Policy/return time scale (default 0.3: 9 Hz policy reference, slower joint return)",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    args = parser.parse_args(argv)
    if args.publish != args.enable_robot:
        parser.error("Robot publication requires both --publish and --enable-robot")
    if args.capture_only and (args.restore or args.infer):
        parser.error("--capture-only cannot combine with --restore/--infer")
    # Preserve the older run_labs_client.sh behavior when no operation was selected.
    if not args.capture_only and not (args.restore or args.infer):
        args.restore = args.infer = True
    if args.capture_only and args.publish:
        parser.error("--capture-only cannot publish")
    if args.infer and (not args.url or not args.url.startswith(("ws://", "wss://"))):
        parser.error("--url ws://.../infer or wss://.../infer is required")
    if not args.capture_only and args.dataset is None:
        parser.error("--dataset is required for the Labs contract and episode-start target")
    if args.start_episode < 0:
        parser.error("--start-episode must be nonnegative")
    if not 0 < args.speed <= 1 or not np.isfinite(args.timeout) or args.timeout <= 0 or args.max_chunks < 1:
        parser.error("Require 0 < speed <= 1, positive timeout and max-chunks")
    return args


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
